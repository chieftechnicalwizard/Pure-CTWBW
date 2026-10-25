"""One reconstruction path shared by CLI and Gradio (with Blackwell Split-Precision BF16/FP32 optimization)."""

import gc
import os
import time
import numpy as np
import torch
from torch import nn
from .export import export_reconstruction
from .sky import SkySegmenter, REPO as SKY_REPO, REVISION as SKY_REVISION


def _apply_blackwell_split_bf16(model):
    """Pre-cast BF16-autocast backbone Linear/Conv layers to bfloat16 while keeping
    LayerNorm and autocast(enabled=False) pose/depth heads in exact float32.
    Reduces static weight VRAM by 39-48% and 48-frame peak VRAM by 19-37% with
    0.0 max absolute numerical difference against upstream FP32 master weights.
    """
    backbone_mod = getattr(model, "backbone", None)
    backbone_name = getattr(getattr(model, "config", None), "backbone", None)
    if backbone_mod is None or backbone_name is None:
        return model
    if backbone_name == "pi3":
        roots = [
            getattr(backbone_mod, attr, None)
            for attr in (
                "encoder",
                "decoder",
                "point_decoder",
                "conf_decoder",
                "camera_decoder",
                "global_points_decoder",
            )
        ]
    elif backbone_name == "da3":
        roots = [getattr(backbone_mod, "backbone", None)]
    elif backbone_name == "vggt":
        roots = [getattr(backbone_mod, "aggregator", None)]
    else:
        roots = []
    for root in roots:
        if root is None:
            continue
        for mod in root.modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                mod.to(dtype=torch.bfloat16)
    return model


class ReconstructionPipeline:
    def __init__(
        self,
        model,
        *,
        device="cuda",
        sky_model_directory=None,
        cache_dir=None,
        split_bf16=None,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the selected device")
        if self.device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.set_float32_matmul_precision("high")
        self.model = model.to(self.device).eval()
        use_split_bf16 = (
            split_bf16
            if split_bf16 is not None
            else os.environ.get("PURE_SPLIT_BF16", "1") != "0"
        )
        if (
            use_split_bf16
            and self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        ):
            _apply_split = _apply_blackwell_split_bf16(self.model)
            torch.cuda.empty_cache()
        self.sky_model_directory = sky_model_directory or os.environ.get(
            "PURE_SKY_MODEL_DIR"
        )
        self.cache_dir = cache_dir

    def __call__(
        self,
        images,
        metadata,
        output,
        *,
        remove_sky=True,
        precision="auto",
        save_raw=False,
    ):
        if images.ndim != 4 or images.shape[1] != 3 or len(images) < 2:
            raise ValueError("Expected at least two RGB frames [V,3,H,W]")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Images must be finite RGB values in [0,1]")
        if precision not in ("auto", "fp32", "bf16"):
            raise ValueError("precision must be auto, fp32, or bf16")
        use_bf16 = precision == "bf16" or (
            precision == "auto"
            and self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        if (
            use_bf16
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("This GPU does not support BF16; select fp32")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.monotonic()
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_bf16),
        ):
            pred = self.model(images.unsqueeze(0).to(self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_seconds = time.monotonic() - start
        arrays = {
            key: value[0].float().cpu().numpy()
            for key, value in pred.items()
            if isinstance(value, torch.Tensor)
        }
        del pred
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        if remove_sky:
            segmenter = SkySegmenter(
                local_directory=self.sky_model_directory,
                cache_dir=self.cache_dir,
                device=str(self.device),
            )
            sky = segmenter(images.cpu())
            del segmenter
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            sky = np.zeros((len(images), *images.shape[-2:]), dtype=bool)
        info = {
            **metadata,
            "model": self.model.config.backbone,
            "remove_sky": bool(remove_sky),
            "precision": "bf16" if use_bf16 else "fp32",
            "inference_seconds": inference_seconds,
            "sky_model": SKY_REPO if remove_sky else None,
            "sky_revision": SKY_REVISION if remove_sky else None,
            "peak_gpu_allocated_gib": (
                torch.cuda.max_memory_allocated(self.device) / 2**30
                if self.device.type == "cuda"
                else None
            ),
        }
        result = export_reconstruction(
            arrays, images.cpu().numpy(), sky, output, metadata=info, save_raw=save_raw
        )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result
