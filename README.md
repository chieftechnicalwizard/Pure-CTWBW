# Pure-CTWBW: Refining Feed-forward 3D Reconstruction with Unlabeled Post-training (NVIDIA Blackwell Edition)

<p align="center">
  <a href="https://arxiv.org/abs/2610.14698"><img src="https://img.shields.io/badge/arXiv-2610.14698-b31b1b.svg" alt="arXiv Paper"></a> &nbsp;
  <a href="https://funnyguy525.github.io/PURE-Page/"><img src="https://img.shields.io/badge/Project-Page-Green" alt="Project Page"></a> &nbsp;
  <a href="https://huggingface.co/cyun9286/PURE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Official%20Weights-blue" alt="Hugging Face Weights"></a> &nbsp;
  <a href="https://huggingface.co/spaces/cyun9286/PURE-Studio"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Studio-Try%20the%20Demo-orange" alt="Open PURE Studio"></a> &nbsp;
  <a href="docs/BLACKWELL_OPTIMIZATIONS.md"><img src="https://img.shields.io/badge/NVIDIA%20Blackwell-RTX%205070%20Ti%20%26%20DGX%20Spark%20GB10-76b900" alt="Blackwell Optimizations"></a>
</p>

> **Pure-CTWBW** (*ChiefTechnicalWizard Blackwell Edition*) is an optimized fork of [**IGL-HKUST/PURE**](https://github.com/IGL-HKUST/PURE) (*ACM Multimedia 2026*) engineered for **NVIDIA Blackwell** architectures—specifically **GeForce RTX 5070 Ti (`16 GB GDDR7`, `x86_64`)** and **NVIDIA DGX Spark Grace Blackwell `GB10` (`128 GB Unified Memory`, `ARM64`)**—with **Split-Precision `bfloat16`/`float32` Weight Residency**, **FlashAttention-2 SDPA $O(N)$ scaling**, **inter-stage CUDA virtual page reclamation**, and **100% air-gapped offline execution**.
>
> *No model weights are mirrored in this repository; all weights are downloaded directly from the official upstream Hugging Face repositories ([`cyun9286/PURE`](https://huggingface.co/cyun9286/PURE), [`FunnyGuy525/RefinedModel`](https://huggingface.co/FunnyGuy525/RefinedModel), and [`nvidia/segformer-b2-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)).*

---

## ⚡ NVIDIA Blackwell Optimizations (`RTX 5070 Ti 16 GB` & `DGX Spark GB10 128 GB`)

<p align="center">
  <img src="assets/blackwell_vram_benchmark.png" alt="Pure-CTWBW Blackwell 48-Frame VRAM and Latency Benchmark" width="100%">
</p>

<p align="center">
  <img src="assets/blackwell_architecture_pipeline.png" alt="Pure-CTWBW Split-Precision Blackwell Architecture and Dual-Node Deployment" width="100%">
</p>

### Key Enhancements Over Upstream (`0.0` Max Numerical Difference)

1. **Split-Precision `bfloat16`/`float32` Weight Residency (`pure/pipeline.py`):**
   - Upstream PURE loads all `989M–1.21B` parameters in `float32` (`3.45–4.58 GiB`) and dynamically casts backbone weights to `bfloat16` inside `torch.autocast`, holding both `float32` master weights and temporary `bfloat16` cast buffers in VRAM simultaneously.
   - Because all three adapters (`pi3`, `da3`, `vggt`) wrap their final camera-pose and depth-refiner heads (`DualDPT`, `CameraDec`, `DepthRefiner`, `CameraHead`, `DPTHead`) in `with torch.autocast(enabled=False):` using `.float()` tensors for sub-millimeter $SE(3)$ pose accuracy, `_apply_blackwell_split_bf16()` pre-casts `nn.Linear`/`nn.Conv2d`/`nn.ConvTranspose2d` layers **only inside the BF16-autocast vision & alternating-attention backbones** while keeping `nn.LayerNorm` and all `autocast(enabled=False)` pose/depth heads in exact `float32`.
   - **Result:** Cuts static weight VRAM by **39%–48%**, slashes 48-frame peak GPU VRAM by **19.4%–37.4%** (saving **`-3.79 GiB`** on `da3-Giant`), and achieves **`max_abs_diff = 0.000000e+00` (bit-for-bit identical outputs)** against upstream!
2. **Full 48-Frame Studio & `64–96+` Frame CLI on `16 GB` GeForce RTX 5070 Ti:**
   - Upstream documentation recommends a **24 GB GPU (A10G)** for 48-frame reconstructions.
   - With Split-Precision weight residency + Blackwell FlashAttention-2 SDPA (`O(N)` memory scaling) + inter-stage `torch.cuda.empty_cache()` before `SegFormer-B2` sky segmentation, **48-frame peak VRAM drops to just `5.26 GiB` (`vggt`), `6.34 GiB` (`da3`), and `6.87 GiB` (`pi3`)**—fitting effortlessly on a **16 GB GeForce RTX 5070 Ti** even when sharing the GPU with other services, and unlocking **`64–96+` frame CLI reconstructions** when dedicated!
3. **Inference Checkpoint Bypass (`pure/models/pi3/layers/transformer_head.py`):**
   - Skips unconditional `torch.utils.checkpoint.checkpoint` calls in `pi3` upsample blocks during `torch.inference_mode()`, reducing 48-frame latency by **6.8%–8.3%**.
4. **100% Air-Gapped / Offline Execution (`scripts/hub/stage_offline_weights.sh`):**
   - `SkySegmenter` (`pure/sky.py`) and `scripts/demo/reconstruct.py` natively resolve `$PURE_MODEL_ROOT` and `$PURE_SKY_MODEL_DIR` across CLI, Studio Web UI, and Python API calls with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

### Measured 48-Frame Benchmark Table (`518 × 294`, `7,308,956` Exported 3D Points)

| Model (`48 Frames`) | Params | Upstream Static VRAM | **CTWBW Static VRAM** | Upstream 48f Peak VRAM | **CTWBW 48f Peak VRAM** | **VRAM Saved** | Upstream Latency (`GB10`) | **CTWBW Latency (`GB10`)** | Max Abs Diff |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`vggt`** | `1.16B` | `4.35 GiB` | **`2.65 GiB` (-39%)** | `6.94 GiB` | **`5.26 GiB`** | **`-1.68 GiB` (-24.2%)** | `5.93 s` | **`5.44 s` (-8.3%)** | `0.000000e+00` |
| **`da3`** | `1.21B` | `4.58 GiB` | **`2.42 GiB` (-47%)** | `10.13 GiB` | **`6.34 GiB`** | **`-3.79 GiB` (-37.4%)** | `6.48 s` | **`5.94 s` (-8.3%)** | `0.000000e+00` |
| **`pi3`** | `989M` | `3.45 GiB` | **`1.79 GiB` (-48%)** | `8.52 GiB` | **`6.87 GiB`** | **`-1.65 GiB` (-19.4%)** | `8.81 s` | **`8.21 s` (-6.8%)** | `0.000000e+00` |

See the full architectural deep dive in [**`docs/BLACKWELL_OPTIMIZATIONS.md`**](docs/BLACKWELL_OPTIMIZATIONS.md).

---

## Original Authors & Paper (`ACM MM 2026`)

<div align="center">

<p>
  Yaolong Zhu<sup>1</sup>,
  <a href="https://penghtyx.github.io/yuki-lipeng/">Peng Li</a><sup>2</sup>,
  <a href="https://github.com/jiah-cloud">Jiahao Lu</a><sup>2,▲</sup>,
  <a href="https://afterjourney00.github.io/">Chengfeng Zhao</a><sup>2</sup>,
  Mengfei Li<sup>2</sup>,
  Chao Wang<sup>3</sup>,
  Wenyue Chen<sup>1</sup>,<br>
  Zekai Gu<sup>2</sup>,
  Chengwei Ren<sup>2</sup>,
  <a href="https://www.ece.pku.edu.cn/info/1046/2147.htm">Ronggang Wang</a><sup>1,✉</sup>,
  <a href="https://liuyuan-pal.github.io/">Yuan Liu</a><sup>2,✉</sup>
</p>

<p>
  <sup>1</sup> Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology, Shenzhen Graduate School, Peking University<br>
  <sup>2</sup> The Hong Kong University of Science and Technology<br>
  <sup>3</sup> Pengcheng Laboratory<br>
  <sup>▲</sup> Project lead &nbsp;&nbsp;·&nbsp;&nbsp; <sup>✉</sup> Co-corresponding authors
</p>

</div>

<p align="center">
  <b>Reconstruct 3D points, depth and camera poses from video or images with Pi3, DA3 Giant and VGGT.</b>
</p>

<p align="center">
  <a href="https://funnyguy525.github.io/PURE-Page/#long"><img src="assets/teaser-15fps-v2.gif" alt="PURE 15 fps preview: growing overhead map expands into the complete 3D reconstruction" width="100%"></a>
</p>

## Pipeline

<p align="center">
  <img src="assets/pipeline.jpg" alt="PURE post-training pipeline" width="100%">
</p>

---

## 🚀 Quickstart (Online & 100% Air-Gapped Offline)

### 1. Official Model Weight Sources (Not Mirrored in Git)
All model weights are hosted on the official Hugging Face Hub repositories:
* **Refined PURE Backbones (`pi3`, `da3`, `vggt` — Safetensors):** <https://huggingface.co/cyun9286/PURE>
* **Original Pinned Checkpoints (`pi3.bin`, `da3.bin`, `vggt.bin`):** <https://huggingface.co/FunnyGuy525/RefinedModel>
* **Semantic Sky Segmenter (`SegFormer-B2 ADE20K`):** <https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512>

### 2. Stage Weights for 100% Offline / Air-Gapped Execution
```bash
# Downloads pi3, da3, vggt + SegFormer-B2 from Hugging Face and verifies SHA256 checksums:
bash scripts/hub/stage_offline_weights.sh ./offline_weights

export PURE_MODEL_ROOT="$PWD/offline_weights/models"
export PURE_SKY_MODEL_DIR="$PWD/offline_weights/sky_segformer_b2"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

### 3. Launch PURE Reconstruction Studio (Web App)
```bash
pip install -e '.[space]'
PORT=7875 python app.py
```
Open **`http://localhost:7875`** and upload a video or image sequence (2–48 frames). Outputs interactive 3D `.glb` preview, sky-filtered `.ply` point cloud, `cameras.npz` poses, RGB/depth `.mp4` videos, and a complete `.zip` archive.

### 4. Command-Line Reconstruction
```bash
pip install -e '.[inference]'

# Using staged local weights ($PURE_MODEL_ROOT and $PURE_SKY_MODEL_DIR) or Hugging Face ID:
python scripts/demo/reconstruct.py \
  --pretrained cyun9286/PURE --subfolder da3 \
  --video input.mp4 --frames 48 --output outputs/my-video
```

### 5. Python API
```python
import torch
from pure import PURE, PUREImageProcessor
from pure.pipeline import ReconstructionPipeline

model = PURE.from_pretrained("cyun9286/PURE", subfolder="da3")
pipeline = ReconstructionPipeline(model, device="cuda")  # Automatically applies Blackwell Split-BF16/FP32
images, meta = PUREImageProcessor().video("input.mp4", max_frames=48, return_metadata=True)
result = pipeline(images, meta, "outputs/api-run", precision="bf16")
```

---

## Training

Prepare aligned teacher caches using the [training guide](docs/TRAINING_INFERENCE_ZH.md) and [data format](docs/DATA_FORMAT.md), then run:

```bash
pip install -e '.[training]'
python scripts/hub/download_weights.py --model da3 --output checkpoints
bash scripts/train/run_train.sh \
  --config configs/da3_posttrain.json --data data/da3_train \
  --checkpoint checkpoints/da3.bin --output outputs/train-da3
```

---

## Citation

```bibtex
@inproceedings{zhu2026pure,
  author    = {Zhu, Yaolong and Li, Peng and Lu, Jiahao and Zhao, Chengfeng and Li, Mengfei and Wang, Chao and Chen, Wenyue and Gu, Zekai and Ren, Chengwei and Wang, Ronggang and Liu, Yuan},
  title     = {{PURE}: Refining Feed-forward {3D} Reconstruction with Unlabeled Post-training},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  series    = {MM '26},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3767308.3836296},
  url       = {https://doi.org/10.1145/3767308.3836296}
}
```

## Acknowledgements

PURE builds on [VGGT](https://github.com/facebookresearch/vggt), [Pi3](https://github.com/yyfz/Pi3), and [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3). Pseudo supervision uses [MoGe](https://github.com/microsoft/MoGe), [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2), and [GIM](https://github.com/xuelunshen/gim). The scale-and-shift solver follows [Track4World](https://github.com/TencentARC/Track4World).

---

## 💖 Sponsors & Support

Sponsorship and donations support Shekinah Shine maintenance, sanctuary groundskeeping, hardware R&D, and open-source development:

* **PayPal Donate (Shekinah Shine):** [Donate via PayPal](https://www.paypal.com/donate/?business=DQWR2UNVNWPQU&no_recurring=0&item_name=SHEKINAH+SHINE+MAINTENANCE%2C+GARDEN%2C+AND+GROUNDSKEEPING&currency_code=USD)
* **Cash App:** [`$chieftechnicalwizard`](https://cash.app/$chieftechnicalwizard)
* **Venmo:** [`@Christopher-CTW`](https://venmo.com/u/Christopher-CTW)
* **GitHub Sponsors:** [github.com/sponsors/chieftechnicalwizard](https://github.com/sponsors/chieftechnicalwizard)
* **Website:** [shekinahshine.org](https://www.shekinahshine.org/)
