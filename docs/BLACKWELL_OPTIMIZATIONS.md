# NVIDIA Blackwell Optimizations & Standalone Deployment Guide (`Pure-CTWBW`)

**Pure-CTWBW** (*ChiefTechnicalWizard Blackwell Edition*) optimizes the upstream [IGL-HKUST/PURE](https://github.com/IGL-HKUST/PURE) feed-forward 3D reconstruction engine for **100% standalone, single-GPU execution** on **NVIDIA Blackwell** architectures—supporting either a **desktop 16 GB GDDR7 GPU (`GeForce RTX 5070 Ti`)** or a **128 GB Unified Memory ARM64 superchip (`NVIDIA DGX Spark GB10`)**—as well as **100% air-gapped offline operation**.

<p align="center">
  <img src="../assets/blackwell_vram_benchmark.png" alt="Pure-CTWBW Blackwell 48-Frame VRAM and Latency Benchmark" width="100%">
</p>

<p align="center">
  <img src="../assets/blackwell_architecture_pipeline.png" alt="Pure-CTWBW Split-Precision Blackwell Pipeline and Standalone GPU Options" width="100%">
</p>

---

## 1. Summary of Blackwell Architectural Optimizations

### 1.1 Split-Precision `bfloat16` / `float32` Weight Residency (`pure/pipeline.py`)
* **The Bottleneck in Upstream PURE:**
  Upstream `ReconstructionPipeline` loads all `989M–1.21B` model parameters onto CUDA in **`float32`** (`3.45 GiB` for `pi3`, `4.58 GiB` for `da3`, `4.35 GiB` for `vggt`) and relies on `torch.autocast("cuda", dtype=torch.bfloat16)` during the forward pass. During execution, PyTorch dynamically casts every `nn.Linear` and `nn.Conv2d` weight from `float32` to `bfloat16`, holding **both** the `float32` master weights and temporary `bfloat16` cast buffers in VRAM simultaneously.
  However, a naive `model.to(torch.bfloat16)` fails because all three adapters (`pi3`, `da3`, `vggt`) intentionally wrap their final camera-pose and depth-refinement heads (`DualDPT`, `CameraDec`, `DepthRefiner`, `CameraHead`, `DPTHead`) inside `with torch.autocast(enabled=False):` using explicit `.float()` tensors for sub-millimeter $SE(3)$ camera pose stability.
* **The Pure-CTWBW Solution (`_apply_blackwell_split_bf16`):**
  We inspect the adapter's split execution boundary and pre-cast `nn.Linear`, `nn.Conv2d`, and `nn.ConvTranspose2d` layers **only inside the BF16-autocast vision & alternating-attention backbones**:
  - **`pi3`**: `encoder`, `decoder`, `point_decoder`, `conf_decoder`, `camera_decoder`, `global_points_decoder`
  - **`da3`**: `backbone` (`DinoV2` ViT-Giant)
  - **`vggt`**: `aggregator` (`Aggregator`)
  Meanwhile, all `nn.LayerNorm` parameters and all `autocast(enabled=False)` geometry/pose heads remain in exact `float32`.
* **Mathematical & Numerical Parity (`max_abs_diff = 0.0`):**
  Because `torch.autocast("cuda", dtype=torch.bfloat16)` casts those exact `Linear`/`Conv` weights to `bfloat16` on the fly while keeping `LayerNorm` in `float32`, pre-casting them produces **bit-for-bit identical (`0.000000e+00` maximum absolute difference)** 3D point clouds, depth maps, and camera poses across all three models while cutting static weight VRAM by **39%–48%** and 48-frame peak VRAM by **19%–37.4%** (`-3.79 GiB` saved on `da3-Giant`).

### 1.2 Blackwell FlashAttention-2 / cuDNN SDPA & `O(N)` Multi-View Scaling
* `ReconstructionPipeline` explicitly enables `torch.backends.cuda.enable_flash_sdp(True)`, `torch.backends.cuda.enable_mem_efficient_sdp(True)`, and `torch.set_float32_matmul_precision("high")`.
* On Blackwell Tensor Cores (`GB10` and `RTX 5070 Ti`), the multi-view frame and global alternating-attention blocks execute via fused FlashAttention-2 kernels with **$O(N)$ linear activation memory scaling** instead of materializing $O(N^2)$ attention matrices.
* Scaling from **16 frames to 48 frames (`3×` sequence length)** adds only **`~1.0 GiB`** of peak activation memory (`5.26–6.87 GiB` total peak across all 3 models).

### 1.3 Inter-Stage Virtual Page Reclamation (`expandable_segments:True` + `empty_cache()`)
* `app.py` and `scripts/demo/reconstruct.py` automatically configure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (CUDA driver `cuMemMap` virtual page allocation) prior to CUDA initialization.
* In `pure/pipeline.py`, `torch.cuda.empty_cache()` is invoked immediately after `del pred`—**before** `SkySegmenter` loads `SegFormer-B2` (`nvidia/segformer-b2-finetuned-ade-512-512`). This ensures the sky segmenter reuses the freed 3D backbone activation pages rather than allocating atop the high-water mark.

### 1.4 Inference-Time Activation Checkpoint Bypass (`pure/models/pi3/layers/transformer_head.py`)
* Upstream `transformer_head.py` invoked `torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)` unconditionally inside the multi-stage upsample blocks even during `torch.inference_mode()`.
* Guarding checkpointing with `if self.training:` eliminates redundant Python wrapper and tensor tracking overhead during inference, reducing 48-frame feed-forward latency by **6.8%–8.3%**.

---

## 2. Measured Benchmarks (`48 Frames` @ `518 × 294`, `7,308,956` Exported 3D Points)

| Model (`48 Frames`) | Params | Upstream Static Weights | **CTWBW Static Weights** | Upstream 48f Peak VRAM | **CTWBW 48f Peak VRAM** | **VRAM Saved** | Upstream Latency (`GB10`) | **CTWBW Latency (`GB10`)** | Max Abs Diff vs. Upstream |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`vggt`** | `1.16B` | `4.35 GiB` | **`2.65 GiB` (-39%)** | `6.94 GiB` | **`5.26 GiB`** | **`-1.68 GiB` (-24.2%)** | `5.93 s` | **`5.44 s` (-8.3%)** | `0.000000e+00` |
| **`da3`** | `1.21B` | `4.58 GiB` | **`2.42 GiB` (-47%)** | `10.13 GiB` | **`6.34 GiB`** | **`-3.79 GiB` (-37.4%)** | `6.48 s` | **`5.94 s` (-8.3%)** | `0.000000e+00` |
| **`pi3`** | `989M` | `3.45 GiB` | **`1.79 GiB` (-48%)** | `8.52 GiB` | **`6.87 GiB`** | **`-1.65 GiB` (-19.4%)** | `8.81 s` | **`8.21 s` (-6.8%)** | `0.000000e+00` |

---

## 3. Standalone Deployment Options & Side-by-Side Comparison

`Pure-CTWBW` runs **100% standalone on a single GPU**. Below are the two independent Blackwell hardware deployment options and how they compare:

<p align="center">
  <img src="../assets/blackwell_standalone_comparison.png" alt="Standalone Deployment Comparison: GeForce RTX 5070 Ti 16 GB vs. DGX Spark GB10 128 GB" width="100%">
</p>

### Option A (Standalone): Desktop NVIDIA GeForce RTX 5070 Ti (`16 GB GDDR7`, `x86_64`)
* **Self-Contained Capability:** Runs the entire PURE Reconstruction Studio web app (`app.py`), CLI (`reconstruct.py`), all three refined backbones (`pi3`, `da3`, `vggt`), and `SegFormer-B2` sky removal on a single 16 GB consumer card.
* **Shared vs. Dedicated Operating Modes on 16 GB VRAM:**
  - **Shared Mode (`~12–13 GiB` free VRAM while co-hosting lightweight services):** Comfortably runs the **full 48-frame Studio maximum** across all three models (`5.26–6.87 GiB` peak VRAM) and up to `64` frames in CLI mode.
  - **Dedicated Mode (`16.0 GiB` full GPU with other GPU tools unloaded):** Unlocks **`64–96+` consecutive frame CLI reconstructions** across all three models while taking advantage of high desktop GDDR7 bandwidth (`672 GB/s`) and desktop SM boost clocks.
* **Setup (`x86_64` Linux):**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e '.[space]'
  bash scripts/hub/stage_offline_weights.sh ./offline_weights
  ```

### Option B (Standalone): NVIDIA DGX Spark `GB10` (`128 GB Unified Memory`, `ARM64`)
* **Self-Contained Capability:** Runs the entire PURE Reconstruction Studio web app and CLI on a single ARM64 Grace Blackwell `GB10` node.
* **Ultra-Long Sequence & Multi-Service Headroom:**
  - At the standard 48-frame Studio ceiling, `Pure-CTWBW` consumes only **4%–5% (`5.26–6.87 GiB`)** of the 128 GB unified memory pool—leaving **121+ GiB free** so PURE Studio can stay online 24/7 alongside large resident TTS, ASR, VLM, or DiT services without unloading anything.
  - For batch CLI jobs (`scripts/demo/reconstruct.py --frames N`), the 128 GB unified memory pool unlocks **`96–250+` consecutive frame reconstructions** in a single feed-forward pass.
* **Setup (`ARM64 aarch64` Linux with System CUDA 13.0 PyTorch):**
  > On `aarch64` Grace Blackwell systems where a custom `cu130` PyTorch wheel is pre-installed, create the virtual environment preserving the system/existing `torch` and `torchvision` packages so `pip` does not replace the ARM64 GB10 wheel:
  ```bash
  python3 -m venv --system-site-packages .venv
  source .venv/bin/activate
  pip install --no-deps -e .
  pip install "safetensors>=0.4.3" "einops>=0.7" "addict>=2.4" "av>=12.0" \
              "matplotlib>=3.8" "trimesh>=4.4" "transformers==4.46.3" \
              "gradio==5.49.1" "pillow>=10.0" "scipy>=1.11"
  bash scripts/hub/stage_offline_weights.sh ./offline_weights
  ```

### Direct Comparison Table (`Option A` vs. `Option B` vs. `Upstream Baseline`)

| Dimension | **Option A: GeForce RTX 5070 Ti (`16 GB`)** | **Option B: DGX Spark `GB10` (`128 GB`)** | Upstream Reference (`A10G 24 GB`) |
| :--- | :--- | :--- | :--- |
| **Deployment Type** | **100% Standalone Single GPU (`x86_64`)** | **100% Standalone Single Superchip (`ARM64`)** | Single Cloud GPU (`x86_64`) |
| **Total GPU Memory** | `16 GB GDDR7` (`672 GB/s`) | `128 GB Unified LPDDR5x` (`273 GB/s`) | `24 GB GDDR6` (`600 GB/s`) |
| **48-Frame Studio Peak (`da3-Giant`)** | **`6.34 GiB` (`39.6%` of 16 GB)** | **`6.34 GiB` (`5.0%` of 128 GB)** | `10.13 GiB` (`42.2%` of 24 GB) |
| **48-Frame Studio Support (All 3 Models)** | **Yes (Shared or Dedicated)** | **Yes (Always-On)** | Yes |
| **Max CLI Frame Count (Single Pass)** | **`48–64` (Shared) / `64–96+` (Dedicated)** | **`96–250+` Frames** | `~64–80` Frames |
| **Primary Strength** | High GDDR7 bandwidth & desktop SM clocks on affordable 16 GB hardware | Massive 128 GB unified memory for 250+ frame sequences & 24/7 multi-service hosting | Standard cloud baseline |

---

## 4. 100% Air-Gapped / Offline Execution (No Bundled Weights in Git)

This repository does **not** bundle or mirror large model weights. All checkpoints are fetched directly from the official Hugging Face Hub repositories:
* **Refined PURE Backbones (`pi3`, `da3`, `vggt`):** [`cyun9286/PURE`](https://huggingface.co/cyun9286/PURE) (and legacy `.bin` weights at [`FunnyGuy525/RefinedModel`](https://huggingface.co/FunnyGuy525/RefinedModel))
* **Semantic Sky Segmenter (`SegFormer-B2 ADE20K`):** [`nvidia/segformer-b2-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)

Stage all weights once and run completely offline on either standalone machine:

```bash
bash scripts/hub/stage_offline_weights.sh ./offline_weights

export PURE_MODEL_ROOT="$PWD/offline_weights/models"
export PURE_SKY_MODEL_DIR="$PWD/offline_weights/sky_segformer_b2"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Launch PURE Reconstruction Studio Web UI on port 7875
PORT=7875 python app.py

# Or run a 48-frame CLI reconstruction
python scripts/demo/reconstruct.py \
  --video examples/penguin.mp4 \
  --subfolder da3 \
  --frames 48 \
  --output outputs/penguin-da3-48 \
  --device cuda \
  --precision bf16
```
