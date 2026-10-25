# Pure-CTWBW: Refining Feed-forward 3D Reconstruction with Unlabeled Post-training (NVIDIA Blackwell Edition)

<p align="center">
  <a href="https://arxiv.org/abs/2610.14698"><img src="https://img.shields.io/badge/arXiv-2610.14698-b31b1b.svg" alt="arXiv Paper"></a> &nbsp;
  <a href="https://funnyguy525.github.io/PURE-Page/"><img src="https://img.shields.io/badge/Project-Page-Green" alt="Project Page"></a> &nbsp;
  <a href="https://huggingface.co/cyun9286/PURE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Official%20Weights-blue" alt="Hugging Face Weights"></a> &nbsp;
  <a href="https://huggingface.co/spaces/cyun9286/PURE-Studio"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Studio-Try%20the%20Demo-orange" alt="Open PURE Studio"></a> &nbsp;
  <a href="docs/BLACKWELL_OPTIMIZATIONS.md"><img src="https://img.shields.io/badge/NVIDIA%20Blackwell-RTX%205070%20Ti%20%7C%20DGX%20Spark%20GB10-76b900" alt="Blackwell Optimizations"></a> &nbsp;
  <a href="https://www.shekinahshine.org/?donate=paypal"><img src="docs/badges/paypal-donate-pill.svg" alt="Donate via PayPal" height="20"></a>
</p>

> **Pure-CTWBW** (*ChiefTechnicalWizard Blackwell Edition*) is an optimized fork of [**IGL-HKUST/PURE**](https://github.com/IGL-HKUST/PURE) (*ACM Multimedia 2026*) engineered for **NVIDIA Blackwell** hardware across three flexible deployment options:
>
> - **Option A (Standalone Desktop GPU) — `GeForce RTX 5070 Ti 16 GB GDDR7` (`x86_64`)**: Runs the complete 48-frame Studio web app, CLI, and all three backbones (`pi3`, `da3`, `vggt`) + `SegFormer-B2` sky removal 100% standalone on a single 16 GB desktop GPU (`5.26–6.87 GiB` peak VRAM—eliminating the upstream 24 GB A10G requirement).
> - **Option B (Standalone Grace Blackwell Superchip) — `NVIDIA DGX Spark GB10 128 GB Unified` (`ARM64`)**: Runs the complete 48-frame Studio web app, CLI, and ultra-long `96–250+` frame single-pass sequences 100% standalone on a single ARM64 Grace Blackwell node.
> - **Option C (Dual-Node Heterogeneous Cluster) — `Tier 1 DGX Spark GB10 (128 GB) + Tier 2 RTX 5070 Ti (16 GB)`**: Combines both nodes (`144 GB` combined memory pool) to isolate always-on 24/7 Studio serving & `96–250+` frame ultra-long jobs on Tier 1 while offloading parallel `2–96` frame CLI batches & multi-model ensembles to Tier 2.
>
> *No model weights are mirrored in this repository; all weights are downloaded directly from the official upstream Hugging Face repositories ([`cyun9286/PURE`](https://huggingface.co/cyun9286/PURE), [`FunnyGuy525/RefinedModel`](https://huggingface.co/FunnyGuy525/RefinedModel), and [`nvidia/segformer-b2-finetuned-ade-512-512`](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)).*

---

## ⚡ NVIDIA Blackwell Optimizations & Benchmarks

<p align="center">
  <img src="assets/blackwell_vram_benchmark.png" alt="Pure-CTWBW Blackwell 48-Frame VRAM and Latency Benchmark" width="100%">
</p>

<p align="center">
  <img src="assets/blackwell_architecture_pipeline.png" alt="Pure-CTWBW Split-Precision Blackwell Pipeline and Standalone GPU Options" width="100%">
</p>

### Key Enhancements Over Upstream (Up to 37.4% Less VRAM · 8.3% Faster · 100% Lossless 3D Quality)

1. **Split-Precision `bfloat16`/`float32` Weight Residency (`pure/pipeline.py`):**
   - **Why Memory Drops by 37%–48% with Zero Quality Loss:** Normally, shrinking a 3D model's VRAM footprint by 37%–48% (via quantization or naive half-precision casting) degrades point cloud sharpness and causes camera poses to drift. In upstream PURE, all `989M–1.21B` parameters are loaded into VRAM as `32-bit` (`float32`, taking `3.45–4.58 GiB`), and during every forward pass, `torch.autocast` converts the `1.0B+` vision backbone weights to `16-bit` (`bfloat16`) on the fly—wasting VRAM by holding **both** the `32-bit` master weights and temporary `16-bit` copies at the same time.
   - Meanwhile, all three adapters (`pi3`, `da3`, `vggt`) wrap their final camera-pose and depth-refiner heads (`DualDPT`, `CameraDec`, `DepthRefiner`, `CameraHead`, `DPTHead`) in `with torch.autocast(enabled=False):` using explicit `32-bit` `.float()` tensors for sub-millimeter $SE(3)$ pose accuracy.
   - Our `_apply_blackwell_split_bf16()` optimizer pre-converts **only** the `16-bit` vision & alternating-attention backbone (`nn.Linear`/`nn.Conv2d`/`nn.ConvTranspose2d`) weights on load while leaving `nn.LayerNorm` and all `32-bit` (`autocast(enabled=False)`) camera-pose and depth-refiner heads untouched.
   - **Result:** The GPU Tensor Cores execute the **exact same mathematical operations** down to the last bit—meaning all `7,308,956` reconstructed `(X, Y, Z)` 3D coordinates are **100% mathematically identical (`0.000000` coordinate drift, zero quality loss)** to the unoptimized model while cutting static weight VRAM by **39%–48%** and 48-frame peak GPU VRAM by **19.4%–37.4%** (saving **`-3.79 GiB`** on `da3-Giant`, allowing a **16 GB RTX 5070 Ti** to run workloads that originally required a **24 GB A10G**).
2. **Blackwell FlashAttention-2 SDPA (`O(N)` Multi-View Scaling) & Inter-Stage Page Reclamation:**
   - Enforces fused FlashAttention-2 / cuDNN SDPA kernels (`torch.backends.cuda.enable_flash_sdp(True)`) so multi-view attention scales linearly ($O(N)$) with frame count (going from 16 to 48 frames adds only `~1.0 GiB` of activation memory).
   - Automatically enables `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and runs `torch.cuda.empty_cache()` between the 3D backbone forward pass and the `SegFormer-B2` sky segmentation pass so their peak VRAM allocations never stack.
3. **Inference Checkpoint Bypass (`pure/models/pi3/layers/transformer_head.py`):**
   - Skips unconditional `torch.utils.checkpoint.checkpoint` calls in `pi3` upsample blocks during `torch.inference_mode()`, reducing 48-frame latency by **6.8%–8.3%**.
4. **100% Air-Gapped / Offline Execution (`scripts/hub/stage_offline_weights.sh`):**
   - `SkySegmenter` (`pure/sky.py`) and `scripts/demo/reconstruct.py` natively resolve `$PURE_MODEL_ROOT` and `$PURE_SKY_MODEL_DIR` across CLI, Studio Web UI, and Python API calls with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

### Measured 48-Frame Benchmark Table (`518 × 294`, `7,308,956` Exported 3D Points)

| Model (`48 Frames`) | Params | Upstream Static VRAM | **CTWBW Static VRAM** | Upstream 48f Peak VRAM | **CTWBW 48f Peak VRAM** | **VRAM Saved** | Upstream Latency (`GB10`) | **CTWBW Latency (`GB10`)** | **3D Output Quality** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`vggt`** | `1.16B` | `4.35 GiB` | **`2.65 GiB` (-39%)** | `6.94 GiB` | **`5.26 GiB`** | **`-1.68 GiB` (-24.2%)** | `5.93 s` | **`5.44 s` (-8.3%)** | **100% Lossless (`0.0` drift)** |
| **`da3`** | `1.21B` | `4.58 GiB` | **`2.42 GiB` (-47%)** | `10.13 GiB` | **`6.34 GiB`** | **`-3.79 GiB` (-37.4%)** | `6.48 s` | **`5.94 s` (-8.3%)** | **100% Lossless (`0.0` drift)** |
| **`pi3`** | `989M` | `3.45 GiB` | **`1.79 GiB` (-48%)** | `8.52 GiB` | **`6.87 GiB`** | **`-1.65 GiB` (-19.4%)** | `8.81 s` | **`8.21 s` (-6.8%)** | **100% Lossless (`0.0` drift)** |

---

## 🖥️ Deployment Options: Standalone Single-GPU (Options A & B) & Dual-Node Cluster (Option C)

`Pure-CTWBW` runs **100% standalone on a single GPU** (Option A or Option B), or can be deployed across **both nodes simultaneously** as a **Dual-Node Heterogeneous Blackwell Cluster** (Option C).

### 1. Standalone Deployment Comparison (`Option A: RTX 5070 Ti 16 GB` vs. `Option B: DGX Spark GB10 128 GB`)

<p align="center">
  <img src="assets/blackwell_standalone_comparison.png" alt="Standalone Deployment Comparison: GeForce RTX 5070 Ti 16 GB vs. DGX Spark GB10 128 GB" width="100%">
</p>

### 2. Multi-Node Cluster Topology (`Option C: Tier 1 DGX Spark GB10 + Tier 2 GeForce RTX 5070 Ti`)

<p align="center">
  <img src="assets/blackwell_multinode_cluster.png" alt="Option C: Dual-Node Blackwell Cluster (Tier 1 GB10 + Tier 2 RTX 5070 Ti)" width="100%">
</p>

### 3. Complete Deployment Matrix (`Options A, B, and C` vs. `Upstream Baseline`)

| Feature / Metric | **Option A (Standalone): GeForce RTX 5070 Ti (`16 GB`)** | **Option B (Standalone): NVIDIA DGX Spark `GB10` (`128 GB`)** | **Option C (Dual-Node Cluster): Tier 1 `GB10` + Tier 2 `RTX 5070 Ti`** | Upstream Reference Baseline (`A10G 24 GB`) |
| :--- | :--- | :--- | :--- | :--- |
| **System Architecture** | Single Desktop / Workstation GPU (`x86_64`) | Single Grace Blackwell Superchip (`ARM64 aarch64`) | Heterogeneous 2-Node Blackwell Cluster (`ARM64` + `x86_64`) | Single Cloud / Datacenter GPU (`x86_64`) |
| **Memory Pool & Bandwidth** | **`16 GB GDDR7`** (`~672 GB/s`) | **`128 GB Unified LPDDR5x`** (`~273 GB/s`) | **`144 GB Combined Pool`** (`128 GB Unified` + `16 GB GDDR7`) | `24 GB GDDR6` (`~600 GB/s`) |
| **48-Frame Peak Memory Used (`da3` / `pi3` / `vggt`)** | **`6.34` / `6.87` / `5.26 GiB`** (`33%–43%` of 16 GB) | **`6.34` / `6.87` / `5.26 GiB`** (`4%–5%` of 128 GB) | **`5.26–6.87 GiB` per active node** (Bit-for-bit identical outputs across both nodes) | `10.13` / `8.52` / `6.94 GiB` (`FP32` master weights) |
| **Studio Web UI (`2–48` Frames, All 3 Models)** | **100% Supported Standalone** (Shared or Dedicated) | **100% Supported Standalone** (Always-On) | **Always-On on Tier 1 (`GB10`)** + Zero-Contention CLI Batch Worker on Tier 2 (`5070 Ti`) | Supported |
| **Max Consecutive Frame Ceiling (CLI)** | **`48–64` frames (Shared)** · **`64–96+` frames (Dedicated 16 GB)** | **`96–250+` frames** (Single-pass ultra-long sequences) | **`64–96+` frames on Tier 2** + **`96–250+` frames on Tier 1** in parallel | `~64–80` frames |
| **Best Fit Use Case** | Fast interactive desktop/workstation 3D reconstruction (`2–96` frames) on consumer 16 GB hardware | Always-on 24/7 Studio appliance, ultra-long `96–250+` frame CLI sequences, or co-hosting multiple large AI models | High-throughput parallel multi-backbone ensembles (`pi3` + `da3` + `vggt` simultaneously), Studio/CLI isolation, and LAN-synced air-gapped operation | Standard cloud Space deployment |

See the full architectural deep dive and setup instructions for all three deployment options in [**`docs/BLACKWELL_OPTIMIZATIONS.md`**](docs/BLACKWELL_OPTIMIZATIONS.md).

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

## Sponsors & Donations

Sponsorship and donations support Shekinah Shine maintenance, sanctuary groundskeeping, AI broadcast infrastructure, and open-source development.

<p>
  <a href="https://www.shekinahshine.org/?donate=paypal"><img src="docs/badges/paypal-donate-pill.svg" alt="Donate via PayPal" height="24" align="absmiddle" /></a> &nbsp;·&nbsp;
  <a href="https://www.shekinahshine.org/?donate=paypal"><img src="docs/badges/icon-paypal.svg" width="16" height="16" align="absmiddle" alt="" /> PayPal Donate (Shekinah Shine)</a> &nbsp;·&nbsp;
  <a href="https://www.shekinahshine.org/?donate=cashapp"><img src="docs/badges/icon-cashapp.svg" width="16" height="16" align="absmiddle" alt="" /> Cash App</a> &nbsp;·&nbsp;
  <a href="https://www.shekinahshine.org/?donate=venmo"><img src="docs/badges/icon-venmo.svg" width="16" height="16" align="absmiddle" alt="" /> Venmo</a> &nbsp;·&nbsp;
  <a href="https://github.com/sponsors/chieftechnicalwizard"><img src="docs/badges/icon-github-sponsors.svg" width="16" height="16" align="absmiddle" alt="" /> GitHub Sponsors</a> &nbsp;·&nbsp;
  <a href="https://www.shekinahshine.org/"><img src="docs/badges/icon-shekinah.svg" width="16" height="16" align="absmiddle" alt="" /> Shekinah Shine Website</a>
</p>
