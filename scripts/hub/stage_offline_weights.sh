#!/usr/bin/env bash
# Stage official PURE Safetensors weights (cyun9286/PURE) and SegFormer-B2 sky
# segmentation weights (nvidia/segformer-b2-finetuned-ade-512-512) into a local
# directory for 100% air-gapped offline execution, and verify SHA256 checksums.
set -euo pipefail

DEST_ROOT="${1:-./offline_weights}"
MODELS="${PURE_STAGE_MODELS:-pi3 da3 vggt}"

mkdir -p "${DEST_ROOT}/models" "${DEST_ROOT}/sky_segformer_b2"

echo "=== [1/2] Staging SegFormer-B2 Sky Segmenter (nvidia/segformer-b2-finetuned-ade-512-512) ==="
SKY_REV="de01bae28967510f9ddd496c60a969357195400c"
for f in config.json preprocessor_config.json pytorch_model.bin; do
  until curl -L --fail --retry 10 --retry-delay 3 -C - \
    "https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512/resolve/${SKY_REV}/${f}" \
    -o "${DEST_ROOT}/sky_segformer_b2/${f}"; do
    echo "Retrying ${f} in 3s..."
    sleep 3
  done
done
echo "187ca07bea003a5717c63d04ea90b07f33cd033c0ebf44b4b89fce5070d6c8f3  ${DEST_ROOT}/sky_segformer_b2/pytorch_model.bin" | sha256sum -c -

echo "=== [2/2] Staging PURE Refined Backbones (cyun9286/PURE: ${MODELS}) ==="
declare -A EXPECTED_SHA256=(
  ["pi3"]="b7a942d16c5b211a628092092544256517a444d2dcaff3a492b1c950630c4164"
  ["da3"]="ecd155f637181c4842e1d5442ed0324f1beea0286635fcb0fab5ce6de00ba882"
  ["vggt"]="679ea3b847752ccb323d22428afd9f5f576f9c527619886888083de8db8fdaf9"
)

for m in ${MODELS}; do
  mkdir -p "${DEST_ROOT}/models/${m}"
  for f in config.json preprocessor_config.json model.safetensors; do
    echo "Downloading ${m}/${f}..."
    until curl -L --fail --retry 10 --retry-delay 3 -C - \
      "https://huggingface.co/cyun9286/PURE/resolve/main/${m}/${f}" \
      -o "${DEST_ROOT}/models/${m}/${f}"; do
      echo "Retrying ${m}/${f} in 3s..."
      sleep 3
    done
  done
  if [[ -n "${EXPECTED_SHA256[$m]:-}" ]]; then
    echo "${EXPECTED_SHA256[$m]}  ${DEST_ROOT}/models/${m}/model.safetensors" | sha256sum -c -
  fi
done

cat << EOF

===============================================================================
All requested models and sky segmenter staged and SHA256-verified!
To run PURE-CTWBW 100% offline (air-gapped):

  export PURE_MODEL_ROOT="$(cd "${DEST_ROOT}/models" && pwd)"
  export PURE_SKY_MODEL_DIR="$(cd "${DEST_ROOT}/sky_segformer_b2" && pwd)"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
===============================================================================
EOF
