#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

TARGET_MODE="${1:-final_delta}"

DATASET_PATH="third_party/forcelens_dp/data/flip/flip_obj_4_stage2_all"
PSEUDO_LABEL_NAME="viewforce_pseudo_force_fz.npz"
VIEWFORCE_CKPT="checkpoints/viewforce_fz_edge_only_valmul10_trim01_v1/best.pt"
OUTPUT_DIR="checkpoints/flip_chunk_force_obj4_stage2_${TARGET_MODE}_v1"

python scripts/precompute_flip_pseudo_force.py \
  --dataset-path "${DATASET_PATH}" \
  --viewforce-ckpt "${VIEWFORCE_CKPT}" \
  --output-name "${PSEUDO_LABEL_NAME}" \
  --force-keys Fz \
  --mask-mode sam2 \
  --sam2-model small \
  --sam2-repo third_party/sam2 \
  --sam2-ckpt third_party/sam2/checkpoints/sam2.1_hiera_small.pt \
  --device cuda:1 \
  --batch-size 64

python scripts/train_flip_delta_force.py \
  --dataset-path "${DATASET_PATH}" \
  --pseudo-label-name "${PSEUDO_LABEL_NAME}" \
  --force-key Fz \
  --force-mode magnitude \
  --action-mode obs_delta_pos_gripper \
  --obs-steps 2 \
  --pred-horizon 16 \
  --image-size 240 320 \
  --image-normalization minus_one_one \
  --include-agent-pos \
  --target-mode "${TARGET_MODE}" \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --cum-loss-weight 0.0 \
  --device cuda:1 \
  --output "${OUTPUT_DIR}" \
  --wandb-project force_estimation \
  --wandb-run "flip_obj4_stage2_chunk_force_${TARGET_MODE}_v1"

