#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_PATH="third_party/forcelens_dp/data/flip/flip_obj_4_stage2_all"
PSEUDO_LABEL_NAME="viewforce_pseudo_force_fz.npz"
VIEWFORCE_CKPT="checkpoints/viewforce_fz_edge_only_valmul10_trim01_v1/best.pt"
OUTPUT_DIR="checkpoints/flip_delta_force_obj4_stage2_v2_dp_obs"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:1}"

"${PYTHON_BIN}" scripts/precompute_flip_pseudo_force.py \
  --dataset-path "${DATASET_PATH}" \
  --viewforce-ckpt "${VIEWFORCE_CKPT}" \
  --output-name "${PSEUDO_LABEL_NAME}" \
  --force-keys Fz \
  --mask-mode sam2 \
  --sam2-model small \
  --sam2-repo third_party/sam2 \
  --sam2-ckpt third_party/sam2/checkpoints/sam2.1_hiera_small.pt \
  --device "${DEVICE}" \
  --batch-size 64

"${PYTHON_BIN}" scripts/train_flip_delta_force.py \
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
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --cum-loss-weight 1.0 \
  --device "${DEVICE}" \
  --output "${OUTPUT_DIR}" \
  --wandb-project force_estimation \
  --wandb-run flip_obj4_stage2_delta_force_v2_dp_obs
