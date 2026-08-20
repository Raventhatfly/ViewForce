#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage: scripts/train_berry_force_critic.sh VARIANT [TARGET_MODE]

Train a berry force-prediction critic using one of the supported experiment
profiles:

  standard     RGB critic, 16-step prediction horizon
  action_only  Action-only critic, 4-step prediction horizon
  edge         Edge-image critic, 16-step prediction horizon
  h4           RGB critic, 4-step prediction horizon

TARGET_MODE defaults to final_delta for standard, future_peak for action_only,
and max_delta for edge and h4. Environment overrides: PYTHON_BIN, DEVICE,
REBUILD, and PRECOMPUTE_OVERWRITE.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

VARIANT="$1"
shift

PYTHON_BIN="${PYTHON_BIN:-/home/wfy/miniforge3/envs/robodiff/bin/python}"
DEVICE="${DEVICE:-cuda:1}"
REBUILD="${REBUILD:-0}"
PRECOMPUTE_OVERWRITE="${PRECOMPUTE_OVERWRITE:-0}"

FORCELENS_DP_ROOT="third_party/forcelens_dp"
PSEUDO_LABEL_NAME="viewforce_pseudo_force_fz.npz"
VIEWFORCE_CKPT="checkpoints/viewforce_fz_edge_only_valmul10_trim01_v1/best.pt"

BUILD_ARGS=()
PRECOMPUTE_PROFILE_ARGS=()
TRAIN_PROFILE_ARGS=()

case "${VARIANT}" in
  standard)
    TARGET_MODE="${1:-final_delta}"
    DATASET_PATH="${FORCELENS_DP_ROOT}/data/pick/berry_force_prediction_all"
    OUTPUT_DIR="checkpoints/berry_force_prediction_${TARGET_MODE}_v1"
    PREVIEW_DIR="reports/berry_force_prediction_pseudo_force_preview"
    BUILD_ARGS+=(--overwrite)
    TRAIN_PROFILE_ARGS+=(
      --pred-horizon 16
      --image-normalization minus_one_one
      --include-agent-pos
      --batch-size 32
    )
    ;;
  action_only)
    TARGET_MODE="${1:-future_peak}"
    DATASET_PATH="${FORCELENS_DP_ROOT}/data/pick/berry_force_prediction_all_action_only"
    OUTPUT_DIR="checkpoints/berry_force_prediction_${TARGET_MODE}_action_only_v1"
    PREVIEW_DIR="reports/berry_force_prediction_action_only_pseudo_force_preview"
    BUILD_ARGS+=(
      --policy-output data/pick/berry_stage2_policy_action_only_tmp
      --verifier-output data/pick/berry_force_prediction_all_action_only
      --overwrite
    )
    TRAIN_PROFILE_ARGS+=(
      --pred-horizon 4
      --critic-image-mode none
      --batch-size 64
    )
    if [[ "${TARGET_MODE}" == "future_peak" ]]; then
      TRAIN_PROFILE_ARGS+=(--include-current-force)
    fi
    ;;
  edge)
    TARGET_MODE="${1:-max_delta}"
    DATASET_PATH="${FORCELENS_DP_ROOT}/data/pick/berry_force_prediction_all_edge"
    OUTPUT_DIR="checkpoints/berry_force_prediction_${TARGET_MODE}_edge_v1"
    PREVIEW_DIR="reports/berry_force_prediction_edge_pseudo_force_preview"
    EDGE_OBS_NAME="viewforce_edge_obs.npz"
    BUILD_ARGS+=(
      --policy-output data/pick/berry_stage2_policy_edge_tmp
      --verifier-output data/pick/berry_force_prediction_all_edge
      --overwrite
    )
    PRECOMPUTE_PROFILE_ARGS+=(--edge-output-name "${EDGE_OBS_NAME}")
    TRAIN_PROFILE_ARGS+=(
      --pred-horizon 16
      --image-normalization zero_one
      --critic-image-mode edge
      --edge-obs-name "${EDGE_OBS_NAME}"
      --include-agent-pos
      --batch-size 32
    )
    ;;
  h4)
    TARGET_MODE="${1:-max_delta}"
    DATASET_PATH="${FORCELENS_DP_ROOT}/data/pick/berry_force_prediction_all_h4"
    OUTPUT_DIR="checkpoints/berry_force_prediction_${TARGET_MODE}_h4_v1"
    PREVIEW_DIR="reports/berry_force_prediction_h4_pseudo_force_preview"
    BUILD_ARGS+=(
      --policy-output data/pick/berry_stage2_policy_h4_tmp
      --verifier-output data/pick/berry_force_prediction_all_h4
      --overwrite
    )
    TRAIN_PROFILE_ARGS+=(
      --pred-horizon 4
      --image-normalization minus_one_one
      --critic-image-mode rgb
      --include-agent-pos
      --batch-size 32
    )
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown berry critic variant: ${VARIANT}" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ $# -gt 1 ]]; then
  echo "Unexpected argument: $2" >&2
  usage >&2
  exit 2
fi

PRECOMPUTE_EXTRA_ARGS=()
if [[ "${PRECOMPUTE_OVERWRITE}" == "1" ]]; then
  PRECOMPUTE_EXTRA_ARGS+=(--overwrite)
fi

if [[ "${REBUILD}" == "1" || ! -d "${DATASET_PATH}/episodes" ]]; then
  (
    cd "${FORCELENS_DP_ROOT}"
    "${PYTHON_BIN}" scripts/build_berry_staged_datasets.py "${BUILD_ARGS[@]}"
  )
else
  echo "Reusing existing verifier dataset: ${DATASET_PATH}"
fi

"${PYTHON_BIN}" scripts/precompute_flip_pseudo_force.py \
  --dataset-path "${DATASET_PATH}" \
  --viewforce-ckpt "${VIEWFORCE_CKPT}" \
  --output-name "${PSEUDO_LABEL_NAME}" \
  "${PRECOMPUTE_PROFILE_ARGS[@]}" \
  --force-keys Fz \
  --mask-mode sam2 \
  --sam2-model small \
  --sam2-repo third_party/sam2 \
  --sam2-ckpt third_party/sam2/checkpoints/sam2.1_hiera_small.pt \
  --device "${DEVICE}" \
  --batch-size 64 \
  --preview-dir "${PREVIEW_DIR}" \
  --preview-frames 6 \
  "${PRECOMPUTE_EXTRA_ARGS[@]}"

"${PYTHON_BIN}" scripts/train_flip_delta_force.py \
  --dataset-path "${DATASET_PATH}" \
  --pseudo-label-name "${PSEUDO_LABEL_NAME}" \
  --force-key Fz \
  --force-mode magnitude \
  --action-mode obs_delta_pos_gripper \
  --obs-steps 2 \
  --image-size 240 320 \
  "${TRAIN_PROFILE_ARGS[@]}" \
  --target-mode "${TARGET_MODE}" \
  --epochs 100 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --cum-loss-weight 0.0 \
  --device "${DEVICE}" \
  --output "${OUTPUT_DIR}" \
  --no-wandb
