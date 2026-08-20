#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/train_berry_staged_force_prediction_action_only.sh" \
  future_peak \
  "$@"
