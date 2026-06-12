#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
CHECKPOINT=${1:-"checkpoints/model_ep0100.pt"}
python src/distribution_shift.py --checkpoint "$CHECKPOINT" "$@"
