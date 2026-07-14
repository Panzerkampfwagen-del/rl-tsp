#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
CHECKPOINT=${1:-"checkpoints/model_ep0100.pt"}
[ "$#" -gt 0 ] && shift   # drop the positional checkpoint; forward any remaining flags
python src/distribution_shift.py --checkpoint "$CHECKPOINT" "$@"
