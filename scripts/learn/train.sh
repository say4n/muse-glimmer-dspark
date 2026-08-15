#!/usr/bin/env bash
# Train the DSpark drafter against Qwen3-4B-Instruct-2507 on a small cache.
#
# Small anchors/block so a full epoch is minutes, not hours — you can watch
# acceptance rise across epochs and see what each loss term does.
#
# Usage: bash scripts/learn/train.sh --cache <dir> [--epochs N]
# Env: REPO_DIR, CACHE_DIR, EPOCHS
set -euo pipefail

# Infer repo root from this script's location: <repo>/scripts/learn/..
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}
CACHE_DIR=${CACHE_DIR:-}
EPOCHS=${EPOCHS:-5}

while [ $# -gt 0 ]; do
    case "$1" in
        --cache) CACHE_DIR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
if [ -z "${CACHE_DIR}" ]; then
    echo "CACHE_DIR not set; pass --cache <dir>" >&2
    exit 1
fi

cd "${REPO_DIR}"
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

PYTORCH_ALLOC_CONF=expandable_segments:True uv run python train.py \
    --config config/dspark/dspark_qwen3_4b_instruct_2507.py \
    --opts "data.target_cache_path=${CACHE_DIR}" \
    --opts "train.global_batch_size=32" \
    --opts "train.local_batch_size=1" \
    --opts "train.num_train_epochs=${EPOCHS}" \
    --opts "logging.checkpointing_steps=200" \
    --opts "train.torch_compile=False" \
    --opts "train.include_optimizer_state=False" \
    --opts "model.num_anchors=32" \
    --opts "model.block_size=7"
