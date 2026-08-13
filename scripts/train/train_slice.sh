#!/usr/bin/env bash
# Train the DSpark drafter on a 10k-row slice with near-real geometry
# (anchors=16, block=16) that fits a single ~96GB GPU.
#
# Usage: bash scripts/train/train_slice.sh [--cache <dir>]
# Env: REPO_DIR, CACHE_DIR, EPOCHS.
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/muse-glimmer-dspark}
CACHE_DIR=${CACHE_DIR:-}
EPOCHS=${EPOCHS:-1}

# --- parse --cache ---
while [ $# -gt 0 ]; do
    case "$1" in
        --cache) CACHE_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
if [ -z "${CACHE_DIR}" ]; then
    echo "CACHE_DIR not set; pass --cache <dir>" >&2
    exit 1
fi

cd "${REPO_DIR}"

# Some hosts fail to bind the default TCPStore address in init_dist; set the
# distributed env explicitly (single-node, single-GPU defaults).
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

PYTORCH_ALLOC_CONF=expandable_segments:True uv run python train.py \
    --config config/dspark/dspark_muse_glimmer_30b.py \
    --opts "data.target_cache_path=${CACHE_DIR}" \
    --opts "train.global_batch_size=32" \
    --opts "train.local_batch_size=1" \
    --opts "train.num_train_epochs=${EPOCHS}" \
    --opts "logging.checkpointing_steps=300" \
    --opts "train.torch_compile=False" \
    --opts "train.include_optimizer_state=False" \
    --opts "model.num_anchors=16" \
    --opts "model.block_size=16"
