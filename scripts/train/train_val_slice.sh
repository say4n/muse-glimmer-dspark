#!/usr/bin/env bash
# Train the DSpark drafter on a small validation slice of the Muse-Glimmer
# target cache. Tuned for a single ~80GB GPU (e.g. A100 SXM4):
#   - num_anchors + block_size lowered so the DSpark loss (fp32 over the
#     full 202k vocab) fits in memory
#   - expandable_segments avoids PyTorch allocator fragmentation that OOMs
#     the backward pass even when there appears to be free VRAM
set -euo pipefail

# Some hosts fail to bind the default TCPStore address in init_dist; set the
# distributed env explicitly (single-node, single-GPU defaults).
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

cache_dir=${cache_dir:-${HOME}/muse-glimmer-dspark/cache_val}

PYTORCH_ALLOC_CONF=expandable_segments:True uv run python train.py \
    --config config/dspark/dspark_muse_glimmer_30b.py \
    --opts "data.target_cache_path=${cache_dir}" \
    --opts "train.global_batch_size=8" \
    --opts "train.local_batch_size=1" \
    --opts "train.num_train_epochs=2" \
    --opts "logging.checkpointing_steps=50" \
    --opts "train.torch_compile=False" \
    --opts "train.include_optimizer_state=False" \
    --opts "model.num_anchors=8" \
    --opts "model.block_size=8"
