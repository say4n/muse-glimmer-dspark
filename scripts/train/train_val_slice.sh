#!/usr/bin/env bash
# Train the DSpark drafter on a small validation slice of the Muse-Glimmer
# target cache. Tuned for a single ~80GB GPU (e.g. A100 SXM4):
#   - num_anchors + block_size lowered so the DSpark loss (fp32 over the
#     full 202k vocab) fits in memory
#   - expandable_segments avoids PyTorch allocator fragmentation that OOMs
#     the backward pass even when there appears to be free VRAM
set -euo pipefail

cache_dir=${cache_dir:-${HOME}/muse-glimmer-dspark/cache_val}

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python train.py \
    --config config/dspark/dspark_muse_glimmer_30b.py \
    --opts "data.target_cache_path=${cache_dir}" \
    --opts "train.global_batch_size=8" \
    --opts "train.local_batch_size=1" \
    --opts "train.num_train_epochs=2" \
    --opts "logging.checkpointing_steps=50" \
    --opts "train.torch_compile=False" \
    --opts "model.num_anchors=8" \
    --opts "model.block_size=8"
