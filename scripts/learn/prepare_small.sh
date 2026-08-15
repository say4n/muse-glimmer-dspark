#!/usr/bin/env bash
# Prepare a small training slice for the Qwen3-4B-Instruct-2507 playground.
#
# Skips the sglang regen step entirely: raw open-perfectblend conversations
# are fine for learning speculative-decoding dynamics (the drafter just needs
# *some* assistant text to distill against).
#
# Usage: bash scripts/learn/prepare_small.sh
# Env: REPO_DIR, DATA_ROOT, NUM_SAMPLES, MAX_LENGTH
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/muse-glimmer-dspark}
DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/train_datasets}
NUM_SAMPLES=${NUM_SAMPLES:-500}
MAX_LENGTH=${MAX_LENGTH:-2048}

cd "${REPO_DIR}"

# download + split (idempotent)
uv run python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --train-output-path "${DATA_ROOT}/perfectblend_train.jsonl" \
    --test-output-dir eval_datasets \
    --skip-existing

# slice
slice_path="${DATA_ROOT}/learn_slice_${NUM_SAMPLES}.jsonl"
head -n "${NUM_SAMPLES}" "${DATA_ROOT}/perfectblend_train.jsonl" > "${slice_path}"

# target cache
cache_dir="${DATA_ROOT}/cache_learn_${NUM_SAMPLES}_len${MAX_LENGTH}"
uv run python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3_4b_instruct_2507.py \
    --train-data-path "${slice_path}" \
    --output-dir "${cache_dir}" \
    --local-batch-size 4 \
    --num-workers 2 \
    --opts "data.max_length=${MAX_LENGTH}"

echo "Done. Cache: ${cache_dir}"
echo "Train with: bash scripts/learn/train.sh --cache ${cache_dir}"
