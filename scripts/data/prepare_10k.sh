#!/usr/bin/env bash
# Prepare a 10k-row dataset slice for Muse-Glimmer DSpark training:
# download+split -> slice -> (regen via sglang) -> target cache.
#
# Usage:
#   1. bash scripts/data/prepare_10k.sh --regen   # full flow
#   2. bash scripts/data/prepare_10k.sh           # skip regen (raw perfectblend)
#
# Env: NUM_WORKERS (sglang workers, default 1), REPO_DIR, DATA_ROOT,
#      MAX_LENGTH (cache token cap; cache size scales ~linearly with this).
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/muse-glimmer-dspark}
DATA_ROOT=${DATA_ROOT:-${REPO_DIR}/train_datasets}
NUM_SAMPLES=${NUM_SAMPLES:-10000}
NUM_WORKERS=${NUM_WORKERS:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
DO_REGEN=0
if [[ "${1:-}" == "--regen" ]]; then
    DO_REGEN=1
fi

cd "${REPO_DIR}"
mkdir -p "${DATA_ROOT}/muse_glimmer_30b"

# --- 1. download + split (idempotent) ---
uv run python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --train-output-path "${DATA_ROOT}/perfectblend_train.jsonl" \
    --test-output-dir eval_datasets \
    --skip-existing

# --- 2. slice ---
slice_path="${DATA_ROOT}/slice_${NUM_SAMPLES}.jsonl"
if [ ! -s "${slice_path}" ]; then
    head -n "${NUM_SAMPLES}" "${DATA_ROOT}/perfectblend_train.jsonl" > "${slice_path}"
fi
echo "Slice: ${slice_path} ($(wc -l < "${slice_path}") rows)"

# --- 3. regen answers through sglang (optional) ---
regen_path="${DATA_ROOT}/muse_glimmer_30b/slice_${NUM_SAMPLES}_regen.jsonl"
if [ "${DO_REGEN}" -eq 1 ]; then
    echo "Starting sglang (${NUM_WORKERS} worker(s))..."
    NUM_WORKERS=${NUM_WORKERS} uv run --project serving \
        bash scripts/data/launch_sglang_server.sh > /tmp/sglang_launch.log 2>&1 &
    echo "Waiting for sglang to be ready..."
    for i in $(seq 1 120); do
        if curl -sf http://127.0.0.1:30000/v1/models > /dev/null 2>&1; then
            break
        fi
        sleep 5
    done
    uv run python scripts/data/generate_train_data.py \
        --model meta-models/Muse-Glimmer-30B \
        --server-address 127.0.0.1:30000 \
        --concurrency 32 --temperature 0.7 --top-p 0.95 --top-k 64 \
        --max-tokens 4096 --disable-thinking --resume \
        --num-samples "${NUM_SAMPLES}" \
        --input-file-path "${slice_path}" \
        --output-file-path "${regen_path}"
    echo "Stopping sglang..."
    pgrep -f 'sglang serve' | xargs -r kill || true
    sleep 5
    train_path="${regen_path}"
else
    echo "Skipping regen; using raw slice."
    train_path="${slice_path}"
fi

# --- 4. target cache ---
cache_dir="${DATA_ROOT}/cache_slice_${NUM_SAMPLES}_len${MAX_LENGTH}"
uv run python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_muse_glimmer_30b.py \
    --train-data-path "${train_path}" \
    --output-dir "${cache_dir}" \
    --local-batch-size 4 \
    --num-workers 2 \
    --opts "data.max_length=${MAX_LENGTH}"

echo "Done. Cache at: ${cache_dir}"
echo "Train with: bash scripts/train/train_slice.sh --cache ${cache_dir}"
