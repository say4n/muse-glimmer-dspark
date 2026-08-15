#!/usr/bin/env bash
# Eval the DSpark drafter against Qwen3-4B-Instruct-2507.
#
# Usage: bash scripts/learn/eval.sh --draft <ckpt> [--samples N]
# Env: REPO_DIR, DRAFT_PATH, SAMPLES
set -euo pipefail

REPO_DIR=${REPO_DIR:-/workspace/muse-glimmer-dspark}
DRAFT_PATH=${DRAFT_PATH:-}
SAMPLES=${SAMPLES:-30}

while [ $# -gt 0 ]; do
    case "$1" in
        --draft) DRAFT_PATH="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
if [ -z "${DRAFT_PATH}" ]; then
    echo "DRAFT_PATH not set; pass --draft <ckpt>" >&2
    exit 1
fi

cd "${REPO_DIR}"
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29501}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

PYTORCH_ALLOC_CONF=expandable_segments:True uv run python eval.py \
    --target_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --draft_name_or_path "${DRAFT_PATH}" \
    --max-new-tokens 128 \
    --temperature 0.7 \
    --max-samples "${SAMPLES}"
