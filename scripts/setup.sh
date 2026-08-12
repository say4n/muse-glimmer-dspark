#!/usr/bin/env bash
# Provision a fresh GPU box (Linux/CUDA) to run the Muse-Glimmer DSpark pipeline.
#   - installs uv (if missing)
#   - installs Rust (required to build sglang from git main, which has the
#     day-0 muse_glimmer backend; PyPI sglang predates it)
#   - syncs the main project env + the isolated sglang serving env
# Usage: bash scripts/setup.sh   (idempotent; safe to re-run)
set -euo pipefail

# --- uv ---
if ! command -v uv > /dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# --- Rust (needed for sglang's custom ops build) ---
if ! command -v cargo > /dev/null 2>&1 && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
    echo "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="$HOME/.cargo/bin:$PATH"
fi
export PATH="$HOME/.cargo/bin:$PATH"
cargo --version

# --- Project environments ---
echo "Syncing main project env..."
uv sync --extra data

echo "Syncing sglang serving env..."
uv sync --project serving

echo "Setup complete. Verify with:"
echo "  uv run python -c 'import torch; print(torch.cuda.is_available())'"
echo "  uv run --project serving python -c 'import sglang; print(sglang.__version__)'"
