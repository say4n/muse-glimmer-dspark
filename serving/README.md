# Serving environment

`sglang` pins `torch==2.8.0`, which conflicts with the main project's
`torch==2.9.1` + `transformers>=5.15`. This sub-project keeps sglang in its
own uv environment and lockfile so the two never clobber each other.

```bash
# Create the isolated env + lockfile
uv sync --project serving

# Start sglang servers for data regeneration
uv run --project serving bash ../scripts/data/launch_sglang_server.sh
```

`NUM_WORKERS` controls how many GPUs are used (Colab single GPU: `NUM_WORKERS=1`).
