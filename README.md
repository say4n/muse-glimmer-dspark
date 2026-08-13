# muse-glimmer-dspark

Meta's **Muse-Glimmer-30B** with a **DSpark** speculative drafter instead of
the shipped DFlash drafter.

This is a fork of [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec)
(MIT) that adds a `MuseGlimmerDSparkModel` target-model family, trained against
`meta-models/Muse-Glimmer-30B` using DeepSeek's DSpark recipe (see
[`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)).

> The DSpark module shipped inside `DeepSeek-V4-Flash-DSpark` cannot be lifted
> onto Muse-Glimmer — its weights are coupled to DeepSeek-V4's vocab, hidden
> size, and layer count. This repo **trains a fresh DSpark drafter against
> Muse-Glimmer** instead. See [DESIGN.md](DESIGN.md) for details.

## Setup

On a fresh GPU box (Linux/CUDA):

```bash
bash scripts/setup.sh     # installs uv + rust, syncs main + sglang envs
```

This handles the two gotchas: sglang must come from git `main` (PyPI predates
the `muse_glimmer` backend) and needs a Rust toolchain to build its custom ops.

Training requires a CUDA build of torch (the default PyPI wheel works on Linux;
adjust if your box needs a different wheel). flex_attention (triton) is used
for drafter training.

## Pipeline

```bash
# 0. Log in to HF (Muse-Glimmer is a gated repo)
huggingface-cli login

# 1. Download + split the dataset (CPU-only)
uv run python scripts/data/download_and_split.py \
  --dataset-name mlabonne/open-perfectblend \
  --train-output-path train_datasets/perfectblend_train.jsonl \
  --test-output-dir eval_datasets \
  --skip-existing

# 2. Start sglang servers (one per GPU; keep this running in another terminal)
NUM_WORKERS=1 uv run --project serving bash scripts/data/launch_sglang_server.sh

# 3. Regenerate answers with Muse-Glimmer through sglang
uv run python scripts/data/generate_train_data.py \
  --model meta-models/Muse-Glimmer-30B \
  --server-address 127.0.0.1:30000 \
  --concurrency 32 --temperature 0.7 --top-p 0.95 --top-k 64 \
  --max-tokens 4096 --disable-thinking --resume \
  --input-file-path train_datasets/perfectblend_train.jsonl \
  --output-file-path train_datasets/muse_glimmer_30b/perfectblend_train_regen.jsonl

# 4. Stop sglang, then build the target cache (frees the GPU)
uv run python scripts/data/prepare_target_cache.py \
  --config config/dspark/dspark_muse_glimmer_30b.py \
  --train-data-path train_datasets/muse_glimmer_30b/perfectblend_train_regen.jsonl \
  --output-dir ~/.cache/deepspec/muse_glimmer_30b_target_cache

# 5. Train the DSpark drafter against Muse-Glimmer
bash scripts/train/train.sh

# 6. Evaluate speculative-decoding acceptance
bash scripts/eval/eval.sh
```

> `scripts/data/prepare_data_muse_glimmer.sh` orchestrates the steps above but
> does **not** manage the sglang server lifecycle — you must start it before
> the regeneration step and stop it before the cache step. The steps are listed
> explicitly here so the ordering is clear.

For a quick single-GPU validation of the whole loop, use
`scripts/train/train_val_slice.sh` (small anchors/block + no optimizer state in
checkpoints) and pass `--max-samples 10` to `eval.py`.

> **Rebuilding after VM shutdown.** GPU instances are ephemeral: the target
> cache, regenerated training data, checkpoints, and eval outputs live only on
> the rented machine and are lost when it is shut down (the 60GB model is
> re-downloaded into the HF cache). None of it is in the repo — treat it as
> reproducible build output. To start fresh on a new box: `bash scripts/setup.sh`,
> re-run the data prep (`prepare_data.sh` / `prepare_data_muse_glimmer.sh`),
> then train + eval. The cache for the full 1.3M-row dataset is TB-scale; use a
> sample slice for validation.

See [DESIGN.md](DESIGN.md) for architecture, cache sizing, and verification
status.

## Layout

- `deepspec/modeling/dspark/muse_glimmer/` — `MuseGlimmerDSparkModel` + config builder
- `deepspec/trainer/dspark_trainer.py` — `MuseGlimmerDSparkTrainer`
- `deepspec/eval/dspark/evaluator.py` — `MuseGlimmerDSparkEvaluator`
- `config/dspark/dspark_muse_glimmer_30b.py` — training config
- rest — vendored DeepSpec core (MIT, see [NOTICE](NOTICE))

## License

Apache-2.0. Vendored DeepSpec code remains MIT (see [NOTICE](NOTICE)).
