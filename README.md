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
# 1. Data prep: download + regenerate answers + build target cache
bash scripts/data/prepare_data_muse_glimmer.sh

# 2. Train the DSpark drafter against Muse-Glimmer
bash scripts/train/train.sh

# 3. Evaluate speculative-decoding acceptance
bash scripts/eval/eval.sh
```

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
