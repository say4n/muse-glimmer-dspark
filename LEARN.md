# Learning playground: DSpark on Qwen3-4B-Instruct-2507

Use this to learn speculative-decoding dynamics on **free hardware** (a 4B
target is tiny) before investing more compute in Muse-Glimmer.

## Why this target

- Dense Qwen3 (36 layers) — the existing `Qwen3DSparkModel` works unchanged.
- ~8GB bf16 — fits a free Colab GPU, or CPU at small batch.
- Plain causal LM (no multimodal wrapper), so the loop is minimal.

## Run it

```bash
# 1. data: 500-sample slice -> small cache (no sglang needed)
bash scripts/learn/prepare_small.sh          # NUM_SAMPLES=500, MAX_LENGTH=2048

# 2. train: 5 epochs, each ~minutes
bash scripts/learn/train.sh --cache train_datasets/cache_learn_500_len2048

# 3. eval: acceptance metrics on a few tasks
bash scripts/learn/eval.sh --draft ~/checkpoints/muse-glimmer-dspark-learn/dspark_qwen3_4b_instruct_2507/step_latest
```

## What to watch (this is the lesson)

Each run teaches you something specific. Watch the training log and the eval
tables together:

1. **Loss curve.** Expect an initial cliff (13 → ~3 in the first steps: the
   drafter learns "write like the target" = CE term), then a slow grind
   (distribution matching = L1 term). That slow grind is what drives acceptance.
2. **accept len / accept rate.** After 1 epoch, acceptance length should be
   meaningfully >1.0. Watch it climb across 2, 3, 5 epochs. If it plateaus low,
   the drafter needs more capacity/data — that's the signal.
3. **Per-position accept rate (acc@0, acc@1, ...).** Early tokens accepted more,
   later ones rejected more — the confidence head should predict this ranking.
4. **Confidence AUC.** If it's well above 0.5, the confidence head learned to
   rank acceptance — that's the mechanism behind DSpark's block-length control.

## Experiments to try once the baseline works

- **Toggle `confidence_head_alpha`** (0 vs 1): watch accept length change.
- **Toggle `markov_rank`** (0 vs 256): the markov head's autoregressive
  refinement should lift acc@1+.
- **Change `block_size`** (4 vs 16): bigger blocks propose more, but acceptance
  per position falls — the confidence head decides how much to actually use.
- **Change data size** (500 → 5000): watch acceptance saturate with data.

## Key files

| File | Purpose |
|---|---|
| `config/dspark/dspark_qwen3_4b_instruct_2507.py` | Playground training config |
| `scripts/learn/prepare_small.sh` | Slice + cache (no regen) |
| `scripts/learn/train.sh` | Train drafter, fast iteration |
| `scripts/learn/eval.sh` | Acceptance eval |

The DeepSpec README and the DSpark paper (arXiv 2607.05147) cover the theory
behind the acceptance math (`min(1, p_target/p_draft)`).
