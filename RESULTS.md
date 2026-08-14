# Results: DSpark drafter for Muse-Glimmer-30B

Speculative-decoding evaluation of a DSpark drafter trained against
[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B).

## Setup

- **Target**: `meta-models/Muse-Glimmer-30B` (text LM, language-model-only)
- **Drafter**: `MuseGlimmerDSparkModel` — block_size=16, num_anchors=16,
  5 draft layers, target layers `{1, 13, 25, 37, 49}`, markov rank 256,
  confidence head on
- **Training data**: 4591-sample slice of open-perfectblend regenerated
  through Muse-Glimmer (via sglang), token cap 2048
- **Training**: global_batch 32, ~400 steps (1 partial epoch)
- **Eval**: DeepSpec speculative decoder, max_new_tokens=128, temperature=0.7,
  batch=1, 50 samples/task

All artifacts (checkpoints, data, eval) are mirrored on Hugging Face:
[`say4n/muse-glimmer-dspark-10k`](https://huggingface.co/say4n/muse-glimmer-dspark-10k).

## Acceptance results

| dataset | draft/proposal | accept len | verify rate | acc@0 | acc@1 | acc@2 |
|---|---|---|---|---|---|---|
| gsm8k    | 16+1 | 1.15 | 0.0674 | 0.1234 | 0.0222 | 0.0009 |
| math500  | 16+1 | 1.16 | 0.0681 | 0.1339 | 0.0215 | 0.0022 |
| aime25   | 16+1 | 1.15 | 0.0676 | 0.1304 | 0.0164 | 0.0018 |
| humaneval| 16+1 | 1.17 | 0.0686 | 0.1301 | 0.0293 | 0.0040 |

## Confidence-head calibration (acceptance AUC)

| dataset | AUC | samples | proposals |
|---|---|---|---|
| gsm8k    | 0.8525 | 50 | 5592 |
| math500  | 0.7966 | 50 | 5533 |
| aime25   | 0.7455 | 30 | 3344 |
| humaneval| 0.7811 | 50 | — |

## Interpretation

- **Acceptance ~1.15** (vs 1.0 = no speculation) at 400 steps, with
  first-token accept rate ~13%. This is a real but early-stage signal — the
  drafter is learning to propose tokens the target accepts.
- **The confidence head is the strongest result**: acceptance-prediction AUC
  of 0.75–0.85 across tasks even at this small training scale. This means the
  DSpark confidence-scheduling mechanism is learning to rank tokens by
  acceptance probability, which is what makes block length control work.
- **Still far from production**: 1 partial epoch of a 5k slice is a fraction
  of DeepSpec-scale training (10 epochs, global_batch 512, larger data). A
  fully-trained drafter should reach substantially higher acceptance; these
  numbers validate the machinery, not the ceiling.

## What would push acceptance up

1. Train to convergence (10 epochs / global_batch 512 / more data).
2. Larger `num_anchors` (the loss budget here capped it at 16 for VRAM).
3. Higher `max_length` cache (2048 cap here truncates long contexts).
