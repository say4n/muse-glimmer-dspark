# Design: DSpark speculative decoder for Muse-Glimmer-30B

## Goal

Replace the **DFlash block-diffusion drafter** shipped with
[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)
with a **DSpark semi-autoregressive drafter**, following DeepSeek's DeepSpec
implementation and the recipe behind
[`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark).

This repo is a fork of [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec)
(MIT) with a new `MuseGlimmerDSparkModel` target-model family added.

## Why a new drafter is required

A speculative drafter is *trained against a specific target model*. The DSpark
module inside `DeepSeek-V4-Flash-DSpark` (the `mtp.2.*` weights) is coupled to
DeepSeek-V4-Flash:

| Property | DeepSeek-V4-Flash | Muse-Glimmer-30B |
|---|---|---|
| vocab size | 129,280 | 202,048 |
| text hidden size | 4,096 | 6,656 |
| text layers | 43 | 52 |
| attention | MoE / CSA-HCA hybrid | dense, gated, GQA 32:2 |
| logit transform | plain | `output_multiplier` + tanh softcap (T=20) |

The embeddings/LM-head are frozen copies of the target's, the backbone reads
target hidden states from configured layers, and the markov/confidence heads
live in the target's hidden/vocab space. None of those transfer across models,
so the drafter must be **trained from scratch against Muse-Glimmer** using the
DeepSpec pipeline (data prep → target cache → DSpark training → eval).

## DSpark architecture recap (as implemented in DeepSpec)

DSpark is a block-wise, semi-autoregressive drafter with confidence-scheduled
verification. At every decoding step it proposes up to `block_size` tokens in
a *single* draft-model forward, then refines them autoregressively with a small
markov head before the target verifies them in parallel.

Three components on top of a small transformer backbone:

1. **Context projection** — `fc: (len(target_layer_ids) * H) -> H` plus
   `hidden_norm`, turning concatenated hidden states from selected *target*
   layers into a single context embedding. DeepSeek uses `[40, 41, 42]`; the
   Muse-Glimmer DFlash drafter (and our default) uses `{1, 13, 25, 37, 49}`.
2. **Markov head** — an `Embedding(V) -> R` + `Linear(R -> V)` bias added to
   the block logits, enabling the cheap step-by-step refinement of the
   parallel block proposal (`markov_rank=256`, `vanilla`).
3. **Confidence head** — an `AcceptRatePredictor` over backbone features
   (optionally concatenated with markov embeddings) that predicts per-token
   acceptance probability. A threshold selects how many of the drafted tokens
   are actually proposed for verification, saving target FLOPs.

Training uses the *anchor/target* distillation loss
(`ce_loss_alpha * CE + l1_loss_alpha * L1 + confidence_head_alpha * BCE`),
where every anchor block is supervised against the target's own hidden-state
logits (`aligned_target_logits`).

## Muse-Glimmer DSpark model

`deepspec/modeling/dspark/muse_glimmer/`

- `config.py` — `build_draft_config(target_config, model_args)` clones
  `target_config.text_config`, keeps Muse-Glimmer's text hyperparameters
  (RoPE θ=500k, gated attention, `final_logit_softcapping`, `output_multiplier`,
  centered RMSNorm layout) and overlays draft-specific fields
  (`num_hidden_layers=5`, `block_size=16`, `target_layer_ids`,
  `mask_token_id=201818`, markov/confidence settings). Defaults mirror the
  released DFlash drafter so results are comparable.
- `modeling.py` — `MuseGlimmerDSparkModel`:
  - `embed_tokens` + `lm_head` are **frozen copies** of the target text
    model's (`initialize_embeddings_and_head`).
  - `MuseGlimmerDSparkAttention` reuses Muse-Glimmer's gated attention
    (QK-norm + `qk_scale_factor` + `gate_proj`) with the DSpark K/V pattern:
    keys/values are the concatenation of target context states and draft block
    states.
  - `compute_logits` reproduces the target's logit transform
    (`output_multiplier`, then `T * tanh(x/T)`) so draft and target logits are
    comparable for acceptance / L1 terms.
  - Backbone, markov head, and confidence head from `deepspec.modeling.dspark.*`.

`MuseGlimmerDSparkModel` implements the same interface as the qwen3/gemma4
variants (`_forward_backbone`, `compute_logits`, `sample_draft_tokens`,
`predict_confidence_step`), so the shared trainer/eval/draft_ops code works
unchanged.

## Pipeline (mirrors DeepSpec)

```
1. Data          scripts/data/prepare_data_muse_glimmer.sh
                 download + split open-perfectblend, regenerate answers with
                 sglang (Muse-Glimmer sampling: T=0.7, top_p=0.95, top_k=64),
                 then prepare_target_cache.py runs the target once per sample,
                 capturing input_ids + layer hidden states + loss mask.
                 -> ~/.cache/deepspec/muse_glimmer_30b_target_cache

2. Train         scripts/train/train.sh  (config/dspark/dspark_muse_glimmer_30b.py)
                 DSpark loss over cached target states. Embeddings/LM-head frozen.
                 -> ~/checkpoints/muse-glimmer-dspark/dspark_block16_muse_glimmer_30b/

3. Eval          scripts/eval/eval.sh  (deepspec/eval/dspark/MuseGlimmerDSparkEvaluator)
                 Speculative-decoding acceptance metrics (accept length, verify
                 rate, per-position acceptance) over gsm8k/math500/aime25/etc.
```

Note on storage: the target cache stores per-token hidden states for every
training sample; the DeepSpec README warns this can reach TB scale for large
targets. For Muse-Glimmer (H=6656, 5 target layers) a 4096-token sample needs
~4096 * 5 * 6656 * 2B ≈ 273 MB. Plan disk accordingly (`--max-shard-bytes`,
`--num-workers`, `--local-batch-size` are tunable).

## Key files

| File | Purpose |
|---|---|
| `deepspec/modeling/dspark/muse_glimmer/config.py` | Draft-config builder from target config |
| `deepspec/modeling/dspark/muse_glimmer/modeling.py` | `MuseGlimmerDSparkModel` (+ attention, decoder layer) |
| `deepspec/trainer/dspark_trainer.py` | `MuseGlimmerDSparkTrainer` |
| `deepspec/eval/dspark/evaluator.py` | `MuseGlimmerDSparkEvaluator` (loads the multimodal target correctly) |
| `deepspec/eval/dspark/draft_ops.py` | Shared DSpark proposal/verification ops |
| `deepspec/data/parser.py` | `muse_glimmer` chat template registration |
| `scripts/data/prepare_target_cache.py` | Target hidden-state cache (handles `muse_glimmer`) |
| `config/dspark/dspark_muse_glimmer_30b.py` | Training config |

## Design decisions

1. **Mirror the DFlash drafter's geometry.** `block_size=16`, 5 draft layers,
   `target_layer_ids={1,13,25,37,49}` match the shipped DFlash drafter so that
   DSpark vs DFlash is the only variable in comparisons. These are overridable
   via the config / `--opts`.
2. **Preserve Muse-Glimmer text semantics in the draft backbone.** Gated
   attention, QK-norm, centered RMSNorm placement, RoPE θ, and the logit
   softcap/multiplier are all reused so the drafter operates in the same
   representational space as the target.
3. **Multimodal target, text-only drafting.** The target is
   `MuseGlimmerForConditionalGeneration`. Drafting operates on the text LM
   hidden states only; image tokens are handled by the target prefill (out of
   scope for drafter training). `MuseGlimmerDSparkEvaluator` loads the target
   via `MuseGlimmerForConditionalGeneration` rather than
   `AutoModelForCausalLM`.
4. **flex_attention for training, sdpa for eval** (same split as DeepSpec):
   the block-diagonal DSpark mask is built with `create_dspark_attention_mask`;
   during eval the drafter runs with a KV cache and `is_causal=False`.

## Verification status

- [x] `MuseGlimmerDSparkModel` imports and instantiates from the real
      `meta-models/Muse-Glimmer-30B` config.
- [x] Eval path smoke-tested on CPU: `_forward_backbone` + `compute_logits` +
      `sample_draft_tokens` + `predict_confidence_step` with a DynamicCache.
- [x] Training path smoke-tested on CPU (eager mask): DSpark loss computed and
      gradients flow to backbone/markov/confidence heads; embeddings frozen.
- [x] End-to-end validation on GPU (RTX PRO 6000 Blackwell, 96GB): target
      cache built from 500 samples, drafter trained (loss 11.1 → 2.61 over 120
      steps), checkpoint saved, speculative eval ran (gsm8k smoke, acceptance
      length ~1.06 at the tiny training budget).

## Operational notes (from a full GPU run)

1. **sglang must come from git `main`**, not PyPI. PyPI 0.5.17 predates the
   `muse_glimmer` backend (day-0 in sglang main). sglang main also requires a
   Rust toolchain to build its custom ops. `scripts/setup.sh` installs
   uv + rust and syncs both environments. Launch with
   `--language-model-only --reasoning-parser muse --tool-call-parser muse`
   (see `scripts/data/launch_sglang_server.sh`).
2. **Drafter training is VRAM-heavy.** The DSpark loss materialises fp32
   tensors over the full 202k vocab: per tensor that is
   `num_anchors * block_size * 202048 * 4 bytes`. With
   `num_anchors=512, block_size=16` that is ~6.6GB per tensor and several are
   live at once. The `BF16Optimizer` also keeps fp32 master + Adam states
   (~14B/param). Budget ~48GB for weights+optimizer plus activations/loss.
   `scripts/train/train_val_slice.sh` uses `num_anchors=8, block_size=8` and
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (avoids allocator
   fragmentation that OOMs the backward pass even with nominally free VRAM).
   Note: the env var is `PYTORCH_CUDA_ALLOC_CONF` in torch 2.9 (the older
   `PYTORCH_CUDA_ALLOC_CONF` spelling logs a deprecation warning).
3. **Checkpoints can fill a small disk.** The saved optimizer state (fp32
   master + Adam moments) is ~32GB for this drafter on top of the ~11GB model.
   Set `train.include_optimizer_state=False` for validation runs — eval only
   needs `config.json` + `model.safetensors`. (If this flag is missing from
   your config, add it first; `--opts` rejects unknown keys.)
4. **Distributed init** needs explicit `MASTER_ADDR`/`MASTER_PORT`/`RANK`/
   `WORLD_SIZE` on some hosts (the default TCPStore bind can fail).
5. **Eval is batch=1 by design** (`generate_decoding_sample` asserts bsz=1),
   so GPU utilization is low and throughput is latency-bound. Use
   `--max-samples N` to smoke-test quickly; ~10-20s per sample.

## Out of scope (for now)

- Porting the `DeepSeek-V4-Flash-DSpark` weights (impossible without
  retraining — see above).
- SGLang/vLLM serving integration of the combined target+drafter.
- Multimodal (image) drafting; the drafter sees text hidden states only.
