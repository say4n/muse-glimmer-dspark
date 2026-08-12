import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"

# Muse Glimmer publishes the DFlash drafter's design choices (block size 16,
# five draft layers, target feature layers {1, 13, 25, 37, 49}). DSpark reuses
# the same target-feature plumbing, so we keep the same defaults for an
# apples-to-apples comparison with the shipped DFlash drafter.
DEFAULT_BLOCK_SIZE = 16
DEFAULT_NUM_DRAFT_LAYERS = 5
DEFAULT_TARGET_LAYER_IDS = [1, 13, 25, 37, 49]


def get_muse_glimmer_text_config(target_config):
    assert target_config.model_type in ("muse_glimmer",), (
        "MuseGlimmer DSpark expects a MuseGlimmer top-level target config, "
        f"got model_type={target_config.model_type!r}."
    )
    text_config = target_config.text_config
    assert text_config.model_type in ("muse_glimmer_text",), (
        "MuseGlimmer DSpark expects target_config.text_config.model_type to be "
        f"'muse_glimmer_text', got {text_config.model_type!r}."
    )
    return copy.deepcopy(text_config)


def _validate_required_text_fields(text_config) -> None:
    required_fields = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_activation",
        "attention_bias",
        "attention_dropout",
        "initializer_range",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_parameters",
        "layer_rope_theta",
        "sliding_window",
        "qk_scale_factor",
        "final_logit_softcapping",
        "output_multiplier",
        "post_norm_eps",
    )
    for field in required_fields:
        assert hasattr(text_config, field), (
            f"target_config.text_config.{field} must be provided."
        )


def build_draft_config(target_config, model_args):
    draft_config = get_muse_glimmer_text_config(target_config)
    _validate_required_text_fields(draft_config)

    num_target_layers = int(draft_config.num_hidden_layers)
    num_draft_layers = int(
        getattr(model_args, "num_draft_layers", DEFAULT_NUM_DRAFT_LAYERS)
    )
    layer_types = ["full_attention"] * num_draft_layers

    if not hasattr(model_args, "target_layer_ids"):
        raise ValueError("target_layer_ids must be provided.")
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head and not hasattr(
        model_args,
        "confidence_head_with_markov",
    ):
        raise ValueError(
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )

    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0 and not hasattr(model_args, "markov_head_type"):
        raise ValueError(
            "markov_head_type must be provided when markov_rank > 0."
        )

    draft_config.architectures = ["MuseGlimmerDSparkModel"]
    draft_config.target_model_type = str(target_config.model_type)
    draft_config.target_text_model_type = str(draft_config.model_type)
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(
        getattr(model_args, "block_size", DEFAULT_BLOCK_SIZE)
    )
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config.layer_rope_theta = [
        float(draft_config.rope_parameters["rope_theta"])
    ] * num_draft_layers
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_NUM_DRAFT_LAYERS",
    "DEFAULT_TARGET_LAYER_IDS",
    "build_draft_config",
    "get_muse_glimmer_text_config",
]
