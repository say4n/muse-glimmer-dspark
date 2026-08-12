import os

from deepspec.trainer import MuseGlimmerDSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR, MUSE_GLIMMER_30B

from deepspec.modeling.dspark.muse_glimmer.config import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_NUM_DRAFT_LAYERS,
    DEFAULT_TARGET_LAYER_IDS,
)

project_name = "muse-glimmer-dspark"
exp_name = "dspark_block16_muse_glimmer_30b"
seed = 42

model = dict(
    target_model_name_or_path=MUSE_GLIMMER_30B,
    block_size=DEFAULT_BLOCK_SIZE,
    num_draft_layers=DEFAULT_NUM_DRAFT_LAYERS,
    target_layer_ids=DEFAULT_TARGET_LAYER_IDS,
    mask_token_id=201818,
    num_anchors=512,

    # markov head
    markov_rank=256,
    markov_head_type="vanilla",

    # confidence head
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    # loss
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=MuseGlimmerDSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="muse_glimmer",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(
        BASE_CKPT_DIR,
        project_name,
        exp_name,
    )
    logging_cfg["tensorboard_dir"] = os.path.join(
        BASE_TB_DIR,
        project_name,
        exp_name,
    )
    cfg["logging"] = logging_cfg
    return cfg
