_base_ = [
    '../../_base_/models/upernet_my_r50.py', 
    "../../_base_/datasets/siim.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_20k.py",
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(frozen_stages=4),
)

optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=1e-3,
    # lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    # type="SGD",
    # lr=1e-4,
    # constructor="LayerDecayOptimizerConstructor",
    # paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type="PolyLR", eta_min=0.0, power=1.0, begin=1500, end=20000, by_epoch=False),
]

optim_wrapper = dict(
    type='AmpOptimWrapper', 
    optimizer=optimizer,
    # constructor="LayerDecayOptimizerConstructor",
    # paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
)

# mixed precision
fp16 = dict(loss_scale="dynamic")

train_cfg = dict(type="IterBasedTrainLoop", max_iters=20000, val_interval=2000)

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=8)
val_dataloader = dict(batch_size=1)
test_dataloader = dict(batch_size=1)

# default_hooks = dict(
#     checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=2000)
# )

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=-1, save_best="mDice")
)

val_evaluator = dict(
    type="IoUMetric", iou_metrics=["mDice", "mIoU", "mFscore"]
)
test_evaluator = val_evaluator

randomness = {"seed": 42}