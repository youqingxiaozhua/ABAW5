_base_ = [
    '../_base_/datasets/abaw5_224.py',
    '../_base_/default_runtime.py'
]

ce_head = dict(
    type='LinearClsHead',
    num_classes=8,
    in_channels=512,
    loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    topk=(1,),
)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        in_channels=3,
        num_stages=4,
        out_indices=[3],
        norm_cfg=dict(type='SyncBN'),
        frozen_stages=-1),
    neck=dict(type='GlobalAveragePooling'),
    head=ce_head)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_size = 224
data = dict(
    samples_per_gpu=64,  # total 32*4=128
    train=dict(
        dataset=dict(pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', size=img_size, scale=(0.6, 1.0), ratio=(1.0, 1.0)),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ])
    ),
    val=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='Resize', size=int(256 / 224 * img_size)),
        dict(type='CenterCrop', crop_size=img_size),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img', ])
    ]),
    test=dict(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='Resize', size=int(256 / 224 * img_size)),
        dict(type='CenterCrop', crop_size=img_size),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img', ])
    ]),
)

# fp16 settings
fp16 = dict(loss_scale=512.)
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing', min_lr=1e-5,
            warmup='linear',
            warmup_iters=100,
            warmup_ratio=0.1,
            warmup_by_epoch=False,)

runner = dict(max_iters=8000)
evaluation = dict(interval=4000)
checkpoint_config = dict(create_symlink=False, max_keep_ckpts=1, by_epoch=False, interval=5000)

load_from = 'weights/ContraWarping_R18_BYOL_m6_ldmk_ep50.pth'
