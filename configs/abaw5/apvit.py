_base_ = [
    '../_base_/datasets/abaw5_112.py',
    '../_base_/default_runtime.py'
]

ce_head = dict(
    type='LinearClsHead',
    num_classes=8,
    in_channels=768,
    loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    topk=(1,),
)

vit_small = dict(img_size=112, patch_size=16, embed_dim=768, num_heads=8, mlp_ratio=3, qkv_bias=False, norm_layer_eps=1e-6)

model = dict(
    type='PoolingVitClassifier',
    extractor=dict(
        type='HierarchicalIRSE',
        input_size=(112, 112),
        num_layers=50,
        pretrained='weights/backbone_ir50_ms1m_epoch120.pth',
        mode='ir',
        return_index=[2],   # only use the first 3 stages
        return_type='Tuple',
    ),
    convert=None,
    vit=dict(
        type='PoolingViT',
        pretrained='weights/vit_small_p16_224-15ec54c9.pth',
        input_type='feature',
        patch_num=196,
        in_channels=[256],
        attn_method='SUM_ABS_1',
        sum_batch_mean=False,
        cnn_pool_config=dict(keep_num=160, exclude_first=False),
        vit_pool_configs=dict(keep_rates=[1.] * 4 + [0.9] * 4, exclude_first=True, attn_method='SUM'),  # None by default
        depth=8,
        **vit_small,
    ),
    head=ce_head
)

data = dict(
    samples_per_gpu=64,  # total 32*4=128
)

# fp16 settings
fp16 = dict(loss_scale=512.)
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.01)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing', min_lr=0,
            warmup='linear',
            warmup_iters=100,
            warmup_ratio=0.1,
            warmup_by_epoch=False,)

runner = dict(max_iters=8000)
evaluation = dict(interval=4000)
checkpoint_config = dict(create_symlink=False, max_keep_ckpts=1, by_epoch=False, interval=5000)

# load_from = 'weights/APViT_RAF-3eeecf7d.pth'
