# dataset settings
dataset_type = 'ABAW5'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_size = 112

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=img_size),
    dict(type='RandomRotate', prob=0.5, degree=6),
    dict(type='RandomResizedCrop', size=img_size, scale=(0.8, 1.0), ratio=(1. / 1., 1. / 1.)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomGrayscale', gray_prob=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4)
        ],
        p=0.8),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=0.5),
    dict(
        type='RandomErasing',
        erase_prob=0.5,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', ])
]

base_path = 'data/ABAW5/'
# image_path = base_path + 'cropped_aligned'
image_path = base_path + 'aligned_XUE'

task = 'EXPR_Classification'
# task = 'AU_Detection'
# task = 'VA_Estimation'

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=8,
    cross_val=(
        'data/ABAW5/annotations/EXPR_Classification_Challenge/5fold/{}_train.txt',
        'data/ABAW5/annotations/EXPR_Classification_Challenge/5fold/{}_val.txt',
        ),
    val=dict(
        type=dataset_type,
        data_prefix=image_path,
        ann_file=base_path + 'annotations/EXPR_Classification_Challenge/Validation_Set',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=image_path,
        ann_file=base_path + 'annotations/EXPR_Classification_Challenge/Validation_Set',
        pipeline=test_pipeline),
)

# bs, iter
# 128, 4.5k

balanced_dataset = True
if task == 'EXPR_Classification':
    metrics = ['accuracy', 'f1_score', 'class_accuracy']
    save_best='f1_score.mean'
    rule='greater'
elif task == 'AU_Detection':
    metrics = ['f1_score']
    save_best='f1_score.mean'
    rule='greater'
    balanced_dataset = False
elif task == 'VA_Estimation':
    metrics = ['MSE', 'CCC']
    save_best='ccc_va'
    rule='greater'
    balanced_dataset = False
else:
    raise ValueError('invalid task value')

train_dataset = dict(
        type=dataset_type,
        data_prefix=image_path,
        ann_file=base_path + 'annotations/EXPR_Classification_Challenge/Train_Set',
        pipeline=train_pipeline)


if balanced_dataset:
    data['train'] = dict(
        type='ClassBalancedDataset',
        oversample_thr=0.15,
        method='sqrt',
        # method='reciprocal',
        dataset=train_dataset
    )
else:
    data['train'] = train_dataset

evaluation = dict(interval=1000, metric=metrics, metric_options=dict(average_mode='none'), save_best=save_best, rule=rule)
runner = dict(type='IterBasedRunner', max_iters=30000)  # 6k = 1 epoch when bs=128
lr_config = dict(by_epoch=False,)

