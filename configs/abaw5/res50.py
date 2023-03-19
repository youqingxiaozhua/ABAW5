_base_ = [
    './res18.py',
]


model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=2,
        ),
    neck=dict(type='SpatialFlatten', ),
    head=dict(
        in_channels=2304,
        # in_channels=2048,
    )
)

evaluation = dict(interval=4000)

