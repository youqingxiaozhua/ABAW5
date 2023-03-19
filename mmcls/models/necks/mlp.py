# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import NECKS


@NECKS.register_module()
class SpatialFlatten(nn.Module):
    """
    Keep the spatial information of the feature map.
    [2048, 7, 7] -> [256, 7, 7] -> [256, 3, 3] -> [256*9]

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, in_channels=2048, ):
        super().__init__()
        
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        act_cfg = dict(type='ReLU')

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=256,
            kernel_size=2,
            padding=0,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            256, 256, 2,
            stride=2,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            x = inputs[-1]
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.flatten(start_dim=1)
            return (x, )
        else:
            raise TypeError('neck inputs should be tuple')
        return outs
