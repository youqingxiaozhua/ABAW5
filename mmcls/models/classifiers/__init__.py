# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .apvit import PoolingVitClassifier

__all__ = ['BaseClassifier', 'ImageClassifier', 'PoolingVitClassifier']
