import torch
import torch.nn as nn

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier



@CLASSIFIERS.register_module()
class PoolingVitClassifier(BaseClassifier):

    def __init__(self, extractor, convert, vit, neck=None, head=None, pretrained=None, freeze_backbone=False):
        super().__init__()
        if extractor:
            self.extractor = build_backbone(extractor)
        if freeze_backbone and extractor:
            print('freeze backbone: %s' % extractor['type'])
            self.extractor.eval()
            for param in self.extractor.parameters():
                    param.requires_grad = False
        if convert:
            self.convert = build_neck(convert)
        self.vit:nn.Module = build_backbone(vit)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        # super().init_weights(pretrained)
        # self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        aux_loss = dict()
        if hasattr(self, 'extractor'):
            x = self.extractor(img)
        else:
            x = img
        if hasattr(self, 'convert'):
            x = self.convert(x)
        else:
            x = dict(x=x)
        x = self.vit(**x)
        if isinstance(x, dict):
            aux_loss.update(x['loss'])
            x = x['x']
        if self.with_neck:
            x = self.neck(x)
        return x, aux_loss
    
    def extract_attn_map(self, img):
        if hasattr(self, 'extractor'):
            x = self.extractor(img)
        else:
            x = img
        if hasattr(self, 'convert'):
            x = self.convert(x)
        else:
            x = dict(x=x)
        x = self.vit(**x)
        return x['attn_map']

    def forward_train(self, img, gt_label, au_label=None, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x, aux_loss = self.extract_feat(img)

        if au_label is None:
            losses = self.head.forward_train(x, gt_label)
        else:
            losses = self.head.forward_train(x, gt_label, au_label)
        # losses['ce_loss'] = losses['loss']
        # losses['loss'] *= 0.
        # losses['aux_loss'] = aux_loss
        losses.update(aux_loss)

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        x, _ = self.extract_feat(img)
        return self.head.simple_test(x)
    
    def inference(self, img, **kwargs):
        x, _ = self.extract_feat(img)
        x = self.head.extract_feat(x)
        return x
    
    def aug_test(self, imgs, **kwargs): # TODO: pull request: add aug test to mmcls
        logit = self.inference(imgs[0], **kwargs)
        for i in range(1, len(imgs)):
            cur_logit = self.inference(imgs[i])
            logit += cur_logit
        logit /= len(imgs)
        # pred = F.softmax(logit, dim=1)
        pred = logit
        pred = pred.cpu().numpy()
        # unravel batch dim
        pred = list(pred)
        return pred
