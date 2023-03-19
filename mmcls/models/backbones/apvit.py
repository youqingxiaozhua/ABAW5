import os
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_state_dict
from mmcv.cnn.bricks.drop import DropPath
from mmcv.cnn.utils.weight_init import trunc_normal_

from mmcls.utils import get_root_logger
from ..builder import BACKBONES
from .base_backbone import BaseBackbone


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def resize_pos_embed_v2(posemb:torch.Tensor, token_num_new:int, num_tokens=1):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger = get_root_logger()
    _logger.info('Resized position embedding: %s to %s', posemb.shape, token_num_new)
    ntok_new = token_num_new
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def top_pool(x:torch.Tensor, dim=1, keep_num:int=None, keep_rate=None, alpha1=1, alpha2=0, exclude_first=True, **kwargs):
    """
    根据输入x 的值和方差来选择 topk 个元素，并返回其 index, index的shape为[B, keep_num, dim]
    选择标准为 alpha1 * mean + alpha2 * std
    args:
        exclude_first: if set to True, will return the index of the first element and the top k-1 elements
    """
    # print('random weight')
    # x = torch.rand(x.shape, device=x.device)
    assert x.ndim == 3, 'input x must have 3 dimensions(B, N, C)'
    assert not (keep_num is not None and keep_rate is not None), 'keep_num and keep_rate can not be assigned on the same time'
    assert not (keep_num is None and keep_rate is None)
    B, N, C = x.shape
    if exclude_first is True:
        x = x[:, 1:, :]
        N -= 1
    if keep_num is None:
        keep_num = max(int(N * keep_rate), 1)
    
    if N == keep_num:
        return None

    mean_weight = x.mean(dim=-1)
    if C == 1:
        std_weight = torch.zeros((B, N)).to(mean_weight.device)
    else:
        std_weight = x.std(dim=-1)
    pool_weight = alpha1 * mean_weight + alpha2 * std_weight
    pool_weight = pool_weight.unsqueeze(-1).expand(B, N, dim)

    if exclude_first is False:
        try:
            _, keep_index = torch.topk(pool_weight, k=keep_num, dim=1, sorted=False)
        except Exception as e:
            print(e)
            print('pool_weight', pool_weight.shape)
            print('k', keep_num)
            exit()
        keep_index, _ = torch.sort(keep_index, dim=1)
    else:
        # pool_weight = pool_weight[:, 1:, ...]
        _, keep_index = torch.topk(pool_weight, k=keep_num, dim=1, sorted=False)
        keep_index, _ = torch.sort(keep_index, dim=1)
        keep_index = torch.cat([torch.zeros([B, 1, dim]).type(torch.int16).to(keep_index.device), keep_index + 1], dim=1)
    return keep_index


class LANet(nn.Module):
    def __init__(self, channel_num, ratio=16):
        super().__init__()
        assert channel_num % ratio == 0, f"input_channel{channel_num} must be exact division by ratio{ratio}"
        self.channel_num = channel_num
        self.ratio = ratio
        self.relu = nn.ReLU(inplace=True)

        self.LA_conv1 = nn.Conv2d(channel_num, int(channel_num / ratio), kernel_size=1)
        self.bn1 = nn.BatchNorm2d(int(channel_num / ratio))
        self.LA_conv2 = nn.Conv2d(int(channel_num / ratio), 1, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        LA = self.LA_conv1(x)
        LA = self.bn1(LA)
        LA = self.relu(LA)
        LA = self.LA_conv2(LA)
        LA = self.bn2(LA)
        LA = self.sigmoid(LA)
        return LA
        # LA = LA.repeat(1, self.channel_num, 1, 1)
        # x = x*LA

        # return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
                 pool_config=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pool_config = pool_config


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, head_num, token_num, token_num]

        if self.pool_config:
            attn_method = self.pool_config.get('attn_method')
            if attn_method == 'SUM_ABS_1':
                attn_weight = attn[:, :, 0, :].transpose(-1, -2)    # [B, token_num, head_num]
                attn_weight = torch.sum(torch.abs(attn_weight), dim=-1).unsqueeze(-1)
            elif attn_method == 'SUM':
                attn_weight = attn[:, :, 0, :].transpose(-1, -2)    # [B, token_num, head_num]
                attn_weight = torch.sum(attn_weight, dim=-1).unsqueeze(-1)
            elif attn_method == 'MAX':
                attn_weight = attn[:, :, 0, :].transpose(-1, -2)
                attn_weight = torch.max(attn_weight, dim=-1)[0].unsqueeze(-1)
            else:
                raise ValueError('Invalid attn_method: %s' % attn_method)

            # attn_weight = torch.rand(attn_weight.shape, device=attn_weight.device)
            keep_index = top_pool(attn_weight, dim=self.dim, **self.pool_config)
        else:
            keep_index = None

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, keep_index


class PoolingBlock(nn.Module):

    def __init__(self, dim=0, num_heads=0, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_config=None, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            pool_config=pool_config)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        feature, keep_index = self.attn(self.norm1(x))
        x = x + self.drop_path(feature)
        if keep_index is not None:
            if len(keep_index) != x.shape[1]:
                x = x.gather(dim=1, index=keep_index)
                # pooled_x = []
                # for i in range(keep_index.shape[0]):
                #     pooled_x.append(x[i, keep_index[i, :, 0]])
                # x = torch.stack(pooled_x)
                # assert torch.all(torch.eq(quick_x, x))

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@BACKBONES.register_module()
class PoolingViT(BaseBackbone):
    """ 
    
    """
    def __init__(self, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer_eps=1e-5, freeze=False,
                 input_type='image',  pretrained=None, 
                 in_channels=[], patch_num=0,
                 attn_method='SUM_ABS_1',
                 cnn_pool_config=None,
                 vit_pool_configs=None,
                 multi_head_fusion=False,
                 sum_batch_mean=False,
                 **kwargs):
        super().__init__()
        if kwargs:
            print('Unused kwargs: ')
            print(kwargs)
        assert input_type  == 'feature', 'Only suit for hybrid model'
        self.sum_batch_mean = sum_batch_mean
        if sum_batch_mean:
            self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.multi_head_fusion = multi_head_fusion
        self.num_heads = num_heads
        if multi_head_fusion:
            assert vit_pool_configs is None, 'MultiHeadFusion only support original ViT Block, by now'

        self.input_type = input_type
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.projs = nn.ModuleList([nn.Conv2d(in_channels[i], embed_dim, 1,) for i in range(len(in_channels))])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, patch_num, embed_dim), requires_grad=True)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.attn_method = attn_method
        self.cnn_pool_config = cnn_pool_config
        if attn_method == 'LA':
            # self.attn_f = LANet(in_channels[-1], 16)
            self.attn_f = LANet(embed_dim, 16)
            
        elif attn_method == 'SUM':
            self.attn_f = lambda x: torch.sum(x, dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_1':
            self.attn_f = lambda x: torch.sum(torch.abs(x), dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_2':
            self.attn_f = lambda x: torch.sum(torch.pow(torch.abs(x), 2), dim=1).unsqueeze(1)
        elif attn_method == 'MAX':
            self.attn_f = lambda x: torch.max(x, dim=1)[0].unsqueeze(1)
        elif attn_method == 'MAX_ABS_1':
            self.attn_f = lambda x: torch.max(torch.abs(x), dim=1)[0].unsqueeze(1)
        elif attn_method == 'Random':
            self.attn_f = lambda x: x[:, torch.randint(high=x.shape[1], size=(1,))[0], ...].unsqueeze(1)
        else:
            raise ValueError("Unknown attn_method")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if vit_pool_configs is None:

            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, head_fusion=multi_head_fusion,
                    )
                for i in range(depth)])
        else:
            vit_keep_rates = vit_pool_configs['keep_rates']
            self.blocks = nn.ModuleList([
                PoolingBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                    pool_config=dict(keep_rate=vit_keep_rates[i], **vit_pool_configs),
                    )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.s2_pooling = nn.MaxPool2d(kernel_size=2)

        if pretrained:
            self.init_weights(pretrained, patch_num)
        else:
            trunc_normal_(self.patch_pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
        if freeze:
            self.apply(self._freeze_weights)

    def init_weights(self, pretrained, patch_num=0):
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed']     # [1, 197, 768] for base
        patch_pos_embed = pos_embed[:, 1:, :]

        if patch_num != pos_embed.shape[1] - 1:
            logger.warning(f'interpolate pos_embed from {patch_pos_embed.shape[1]} to {patch_num}')
            pos_embed_new = resize_pos_embed_v2(patch_pos_embed, patch_num, 0)
        else:   # 去掉 cls_token
            print('does not need to resize！')
            pos_embed_new = patch_pos_embed
        del state_dict['pos_embed']
        state_dict['patch_pos_embed'] = pos_embed_new
        state_dict['cls_pos_embed'] = pos_embed[:, 0, :].unsqueeze(1)

        if self.multi_head_fusion:
            # convert blocks.0.attn.qkv.weight to blocks.0.attn.qkv.0.weight
            num_groups = self.blocks[0].attn.group_number
            d = self.embed_dim // num_groups
            print('d', d)
            for k in list(state_dict.keys()):
                if k.startswith('blocks.'):
                    keys = k.split('.')
                    if  not (keys[2] == 'attn' and keys[3] == 'qkv'):
                        continue
                    for i in range(num_groups):
                        new_key = f'blocks.{keys[1]}.attn.qkv.{i}.weight'
                        new_value = state_dict[k][i*3*d:(i+1)*3*d, i*d: i*d+d]
                        state_dict[new_key] = new_value

                    del state_dict[k]

        for k in ('patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias'):
            del state_dict[k]
        load_state_dict(self, state_dict, strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _freeze_weights(self, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
        for param in m.parameters():
            param.requires_grad = False

    def forward_features(self, x):
        assert len(x) == 1, '目前只支持1个 stage'
        assert isinstance(x, list) or isinstance(x, tuple)
        if len(x) == 2: # S2, S3
            x[0] = self.s2_pooling(x[0])
        elif len(x) == 3:
            x[0] = nn.MaxPool2d(kernel_size=4)(x[0])
            x[1] = self.s2_pooling(x[1])
        if os.getenv('DEBUG_MODE') == '1':
            print(x[0].shape)

        x = [self.projs[i](x[i]) for i in range(len(x))]
        # x = x[0]
        B, C, H, W = x[-1].shape
        attn_map = self.attn_f(x[-1]) # [B, 1, H, W]
        if self.attn_method == 'LA':
            x[-1] = x[-1] * attn_map    #  to have gradient
        x = [i.flatten(2).transpose(2, 1) for i in x]
        # x = self.projs[0](x).flatten(2).transpose(2, 1)
        # disable the first row and columns
        # attn_map[:, :, 0, :] = 0.
        # attn_map[:, :, :, 0] = 0.
        attn_weight = attn_map.flatten(2).transpose(2, 1)

        # attn_weight = torch.rand(attn_weight.shape, device=attn_weight.device)
        
        x = torch.stack(x).sum(dim=0)   # S1 + S2 + S3
        x = x + self.patch_pos_embed
        
        B, N, C = x.shape
        
        if self.cnn_pool_config is not None:
            keep_indexes = top_pool(attn_weight, dim=C, **self.cnn_pool_config)
            if keep_indexes is not None:
                x = x.gather(dim=1, index=keep_indexes)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        cls_tokens = cls_tokens + self.cls_pos_embed

        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)    # (B, N, dim)
        if os.environ.get('DEBUG_MODE', '0') == '1':
            print('output', x.shape)
        x = x[:, 0]
        if self.sum_batch_mean:
            x = x + x.mean(dim=0) * self.alpha
        loss = dict()
        return x, loss, attn_map

    def forward(self, x, **kwargs):
        x, loss, attn_map = self.forward_features(x)
        return dict(x=x, loss=dict(VitDiv_loss=loss), attn_map=attn_map)

