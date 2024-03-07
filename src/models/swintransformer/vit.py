
# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Vision Transformer implementation."""

# from importlib import import_module
# from easydict import EasyDict as edict
import numpy as np

# import mindspore as ms
# import mindspore.ops as ops
# from mindspore.common.initializer import initializer
# from mindspore.common.parameter import Parameter
# from mindspore.nn import Cell, Dense, Dropout, SequentialCell
# from mindspore import Tensor
# from .representation import*
# MIN_NUM_PATCHES = 4

# class VitConfig:
#     """
#     VitConfig
#     """
#     def __init__(self, configs):
#         self.configs = configs

#         # network init
#         self.network_norm = ms.nn.LayerNorm((configs.normalized_shape,))
#         self.network_init = ms.common.initializer.Normal(sigma=1.0)
#         self.network_dropout_rate = 0.0
#         self.network_pool = 'cls'
#         self.network = ViT

#         # stem
#         self.stem_init = ms.common.initializer.XavierUniform()
#         self.stem = VitStem

#         # body
#         self.body_norm = ms.nn.LayerNorm
#         self.body_drop_path_rate = 0.0
#         self.body = Transformer

#         # body attention
#         self.attention_init = ms.common.initializer.XavierUniform()
#         self.attention_activation = ms.nn.Softmax()
#         self.attention_dropout_rate = 0.0
#         self.attention = Attention

#         # body feedforward
#         self.feedforward_init = ms.common.initializer.XavierUniform()
#         self.feedforward_activation = ms.nn.GELU()
#         self.feedforward_dropout_rate = 0.0
#         self.feedforward = FeedForward

#         # head
#         self.head = origin_head
#         self.head_init = ms.common.initializer.XavierUniform()
#         self.head_dropout_rate = 0.0
#         self.head_norm = ms.nn.LayerNorm((configs.normalized_shape,))
#         self.head_activation = ms.nn.GELU()


# class DropPath(Cell):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """

#     def __init__(self, drop_prob=None, seed=0):
#         super(DropPath, self).__init__()
#         self.keep_prob = 1 - drop_prob
#         seed = min(seed, 0) # always be 0

#         self.shape = ops.Shape()
#         self.ones = ops.Ones()
#         self.dropout = Dropout(p=1 - self.keep_prob)

#     def construct(self, x):
#         if self.training:
#             x_shape = self.shape(x) # B N C
#             mask = self.ones((x_shape[0], 1, 1), x.dtype)
#             x = self.dropout(mask)*x
#         return x


# class BatchDense(Cell):
#     """BatchDense module."""

#     def __init__(self, in_features, out_features, initialization, has_bias=True):
#         super().__init__()
#         self.out_features = out_features
#         self.dense = Dense(in_features, out_features, has_bias=has_bias)
#         self.dense.weight.set_data(initializer(initialization, [out_features, in_features]))
#         self.reshape = ops.Reshape()

#     def construct(self, x):
#         bs, seq_len, d_model = x.shape
#         out = self.reshape(x, (bs * seq_len, d_model))
#         out = self.dense(out)
#         out = self.reshape(out, (bs, seq_len, self.out_features))
#         return out


# class ResidualCell(Cell):
#     """Cell which implements x + f(x) function."""
#     def __init__(self, cell):
#         super().__init__()
#         self.cell = cell

#     def construct(self, x, **kwargs):
#         return self.cell(x, **kwargs) + x


# def pretrain_head(vit_config):
#     """Head for ViT pretraining."""
#     d_model = vit_config.configs.d_model
#     mlp_dim = vit_config.configs.mlp_dim
#     num_classes = vit_config.configs.num_classes

#     dropout_rate = vit_config.head_dropout_rate
#     initialization = vit_config.head_init
#     normalization = vit_config.head_norm
#     activation = vit_config.head_activation

#     dense1 = Dense(d_model, mlp_dim)
#     dense1.weight.set_data(initializer(initialization, [mlp_dim, d_model]))
#     dense2 = Dense(mlp_dim, num_classes)
#     dense2.weight.set_data(initializer(initialization, [num_classes, mlp_dim]))

#     return SequentialCell([
#         normalization,
#         dense1,
#         activation,
#         Dropout(p=dropout_rate),
#         dense2])


# def origin_head(vit_config):
#     """Head for ViT pretraining."""
#     d_model = vit_config.configs.d_model
#     num_classes = vit_config.configs.num_classes
#     initialization = vit_config.head_init
#     dense = Dense(d_model, num_classes)
#     dense.weight.set_data(initializer(initialization, [num_classes, d_model]))
#     return SequentialCell([dense])


# class VitStem(Cell):
#     """Stem layer for ViT."""

#     def __init__(self, vit_config):
#         super().__init__()
#         d_model = vit_config.configs.d_model
#         patch_size = vit_config.configs.patch_size
#         image_size = vit_config.configs.image_size
#         initialization = vit_config.stem_init
#         channels = 3

#         assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
#         num_patches = (image_size // patch_size) ** 2
#         assert num_patches > MIN_NUM_PATCHES, f'your number of patches {num_patches} is too small'
#         patch_dim = channels * patch_size ** 2

#         self.patch_size = patch_size
#         self.reshape = ops.Reshape()
#         self.transpose = ops.Transpose()
#         self.patch_to_embedding =  nn.Conv2d(3, 768, kernel_size=16, stride=16,has_bias=True)

#     def construct(self, img):
#         p = self.patch_size
#         bs, channels, h, w = img.shape
#        # x = self.reshape(img, (bs, channels, h // p, p, w // p, p))
#         #x = self.transpose(x, (0, 2, 4, 1, 3, 5))
#         #x = self.reshape(x, (bs, (h//p)*(w//p), channels*p*p))
#         x = self.patch_to_embedding( img)
#         flatten=nn.Flatten(start_dim=2)
#         x=flatten(x).transpose(0,2,1)
#         return x


# class ViT(Cell):
#     """Vision Transformer implementation."""

#     def __init__(self, vit_config):
#         super().__init__()

#         d_model = vit_config.configs.d_model
#         patch_size = vit_config.configs.patch_size
#         image_size = vit_config.configs.image_size

#         initialization = vit_config.network_init
#         pool = vit_config.network_pool
#         dropout_rate = vit_config.network_dropout_rate
#         norm = vit_config.network_norm

#         stem = vit_config.stem(vit_config)
#         body = vit_config.body(vit_config)
#         self.head = Moment_Probing_ViT(in_dim=768,num_classes=1000)
#         #self.head =nn.Dense(768,1000)
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'
#         num_patches = (image_size // patch_size) ** 2

#         if pool == "cls":
#             self.cls_token = Parameter(initializer(initialization, (1, 1, d_model)),
#                                        name='cls', requires_grad=True)
#             self.pos_embedding = Parameter(initializer(initialization, (1, num_patches + 1, d_model)),
#                                            name='pos_embedding', requires_grad=True)
#             self.tile = ops.Tile()
#             self.cat_1 = ops.Concat(axis=1)
#         else:
#             self.pos_embedding = Parameter(initializer(initialization, (1, num_patches, d_model)),
#                                            name='pos_embedding', requires_grad=True)
#             self.mean = ops.ReduceMean(keep_dims=False)
#         self.pool = pool

#         self.dropout = Dropout(p=dropout_rate)
#         self.stem = stem
#         self.body = body
#         self.norm = norm

#     def construct(self, img):
        
#         #cls=32*3*224*224
#         #sequence = mindspore.numpy.arange(0.0001, 0.0001*cls + 0.0001,0.0001,dtype=mindspore.float32)
#         #tensor = sequence.reshape(32,3,224,224)
#         #img = mindspore.Tensor(tensor, mindspore.float32)
#         ones=ops.Ones()
#         img=ones((32,3,224,224),mindspore.float32)
#         x = self.stem(img)
#         print("patchembed"+str(x))
#         bs, seq_len, _ = x.shape

#         if self.pool == "cls":
#             cls_tokens = self.tile(self.cls_token, (bs, 1, 1))
#             x = self.cat_1((cls_tokens, x)) # now x has shape = (bs, seq_len+1, d)
#             x += self.pos_embedding[:, :(seq_len + 1)]
#         else:
#             x += self.pos_embedding[:, :seq_len]

#         y = ops.cast(x, ms.float32)
#         y = self.dropout(y)
#         x = ops.cast(y, x.dtype)

#         x = self.body(x)
#         print("body"+str(x))
#         if self.norm is not None:
#             x = self.norm(x)
#         cls=x[:, 0]

#         return self.head(cls,x)
#         #return self.head(cls)

# class Attention(Cell):
#     """Attention layer implementation."""

#     def __init__(self, vit_config):
#         super().__init__()
#         d_model = vit_config.configs.d_model
#         dim_head = vit_config.configs.dim_head
#         heads = vit_config.configs.heads


#         activation = vit_config.attention_activation
#         dropout_rate = vit_config.attention_dropout_rate

#         self.dim_head = dim_head
#         self.heads = heads
#         self.scale = Tensor([dim_head ** -0.5])
#         self.qkv=Dense(d_model,3*d_model,has_bias=True)
#         self.dropout = Dropout(p=dropout_rate)
#         self.proj=Dense(d_model,d_model,has_bias=True)
#         self.activation = activation
#         #self.attn_drop =Dropout(0.1)
#         #auxiliary functions
#         self.reshape = ops.Reshape()
#         self.transpose = ops.Transpose()
#         self.mul = ops.Mul()
#         self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
#         self.attn_matmul_v = ops.BatchMatMul()
#         self.softmax_nz =True
          
#     def construct(self, x):
#         '''x size - BxNxd_model'''
#         ones=ops.Ones()
#         x=ones((32,197,768),mindspore.float32)
#         print("x"+str(x.shape))
#         b, n, d, h, hd = x.shape[0], x.shape[1], x.shape[2], self.heads, self.dim_head
#         qkv=self.qkv(x).reshape(b,n,3,h,hd).permute(2,0,3,1,4)
#         q=qkv[0]
#         k=qkv[0]
#         v=qkv[0]
#         atten=self.q_matmul_k(q, k)*self.scale
#         atten=ops.softmax(atten)
#         out=self.attn_matmul_v(atten, v).transpose((0, 2, 1, 3)).reshape(b,n,d)
#         out=self.proj(out)
#         print("attention"+str(out))
#         #out = self.reshape(out, (bs, seq_len, d_model))
#         return out


# class FeedForward(Cell):
#     """FeedForward layer implementation."""

#     def __init__(self, vit_config):
#         super().__init__()

#         d_model = vit_config.configs.d_model
#         hidden_dim = vit_config.configs.mlp_dim

#         initialization = vit_config.feedforward_init
#         activation = vit_config.feedforward_activation
#         dropout_rate = vit_config.feedforward_dropout_rate

#         self.ff1 = BatchDense(d_model, hidden_dim, initialization)
#         self.activation = activation
#         self.dropout = Dropout(p=dropout_rate)
#         self.ff2 = BatchDense(hidden_dim, d_model, initialization)

#     def construct(self, x):
#         y = self.ff1(x)

#         y = self.activation(y)
#         y = self.dropout(y)
#         y = self.ff2(y)
#         y = self.dropout(y)
#         return y


# class Transformer(Cell):
#     """Transformer implementation."""

#     def __init__(self, vit_config):
#         super().__init__()

#         depth = vit_config.configs.depth
#         drop_path_rate = vit_config.body_drop_path_rate

#         dpr = [x.item() for x in np.linspace(0, drop_path_rate, depth)]
#         att_seeds = [np.random.randint(1024) for _ in range(depth)]
#         mlp_seeds = [np.random.randint(1024) for _ in range(depth)]
#         layers = []
#         for i in range(depth):
#             normalization = vit_config.body_norm((vit_config.configs.normalized_shape,))
#             normalization2 = vit_config.body_norm((vit_config.configs.normalized_shape,))
#             attention = vit_config.attention(vit_config)
#             feedforward = vit_config.feedforward(vit_config)

#             if drop_path_rate > 0:
#                 layers.append(
#                     SequentialCell([
#                         ResidualCell(SequentialCell([normalization,
#                                                      attention,
#                                                      DropPath(dpr[i], att_seeds[i])])),
#                         ResidualCell(SequentialCell([normalization2,
#                                                      feedforward,
#                                                      DropPath(dpr[i], mlp_seeds[i])]))
#                     ])
#                 )
#             else:
#                 layers.append(
#                     SequentialCell([
#                         ResidualCell(SequentialCell([normalization,
#                                                      attention])),
#                         ResidualCell(SequentialCell([normalization2,
#                                                      feedforward]))
#                     ])
#                 )

#         self.layers = SequentialCell(layers)

#     def construct(self, x):
#         return self.layers(x)


# def load_function(func_name):
#     """Load function using its name."""
#     modules = func_name.split(".")
#     if len(modules) > 1:
#         module_path = ".".join(modules[:-1])
#         name = modules[-1]
#         module = import_module(module_path)
#         return getattr(module, name)
#     return func_name


# vit_cfg = edict({
#     'd_model': 768,
#     'depth': 12,
#     'heads': 12,
#     'mlp_dim': 3072,
#     'dim_head': 64,
#     'patch_size': 32,
#     'normalized_shape': 768,
#     'image_size': 224,
#     'num_classes': 1001,
# })


# def vit_base_patch16(args):
#     """vit_base_patch16"""
#     vit_cfg.d_model = 768
#     vit_cfg.depth = 12
#     vit_cfg.heads = 12
#     vit_cfg.mlp_dim = 3072
#     vit_cfg.dim_head = vit_cfg.d_model // vit_cfg.heads
#     vit_cfg.patch_size = 16
#     vit_cfg.normalized_shape = vit_cfg.d_model
#     vit_cfg.image_size = 224
#     vit_cfg.num_classes = 1000

#     #if args.vit_config_path != '':
#     #    print("get vit_config_path")
#     #    vit_config = load_function(args.vit_config_path)(vit_cfg)
#     #else:
#     #    print("get default_vit_cfg")
#      #   vit_config = VitConfig(vit_cfg)
#     vit_config = VitConfig(vit_cfg)
#     model = vit_config.network(vit_config)
#     return model


# def vit_base_patch32(args):
#     """vit_base_patch32"""
#     vit_cfg.d_model = 768
#     vit_cfg.depth = 12
#     vit_cfg.heads = 12
#     vit_cfg.mlp_dim = 3072
#     vit_cfg.dim_head = vit_cfg.d_model // vit_cfg.heads
#     vit_cfg.patch_size = 32
#     vit_cfg.normalized_shape = vit_cfg.d_model
#     vit_cfg.image_size = args.train_image_size
#     vit_cfg.num_classes = args.class_num

#     if args.vit_config_path != '':
#         print("get vit_config_path")
#         vit_config = load_function(args.vit_config_path)(vit_cfg)
#     else:
#         print("get default_vit_cfg")
#         vit_config = VitConfig(vit_cfg)

#     model = vit_config.network(vit_config)

#     return model

# def get_network(backbone_name, args):
#     """get_network"""
#     if backbone_name == 'vit_base_patch32':
#         backbone = vit_base_patch32(args=args)
#     elif backbone_name == 'vit_base_patch16':
#         backbone = vit_base_patch16(args=args)
#     else:
#         raise NotImplementedError
#     return backbone
import math
import logging
from functools import partial
from collections import OrderedDict
import os
from symbol import parameters 
import mindspore as ms
import mindspore.ops as ops
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell, Dense, Dropout, SequentialCell
from mindspore import Tensor
import mindspore.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv, resolve_pretrained_cfg, checkpoint_seq
from timm.models.layers import trunc_normal_, lecun_normal_, _assert,DropPath
from timm.models.layers.helpers import to_2tuple
from timm.models.registry import register_model
from .representation import *
import mindvision as msd
    
from timm.models.registry import register_model

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Mlp(nn.Cell):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        self.fc1 = Dense(in_features, hidden_features, has_bias=True)
        self.act = act_layer()
        #self.drop1 = ops.dropout()
        self.fc2 = Dense(hidden_features, out_features, has_bias=True)
        #self.drop2 = ops.dropout)

    def construct(self, x):  
        # print("premlp"+str(x))
        # print(ops.sum(x))
        x = self.fc1(x)
        # print("fc1"+str(x))         
        # print(ops.sum(x))        
        x = self.act(x)


        #x = self.drop1(x)
        x = self.fc2(x)

        #x = self.drop2(x)
        # print("postmlp"+str(x))
        # print(ops.sum(x))
        return x

class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Dense(dim, dim * 3, has_bias=True)
        self.attn_drop =attn_drop
        self.proj = Dense(dim, dim, has_bias=True)
        self.proj_drop = proj_drop



    def construct(self, x):
        #ones=ops.Ones()
        #x=ones((32,197,768),mindspore.float32)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(0,1,-1, -2)) * self.scale
        attn = ops.softmax(attn,axis=-1)
        #attn = ops.dropout(attn,self.attn_drop)

        x = (attn @ v).transpose(0,2,1,3).reshape(B, N, C)
        x = self.proj(x)
        #x = ops.dropout(x,self.proj_drop)
        return x


class LayerScale(nn.Cell):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        ones=ops.Ones(dim, mindspore.float32)
        self.gamma = nn.Parameter(init_values * ones)

    def construct(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Cell):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, tuning_mode=None,i=None):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer([dim])
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer([dim])
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.tuning_mode = tuning_mode
        if self.tuning_mode == 'psrp':
            self.psrp = PSRP(dim)


    def construct(self, x):
        if self.tuning_mode == 'psrp':

            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
     
            weight, bias = self.psrp(x)


            x = x + self.drop_path2(self.ls2(bias + (weight + 1)*self.mlp(self.norm2(x))))
        else:
            # print(self.drop_path1)
            # print(self.drop_path2)            
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
#            print("atten"+str(x))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            # print("block"+str(x))
            # print(ops.sum(x))
        return x


class ResPostBlock(nn.Cell):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def construct(self, x):
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelBlock(nn.Cell):

    def __init__(
            self, dim, num_heads, num_parallel=2, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.SequentialCell(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.SequentialCell(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward(self, x):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def construct(self, x):
        return self._forward(x)


class PatchEmbed(nn.Cell):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, tuning_mode=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.norm_layer = norm_layer

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,has_bias=True,pad_mode='valid')
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.tuning_mode = tuning_mode



    def construct(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        x = self.proj(x) 
        if self.flatten:
            x = x.flatten(start_dim=2).transpose(0,2, 1)  # BCHW -> BNC
        
        x = self.norm(x)
        return x



class VisionTransformer(nn.Cell):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='',
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, tuning_mode='linear_probe', probing_mode='mp'): 
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = nn.LayerNorm
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 if class_token else 0
        self.grad_checkpointing = False 

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, tuning_mode=tuning_mode)
        num_patches = self.patch_embed.num_patches

        self.cls_token = mindspore.Parameter(ops.zeros((1, 1, embed_dim))) if self.num_tokens > 0 else None
        self.pos_embed = mindspore.Parameter(ops.randn(1, num_patches + self.num_tokens, embed_dim) * .02)
        #self.pos_drop = ops.dropout(p=drop_rate)

        dpr = [x.item() for x in ops.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        self.tuning_mode = tuning_mode
        tuning_mode_list = [tuning_mode] * depth 

        self.probing_mode = probing_mode
        
        self.blocks = nn.SequentialCell(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, tuning_mode=tuning_mode_list[i],i=i)
            for i in range(depth)])


        self.norm = norm_layer([embed_dim]) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer([embed_dim]) if use_fc_norm else nn.Identity()
        
        if self.probing_mode == 'cls_token' or self.probing_mode == 'gap':
            self.head = Dense(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.probing_mode == 'mp':
            self.head = Moment_Probing_ViT(in_dim=self.embed_dim,num_classes=num_classes)
            



  
    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        #ones=ops.Ones()
        #x=ones((32,3,224,224),mindspore.float32)
        # cls=32*3*224*224
        # sequence = mindspore.numpy.arange(0.0001, 0.0001*cls + 0.0001, 0.0001,dtype=mindspore.float32)
        # tensor = sequence.reshape(32,3,224,224)
        # x = mindspore.Tensor(tensor, mindspore.float32)
        x = self.patch_embed(x)
        # print("patchembed"+str(x))
        # print(ops.sum(x))
        if self.cls_token is not None:
            x = ops.cat((self.cls_token.broadcast_to((x.shape[0], -1, -1)), x), axis=1)

        x = x + self.pos_embed
        # print("poseembed"+str(x))
        # print(ops.sum(x))
        # print("clstoken"+str(self.cls_token))
        # print(ops.sum(self.cls_token))
        if self.grad_checkpointing :
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)           
        return x 

    def forward_head(self, x):
        if self.probing_mode == 'mp':
            cls_token = self.fc_norm(x[:, 0])
            return self.head(cls_token, x)
        elif self.probing_mode == 'gap':
            x = x[:, self.num_tokens:].mean(dim=1)
            x = self.fc_norm(x)
            return self.head(x)
        elif self.probing_mode == 'cls_token': 
            x = x[:, 0]
            x = self.fc_norm(x)
            return self.head(x)
        else:
            assert 0, 'please choose from mp, gap, cls_token !'

    def construct(self, x):
        x = self.forward_features(x)
        #print("prehead"+str(x))
        #print(ops.sum(x))
        #ones=ops.Ones()
        #x=ones((32,197,768),mindspore.float32)    
        x = self.forward_head(x)
        #print("head"+str(x))
        #print(ops.sum(x))
        return x 




def vit_base_patch16(args):
    """vit_base_patch16"""
    d_model = 768
    depth = 12
    heads = 12
    mlp_dim = 3072
    dim_head = 768 // 12
    patch_size = 16
    normalized_shape = 768
    image_size = 224
    num_classes = 100

    #if args.vit_config_path != '':
    #    print("get vit_config_path")
    #    vit_config = load_function(args.vit_config_path)(vit_cfg)
    #else:
    #    print("get default_vit_cfg")
     #   vit_config = VitConfig(vit_cfg)
    #vit_config = VitConfig(vit_cfg)
    model = VisionTransformer(num_classes =num_classes)
    return model


if __name__ =='__main__':
    x = torch.randn(size=(2, 3, 224, 224))
    model = vit_base_patch16_224(probing_mode='mp')  
    y = model(x)
    print(y.shape)  
