from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf


def get_conv(in_ch, out_ch, kernel_size, stride, padding, zero_bias=True, zero_weights=False):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
    if zero_bias:
        conv.bias.data.mul_(0.0)
    if zero_weights:
        conv.weight.data.mul_(0.0)
    return conv


def conv_k1s1(in_ch, out_ch, zero_bias=True, zero_weights=False):
    return get_conv(in_ch, out_ch, 1, 1, 0, zero_bias, zero_weights)

def conv_k3s1(in_ch, out_ch, zero_bias=True, zero_weights=False):
    return get_conv(in_ch, out_ch, 3, 1, 1, zero_bias, zero_weights)

def conv_k5s1(in_ch, out_ch, zero_bias=True, zero_weights=False):
    return get_conv(in_ch, out_ch, 5, 1, 2, zero_bias, zero_weights)

def conv_k3s2(in_ch, out_ch):
    return get_conv(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

def patch_downsample(in_ch, out_ch, rate=2):
    return get_conv(in_ch, out_ch, kernel_size=rate, stride=rate, padding=0)


def patch_upsample(in_ch, out_ch, rate=2):
    conv = nn.Sequential(
        get_conv(in_ch, out_ch * (rate ** 2), kernel_size=1, stride=1, padding=0),
        nn.PixelShuffle(rate)
    )
    return conv

def deconv(in_ch, out_ch, kernel_size=5, stride=2, zero_weights=False):
    conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                              output_padding=stride - 1, padding=kernel_size // 2)
    if zero_weights:
        conv.weight.data.mul_(0.0)
    return conv


class Conv1331Block(nn.Module):
    """ Adapted from VDVAE (https://github.com/openai/vdvae)
    - Paper: Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images
    - arxiv: https://arxiv.org/abs/2011.10650
    """
    def __init__(self, in_ch, hidden_ch=None, out_ch=None, use_3x3=True, zero_last=False):
        super().__init__()
        out_ch = out_ch or in_ch
        hidden_ch = hidden_ch or round(in_ch * 0.25)
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.residual = (in_ch == out_ch)
        self.c1 = conv_k1s1(in_ch, hidden_ch)
        self.c2 = conv_k3s1(hidden_ch, hidden_ch) if use_3x3 else conv_k1s1(hidden_ch, hidden_ch)
        self.c3 = conv_k3s1(hidden_ch, hidden_ch) if use_3x3 else conv_k1s1(hidden_ch, hidden_ch)
        self.c4 = conv_k1s1(hidden_ch, out_ch, zero_weights=zero_last)

    def residual_scaling(self, N):
        self.c4.weight.data.mul_(math.sqrt(1 / N))

    def forward(self, x):
        xhat = self.c1(tnf.gelu(x))
        xhat = self.c2(tnf.gelu(xhat))
        xhat = self.c3(tnf.gelu(xhat))
        xhat = self.c4(tnf.gelu(xhat))
        out = (x + xhat) if self.residual else xhat
        return out


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, out_dim=None, kernel_size=7, mlp_ratio=2,
                 residual=True, ls_init_value=1e-6):
        super().__init__()
        # depthwise conv
        pad = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        # layer norm
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.norm.affine = True # for FLOPs computing
        # MLP
        from timm.layers.mlp import Mlp
        hidden = int(mlp_ratio * dim)
        out_dim = out_dim or dim
        self.mlp = Mlp(dim, hidden_features=hidden, out_features=out_dim, act_layer=nn.GELU)
        # layer scaling
        if ls_init_value >= 0:
            self.gamma = nn.Parameter(torch.full(size=(1, out_dim, 1, 1), fill_value=1e-6))
        else:
            self.gamma = None

        self.residual = residual
        self.requires_embedding = True

    def forward(self, x):
        shortcut = x
        # depthwise conv
        x = self.conv_dw(x)
        # layer norm
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        # MLP
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # scaling
        if self.gamma is not None:
            x = x.mul(self.gamma)
        if self.residual:
            x = x + shortcut
        return x


class BottomUpEncoder(nn.Module):
    def __init__(self, blocks, dict_key='height'):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)
        self.dict_key = dict_key

    @torch.no_grad()
    def _get_dict_key(self, feature, x=None):
        if self.dict_key == 'height':
            key = int(feature.shape[2])
        elif self.dict_key == 'stride':
            key = round(x.shape[2] / feature.shape[2])
        else:
            raise ValueError(f'Unknown key: self.dict_key={self.dict_key}')
        return key

    def forward(self, x):
        feature = x
        enc_features = OrderedDict()
        for i, block in enumerate(self.enc_blocks):
            feature = block(feature)
            key = self._get_dict_key(feature, x)
            enc_features[key] = feature
        return enc_features
