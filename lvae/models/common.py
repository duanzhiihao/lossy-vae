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


class SetKey(nn.Module):
    """ A dummy layer that is used to mark the position of a layer in the network.
    """
    def __init__(self, key):
        super().__init__()
        self.key = key

    def forward(self, x):
        return x


class CompresionStopFlag(nn.Module):
    """ A dummy layer that is used to mark the stop position of encoding bits.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class FeatureExtracter(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x):
        features = OrderedDict()
        for i, block in enumerate(self.enc_blocks):
            if isinstance(block, SetKey):
                features[block.key] = x
            else:
                x = block(x)
        return features


class FeatureExtractorWithEmbedding(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x, emb=None):
        features = OrderedDict()
        for i, block in enumerate(self.enc_blocks):
            if isinstance(block, SetKey):
                features[block.key] = x
            elif getattr(block, 'requires_embedding', False):
                x = block(x, emb)
            else:
                x = block(x)
        return x, features


def sinusoidal_embedding(values: torch.Tensor, dim=256, max_period=64):
    assert values.dim() == 1 and (dim % 2) == 0
    exponents = torch.linspace(0, 1, steps=(dim // 2))
    freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
    args = values.view(-1, 1) * freqs.view(1, dim//2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class ConvNeXtBlockAdaLN(nn.Module):
    default_embedding_dim = 256
    def __init__(self, dim, embed_dim=None, out_dim=None, kernel_size=7, mlp_ratio=2,
                 residual=True, ls_init_value=1e-6):
        super().__init__()
        # depthwise conv
        pad = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        # layer norm
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm.affine = False # for FLOPs computing
        # AdaLN
        embed_dim = embed_dim or self.default_embedding_dim
        self.embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, 2*dim),
            nn.Unflatten(1, unflattened_size=(1, 1, 2*dim))
        )
        # MLP
        hidden = int(mlp_ratio * dim)
        out_dim = out_dim or dim
        from timm.layers.mlp import Mlp
        self.mlp = Mlp(dim, hidden_features=hidden, out_features=out_dim, act_layer=nn.GELU)
        # layer scaling
        if ls_init_value >= 0:
            self.gamma = nn.Parameter(torch.full(size=(1, out_dim, 1, 1), fill_value=1e-6))
        else:
            self.gamma = None

        self.residual = residual
        self.requires_embedding = True

    def forward(self, x, emb):
        shortcut = x
        # depthwise conv
        x = self.conv_dw(x)
        # layer norm
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        # AdaLN
        embedding = self.embedding_layer(emb)
        shift, scale = torch.chunk(embedding, chunks=2, dim=-1)
        x = x * (1 + scale) + shift
        # MLP
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        # scaling
        if self.gamma is not None:
            x = x.mul(self.gamma)
        if self.residual:
            x = x + shortcut
        return x

