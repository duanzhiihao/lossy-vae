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
        return features


def sinusoidal_embedding(values: torch.Tensor, dim=256, max_period=64):
    assert values.dim() == 1 and (dim % 2) == 0
    exponents = torch.linspace(0, 1, steps=(dim // 2))
    freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
    args = values.view(-1, 1) * freqs.view(1, dim//2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class Permute(nn.Module):
    def __init__(self, *dims: tuple):
        """ Permute dimensions of a tensor
        """
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, dims=self.dims)


class LayerScale(nn.Module):
    def __init__(self, *shape, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(*shape))

    def forward(self, x):
        return x * self.gamma


class AdaptiveLayerNorm(nn.Module):
    """ Channel-last LayerNorm with adaptive affine parameters that depend on the \
        input embedding.
    """
    def __init__(self, dim: int, embed_dim: int):
        super().__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, 2*dim),
            # nn.Unflatten(dim=1, unflattened_size=(1, 1, 2*dim))
        )
        # TODO: initialize the affine parameters such that the initial transform is identity
        # self.embedding_layer[1].weight.data.mul_(0.01)
        # self.embedding_layer[1].bias.data.fill_(0.0)

    def forward(self, x, emb):
        # x: (B, ..., dim), emb: (B, embed_dim)
        x = self.layer_norm(x)
        scale, shift = self.embedding_layer(emb).chunk(2, dim=1) # (B, dim) x 2
        # (B, dim) -> (B, ..., dim)
        scale = torch.unflatten(scale, dim=1, sizes=[1] * (x.dim() - 2) + [self.dim])
        shift = torch.unflatten(shift, dim=1, sizes=[1] * (x.dim() - 2) + [self.dim])
        x = x * (1 + scale) + shift
        return x


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


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """ A simple implementation of the scaled dot product attention.

    Args:
        q (torch.Tensor): query, shape (..., n1, c1)
        k (torch.Tensor): key, shape (..., n2, c1)
        v (torch.Tensor): value, shape (..., n2, c2)

    Returns:
        torch.Tensor: output, shape (..., n1, c2)
    """
    assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2], f'{q.shape=}, {k.shape=}, {v.shape=}'
    assert (q.shape[-1] == k.shape[-1]) and (k.shape[-2] == v.shape[-2])
    # Batch, ..., N_tokens, C
    attn = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    attn = torch.softmax(attn, dim=-1)
    out = attn @ v
    assert out.shape[-2:] == (q.shape[-2], v.shape[-1])
    return out, attn


class MultiheadAttention(nn.Module):
    def __init__(self, in_dims: int, num_heads: int, attn_dim=None):
        super().__init__()

        if isinstance(in_dims, int):
            q_in, k_in, v_in = in_dims, in_dims, in_dims
        else:
            assert len(in_dims) == 3, f'Invalid {in_dims=}'
            q_in, k_in, v_in = in_dims

        attn_dim = attn_dim or q_in
        assert attn_dim % num_heads == 0, f"{num_heads=} must divide {attn_dim=}."

        self.num_heads = num_heads
        self.q_proj = nn.Linear(q_in, attn_dim)
        self.k_proj = nn.Linear(k_in, attn_dim)
        self.v_proj = nn.Linear(v_in, attn_dim)
        self.out_proj = nn.Linear(attn_dim, q_in)

    def split_heads(self, x: torch.Tensor):
        return x.unflatten(-1, sizes=[self.num_heads, -1]).transpose(-2, -3)

    def combine_heads(self, x):
        return x.transpose(-2, -3).flatten(-2, -1) # (..., N, C)

    def forward(self, q, k, v, return_attn=False):
        assert q.shape[:-2] == k.shape[:-2] == v.shape[:-2], f'{q.shape=}, {k.shape=}, {v.shape=}'
        # Input projections
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        # Separate into heads
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        # Attention
        out, attn = scaled_dot_product_attention(q, k, v)
        # Output
        out = self.combine_heads(out) # (..., N, C)
        out = self.out_proj(out)
        if return_attn:
            return out, attn
        return out
