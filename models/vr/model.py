from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict, defaultdict
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision as tv
import torchvision.transforms.functional as tvf

from utils.image import crop_divisible_by, pad_divisible_by
import models.common as common
import models.entropy_coding as entropy_coding


MAX_LMB = 8192
# EMBED_MAX_PERIOD = 128
# lmb     is from 1 to MAX_LMB=8192
# log_lmb is from 0 to log(8192) ~= 9.01
# values  if from 0 to EMBED_MAX_PERIOD
# so we should scale log_lmb by EMBED_MAX_PERIOD / log(MAX_LMB)
# LOG_LMB_SCALE = EMBED_MAX_PERIOD / math.log(MAX_LMB)


def sinusoidal_embedding(values: torch.Tensor, dim=256, max_period=128):
    assert values.dim() == 1 and (dim % 2) == 0
    exponents = torch.linspace(0, 1, steps=(dim // 2))
    freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
    args = values.view(-1, 1) * freqs.view(1, dim//2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class VarLambdaMSEOutputNet():
    def __init__(self):
        super().__init__()
        self.loss_name = 'mse'

    def forward_loss(self, x_hat, x_tgt, mse_lmb):
        """ compute MSE loss

        Args:
            x_hat (torch.Tensor): reconstructed image
            x_tgt (torch.Tensor): original image
        """
        assert x_hat.shape == x_tgt.shape
        mse = tnf.mse_loss(x_hat, x_tgt, reduction='none').mean(dim=(1,2,3)) # (B,3,H,W) -> (B,)
        loss = mse * mse_lmb
        return loss, x_hat

    def mean(self, x_hat, temprature=None):
        return x_hat
    sample = mean


class MyConvNeXtBlockAdaLN(nn.Module):
    def __init__(self, dim, embed_dim, out_dim=None, kernel_size=7, mlp_ratio=2,
                 residual=True, ls_init_value=1e-6):
        super().__init__()
        # depthwise conv
        pad = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        # layer norm
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.norm.affine = False # for FLOPs computing
        # AdaLN
        self.embedding_layer = nn.Sequential(
            nn.GELU(),
            nn.Linear(embed_dim, 2*dim),
            nn.Unflatten(1, unflattened_size=(1, 1, 2*dim))
        )
        # MLP
        hidden = int(mlp_ratio * dim)
        out_dim = out_dim or dim
        from timm.models.layers.mlp import Mlp
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

class MyConvNeXtAdaLNPatchDown(MyConvNeXtBlockAdaLN):
    def __init__(self, in_ch, out_ch, down_rate=2, **kwargs):
        super().__init__(in_ch, **kwargs)
        self.downsapmle = common.get_conv(in_ch, out_ch, kernel_size=down_rate,
                                             stride=down_rate, padding=0)

    def forward(self, x, emb):
        x = super().forward(x, emb)
        out = self.downsapmle(x)
        return out

class MyConvNeXtResUpsample(MyConvNeXtBlockAdaLN):
    def __init__(self, in_ch, out_ch, embed_dim, up_rate=2, **kwargs):
        _out = out_ch * up_rate * up_rate
        super().__init__(dim=in_ch, embed_dim=embed_dim, out_dim=_out, residual=False, **kwargs)
        self.shortcut = common.conv_k1s1(in_ch, out_ch)

    def forward(self, x, emb):
        shortcut = tnf.interpolate(self.shortcut(x), scale_factor=2, mode='nearest')
        res = super().forward(x, emb)
        res = tnf.pixel_shuffle(res, upscale_factor=2)
        out = shortcut + res
        return out

class MyConvNeXtBlockAdaLNSoftPlus(MyConvNeXtBlockAdaLN):
    log_2 = math.log(2)
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
        x = x * tnf.softplus(scale, beta=self.log_2) + shift
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
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x, emb=None):
        feature = x
        enc_features = dict()
        for i, block in enumerate(self.enc_blocks):
            if getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
            res = int(feature.shape[2])
            enc_features[res] = feature
        return enc_features


class VRLatentBlock3Pos(nn.Module):
    def __init__(self, width, zdim, embed_dim, enc_width=None, kernel_size=7, mlp_ratio=2):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        enc_width = enc_width or width
        concat_ch = (width * 2) if (enc_width is None) else (width + enc_width)
        self.resnet_front = MyConvNeXtBlockAdaLN(width, embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end   = MyConvNeXtBlockAdaLN(width, embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.posterior0 = MyConvNeXtBlockAdaLN(enc_width, embed_dim, kernel_size=kernel_size)
        self.posterior1 = MyConvNeXtBlockAdaLN(width,     embed_dim, kernel_size=kernel_size)
        self.posterior2 = MyConvNeXtBlockAdaLN(width,     embed_dim, kernel_size=kernel_size)
        self.post_merge = common.conv_k1s1(concat_ch, width)
        self.posterior  = common.conv_k3s1(width, zdim)
        self.z_proj     = common.conv_k1s1(zdim, width)
        self.prior      = common.conv_k1s1(width, zdim*2)
        self.discrete_gaussian = entropy_coding.DiscretizedGaussian()

    def transform_prior(self, feature, lmb_embedding):
        """ prior p(z_i | z_<i)

        Args:
            feature (torch.Tensor): feature map
        """
        feature = self.resnet_front(feature, lmb_embedding)
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        return feature, pm, pv

    def transform_posterior(self, feature, enc_feature, lmb_embedding):
        """ posterior q(z_i | z_<i, x)

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        enc_feature = self.posterior0(enc_feature, lmb_embedding)
        feature = self.posterior1(feature, lmb_embedding)
        merged = torch.cat([feature, enc_feature], dim=1)
        merged = self.post_merge(merged)
        merged = self.posterior2(merged, lmb_embedding)
        qm = self.posterior(merged)
        return qm

    def fuse_feature_and_z(self, feature, z, lmb_embedding=None):
        feature = feature + self.z_proj(z)
        return feature

    def transform_end(self, feature, lmb_embedding):
        feature = self.resnet_end(feature, lmb_embedding)
        return feature

    def forward_train(self, feature, enc_feature, lmb_embedding, get_latents=False):
        """ training mode

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        feature, pm, pv = self.transform_prior(feature, lmb_embedding)
        # posterior q(z|x)
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        qm = self.transform_posterior(feature, enc_feature, lmb_embedding)
        # compute KL divergence
        if self.training:
            z_sample = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
            log_prob = entropy_coding.gaussian_log_prob_mass(pm, pv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
            kl = -1.0 * log_prob
        else:
            z_sample, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
            kl = -1.0 * torch.log(probs)
        # add the new information to feature
        feature = self.fuse_feature_and_z(feature, z_sample, lmb_embedding)
        feature = self.transform_end(feature, lmb_embedding)
        if get_latents:
            return feature, dict(z=z_sample.detach(), kl=kl)
        return feature, dict(kl=kl)

    def forward_uncond(self, feature, lmb_embedding, t=1.0, latent=None):
        feature, pm, pv = self.transform_prior(feature, lmb_embedding)
        pv = pv * t # modulate the prior scale by the temperature t
        if latent is None: # normal case. Just sampling.
            z_sample = pm + pv * torch.randn_like(pm) + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
        else: # if `latent` is provided, directly use it.
            assert pm.shape == latent.shape
            z_sample = latent
        feature = self.fuse_feature_and_z(feature, z_sample, lmb_embedding)
        feature = self.transform_end(feature, lmb_embedding)
        return feature

    def update(self):
        self.discrete_gaussian.update()

    def compress(self, feature, enc_feature, lmb_embedding):
        feature, pm, pv = self.transform_prior(feature, lmb_embedding)
        # posterior q(z|x)
        qm = self.transform_posterior(feature, enc_feature, lmb_embedding)
        # compress
        indexes = self.discrete_gaussian.build_indexes(pv)
        strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
        zhat = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
        # add the new information to feature
        feature = self.fuse_feature_and_z(feature, zhat, lmb_embedding)
        feature = self.transform_end(feature, lmb_embedding)
        return feature, strings

    def decompress(self, feature, lmb_embedding, strings):
        feature, pm, pv = self.transform_prior(feature, lmb_embedding)
        # decompress
        indexes = self.discrete_gaussian.build_indexes(pv)
        zhat = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        # add the new information to feature
        feature = self.fuse_feature_and_z(feature, zhat, lmb_embedding)
        feature = self.transform_end(feature, lmb_embedding)
        return feature


class TopDownDecoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.dec_blocks = nn.ModuleList(blocks)

        width = self.dec_blocks[0].in_channels
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))

    def get_bias(self, nhw_repeat=(1,1,1)):
        nB, nH, nW = nhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        # feature = torch.zeros(nB, self.initial_width, nH, nW, device=self._dummy.device)
        return feature

    def forward(self, enc_features, emb, get_latents=False):
        stats = []
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(nhw_repeat=(nB, nH, nW))
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_train'):
                res = int(feature.shape[2])
                f_enc = enc_features[res]
                feature, block_stats = block.forward_train(feature, f_enc, emb, get_latents=get_latents)
                stats.append(block_stats)
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
        return feature, stats

    def forward_uncond(self, emb, nhw_repeat=(1, 1, 1), t=1.0):
        feature = self.get_bias(nhw_repeat=nhw_repeat)
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_uncond'):
                feature = block.forward_uncond(feature, emb, t)
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
        return feature

    def forward_with_latents(self, emb, latents, nhw_repeat=None, t=1.0):
        if nhw_repeat is None:
            nB, _, nH, nW = latents[0].shape
            feature = self.get_bias(nhw_repeat=(nB, nH, nW))
        else: # use defined
            feature = self.get_bias(nhw_repeat=nhw_repeat)
        idx = 0
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_uncond'):
                feature = block.forward_uncond(feature, emb, t, latent=latents[idx])
                idx += 1
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
        return feature

    def update(self):
        for block in self.dec_blocks:
            if hasattr(block, 'update'):
                block.update()

    def compress(self, enc_features, lmb_embedding):
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(nhw_repeat=(nB, nH, nW))
        strings_all = []
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'compress'):
                # res = block.up_rate * feature.shape[2]
                res = feature.shape[2]
                f_enc = enc_features[res]
                feature, strs_batch = block.compress(feature, f_enc, lmb_embedding)
                strings_all.append(strs_batch)
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, lmb_embedding)
            else:
                feature = block(feature)
        # save smallest feature shape
        strings_all.append((nB, nH, nW))
        return strings_all, feature

    def decompress(self, compressed_object: list, lmb_embedding):
        nB, nH, nW = compressed_object[-1]
        feature = self.get_bias(nhw_repeat=(nB, nH, nW))
        str_i = 0
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'decompress'):
                strs_batch = compressed_object[str_i]
                str_i += 1
                feature = block.decompress(feature, lmb_embedding, strs_batch)
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, lmb_embedding)
            else:
                feature = block(feature)
        assert str_i == len(compressed_object) - 1, f'decoded={str_i}, len={len(compressed_object)}'
        return feature


class VariableRateLossyVAE(nn.Module):
    log2_e = math.log2(math.e)

    def __init__(self, config: dict):
        super().__init__()
        if 'encoder' in config:
            self.encoder = config.pop('encoder')
        else:
            self.encoder = BottomUpEncoder(blocks=config.pop('enc_blocks'))
        self.decoder = TopDownDecoder(blocks=config.pop('dec_blocks'))
        self.out_net = config.pop('out_net')

        self._set_lmb_embedding(config)

        self.im_shift = float(config['im_shift'])
        self.im_scale = float(config['im_scale'])
        self.max_stride = config['max_stride']

        self._stats_log = dict()
        self._log_images = config.get('log_images', None)
        self._log_smpl_k = [1, 2]
        self._flops_mode = False
        self.compressing = False
        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

    def _set_lmb_embedding(self, config):
        self.log_lmb_range = tuple(config['log_lmb_range'])
        assert len(self.log_lmb_range) == 2
        self.lmb_embed_dim = config['lmb_embed_dim']
        self.lmb_embedding = nn.Sequential(
            # nn.Linear(self.lmb_embed_dim[0], 1024),
            # nn.GELU(),
            # nn.Linear(1024, 1024),
            # nn.GELU(),
            # nn.Linear(1024, self.lmb_embed_dim[1]),
            nn.Linear(self.lmb_embed_dim[0], self.lmb_embed_dim[1]),
            nn.GELU(),
            nn.Linear(self.lmb_embed_dim[1], self.lmb_embed_dim[1]),
        )
        self._default_log_lmb = (self.log_lmb_range[0] + self.log_lmb_range[1]) / 2
        # experiment
        self._sin_period = config['sin_period']
        self.LOG_LMB_SCALE = self._sin_period / math.log(MAX_LMB)

    def preprocess_input(self, im: torch.Tensor):
        """ Shift and scale the input image

        Args:
            im (torch.Tensor): a batch of images, values should be between (0, 1)
        """
        assert (im.dim() == 4) and (0 <= im.min() <= im.max() <= 1) and not im.requires_grad
        x = im.clone().add_(self.im_shift).mul_(self.im_scale)
        return x

    def process_output(self, x: torch.Tensor):
        """ scale the decoder output from range (-1, 1) to (0, 1)

        Args:
            x (torch.Tensor): network decoder output, values should be between (-1, 1)
        """
        assert not x.requires_grad
        im_hat = x.clone().clamp_(min=-1.0, max=1.0).mul_(0.5).add_(0.5)
        return im_hat

    def preprocess_target(self, im: torch.Tensor):
        """ Shift and scale the image to make it reconstruction target

        Args:
            im (torch.Tensor): a batch of images, values should be between (0, 1)
        """
        assert (im.dim() == 4) and (0 <= im.min() <= im.max() <= 1) and not im.requires_grad
        x = im.clone().add_(-0.5).mul_(2.0)
        return x

    @torch.no_grad()
    def _forward_flops(self, im, lmb_embedding):
        im = im.uniform_(0, 1)
        if self._flops_mode == 'compress':
            compressed_obj = self.compress(im)
        elif self._flops_mode == 'decompress':
            n, h, w = im.shape[0], im.shape[2]//self.max_stride, im.shape[3]//self.max_stride
            samples = self.uncond_sample(nhw_repeat=(n,h,w))
        elif self._flops_mode == 'end-to-end':
            x = self.preprocess_input(im)
            x_target = self.preprocess_target(im)
            enc_features = self.encoder(x, lmb_embedding)
            feature, stats_all = self.decoder(enc_features, lmb_embedding)
            lmb = math.exp(self._default_log_lmb)
            out_loss, x_hat = self.out_net.forward_loss(feature, x_target, lmb)
        else:
            raise ValueError(f'Unknown self._flops_mode: {self._flops_mode}')
        return

    def sample_log_lmb(self, n):
        low, high = self.log_lmb_range
        log_lmb = low + (high-low) * torch.rand(n, device=self._dummy.device)
        return log_lmb

    def expand_log_lmb(self, log_lmb, n):
        log_lmb = torch.full(size=(n,), fill_value=float(log_lmb), device=self._dummy.device)
        return log_lmb

    def _get_lmb_embedding(self, log_lmb, n=None):
        if log_lmb is None: # use default log-lambda
            log_lmb = self._default_log_lmb
        if isinstance(log_lmb, (int, float)): # expand
            assert n is not None, f'log_lmb={log_lmb}, n must be provided'
            log_lmb = self.expand_log_lmb(log_lmb, n=n)
        else: # tensor
            assert n is None, f'log_lmb={log_lmb}, n should not be provided'
            assert isinstance(log_lmb, torch.Tensor) and (log_lmb.dim() == 1)
        # scaled = log_lmb * LOG_LMB_SCALE
        scaled = log_lmb * self.LOG_LMB_SCALE
        embedding = sinusoidal_embedding(scaled, dim=self.lmb_embed_dim[0], max_period=self._sin_period)
        embedding = self.lmb_embedding(embedding)
        return embedding

    def forward(self, batch, log_lmb=None, return_rec=False):
        device = self._dummy.device
        if isinstance(batch, (tuple, list)):
            im, label = batch
        else:
            im = batch
        im = im.to(device)
        nB, imC, imH, imW = im.shape # batch, channel, height, width

        # ================ get lambda ================
        if self.training:
            assert log_lmb is None
            log_lmb = self.sample_log_lmb(n=nB)
            lmb_embedding = self._get_lmb_embedding(log_lmb)
        else:
            lmb_embedding = self._get_lmb_embedding(log_lmb, n=nB)

        # ================ computing flops ================
        if self._flops_mode:
            return self._forward_flops(im, lmb_embedding)

        # ================ Forward pass ================
        # preprocessing
        x = self.preprocess_input(im)
        x_target = self.preprocess_target(im)
        # feedforward
        enc_features = self.encoder(x, lmb_embedding)
        feature, stats_all = self.decoder(enc_features, lmb_embedding)
        lmb = torch.exp(log_lmb) if isinstance(log_lmb, torch.Tensor) else math.exp(log_lmb)
        out_loss, x_hat = self.out_net.forward_loss(feature, x_target, lmb)

        # ================ Compute Loss ================
        kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        ndims = imC * imH * imW
        kl = sum(kl_divergences) / ndims
        loss = (kl + out_loss).mean(0) # rate + distortion

        # ================ Logging ================
        with torch.no_grad():
            nats_per_dim = kl.detach().cpu().mean(0).item()
            im_hat = self.process_output(x_hat.detach())
            im_mse = tnf.mse_loss(im_hat, im, reduction='mean')
            psnr = -10 * math.log10(im_mse.item())
            # logging
            kls = torch.stack([kl.mean(0) / ndims for kl in kl_divergences], dim=0)
            bpdim = kls * self.log2_e
            mode = 'train' if self.training else 'eval'
            self._stats_log[f'{mode}_bpdim'] = bpdim.tolist()
            self._stats_log[f'{mode}_bppix'] = (bpdim * imC).tolist()
            channel_kls = [stat['kl'].sum(dim=(2,3)).mean(0).cpu() / (imH * imW) for stat in stats_all]
            self._stats_log[f'{mode}_channels'] = [(kls*self.log2_e).tolist() for kls in channel_kls]
            debug = 1

        stats = OrderedDict()
        stats['loss']  = loss
        stats['kl']    = nats_per_dim
        stats[self.out_net.loss_name] = out_loss.detach().cpu().mean(0).item()
        stats['bppix'] = nats_per_dim * self.log2_e * imC
        stats['psnr']  = psnr
        if return_rec:
            stats['im_hat'] = im_hat
        return stats

    def forward_eval(self, im):
        stats = self.forward(im, log_lmb=self._default_log_lmb)
        return stats

    def uncond_sample(self, nhw_repeat, log_lmb=None, temprature=1.0):
        """ unconditionally sample, ie, generate new images

        Args:
            nhw_repeat (tuple): repeat the initial constant feature n,h,w times
            temprature (float): temprature
        """
        lmb_embedding = self._get_lmb_embedding(log_lmb, n=nhw_repeat[0])
        feature = self.decoder.forward_uncond(lmb_embedding, nhw_repeat, t=temprature)
        x_samples = self.out_net.sample(feature, temprature=temprature)
        im_samples = self.process_output(x_samples)
        return im_samples

    def cond_sample(self, latents, nhw_repeat=None, temprature=1.0):
        """ conditional sampling with latents

        Args:
            latents (torch.Tensor): latent variables
            nhw_repeat (tuple): repeat the constant n,h,w times
            temprature (float): temprature
        """
        feature = self.decoder.forward_with_latents(latents, nhw_repeat, t=temprature)
        x_samples = self.out_net.sample(feature, temprature=temprature)
        im_samples = self.process_output(x_samples)
        return im_samples

    def forward_get_latents(self, im):
        """ forward pass and return all the latent variables
        """
        raise NotImplementedError()
        x = self.preprocess_input(im)
        activations = self.encoder.forward(x)
        _, stats = self.decoder.forward(activations, get_latents=True)
        return stats

    @torch.no_grad()
    def study(self, save_dir, **kwargs):
        save_dir = Path(save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir(parents=False)

        device = next(self.parameters()).device
        # unconditional samples
        for k in self._log_smpl_k:
            num = 6
            im_samples = self.uncond_sample(nhw_repeat=(num,k,k))
            save_path = save_dir / f'samples_k{k}_hw{im_samples.shape[2]}.png'
            tv.utils.save_image(im_samples, fp=save_path, nrow=math.ceil(num**0.5))
        # reconstructions
        for imname in self._log_images:
            impath = f'images/{imname}'
            im = tv.io.read_image(impath).unsqueeze_(0).float().div_(255.0).to(device=device)
            stats = self.forward(im, log_lmb=self._default_log_lmb, return_rec=True)
            to_save = torch.cat([im, stats['im_hat']], dim=0)
            tv.utils.save_image(to_save, fp=save_dir / imname)

    def compress_mode(self, mode=True):
        if mode:
            self.decoder.update()
            if hasattr(self.out_net, 'compress'):
                self.out_net.update()
        self.compressing = mode

    @torch.no_grad()
    def compress(self, im, log_lmb=None):
        if log_lmb is None: # use default log-lambda
            log_lmb = self._default_log_lmb
        lmb_embedding = self._get_lmb_embedding(log_lmb, n=im.shape[0])
        x = self.preprocess_input(im)
        enc_features = self.encoder(x, lmb_embedding)
        strings_all, feature = self.decoder.compress(enc_features, lmb_embedding)
        strings_all.append(log_lmb)
        return strings_all

    @torch.no_grad()
    def decompress(self, compressed_object):
        log_lmb = compressed_object.pop()
        lmb_embedding = self._get_lmb_embedding(log_lmb, n=1)
        feature = self.decoder.decompress(compressed_object, lmb_embedding)
        x_hat = self.out_net.mean(feature)
        im_hat = self.process_output(x_hat)
        return im_hat

    @torch.no_grad()
    def compress_file(self, img_path, output_path):
        # read image
        img = Image.open(img_path)
        img_padded = pad_divisible_by(img, div=self.max_stride)
        device = next(self.parameters()).device
        im = tvf.to_tensor(img_padded).unsqueeze_(0).to(device=device)
        # compress by model
        compressed_obj = self.compress(im)
        compressed_obj.append((img.height, img.width))
        # save bits to file
        with open(output_path, 'wb') as f:
            pickle.dump(compressed_obj, file=f)

    @torch.no_grad()
    def decompress_file(self, bits_path):
        # read from file
        with open(bits_path, 'rb') as f:
            compressed_obj = pickle.load(file=f)
        img_h, img_w = compressed_obj.pop()
        # decompress by model
        im_hat = self.decompress(compressed_obj)
        return im_hat[:, :, :img_h, :img_w]

    @torch.no_grad()
    def _self_evaluate(self, img_paths, log_lmb, pbar=False, log_dir=None):
        pbar = tqdm(img_paths) if pbar else img_paths
        all_image_stats = defaultdict(float)
        self._stats_log = dict()
        if log_dir is not None:
            log_dir = Path(log_dir)
            channel_bpp_stats = OrderedDict()
        for impath in pbar:
            img = Image.open(impath)
            imgh, imgw = img.height, img.width
            img = crop_divisible_by(img, div=self.max_stride)
            # img_padded = pad_divisible_by(img, div=self.max_stride)
            im = tvf.to_tensor(img).unsqueeze_(0).to(device=self._dummy.device)
            stats = self.forward(im, log_lmb=log_lmb, return_rec=True)
            # compute bpp
            bpp_estimated = float(stats['bppix']) * (im.shape[2]*im.shape[3]) / (imgh * imgw)
            # compute psnr
            real = tvf.to_tensor(img)
            fake = stats['im_hat'].cpu().squeeze(0)
            mse = tnf.mse_loss(real, fake, reduction='mean').item()
            psnr = float(-10 * math.log10(mse))
            # accumulate results
            all_image_stats['count'] += 1
            all_image_stats['loss'] += stats['loss'].item()
            all_image_stats['bpp']  += bpp_estimated
            all_image_stats['psnr'] += psnr
            # debugging
            if log_dir is not None:
                ch_stats = [torch.Tensor(l) for l in self._stats_log[f'eval_channels']]
                for i, ch_bpp in enumerate(ch_stats):
                    channel_bpp_stats[i] = channel_bpp_stats.get(i, 0) + ch_bpp
        # average over all images
        count = all_image_stats.pop('count')
        avg_stats = {k: v/count for k,v in all_image_stats.items()}
        avg_stats['lambda'] = math.exp(log_lmb)
        if log_dir is not None:
            self._log_channel_stats(channel_bpp_stats, count, log_dir, log_lmb)

        return avg_stats

    @staticmethod
    def _log_channel_stats(channel_bpp_stats, count, log_dir, log_lmb):
        msg = '=' * 64 + '\n'
        msg += '---- row: latent blocks, colums: channels, avg over images ----\n'
        for k, v in channel_bpp_stats.items():
            channel_bpp_stats[k] = v / count
        for k, ch_bpp in channel_bpp_stats.items():
            msg += ''.join([f'{a:<7.4f} ' for a in ch_bpp.tolist()]) + '\n'
        msg += '---- colums: latent blocks, avg over images ----\n'
        block_bpps = [ch_bpp.sum() for k, ch_bpp in channel_bpp_stats.items()]
        msg += ''.join([f'{a:<7.4f} ' for a in block_bpps]) + '\n'
        lmb = round(math.exp(log_lmb))
        with open(log_dir / f'channel-bppix-lmb{lmb}.txt', mode='a') as f:
            print(msg, file=f)
        with open(log_dir / f'all_lmb_channel_stats.txt', mode='a') as f:
            print(msg, file=f)

    @torch.no_grad()
    def self_evaluate(self, img_dir, log_lmb_range=None, steps=8, log_dir=None):
        img_paths = list(Path(img_dir).rglob('*.*'))
        start, end = self.log_lmb_range if (log_lmb_range is None) else log_lmb_range
        log_lambdas = torch.linspace(start, end, steps=steps).tolist()
        pbar = tqdm(log_lambdas, position=0, ascii=True)
        all_lmb_stats = defaultdict(list)
        if log_dir is not None:
            (Path(log_dir) / 'all_lmb_channel_stats.txt').unlink(missing_ok=True)
        for log_lmb in pbar:
            results = self._self_evaluate(img_paths, log_lmb, log_dir=log_dir)
            msg = f'log_lmb={log_lmb:.3f}, lmb={math.exp(log_lmb):.1f}, results={results}'
            pbar.set_description(msg)
            for k,v in results.items():
                all_lmb_stats[k].append(v)
        return all_lmb_stats
