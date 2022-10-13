from pathlib import Path
from collections import OrderedDict, defaultdict
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision as tv

from compressai.entropy_models import GaussianConditional

# from mycv.utils import AverageMeter, print_to_file
# import mycv.models.vae.blocks as vaeblocks
# import mycv.models.probabilistic.entropy_coding as myec


class FeatureFuseSimple1x1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1x1 = vaeblocks.get_1x1(dim * 2, dim)

        self.in_channels = dim
        self.requires_context = True

    def forward(self, x, context):
        assert x.shape == context.shape, f'x.shape={x.shape}, context.shape={context.shape}'
        concat = torch.cat([x, context], dim=1)
        return self.conv1x1(concat)

class FeatureFuseSimple3x3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = vaeblocks.get_3x3(dim * 2, dim)

        self.in_channels = dim
        self.requires_context = True

    def forward(self, x, context):
        assert x.shape == context.shape, f'x.shape={x.shape}, context.shape={context.shape}'
        concat = torch.cat([x, context], dim=1)
        return self.conv(concat)

class TemporalFeatureFuse(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.resblock = vaeblocks.Conv3x3ResBlock(dim * 2)
        self.conv1x1 = vaeblocks.get_1x1(dim * 2, dim)

        self.in_channels = dim
        self.requires_context = True

    def forward(self, x, context):
        assert x.shape == context.shape, f'x.shape={x.shape}, context.shape={context.shape}'
        concat = torch.cat([x, context], dim=1)
        concat = self.resblock(concat)
        return self.conv1x1(concat)


class TemporalBottomUpEncoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x, past_features):
        feature = x
        enc_features = dict()
        for i, block in enumerate(self.enc_blocks):
            if getattr(block, 'requires_context', False):
                f_past = past_features[int(feature.shape[2])]
                feature = block(feature, f_past)
            else:
                feature = block(feature)
            res = int(feature.shape[2])
            enc_features[res] = feature
        return enc_features


class TemporalTopDownDecoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.dec_blocks = nn.ModuleList(blocks)

        width = self.dec_blocks[0].in_channels
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))

    def get_bias(self, bhw_repeat=(1,1,1)):
        nB, nH, nW = bhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        return feature

    def forward(self, enc_features, past_features, get_latents=False):
        stats = []
        intermediate_features = dict()
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        for i, block in enumerate(self.dec_blocks):
            res = int(feature.shape[2])
            f_enc = enc_features[res]
            f_past = past_features[res] if (past_features is not None) else None
            if hasattr(block, 'forward_train'):
                feature, block_stats = block.forward_train(feature, f_enc, f_past, get_latents=get_latents)
                stats.append(block_stats)
            elif getattr(block, 'requires_context', False):
                feature = block(feature, f_past)
            else:
                feature = block(feature)
            intermediate_features[int(feature.shape[2])] = feature
        return feature, stats, intermediate_features

    def forward_uncond(self, past_features, bhw_repeat=(1, 1, 1), t=1.0):
        feature = self.get_bias(bhw_repeat=bhw_repeat)
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_uncond'):
                res = int(feature.shape[2])
                f_past = past_features[res]
                feature = block.forward_uncond(feature, f_past, t)
            else:
                feature = block(feature)
        return feature

    def forward_with_latents(self, latents, bhw_repeat=None, t=1.0):
        raise NotImplementedError()
        if bhw_repeat is None:
            nB, _, nH, nW = latents[0].shape
            feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        else: # use defined
            feature = self.get_bias(bhw_repeat=bhw_repeat)
        idx = 0
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_uncond'):
                feature = block.forward_uncond(feature, t, latent=latents[idx])
                idx += 1
            else:
                feature = block(feature)
        return feature

    def update(self):
        for block in self.dec_blocks:
            if hasattr(block, 'update'):
                block.update()

    def compress(self, enc_features, lmb_embedding):
        raise NotImplementedError()
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
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
        raise NotImplementedError()
        nB, nH, nW = compressed_object[-1]
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
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


class TemporalLatentBlock(nn.Module):
    def __init__(self, width, zdim, enc_ch=None, kernel_size=7, mlp_ratio=2):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        enc_ch = enc_ch or width
        concat_ch = (width * 2) if (enc_ch is None) else (width + enc_ch)
        self.resnet_front = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        # self.resnet_end   = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end = nn.Sequential(
            vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio),
            vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio),
        )
        self.posterior0   = vaeblocks.MyConvNeXtBlock(enc_ch, kernel_size=kernel_size)
        # self.posterior1   = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.posterior1   = nn.Identity()
        self.posterior2   = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size)
        # self.in_merge   = vaeblocks.get_1x1(width*2, width)
        self.post_merge = vaeblocks.get_1x1(concat_ch, width)
        self.posterior  = vaeblocks.get_3x3(width, zdim)
        self.z_proj     = vaeblocks.get_1x1(zdim, width)
        self.prior      = vaeblocks.get_1x1(width, zdim*2)
        self.discrete_gaussian = GaussianConditional(None, scale_bound=0.11)

    def transform_prior(self, feature, past_feature):
        """ prior p(z_i | z_<i)

        Args:
            feature (torch.Tensor): feature map
        """
        # feature = self.in_merge(torch.cat([feature, past_feature], dim=1))
        feature = self.resnet_front(feature)
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        return feature, pm, pv

    def transform_posterior(self, feature, enc_feature, past_feature):
        """ posterior q(z_i | z_<i, x)

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        enc_feature = self.posterior0(enc_feature)
        feature = self.posterior1(feature)
        merged = torch.cat([feature, enc_feature], dim=1)
        merged = self.post_merge(merged)
        merged = self.posterior2(merged)
        qm = self.posterior(merged)
        return qm

    def fuse_feature_and_z(self, feature, z):
        feature = feature + self.z_proj(z)
        return feature

    def transform_end(self, feature):
        feature = self.resnet_end(feature)
        return feature

    def forward_train(self, feature, enc_feature, past_feature, get_latents=False):
        """ training mode

        Args:
            feature      (torch.Tensor): feature map
            enc_feature  (torch.Tensor): feature map
            past_feature (torch.Tensor): feature map
        """
        feature, pm, pv = self.transform_prior(feature, past_feature)
        # posterior q(z|x)
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        qm = self.transform_posterior(feature, enc_feature, past_feature)
        # compute KL divergence
        if self.training:
            z_sample = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
            log_prob = myec.gaussian_log_prob_mass(pm, pv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
            kl = -1.0 * log_prob
        else:
            z_sample, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
            kl = -1.0 * torch.log(probs)
        # add the new information to feature
        feature = self.fuse_feature_and_z(feature, z_sample)
        feature = self.transform_end(feature)
        if get_latents:
            return feature, dict(z=z_sample.detach(), kl=kl)
        return feature, dict(kl=kl)

    def forward_sampling(self, feature, cond_feature, t=1.0, latent=None):
        feature, pm, pv = self.transform_prior(feature, cond_feature)
        pv = pv * t # modulate the prior scale by the temperature t
        if latent is None: # if no latent provided, sample it from prior.
            z_sample = pm + pv * torch.randn_like(pm) + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
        else: # if `latent` is provided, directly use it.
            assert pm.shape == latent.shape
            z_sample = latent
        feature = self.fuse_feature_and_z(feature, z_sample)
        feature = self.transform_end(feature)
        return feature

    def update(self):
        min_scale = 0.1
        max_scale = 20
        log_scales = torch.linspace(math.log(min_scale), math.log(max_scale), steps=64)
        scale_table = torch.exp(log_scales)
        updated = self.discrete_gaussian.update_scale_table(scale_table)
        self.discrete_gaussian.update()

    def compress(self, feature, enc_feature):
        raise NotImplementedError()
        feature, pm, pv = self.transform_prior(feature)
        # posterior q(z|x)
        qm = self.transform_posterior(feature, enc_feature)
        # compress
        indexes = self.discrete_gaussian.build_indexes(pv)
        strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
        zhat = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
        # add the new information to feature
        feature = self.fuse_feature_and_z(feature, zhat)
        # feature = feature + self.z_proj(zhat)
        feature = self.transform_end(feature)
        return feature, strings

    def decompress(self, feature, strings):
        raise NotImplementedError()
        feature, pm, pv = self.transform_prior(feature)
        # decompress
        indexes = self.discrete_gaussian.build_indexes(pv)
        zhat = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        # add the new information to feature
        feature = self.fuse_feature_and_z(feature, zhat)
        # feature = feature + self.z_proj(zhat)
        feature = self.transform_end(feature)
        return feature


class SpyCodingFlowBlock(nn.Module):
    def __init__(self, in_dim, dim=64, zdim=4, ks=7):
        super().__init__()
        # self.in_channels : int
        # self.out_channels: int

        self.shared = nn.Sequential(
            vaeblocks.conv_k3s1(in_dim, dim),
            vaeblocks.MyConvNeXtBlock(dim, dim, kernel_size=ks)
        )
        self.prior = vaeblocks.Conv1331Block(dim+2, hidden_ch=dim//2, out_ch=zdim*2, zero_last=True)

        ch = round(dim*1.5)
        self.posterior = nn.Sequential(
            vaeblocks.conv_k3s1(2 + dim*2, ch),
            vaeblocks.MyConvNeXtBlock(ch, ch, kernel_size=ks),
            vaeblocks.Conv1331Block(ch, hidden_ch=dim//2, out_ch=zdim)
        )

        self.z_proj = vaeblocks.conv_k1s1(zdim, 2)

        self.discrete_gaussian = myec.DiscretizedGaussian()

    def transform_prior(self, flow, f_prev):
        feature = torch.cat([flow, f_prev], dim=1)
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        return pm, pv

    def transform_posterior(self, flow, f_prev, f_curr):
        feature = torch.cat([flow, f_prev, f_curr], dim=1)
        qm = self.posterior(feature)
        return qm

    def fuse_feature_and_z(self, feature, z):
        feature = feature + self.z_proj(z)
        return feature

    def forward(self, flow, f_prev, f_curr, get_latents=False):
        assert flow.shape[2:4] == f_prev.shape[2:4] == f_curr.shape[2:4]
        f_prev, f_curr = self.shared(torch.cat([f_prev, f_curr], dim=0)).chunk(2, dim=0)
        # prior
        pm, pv = self.transform_prior(flow, f_prev)
        # posterior q(z|x)
        qm = self.transform_posterior(flow, f_prev, f_curr)
        # compute KL divergence
        if self.training:
            z_sample = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
            log_prob = myec.gaussian_log_prob_mass(pm, pv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
            kl = -1.0 * log_prob
        else:
            z_sample, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
            kl = -1.0 * torch.log(probs)
        # add the new information to feature
        flow = self.fuse_feature_and_z(flow, z_sample)
        if get_latents:
            return flow, dict(z=z_sample.detach(), kl=kl)
        return flow, dict(kl=kl)


class CorrBlock(nn.Module):
    def __init__(self, dim=128, strided_corr=False):
        super().__init__()
        from mmcv.ops.correlation import Correlation
        from mmflow.models.decoders.liteflownet_decoder import Upsample
        if strided_corr:
            self.correlation = Correlation(max_displacement=3, stride=2, dilation_patch=2)
            self.stride = 2
            self.upsample_if_needed = Upsample(scale_factor=2, channels=7*7)
        else:
            self.correlation = Correlation(max_displacement=3)
            self.stride = 1
            self.upsample_if_needed = nn.Identity()
        self.act = nn.LeakyReLU(0.1)
        self.pred_flow = nn.Sequential(
            vaeblocks.conv_k3s1(49, dim),
            nn.LeakyReLU(0.1),
            vaeblocks.conv_k3s1(dim, 96),
            nn.LeakyReLU(0.1),
            vaeblocks.conv_k3s1(96, 64),
            nn.LeakyReLU(0.1),
            vaeblocks.conv_k3s1(64, 32),
            nn.LeakyReLU(0.1),
        )

    def forward(self, f_prev, f_curr):
        B, C, H, W = f_prev.shape
        corr = self.correlation(f_prev, f_curr)
        assert corr.shape[-2:] == (H // self.stride, W // self.stride)
        corr = corr.view(B, -1, H // self.stride, W // self.stride)
        out = self.act(corr)
        out = self.upsample_if_needed(out)
        assert out.shape[2:4] == f_prev.shape[2:4]
        out = self.pred_flow(out)
        return out

class CorrelationFlowCodingBlock(nn.Module):
    def __init__(self, in_dim, dim=64, zdim=4, ks=7, corr_dim=128, strided_corr=False):
        super().__init__()
        self.shared = vaeblocks.conv_k3s1(in_dim, dim)
        self.correlation = CorrBlock(corr_dim, strided_corr=strided_corr)

        self.prior = nn.Sequential(
            vaeblocks.conv_k5s1(2, 64),
            nn.LeakyReLU(0.1),
            vaeblocks.conv_k3s1(64, 32),
            nn.LeakyReLU(0.1),
            vaeblocks.conv_k3s1(32, zdim*2),
        )
        self.posterior = nn.Sequential(
            vaeblocks.conv_k3s1(32, zdim),
        )
        self.z_proj = vaeblocks.conv_k1s1(zdim, 2)

        self.discrete_gaussian = myec.DiscretizedGaussian()

    def transform_prior(self, flow, f_prev):
        # feature = torch.cat([flow, f_prev], dim=1)
        pm, plogv = self.prior(flow).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        return pm, pv

    def transform_posterior(self, flow, f_prev, f_curr):
        corr_feature = self.correlation(f_curr, f_prev)
        qm = self.posterior(corr_feature)
        return qm

    def fuse_feature_and_z(self, feature, z):
        feature = feature + self.z_proj(z)
        return feature

    def forward(self, flow, f_prev, f_curr, get_latents=False):
        assert flow.shape[2:4] == f_prev.shape[2:4] == f_curr.shape[2:4]
        f_prev = self.shared(f_prev)
        f_curr = self.shared(f_curr)
        # prior
        pm, pv = self.transform_prior(flow, f_prev)
        # posterior q(z|x)
        qm = self.transform_posterior(flow, f_prev, f_curr)
        # compute KL divergence
        if self.training:
            z_sample = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
            log_prob = myec.gaussian_log_prob_mass(pm, pv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
            kl = -1.0 * log_prob
        else:
            z_sample, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
            kl = -1.0 * torch.log(probs)
        # add the new information to feature
        flow = self.fuse_feature_and_z(flow, z_sample)
        if get_latents:
            return flow, dict(z=z_sample.detach(), kl=kl)
        return flow, dict(kl=kl)


class ResCatRes(nn.Module):
    def __init__(self, width, kernel_size=7, mlp_ratio=2):
        super().__init__()
        self.res0 = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.fuse = vaeblocks.conv_k3s1(width*2, width)
        self.res1 = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)

    def forward(self, inputs):
        feature, f_cond = inputs
        f_cond = self.res0(f_cond)
        feature = torch.cat([feature, f_cond], dim=1)
        feature = self.fuse(feature)
        feature = self.res1(feature)
        return feature

class ResConditional(nn.Module):
    def __init__(self, width, kernel_size=7):
        super().__init__()
        self.res0 = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.fuse = vaeblocks.conv_k3s1(width*2, width)
        self.block = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size, residual=False)

    def forward(self, inputs):
        feature, f_cond = inputs
        f_cond = self.res0(f_cond)
        res = torch.cat([feature, f_cond], dim=1)
        res = self.fuse(res)
        res = self.block(res)
        out = feature + res
        return out


class SpyCodingFrameBlock(nn.Module):
    def __init__(self, width, zdim, kernel_size=7):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        # self.resnet_front = ResCatRes(width, kernel_size=kernel_size)
        # self.resnet_end   = ResCatRes(width, kernel_size=kernel_size)
        self.resnet_front = ResConditional(width, kernel_size=kernel_size)
        self.resnet_end   = ResConditional(width, kernel_size=kernel_size)
        self.prior = vaeblocks.Conv1331Block(width, out_ch=zdim*2, zero_last=True)
        self.posterior = nn.Sequential(
            ResCatRes(width, kernel_size=kernel_size),
            vaeblocks.Conv1331Block(width, out_ch=zdim, zero_last=True)
        )
        self.z_proj = vaeblocks.conv_k1s1(zdim, width)

        self.discrete_gaussian = myec.DiscretizedGaussian()

    def transform_prior(self, feature, f_prev):
        feature = self.resnet_front([feature, f_prev])
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        return feature, pm, pv

    def transform_posterior(self, feature, f_curr):
        qm = self.posterior([feature, f_curr])
        return qm

    def fuse_feature_and_z(self, feature, f_prev, z):
        feature = feature + self.z_proj(z)
        feature = self.resnet_end([feature, f_prev])
        return feature

    def forward(self, feature, f_prev, f_curr, get_latents=False):
        feature, pm, pv = self.transform_prior(feature, f_prev)
        # posterior q(z|x)
        assert feature.shape[2:4] == f_curr.shape[2:4]
        qm = self.transform_posterior(feature, f_curr)
        # compute KL divergence
        if self.training:
            z_sample = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
            log_prob = myec.gaussian_log_prob_mass(pm, pv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
            kl = -1.0 * log_prob
        else:
            z_sample, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
            kl = -1.0 * torch.log(probs)
        # add the new information to feature
        feature = self.fuse_feature_and_z(feature, f_prev, z_sample)
        if get_latents:
            return feature, dict(z=z_sample.detach(), kl=kl)
        return feature, dict(kl=kl)


class ConditionalLatentBlockBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels : int
        self.out_channels: int

        self.resnet_front: nn.Module
        self.resnet_end  : nn.Module
        self.posterior0  : nn.Module
        self.posterior1  : nn.Module
        self.posterior2  : nn.Module
        self.conditional : nn.Module
        self.post_merge  : nn.Module
        self.posterior   : nn.Module
        self.z_proj      : nn.Module
        self.prior       : nn.Module
        self.discrete_gaussian = GaussianConditional(None, scale_bound=0.11)

    def transform_prior(self, feature, cond_feature):
        """ prior p(z_i | z_<i)

        Args:
            feature (torch.Tensor): feature map
        """
        feature = self.resnet_front(feature)
        prior_feature = torch.cat([feature, cond_feature], dim=1)
        pm, plogv = self.prior(prior_feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        return feature, pm, pv

    def transform_posterior(self, feature, enc_feature, cond_feature):
        """ posterior q(z_i | z_<i, x)

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        enc_feature = self.posterior0(enc_feature)
        feature = self.posterior1(feature)
        merged = torch.cat([feature, enc_feature, cond_feature], dim=1)
        merged = self.post_merge(merged)
        merged = self.posterior2(merged)
        qm = self.posterior(merged)
        return qm

    def fuse_feature_and_z(self, feature, z):
        feature = feature + self.z_proj(z)
        feature = self.resnet_end(feature)
        return feature

    def forward_train(self, feature, enc_feature, cond_feature, get_latents=False):
        """ training mode

        Args:
            feature      (torch.Tensor): feature map
            enc_feature  (torch.Tensor): feature map
            cond_feature (torch.Tensor): feature map
        """
        cond_feature = self.conditional(cond_feature)
        feature, pm, pv = self.transform_prior(feature, cond_feature)
        # posterior q(z|x)
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        qm = self.transform_posterior(feature, enc_feature, cond_feature)
        # compute KL divergence
        if self.training:
            z_sample = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
            log_prob = myec.gaussian_log_prob_mass(pm, pv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
            kl = -1.0 * log_prob
        else:
            z_sample, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
            kl = -1.0 * torch.log(probs)
        # add the new information to feature
        feature = self.fuse_feature_and_z(feature, z_sample)
        if get_latents:
            return feature, dict(z=z_sample.detach(), kl=kl)
        return feature, dict(kl=kl)

    def forward_sampling(self, feature, cond_feature, t=1.0, latent=None):
        feature, pm, pv = self.transform_prior(feature, cond_feature)
        pv = pv * t # modulate the prior scale by the temperature t
        if latent is None: # if no latent provided, sample it from prior.
            z_sample = pm + pv * torch.randn_like(pm) + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
        else: # if `latent` is provided, directly use it.
            assert pm.shape == latent.shape
            z_sample = latent
        feature = self.fuse_feature_and_z(feature, z_sample)
        return feature

    def update(self):
        min_scale = 0.1
        max_scale = 20
        log_scales = torch.linspace(math.log(min_scale), math.log(max_scale), steps=64)
        scale_table = torch.exp(log_scales)
        updated = self.discrete_gaussian.update_scale_table(scale_table)
        self.discrete_gaussian.update()

    def compress(self, feature, enc_feature):
        raise NotImplementedError()
        feature, pm, pv = self.transform_prior(feature)
        # posterior q(z|x)
        qm = self.transform_posterior(feature, enc_feature)
        # compress
        indexes = self.discrete_gaussian.build_indexes(pv)
        strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
        zhat = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
        # add the new information to feature
        feature = self.fuse_feature_and_z(feature, zhat)
        # feature = feature + self.z_proj(zhat)
        feature = self.transform_end(feature)
        return feature, strings

    def decompress(self, feature, strings):
        raise NotImplementedError()
        feature, pm, pv = self.transform_prior(feature)
        # decompress
        indexes = self.discrete_gaussian.build_indexes(pv)
        zhat = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        # add the new information to feature
        feature = self.fuse_feature_and_z(feature, zhat)
        # feature = feature + self.z_proj(zhat)
        feature = self.transform_end(feature)
        return feature

class ConditionalLatentBlock2(ConditionalLatentBlockBase):
    def __init__(self, width, zdim, enc_ch=None, kernel_size=7, mlp_ratio=2):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        enc_ch = enc_ch or width
        concat_ch = (width * 3) if (enc_ch is None) else (width*2 + enc_ch)
        self.resnet_front = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end = nn.Sequential(
            vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio),
            vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        )
        self.posterior0   = vaeblocks.MyConvNeXtBlock(enc_ch, kernel_size=kernel_size)
        self.posterior1   = nn.Identity()
        self.posterior2   = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.conditional  = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.post_merge = vaeblocks.get_1x1(concat_ch, width)
        self.posterior  = vaeblocks.get_3x3(width, zdim)
        self.z_proj     = vaeblocks.get_1x1(zdim, width)
        self.prior      = vaeblocks.get_1x1(width*2, zdim*2)

class ConditionalLatentBlock(ConditionalLatentBlockBase):
    def __init__(self, width, zdim, enc_ch=None, kernel_size=7, mlp_ratio=2):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        enc_ch = enc_ch or width
        concat_ch = (width * 3) if (enc_ch is None) else (width*2 + enc_ch)
        self.resnet_front = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end   = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.posterior0   = vaeblocks.MyConvNeXtBlock(enc_ch, kernel_size=kernel_size)
        self.posterior1   = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.posterior2   = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.conditional  = vaeblocks.MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.post_merge = vaeblocks.get_1x1(concat_ch, width)
        self.posterior  = vaeblocks.get_3x3(width, zdim)
        self.z_proj     = vaeblocks.get_1x1(zdim, width)
        self.prior      = vaeblocks.get_1x1(width*2, zdim*2)


class TemporalConditionalModel(nn.Module):
    log2_e = math.log2(math.e)

    def __init__(self, config):
        super().__init__()

        self.encoder = TemporalBottomUpEncoder(blocks=config.pop('enc_blocks'))
        self.decoder = TemporalTopDownDecoder(blocks=config.pop('dec_blocks'))

        if 'distortion_func' in config:
            self.distortion_func = config['distortion_func']
            self.distortion_lmb  = float(config['distortion_lmb'])
            self.distortion_name = str(config['distortion_name'])
        else:
            self.distortion_func = None

        self._stop_monitoring()

    def _stop_monitoring(self):
        self._monitor_flag = False
        self._bpp_stats = None

    def _start_monitoring(self):
        self._monitor_flag = True
        self._bpp_stats = defaultdict(AverageMeter)

    def _save_monitoring_results(self, log_dir, prefix=''):
        # self._bpp_stats[f'bpp-blocks'] = [bpps.sum(0).item() for bpps in channel_bpps]
        # self._bpp_stats[f'bpp-channels'] = [bpps.tolist() for bpps in channel_bpps]
        msg = '---- row: latent blocks, colums: channels, avg over images ----\n'
        keys = sorted(self._bpp_stats.keys())
        for k in keys:
            msg += ''.join([f'{a:<7.4f} ' for a in self._bpp_stats[k].avg.tolist()]) + '\n'
        print_to_file(msg, Path(log_dir)/f'{prefix}bpp-channels.txt', mode='w')
        block_bpps = [self._bpp_stats[k].avg.sum(dim=0).item() for k in keys]
        msg = ''.join([f'{a:<7.4f} ' for a in block_bpps])
        print_to_file(msg, Path(log_dir)/f'{prefix}bpp-blocks.txt', mode='w')

    def forward(self, im, context: dict, log_dir=None):
        nB, imC, imH, imW = im.shape # batch, channel, height, width
        num_pix = float(imH * imW)
        # im = im.to(device=self._dummy.device)

        # ================ Forward pass ================
        enc_features = self.encoder(im, context)
        x_hat, stats_all, dec_features = self.decoder(enc_features, context)

        # ================ Compute loss ================
        kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        bpp = self.log2_e * sum(kl_divergences).mean(0) / num_pix
        if (self.distortion_func is not None):
            loss = bpp + self.distortion_lmb * self.distortion_func(x_hat, im)
        else:
            loss = bpp

        # ================ Logging ================
        with torch.no_grad():
            # im_hat = self.process_output(x_hat.detach())
            im_hat = x_hat.detach()
            im_mse = tnf.mse_loss(im_hat, im, reduction='mean')
            psnr = -10 * math.log10(im_mse.item())
            if self._monitor_flag:
                for i, stat in enumerate(stats_all):
                    bpps = stat['kl'].cpu().sum(dim=(2,3)).mean(0).div_(num_pix).mul_(self.log2_e)
                    self._bpp_stats[f'bpp-ch{i}'].update(bpps)

        stats = OrderedDict()
        stats['loss'] = loss
        stats['bpp']  = float(bpp.detach().item())
        stats['psnr'] = float(psnr)

        context = {
            'x_hat': x_hat,
        }
        return stats, context

    def end_to_end_forward_pass(self, im, context, get_latents=False):
        enc_features = self.encoder(im, context)

        stats = []
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.decoder.get_bias(bhw_repeat=(nB, nH, nW))
        for i, block in enumerate(self.decoder.dec_blocks):
            key = int(feature.shape[2])
            f_enc = enc_features[key]
            f_ctx = context[key] if (context is not None) else None
            if hasattr(block, 'forward_train'):
                feature, block_stats = block.forward_train(feature, f_enc, f_ctx, get_latents=get_latents)
                stats.append(block_stats)
            elif getattr(block, 'requires_context', False):
                feature = block(feature, f_ctx)
            else:
                feature = block(feature)
        return feature, stats

    def cond_sample(self, latents, context, bhw_repeat=None, t=1.0):
        """ conditional sampling with latents

        Args:
            latents (torch.Tensor): latent variables
            bhw_repeat (tuple): repeat the constant n,h,w times
            t (float): temprature
        """
        if bhw_repeat is None:
            nB, _, nH, nW = latents[0].shape
            bhw_repeat = (nB, nH, nW)
        feature = self.decoder.get_bias(bhw_repeat=bhw_repeat)
        idx = 0
        for i, block in enumerate(self.decoder.dec_blocks):
            key = int(feature.shape[2])
            f_ctx = context[key] if (context is not None) else None
            if hasattr(block, 'forward_sampling'):
                feature = block.forward_sampling(feature, f_ctx, t, latent=latents[idx])
                idx += 1
            elif getattr(block, 'requires_context', False):
                feature = block(feature, f_ctx)
            else:
                feature = block(feature)
        im_samples = feature
        return im_samples

    def get_progressive_coding(self, im, context):
        _, stats_all = self.end_to_end_forward_pass(im, context, get_latents=True)
        # kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        # kl = sum(kl_divergences) / (imC * imH * imW)
        # bpdim = (stats_all['elbo'] * math.log2(math.e)).item()
        # bppix = bpdim * 3
        # print(f'bpdim={bpdim}, bppix={bppix}')

        progressive = []
        L = len(stats_all)
        for keep in range(1, L+1):
            latents = [stat['z'] if (i < keep) else None for (i,stat) in enumerate(stats_all)]
            # kl_divs = [stat['kl'] for (i,stat) in enumerate(stats_all) if (i < keep)]
            # kl = sum([kl.sum(dim=(1,2,3)) for kl in kl_divs]) / (imH * imW) * math.log2(math.e)
            sample = self.cond_sample(latents, context, t=0)
            progressive.append(sample)
            # print(f'Keep={keep}, bpp={kl.item()}')
        return progressive

    def save_progressive(self, im, context, save_path, is_flow=False):
        progressive = self.get_progressive_coding(im, context)
        nrow = len(progressive) # number of progressive coding results for one single image
        progressive = torch.stack(progressive, dim=0).transpose_(0,1).flatten(0,1)
        if is_flow:
            progressive = tv.utils.flow_to_image(progressive).float().div(255.0)
        # tv.utils.make_grid(progressive, nrow=nrow)
        tv.utils.save_image(progressive, fp=save_path, nrow=nrow)


class TemporalConditionalModel2(nn.Module):
    log2_e = math.log2(math.e)

    def __init__(self, config):
        super().__init__()

        self.encoder = TemporalBottomUpEncoder(blocks=config.pop('enc_blocks'))
        self.dec_blocks = nn.ModuleList(config.pop('dec_blocks'))
        width = self.dec_blocks[0].in_channels
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))

        if 'distortion_func' in config:
            self.distortion_func = config['distortion_func']
            self.distortion_lmb  = float(config['distortion_lmb'])
            self.distortion_name = str(config['distortion_name'])
        else:
            self.distortion_func = None

        self._stop_monitoring()

    def _stop_monitoring(self):
        self._monitor_flag = False
        self._bpp_stats = None

    def _start_monitoring(self):
        self._monitor_flag = True
        self._bpp_stats = defaultdict(AverageMeter)

    def _save_monitoring_results(self, log_dir, prefix=''):
        log_dir = Path(log_dir)
        if not log_dir.is_dir():
            log_dir.mkdir(parents=True)
        # self._bpp_stats[f'bpp-blocks'] = [bpps.sum(0).item() for bpps in channel_bpps]
        # self._bpp_stats[f'bpp-channels'] = [bpps.tolist() for bpps in channel_bpps]
        msg = '---- row: latent blocks, colums: channels, avg over images ----\n'
        keys = sorted(self._bpp_stats.keys())
        for k in keys:
            msg += ''.join([f'{a:<7.4f} ' for a in self._bpp_stats[k].avg.tolist()]) + '\n'
        print_to_file(msg, log_dir/f'{prefix}bpp-channels.txt', mode='w')
        block_bpps = [self._bpp_stats[k].avg.sum(dim=0).item() for k in keys]
        msg = ''.join([f'{a:<7.4f} ' for a in block_bpps])
        print_to_file(msg, log_dir/f'{prefix}bpp-blocks.txt', mode='a')

    def forward(self, im, context: dict, log_dir=None):
        nB, imC, imH, imW = im.shape # batch, channel, height, width
        num_pix = float(imH * imW)
        # im = im.to(device=self._dummy.device)

        # ================ Forward pass ================
        enc_features = self.encoder(im, context)
        x_hat, stats_all, dec_features = self.top_down_decode(enc_features, context)

        # ================ Compute loss ================
        kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        bpp = self.log2_e * sum(kl_divergences).mean(0) / num_pix
        if (self.distortion_func is not None):
            loss = bpp + self.distortion_lmb * self.distortion_func(x_hat, im)
        else:
            loss = bpp

        stats = OrderedDict()
        stats['loss'] = loss
        stats['bpp']  = float(bpp.detach().item())

        # ================ Logging ================
        with torch.no_grad():
            # im_hat = self.process_output(x_hat.detach())
            im_hat = x_hat.detach()
            if (self.distortion_func is not None):
                im_mse = tnf.mse_loss(im_hat, im, reduction='mean')
                psnr = -10 * math.log10(im_mse.item())
                stats['psnr'] = float(psnr)
            if self._monitor_flag:
                for i, stat in enumerate(stats_all):
                    bpps = stat['kl'].cpu().sum(dim=(2,3)).mean(0).div_(num_pix).mul_(self.log2_e)
                    self._bpp_stats[f'bpp-ch{i}'].update(bpps)

        context = {
            'x_hat': x_hat,
            'context_features': dec_features
        }
        return stats, context

    def get_bias(self, bhw_repeat=(1,1,1)):
        """ get the constant bias for decoding

        Args:
            bhw_repeat (tuple): repeats in (batch, height, width) dimensions. Defaults to (1,1,1).
        """
        nB, nH, nW = bhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        return feature

    def top_down_decode(self, enc_features, ctx_features, get_latents=False):
        stats = []
        intermediate_features = dict()
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        for i, block in enumerate(self.dec_blocks):
            res = int(feature.shape[2])
            f_enc = enc_features[res]
            f_ctx = ctx_features[res] if (ctx_features is not None) else None
            if hasattr(block, 'forward_train'):
                feature, block_stats = block.forward_train(feature, f_enc, f_ctx, get_latents=get_latents)
                stats.append(block_stats)
            elif getattr(block, 'requires_context', False):
                feature = block(feature, f_ctx)
            else:
                feature = block(feature)
            intermediate_features[int(feature.shape[2])] = feature
        intermediate_features.pop(int(feature.shape[2]))
        return feature, stats, intermediate_features

    # def end_to_end_forward_pass(self, im, context, get_latents=False):
    #     enc_features = self.encoder(im, context)

    #     stats = []
    #     nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
    #     feature = self.get_bias(bhw_repeat=(nB, nH, nW))
    #     for i, block in enumerate(self.dec_blocks):
    #         key = int(feature.shape[2])
    #         f_enc = enc_features[key]
    #         f_ctx = context[key] if (context is not None) else None
    #         if hasattr(block, 'forward_train'):
    #             feature, block_stats = block.forward_train(feature, f_enc, f_ctx, get_latents=get_latents)
    #             stats.append(block_stats)
    #         elif getattr(block, 'requires_context', False):
    #             feature = block(feature, f_ctx)
    #         else:
    #             feature = block(feature)
    #     return feature, stats

    def conditional_sample(self, latents, context, bhw_repeat=None, t=1.0):
        """ sampling, conditioned on (possibly a subset of) latents

        Args:
            latents (torch.Tensor): latent variables
            bhw_repeat (tuple): repeat the constant n,h,w times
            t (float): temprature
        """
        if bhw_repeat is None:
            nB, _, nH, nW = latents[0].shape
            bhw_repeat = (nB, nH, nW)
        feature = self.get_bias(bhw_repeat=bhw_repeat)
        idx = 0
        for i, block in enumerate(self.dec_blocks):
            key = int(feature.shape[2])
            f_ctx = context[key] if (context is not None) else None
            if hasattr(block, 'forward_sampling'):
                feature = block.forward_sampling(feature, f_ctx, t, latent=latents[idx])
                idx += 1
            elif getattr(block, 'requires_context', False):
                feature = block(feature, f_ctx)
            else:
                feature = block(feature)
        im_samples = feature
        return im_samples

    def get_progressive_coding(self, im, context):
        # ================ Forward pass ================
        enc_features = self.encoder(im, context)
        _, stats_all, _ = self.top_down_decode(enc_features, context, get_latents=True)
        # _, stats_all = self.end_to_end_forward_pass(im, context, get_latents=True)
        # kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        # kl = sum(kl_divergences) / (imC * imH * imW)
        # bpdim = (stats_all['elbo'] * math.log2(math.e)).item()
        # bppix = bpdim * 3
        # print(f'bpdim={bpdim}, bppix={bppix}')

        progressive = []
        L = len(stats_all)
        for keep in range(1, L+1):
            latents = [stat['z'] if (i < keep) else None for (i,stat) in enumerate(stats_all)]
            # kl_divs = [stat['kl'] for (i,stat) in enumerate(stats_all) if (i < keep)]
            # kl = sum([kl.sum(dim=(1,2,3)) for kl in kl_divs]) / (imH * imW) * math.log2(math.e)
            sample = self.conditional_sample(latents, context, t=0)
            progressive.append(sample)
            # print(f'Keep={keep}, bpp={kl.item()}')
        return progressive

    def save_progressive(self, im, context, save_path, is_flow=False):
        progressive = self.get_progressive_coding(im, context)
        nrow = len(progressive) # number of progressive coding results for one single image
        progressive = torch.stack(progressive, dim=0).transpose_(0,1).flatten(0,1)
        if is_flow:
            progressive = tv.utils.flow_to_image(progressive).float().div(255.0)
        # tv.utils.make_grid(progressive, nrow=nrow)
        tv.utils.save_image(progressive, fp=save_path, nrow=nrow)
