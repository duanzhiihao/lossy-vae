import torch
import torch.nn as nn
import torch.nn.functional as tnf

import lvae.models.entropy_coding as ec
import lvae.models.common as common


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
            common.conv_k3s1(49, dim),
            nn.LeakyReLU(0.1),
            common.conv_k3s1(dim, 96),
            nn.LeakyReLU(0.1),
            common.conv_k3s1(96, 64),
            nn.LeakyReLU(0.1),
            common.conv_k3s1(64, 32),
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
        self.shared = common.conv_k3s1(in_dim, dim)
        self.correlation = CorrBlock(corr_dim, strided_corr=strided_corr)

        self.prior = nn.Sequential(
            common.conv_k5s1(2, 64),
            nn.LeakyReLU(0.1),
            common.conv_k3s1(64, 32),
            nn.LeakyReLU(0.1),
            common.conv_k3s1(32, zdim*2),
        )
        self.posterior = nn.Sequential(
            common.conv_k3s1(32, zdim),
        )
        self.z_proj = common.conv_k1s1(zdim, 2)

        self.discrete_gaussian = ec.DiscretizedGaussian()

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
            log_prob = ec.gaussian_log_prob_mass(pm, pv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
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
        self.res0 = common.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.fuse = common.conv_k3s1(width*2, width)
        self.res1 = common.MyConvNeXtBlock(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)

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
        self.res0 = common.MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.fuse = common.conv_k3s1(width*2, width)
        self.block = common.MyConvNeXtBlock(width, kernel_size=kernel_size, residual=False)

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
        self.prior = common.Conv1331Block(width, out_ch=zdim*2, zero_last=True)
        self.posterior = nn.Sequential(
            ResCatRes(width, kernel_size=kernel_size),
            common.Conv1331Block(width, out_ch=zdim, zero_last=True)
        )
        self.z_proj = common.conv_k1s1(zdim, width)

        self.discrete_gaussian = ec.DiscretizedGaussian()

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
            log_prob = ec.gaussian_log_prob_mass(pm, pv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
            kl = -1.0 * log_prob
        else:
            z_sample, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
            kl = -1.0 * torch.log(probs)
        # add the new information to feature
        feature = self.fuse_feature_and_z(feature, f_prev, z_sample)
        if get_latents:
            return feature, dict(z=z_sample.detach(), kl=kl)
        return feature, dict(kl=kl)
