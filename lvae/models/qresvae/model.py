import pickle
from collections import OrderedDict
from PIL import Image
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torch.distributions as td
import torchvision.transforms.functional as tvf
from compressai.entropy_models import GaussianConditional

import lvae.models.common as common
from lvae.models.entropy_coding import gaussian_log_prob_mass


class GaussianNLLOutputNet(nn.Module):
    def __init__(self, conv_mean, conv_scale, bin_size=1/127.5):
        super().__init__()
        self.conv_mean  = conv_mean
        self.conv_scale = conv_scale
        self.bin_size = bin_size
        self.loss_name = 'nll'

    def forward_loss(self, feature, x_tgt):
        """ compute negative log-likelihood loss

        Args:
            feature (torch.Tensor): feature given by the top-down decoder
            x_tgt (torch.Tensor): original image
        """
        feature = feature.float()
        p_mean = self.conv_mean(feature)
        p_logscale = self.conv_scale(feature)
        p_logscale = tnf.softplus(p_logscale + 16) - 16 # logscale lowerbound
        log_prob = gaussian_log_prob_mass(p_mean, torch.exp(p_logscale), x_tgt, bin_size=self.bin_size)
        assert log_prob.shape == x_tgt.shape
        nll = -log_prob.mean(dim=(1,2,3)) # BCHW -> (B,)
        return nll, p_mean

    def mean(self, feature):
        p_mean = self.conv_mean(feature)
        return p_mean

    def sample(self, feature, mode='continuous', temprature=None):
        p_mean = self.conv_mean(feature)
        p_logscale = self.conv_scale(feature)
        p_scale = torch.exp(p_logscale)
        if temprature is not None:
            p_scale = p_scale * temprature

        if mode == 'continuous':
            samples = p_mean + p_scale * torch.randn_like(p_mean)
        elif mode == 'discrete':
            raise NotImplementedError()
        else:
            raise ValueError()
        return samples

    def update(self):
        self.discrete_gaussian = GaussianConditional(None, scale_bound=0.11)
        device = next(self.parameters()).device
        self.discrete_gaussian = self.discrete_gaussian.to(device=device)
        lower = self.discrete_gaussian.lower_bound_scale.bound.item()
        max_scale = 20
        scale_table = torch.exp(torch.linspace(math.log(lower), math.log(max_scale), steps=128))
        updated = self.discrete_gaussian.update_scale_table(scale_table)
        self.discrete_gaussian.update()

    def _preapre_codec(self, feature, x=None):
        assert not feature.requires_grad
        pm = self.conv_mean(feature)
        pm = torch.round(pm * 127.5 + 127.5) / 127.5 - 1 # workaround to make sure lossless
        plogv = self.conv_scale(feature)
        # scale (-1,1) range to (-127.5, 127.5) range
        pm = pm / self.bin_size
        plogv = plogv - math.log(self.bin_size)
        if x is not None:
            x = x / self.bin_size
        return pm, plogv, x

    def compress(self, feature, x):
        pm, plogv, x = self._preapre_codec(feature, x)
        # compress
        indexes = self.discrete_gaussian.build_indexes(torch.exp(plogv))
        strings = self.discrete_gaussian.compress(x, indexes, means=pm)
        return strings

    def decompress(self, feature, strings):
        pm, plogv, _ = self._preapre_codec(feature)
        # decompress
        indexes = self.discrete_gaussian.build_indexes(torch.exp(plogv))
        x_hat = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        x_hat = x_hat * self.bin_size
        return x_hat


class MSEOutputNet(nn.Module):
    def __init__(self, mse_lmb):
        super().__init__()
        self.mse_lmb = float(mse_lmb)
        self.loss_name = 'mse'

    def forward_loss(self, x_hat, x_tgt):
        """ compute MSE loss

        Args:
            x_hat (torch.Tensor): reconstructed image
            x_tgt (torch.Tensor): original image
        """
        assert x_hat.shape == x_tgt.shape
        mse = tnf.mse_loss(x_hat, x_tgt, reduction='none').mean(dim=(1,2,3)) # (B,3,H,W) -> (B,)
        loss = mse * self.mse_lmb
        return loss, x_hat

    def mean(self, x_hat, temprature=None):
        return x_hat
    sample = mean


class VDBlock(nn.Module):
    """ Adapted from VDVAE (https://github.com/openai/vdvae)
    - Paper: Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images
    - arxiv: https://arxiv.org/abs/2011.10650
    """
    def __init__(self, in_ch, hidden_ch=None, out_ch=None, residual=True,
                 use_3x3=True, zero_last=False):
        super().__init__()
        out_ch = out_ch or in_ch
        hidden_ch = hidden_ch or round(in_ch * 0.25)
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.residual = residual
        self.c1 = common.conv_k1s1(in_ch, hidden_ch)
        self.c2 = common.conv_k3s1(hidden_ch, hidden_ch) if use_3x3 else common.conv_k1s1(hidden_ch, hidden_ch)
        self.c3 = common.conv_k3s1(hidden_ch, hidden_ch) if use_3x3 else common.conv_k1s1(hidden_ch, hidden_ch)
        self.c4 = common.conv_k1s1(hidden_ch, out_ch, zero_weights=zero_last)

    def residual_scaling(self, N):
        # This residual scaling improves stability and performance with many layers
        # https://arxiv.org/pdf/2011.10650.pdf, Appendix Table 3
        self.c4.weight.data.mul_(math.sqrt(1 / N))

    def forward(self, x):
        xhat = self.c1(tnf.gelu(x))
        xhat = self.c2(tnf.gelu(xhat))
        xhat = self.c3(tnf.gelu(xhat))
        xhat = self.c4(tnf.gelu(xhat))
        out = (x + xhat) if self.residual else xhat
        return out

class VDBlockPatchDown(VDBlock):
    def __init__(self, in_ch, out_ch, down_rate=2):
        super().__init__(in_ch, residual=True)
        self.downsapmle = common.patch_downsample(in_ch, out_ch, rate=down_rate)

    def forward(self, x):
        x = super().forward(x)
        out = self.downsapmle(x)
        return out


from timm.models.convnext import ConvNeXtBlock
class MyConvNeXtBlock(ConvNeXtBlock):
    def __init__(self, dim, mlp_ratio=2, **kwargs):
        super().__init__(dim, mlp_ratio=mlp_ratio, **kwargs)
        self.norm.affine = True # this variable is useless. just a workaround for flops computation

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

class MyConvNeXtPatchDown(MyConvNeXtBlock):
    def __init__(self, in_ch, out_ch, down_rate=2, mlp_ratio=2, kernel_size=7):
        super().__init__(in_ch, mlp_ratio=mlp_ratio, kernel_size=kernel_size)
        self.downsapmle = common.patch_downsample(in_ch, out_ch, rate=down_rate)

    def forward(self, x):
        x = super().forward(x)
        out = self.downsapmle(x)
        return out


class BottomUpEncoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x):
        feature = x
        enc_features = dict()
        for i, block in enumerate(self.enc_blocks):
            feature = block(feature)
            res = int(feature.shape[2])
            enc_features[res] = feature
        return enc_features


class QLatentBlockX(nn.Module):
    """ Latent block as described in the paper.
    """
    def __init__(self, width, zdim, enc_width=None, kernel_size=7):
        """
        Args:
            width       (int): number of feature channels
            zdim        (int): number of latent variable channels
            enc_width   (int, optional): number of encoder feature channels. \
                Defaults to `width` if not provided.
            kernel_size (int, optional): convolution kernel size. Defaults to 7.
        """
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        enc_width = enc_width or width
        hidden = int(max(width, enc_width) * 0.25)
        concat_ch = (width * 2) if enc_width is None else (width + enc_width)
        use_3x3 = (kernel_size >= 3)
        self.resnet_front = MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.resnet_end   = MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.posterior = VDBlock(concat_ch, hidden, zdim, residual=False, use_3x3=use_3x3)
        self.prior     = VDBlock(width, hidden, zdim * 2, residual=False, use_3x3=use_3x3,
                                 zero_last=True)
        self.z_proj = nn.Sequential(
            common.conv_k3s1(zdim, hidden//2) if use_3x3 else common.conv_k1s1(zdim, hidden//2),
            nn.GELU(),
            common.conv_k1s1(hidden//2, width),
        )
        self.discrete_gaussian = GaussianConditional(None)

    def residual_scaling(self, N):
        self.z_proj[2].weight.data.mul_(math.sqrt(1 / 3*N))

    def transform_prior(self, feature):
        """ prior p(z_i | z_<i)

        Args:
            feature (torch.Tensor): feature map
        """
        feature = self.resnet_front(feature)
        # prior p(z)
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        return feature, pm, plogv

    def forward_train(self, feature, enc_feature, get_latents=False):
        """ Training mode. Forward pass and compute KL.

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        feature, pm, plogv = self.transform_prior(feature)
        pv = torch.exp(plogv)
        # posterior q(z|x)
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        qm = self.posterior(torch.cat([feature, enc_feature], dim=1))
        # compute KL divergence
        if self.training:
            z_sample = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
            log_prob = gaussian_log_prob_mass(pm, pv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
            kl = -1.0 * log_prob
        else:
            z_sample, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
            kl = -1.0 * torch.log(probs)
        # add the new information to feature
        feature = feature + self.z_proj(z_sample)
        feature = self.resnet_end(feature)
        if get_latents:
            return feature, dict(z=z_sample.detach(), kl=kl)
        return feature, dict(kl=kl)

    def forward_uncond(self, feature, t=1.0, latent=None, paint_box=None):
        """ Sampling mode.

        Args:
            feature   (Tensor): feature map.
            t         (float):  tempreture. Defaults to 1.0.
            latent    (Tensor): latent variable z. Sample it from prior if not provided.
            paint_box (Tensor): masked box for inpainting. (x1, y1, x2, y2).
        """
        feature, pm, plogv = self.transform_prior(feature)
        pv = torch.exp(plogv)
        pv = pv * t # modulate the prior scale by the temperature t
        if latent is None: # normal case. Just sampling.
            z = pm + pv * torch.randn_like(pm) + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
        elif paint_box is not None: # partial sampling for inpainting
            nB, zC, zH, zW = latent.shape
            if min(zH, zW) == 1:
                z = latent
            else:
                x1, y1, x2, y2 = paint_box
                h_slice = slice(round(y1*zH), round(y2*zH))
                w_slice = slice(round(x1*zW), round(x2*zW))
                z_sample = pm + pv * torch.randn_like(pm) + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
                z_patch = z_sample[:, :, h_slice, w_slice]
                z = torch.clone(latent)
                z[:, :, h_slice, w_slice] = z_patch
        else: # if `latent` is provided and `paint_box` is not provided, directly use it.
            assert pm.shape == latent.shape
            z = latent
        feature = feature + self.z_proj(z)
        feature = self.resnet_end(feature)
        return feature

    def update(self):
        """ Prepare for entropy coding. Musted be called before compression.
        """
        min_scale = 0.1
        max_scale = 20
        log_scales = torch.linspace(math.log(min_scale), math.log(max_scale), steps=64)
        scale_table = torch.exp(log_scales)
        updated = self.discrete_gaussian.update_scale_table(scale_table)
        self.discrete_gaussian.update()

    def compress(self, feature, enc_feature):
        """ Forward pass, compression (encoding) mode.

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        feature, pm, plogv = self.transform_prior(feature)
        # posterior q(z|x)
        qm = self.posterior(torch.cat([feature, enc_feature], dim=1))
        # compress
        indexes = self.discrete_gaussian.build_indexes(torch.exp(plogv))
        strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
        zhat = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
        # add the new information to feature
        feature = feature + self.z_proj(zhat)
        feature = self.resnet_end(feature)
        return feature, strings

    def decompress(self, feature, strings):
        """ Forward pass, decompression (decoding) mode.

        Args:
            feature (torch.Tensor): feature map
            strings (list[str]):    encoded bits
        """
        feature, pm, plogv = self.transform_prior(feature)
        # decompress
        indexes = self.discrete_gaussian.build_indexes(torch.exp(plogv))
        zhat = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        # add the new information to feature
        feature = feature + self.z_proj(zhat)
        feature = self.resnet_end(feature)
        return feature


class TopDownDecoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.dec_blocks = nn.ModuleList(blocks)

        width = self.dec_blocks[0].in_channels
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))

        self._init_weights()

    def _init_weights(self):
        total_blocks = len([1 for b in self.dec_blocks if hasattr(b, 'residual_scaling')])
        for block in self.dec_blocks:
            if hasattr(block, 'residual_scaling'):
                block.residual_scaling(total_blocks)

    def forward(self, enc_features, get_latents=False):
        stats = []
        min_res = min(enc_features.keys())
        feature = self.bias.expand(enc_features[min_res].shape)
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_train'):
                res = int(feature.shape[2])
                f_enc = enc_features[res]
                feature, block_stats = block.forward_train(feature, f_enc, get_latents=get_latents)
                stats.append(block_stats)
            else:
                feature = block(feature)
        return feature, stats

    def forward_uncond(self, nhw_repeat=(1, 1, 1), t=1.0):
        nB, nH, nW = nhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_uncond'):
                feature = block.forward_uncond(feature, t)
            else:
                feature = block(feature)
        return feature

    def forward_with_latents(self, latents, nhw_repeat=None, t=1.0, paint_box=None):
        if nhw_repeat is None:
            nB, _, nH, nW = latents[0].shape
            feature = self.bias.expand(nB, -1, nH, nW)
        else: # use defined
            nB, nH, nW = nhw_repeat
            feature = self.bias.expand(nB, -1, nH, nW)
        idx = 0
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_uncond'):
                feature = block.forward_uncond(feature, t, latent=latents[idx], paint_box=paint_box)
                idx += 1
            else:
                feature = block(feature)
        return feature

    def update(self):
        for block in self.dec_blocks:
            if hasattr(block, 'update'):
                block.update()

    def compress(self, enc_features):
        # assert len(self.bias_xs) == 1
        min_res = min(enc_features.keys())
        feature = self.bias.expand(enc_features[min_res].shape)
        strings_all = []
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'compress'):
                # res = block.up_rate * feature.shape[2]
                res = feature.shape[2]
                f_enc = enc_features[res]
                feature, strs_batch = block.compress(feature, f_enc)
                strings_all.append(strs_batch)
            else:
                feature = block(feature)
        return strings_all, feature

    def decompress(self, compressed_object: list):
        # assert len(self.bias_xs) == 1
        smallest_shape = compressed_object[-1]
        feature = self.bias.expand(smallest_shape)
        # assert len(compressed_object) == len(self.dec_blocks)
        str_i = 0
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'decompress'):
                strs_batch = compressed_object[str_i]
                str_i += 1
                feature = block.decompress(feature, strs_batch)
            else:
                feature = block(feature)
        assert str_i == len(compressed_object) - 1, f'decoded={str_i}, len={len(compressed_object)}'
        return feature


class HierarchicalVAE(nn.Module):
    """ Class of general hierarchical VAEs
    """
    log2_e = math.log2(math.e)

    def __init__(self, config: dict):
        """ Initialize model

        Args:
            config (dict): model config dict
        """
        super().__init__()
        self.encoder = BottomUpEncoder(blocks=config.pop('enc_blocks'))
        self.decoder = TopDownDecoder(blocks=config.pop('dec_blocks'))
        self.out_net = config.pop('out_net')

        self.im_shift = float(config['im_shift'])
        self.im_scale = float(config['im_scale'])
        self.max_stride = config['max_stride']

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

        self._stats_log = dict()
        self._flops_mode = False
        self.compressing = False

    def preprocess_input(self, im: torch.Tensor):
        """ Shift and scale the input image

        Args:
            im (torch.Tensor): a batch of images, (N, C, H, W), values between (0, 1)
        """
        assert (im.shape[2] % self.max_stride == 0) and (im.shape[3] % self.max_stride == 0)
        if not self._flops_mode:
            assert (im.dim() == 4) and (0 <= im.min() <= im.max() <= 1) and not im.requires_grad
        x = (im + self.im_shift) * self.im_scale
        return x

    def process_output(self, x: torch.Tensor):
        """ scale the decoder output from range (-1, 1) to (0, 1)

        Args:
            x (torch.Tensor): network decoder output, (N, C, H, W), values between (-1, 1)
        """
        assert not x.requires_grad
        im_hat = x.clone().clamp_(min=-1.0, max=1.0).mul_(0.5).add_(0.5)
        return im_hat

    def preprocess_target(self, im: torch.Tensor):
        """ Shift and scale the image to make it reconstruction target

        Args:
            im (torch.Tensor): a batch of images, (N, C, H, W), values between (0, 1)
        """
        if not self._flops_mode:
            assert (im.dim() == 4) and (0 <= im.min() <= im.max() <= 1) and not im.requires_grad
        x = (im - 0.5) * 2.0
        return x

    def forward(self, im, return_rec=False):
        """ Forward pass for training

        Args:
            im (tensor): image, (B, 3, H, W)
            return_rec (bool, optional): if True, return the reconstructed image \
                in addition to losses. Defaults to False.

        Returns:
            dict: str -> loss
        """
        im = im.to(self._dummy.device)
        x = self.preprocess_input(im)
        x_target = self.preprocess_target(im)

        enc_features = self.encoder(x)
        feature, stats_all = self.decoder(enc_features)
        out_loss, x_hat = self.out_net.forward_loss(feature, x_target)

        if self._flops_mode: # testing flops
            return x_hat

        # ================ Training ================
        nB, imC, imH, imW = im.shape # batch, channel, height, width
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
            channel_bpps = [stat['kl'].sum(dim=(2,3)).mean(0).cpu() / (imH * imW) for stat in stats_all]
            self._stats_log[f'{mode}_channels'] = [(bpps*self.log2_e).tolist() for bpps in channel_bpps]

        stats = OrderedDict()
        stats['loss']  = loss
        stats['kl']    = nats_per_dim
        stats[self.out_net.loss_name] = out_loss.detach().cpu().mean(0).item()
        stats['bppix'] = nats_per_dim * self.log2_e * imC
        stats['psnr']  = psnr
        if return_rec:
            stats['im_hat'] = im_hat
        return stats

    @torch.no_grad()
    def forward_eval(self, *args, **kwargs):
        """ a dummy function for evaluation
        """
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def uncond_sample(self, nhw_repeat, temprature=1.0):
        """ unconditionally sample, ie, generate new images

        Args:
            nhw_repeat (tuple): repeat the initial constant feature n,h,w times
            temprature (float): temprature
        """
        feature = self.decoder.forward_uncond(nhw_repeat, t=temprature)
        x_samples = self.out_net.sample(feature, temprature=temprature)
        im_samples = self.process_output(x_samples)
        return im_samples

    @torch.no_grad()
    def cond_sample(self, latents, nhw_repeat=None, temprature=1.0, paint_box=None):
        """ conditional sampling with latents

        Args:
            latents (torch.Tensor): latent variables
            nhw_repeat (tuple): repeat the constant n,h,w times
            temprature (float): temprature
            paint_box (tuple of floats): (x1,y1,x2,y2), in 0-1 range
        """
        feature = self.decoder.forward_with_latents(latents, nhw_repeat, t=temprature, paint_box=paint_box)
        x_samples = self.out_net.sample(feature, temprature=temprature)
        im_samples = self.process_output(x_samples)
        return im_samples

    def forward_get_latents(self, im):
        """ forward pass and return all the latent variables
        """
        x = self.preprocess_input(im)
        activations = self.encoder.forward(x)
        _, stats = self.decoder.forward(activations, get_latents=True)
        return stats

    @torch.no_grad()
    def inpaint(self, im, paint_box, steps=1, temprature=1.0):
        """ Inpainting

        Args:
            im (tensor): image (with paint_box mased out)
            paint_box (tuple): (x1, y1, x2, y2)
            steps (int, optional): A larger `step` gives a slightly better result.
            temprature (float, optional): tempreture. Defaults to 1.0.

        Returns:
            tensor: inpainted image
        """
        nB, imC, imH, imW = im.shape
        x1, y1, x2, y2 = paint_box
        h_slice = slice(round(y1*imH), round(y2*imH))
        w_slice = slice(round(x1*imW), round(x2*imW))
        im_input = im.clone()
        for i in range(steps):
            stats_all = self.forward_get_latents(im_input)
            latents = [st['z'] for st in stats_all]
            im_sample = self.cond_sample(latents, temprature=temprature, paint_box=paint_box)
            torch.clamp_(im_sample, min=0, max=1)
            im_input = im.clone()
            im_input[:, :, h_slice, w_slice] = im_sample[:, :, h_slice, w_slice]
        return im_sample

    def compress_mode(self, mode=True):
        """ Prepare for entropy coding. Musted be called before compression.
        """
        if mode:
            self.decoder.update()
            if hasattr(self.out_net, 'compress'):
                self.out_net.update()
        self.compressing = mode

    @torch.no_grad()
    def compress(self, im):
        """ compress a batch of images

        Args:
            im (torch.Tensor): a batch of images, (N, C, H, W), values between (0, 1)

        Returns:
            list: [string1, string2, string2, ..., string_N, feature_shape]
        """
        x = self.preprocess_input(im)
        enc_features = self.encoder(x)
        compressed_obj, feature = self.decoder.compress(enc_features)
        min_res = min(enc_features.keys())
        compressed_obj.append(tuple(enc_features[min_res].shape))
        if hasattr(self.out_net, 'compress'): # lossless compression
            x_tgt = self.preprocess_target(im)
            final_str = self.out_net.compress(feature, x_tgt)
            compressed_obj.append(final_str)
        return compressed_obj

    @torch.no_grad()
    def decompress(self, compressed_object):
        """ decompress a compressed_object

        Args:
            compressed_object (list): same as the output of self.compress()

        Returns:
            torch.Tensor: a batch of reconstructed images, (N, C, H, W), values between (0, 1)
        """
        if hasattr(self.out_net, 'compress'): # lossless compression
            feature = self.decoder.decompress(compressed_object[:-1])
            x_hat = self.out_net.decompress(feature, compressed_object[-1])
        else: # lossy compression
            feature = self.decoder.decompress(compressed_object)
            x_hat = self.out_net.mean(feature)
        im_hat = self.process_output(x_hat)
        return im_hat

    @torch.no_grad()
    def compress_file(self, img_path, output_path):
        """ Compress an image file specified by `img_path` and save to `output_path`

        Args:
            img_path    (str): input image path
            output_path (str): output bits path
        """
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
        """ Decompress a bits file specified by `bits_path`

        Args:
            bits_path (str): input bits path

        Returns:
            torch.Tensor: reconstructed image
        """
        # read from file
        with open(bits_path, 'rb') as f:
            compressed_obj = pickle.load(file=f)
        img_h, img_w = compressed_obj.pop()
        # decompress by model
        im_hat = self.decompress(compressed_obj)
        return im_hat[:, :, :img_h, :img_w]


def pad_divisible_by(img, div=64):
    """ Pad an PIL.Image at right and bottom border \
         such that both sides are divisible by `div`.

    Args:
        img (PIL.Image): image
        div (int, optional): `div`. Defaults to 64.

    Returns:
        PIL.Image: padded image
    """
    h_old, w_old = img.height, img.width
    if (h_old % div == 0) and (w_old % div == 0):
        return img
    h_tgt = round(div * math.ceil(h_old / div))
    w_tgt = round(div * math.ceil(w_old / div))
    # left, top, right, bottom
    padding = (0, 0, (w_tgt - w_old), (h_tgt - h_old))
    padded = tvf.pad(img, padding=padding, padding_mode='edge')
    return padded
