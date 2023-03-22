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
from torch.hub import load_state_dict_from_url
from timm.utils import AverageMeter

from lvae.utils.coding import crop_divisible_by, pad_divisible_by
import lvae.models.common as common
import lvae.models.entropy_coding as entropy_coding
from lvae.models.registry import register_model


def sinusoidal_embedding(values: torch.Tensor, dim=256, max_period=64):
    assert values.dim() == 1 and (dim % 2) == 0
    exponents = torch.linspace(0, 1, steps=(dim // 2))
    freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
    args = values.view(-1, 1) * freqs.view(1, dim//2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class ConvNeXtBlockFixRate(nn.Module):
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

class ConvNeXtAdaLNPatchDown(ConvNeXtBlockFixRate):
    def __init__(self, in_ch, out_ch, down_rate=2, **kwargs):
        super().__init__(in_ch, **kwargs)
        self.downsapmle = common.patch_downsample(in_ch, out_ch, rate=down_rate)

    def forward(self, x):
        x = super().forward(x)
        out = self.downsapmle(x)
        return out


class VRLatentBlockBase(nn.Module):
    def __init__(self, width, zdim, enc_width=None, kernel_size=7, mlp_ratio=2):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        self.resnet_front = ConvNeXtBlockFixRate(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end   = ConvNeXtBlockFixRate(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.posterior0 = ConvNeXtBlockFixRate(enc_width, kernel_size=kernel_size)
        self.posterior1 = ConvNeXtBlockFixRate(width, kernel_size=kernel_size)
        self.posterior2 = ConvNeXtBlockFixRate(width, kernel_size=kernel_size)
        enc_width = enc_width or width
        self.post_merge = common.conv_k1s1(width + enc_width, width)
        self.posterior  = common.conv_k3s1(width, zdim)
        self.z_proj     = common.conv_k1s1(zdim, width)
        self.prior      = common.conv_k1s1(width, zdim*2)

        self.discrete_gaussian = entropy_coding.DiscretizedGaussian()
        self.is_latent_block = True

    def transform_prior(self, feature):
        """ prior p(z_i | z_<i)

        Args:
            feature (torch.Tensor): feature map
        """
        feature = self.resnet_front(feature)
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        return feature, pm, pv

    def transform_posterior(self, feature, enc_feature):
        """ posterior q(z_i | z_<i, x)

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        enc_feature = self.posterior0(enc_feature)
        feature = self.posterior1(feature)
        merged = torch.cat([feature, enc_feature], dim=1)
        merged = self.post_merge(merged)
        merged = self.posterior2(merged)
        qm = self.posterior(merged)
        return qm

    def fuse_feature_and_z(self, feature, z):
        # add the new information carried by z to the feature
        feature = feature + self.z_proj(z)
        return feature

    def forward(self, feature, enc_feature=None, mode='trainval',
                get_latent=False, latent=None, t=1.0, strings=None):
        """ a complicated forward function

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        feature, pm, pv = self.transform_prior(feature)

        additional = dict()
        if mode == 'trainval': # training or validation
            qm = self.transform_posterior(feature, enc_feature)
            if self.training: # if training, use additive uniform noise
                z = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
                log_prob = entropy_coding.gaussian_log_prob_mass(pm, pv, x=z, bin_size=1.0, prob_clamp=1e-6)
                kl = -1.0 * log_prob
            else: # if evaluation, use residual quantization
                z, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
                kl = -1.0 * torch.log(probs)
            additional['kl'] = kl
        elif mode == 'sampling':
            if latent is None: # if z is not provided, sample it from the prior
                z = pm + pv * torch.randn_like(pm) * t + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
            else: # if `z` is provided, directly use it.
                assert pm.shape == latent.shape
                z = latent
        elif mode == 'compress': # encode z into bits
            qm = self.transform_posterior(feature, enc_feature)
            indexes = self.discrete_gaussian.build_indexes(pv)
            strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
            z = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
            additional['strings'] = strings
        elif mode == 'decompress': # decode z from bits
            assert strings is not None
            indexes = self.discrete_gaussian.build_indexes(pv)
            z = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        else:
            raise ValueError(f'Unknown mode={mode}')

        feature = self.fuse_feature_and_z(feature, z)
        feature = self.resnet_end(feature)
        if get_latent:
            additional['z'] = z.detach()
        return feature, additional

    def update(self):
        self.discrete_gaussian.update()


class QScaler(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([1,channels,1,1]))
        self.factor = nn.Parameter(torch.ones([1,channels,1,1]))

    def compress(self,x):
        return self.factor * (x - self.bias)

    def decompress(self,x):
        return self.bias + x / self.factor


class LatentBlockQSF(nn.Module):
    def __init__(self, width, zdim, enc_width=None, kernel_size=7, mlp_ratio=2):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        self.resnet_front = ConvNeXtBlockFixRate(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end   = ConvNeXtBlockFixRate(width, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.posterior0 = ConvNeXtBlockFixRate(enc_width, kernel_size=kernel_size)
        self.posterior1 = ConvNeXtBlockFixRate(width, kernel_size=kernel_size)
        self.posterior2 = ConvNeXtBlockFixRate(width, kernel_size=kernel_size)
        enc_width = enc_width or width
        self.post_merge = common.conv_k1s1(width + enc_width, width)
        self.posterior  = common.conv_k3s1(width, zdim)
        self.z_proj     = common.conv_k1s1(zdim, width)
        self.prior      = common.conv_k1s1(width, zdim*2)

        self.qsf_scaler = QScaler(zdim)
        self.discrete_gaussian = entropy_coding.DiscretizedGaussian()
        self.is_latent_block = True

    def transform_prior(self, feature):
        """ prior p(z_i | z_<i)

        Args:
            feature (torch.Tensor): feature map
        """
        feature = self.resnet_front(feature)
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        return feature, pm, pv

    def transform_posterior(self, feature, enc_feature):
        """ posterior q(z_i | z_<i, x)

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        enc_feature = self.posterior0(enc_feature)
        feature = self.posterior1(feature)
        merged = torch.cat([feature, enc_feature], dim=1)
        merged = self.post_merge(merged)
        merged = self.posterior2(merged)
        qm = self.posterior(merged)
        qm = self.qsf_scaler.compress(qm)
        return qm

    def fuse_feature_and_z(self, feature, z):
        # add the new information carried by z to the feature
        feature = feature + self.z_proj(z)
        return feature

    def forward(self, feature, enc_feature=None, mode='trainval',
                get_latent=False, latent=None, t=1.0, strings=None):
        """ a complicated forward function

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        feature, pm, pv = self.transform_prior(feature)

        additional = dict()
        if mode == 'trainval': # training or validation
            qm = self.transform_posterior(feature, enc_feature)
            if self.training: # if training, use additive uniform noise
                z = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
                log_prob = entropy_coding.gaussian_log_prob_mass(pm, pv, x=z, bin_size=1.0, prob_clamp=1e-6)
                kl = -1.0 * log_prob
            else: # if evaluation, use residual quantization
                z, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
                kl = -1.0 * torch.log(probs)
            additional['kl'] = kl
        elif mode == 'sampling':
            if latent is None: # if z is not provided, sample it from the prior
                z = pm + pv * torch.randn_like(pm) * t + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
            else: # if `z` is provided, directly use it.
                assert pm.shape == latent.shape
                z = latent
        elif mode == 'compress': # encode z into bits
            qm = self.transform_posterior(feature, enc_feature)
            indexes = self.discrete_gaussian.build_indexes(pv)
            strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
            z = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
            additional['strings'] = strings
        elif mode == 'decompress': # decode z from bits
            assert strings is not None
            indexes = self.discrete_gaussian.build_indexes(pv)
            z = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        else:
            raise ValueError(f'Unknown mode={mode}')

        z = self.qsf_scaler.decompress(z)
        feature = self.fuse_feature_and_z(feature, z)
        feature = self.resnet_end(feature)
        if get_latent:
            additional['z'] = z.detach()
        return feature, additional

    def update(self):
        self.discrete_gaussian.update()


class FeatureExtractor(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x):
        feature = x
        enc_features = OrderedDict()
        for i, block in enumerate(self.enc_blocks):
            feature = block(feature)
            enc_features[int(feature.shape[2])] = feature
        return enc_features


def mse_loss(fake, real):
    assert fake.shape == real.shape
    return tnf.mse_loss(fake, real, reduction='none').mean(dim=(1,2,3))


class VariableRateLossyVAE(nn.Module):
    log2_e = math.log2(math.e)
    # MAX_LMB = 8192

    def __init__(self, config: dict):
        super().__init__()
        # feature extractor (bottom-up path)
        self.encoder = FeatureExtractor(config.pop('enc_blocks'))
        # latent variable blocks (top-down path)
        self.dec_blocks = nn.ModuleList(config.pop('dec_blocks'))
        width = self.dec_blocks[0].in_channels
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.num_latents = len([b for b in self.dec_blocks if getattr(b, 'is_latent_block', False)])
        # loss function, for computing reconstruction loss
        self.distortion_name = 'mse'
        self.distortion_func = mse_loss
        self.distortion_lmb = float(config['distortion_lmb'])

        self.im_shift = float(config['im_shift'])
        self.im_scale = float(config['im_scale'])
        self.max_stride = config['max_stride']

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

        self.compressing = False
        # self._stats_log = dict()
        self._logging_images = config.get('log_images', [])
        self._flops_mode = False

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
    def _forward_flops(self, im):
        im = im.uniform_(0, 1)
        if self._flops_mode == 'compress':
            compressed_obj = self.compress(im)
        elif self._flops_mode == 'decompress':
            n, h, w = im.shape[0], im.shape[2]//self.max_stride, im.shape[3]//self.max_stride
            samples = self.unconditional_sample(bhw_repeat=(n,h,w))
        elif self._flops_mode == 'end-to-end':
            x_hat, stats_all = self.forward_end2end(im)
        else:
            raise ValueError(f'Unknown self._flops_mode: {self._flops_mode}')
        return

    def get_bias(self, bhw_repeat=(1,1,1)):
        nB, nH, nW = bhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        return feature

    def forward_end2end(self, im: torch.Tensor, get_latents=False):
        x = self.preprocess_input(im)
        # ================ Forward pass ================
        enc_features = self.encoder(x)
        all_block_stats = []
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                key = int(feature.shape[2])
                f_enc = enc_features[key]
                feature, stats = block(feature, enc_feature=f_enc, mode='trainval',
                                       get_latent=get_latents)
                all_block_stats.append(stats)
            else:
                feature = block(feature)
        return feature, all_block_stats

    def forward(self, batch, return_rec=False):
        if isinstance(batch, (tuple, list)):
            im, label = batch
        else:
            im = batch
        im = im.to(self._dummy.device)
        nB, imC, imH, imW = im.shape # batch, channel, height, width

        # ================ computing flops ================
        if self._flops_mode:
            return self._forward_flops(im)

        # ================ Forward pass ================
        x_hat, stats_all = self.forward_end2end(im)

        # ================ Compute Loss ================
        # rate
        kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        ndims = float(imC * imH * imW)
        kl = sum(kl_divergences) / ndims # nats per dimension
        # distortion
        x_target = self.preprocess_target(im)
        distortion = self.distortion_func(x_hat, x_target)
        # rate + distortion
        loss = kl + self.distortion_lmb * distortion
        loss = loss.mean(0)

        stats = OrderedDict()
        stats['loss'] = loss

        # ================ Logging ================
        with torch.no_grad():
            # for training print
            stats['bppix'] = kl.mean(0).item() * self.log2_e * imC
            stats[self.distortion_name] = distortion.mean(0).item()
            im_hat = self.process_output(x_hat.detach())
            im_mse = tnf.mse_loss(im_hat, im, reduction='mean')
            psnr = -10 * math.log10(im_mse.item())
            stats['psnr'] = psnr

        if return_rec:
            stats['im_hat'] = im_hat
        return stats

    def conditional_sample(self, latents, bhw_repeat=None, t=1.0):
        """ sampling, conditioned on (a list of) latents

        Args:
            latents (torch.Tensor): latent variables. If None, do unconditional sampling
            bhw_repeat (tuple): the constant bias will be repeated (batch, height, width) times
            t (float): temprature
        """
        # initialize latents variables
        if latents is None: # unconditional sampling
            latents = [None] * self.num_latents
        if latents[0] is None:
            assert bhw_repeat is not None, f'bhw_repeat should be provided'
            nB, nH, nW = bhw_repeat
        else: # conditional sampling
            assert (len(latents) == self.num_latents)
            nB, _, nH, nW = latents[0].shape
        # initialize lmb and embedding
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        idx = 0
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                feature, _ = block(feature, mode='sampling', latent=latents[idx], t=t)
                idx += 1
            else:
                feature = block(feature)
        assert idx == len(latents)
        im_samples = self.process_output(feature)
        return im_samples

    def unconditional_sample(self, bhw_repeat, t=1.0):
        """ unconditionally sample, ie, generate new images

        Args:
            bhw_repeat (tuple): repeat the initial constant feature n,h,w times
            t (float): temprature
        """
        return self.conditional_sample(latents=None, bhw_repeat=bhw_repeat, t=t)

    @torch.no_grad()
    def study(self, save_dir, **kwargs):
        save_dir = Path(save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir(parents=False)

        # unconditional samples
        for k in [1, 2]:
            num = 6
            im_samples = self.unconditional_sample(bhw_repeat=(num,k,k))
            save_path = save_dir / f'samples_k{k}_hw{im_samples.shape[2]}.png'
            tv.utils.save_image(im_samples, fp=save_path, nrow=math.ceil(num**0.5))
        # reconstructions
        for imname in self._logging_images:
            impath = f'images/{imname}'
            im = tvf.to_tensor(Image.open(impath)).unsqueeze_(0).to(device=self._dummy.device)
            x_hat, _ = self.forward_end2end(im)
            im_hat = self.process_output(x_hat)
            tv.utils.save_image(torch.cat([im, im_hat], dim=0), fp=save_dir / imname)

    def compress_mode(self, mode=True):
        if mode:
            for block in self.dec_blocks:
                if hasattr(block, 'update'):
                    block.update()
        self.compressing = mode

    @torch.no_grad()
    def compress(self, im):
        x = self.preprocess_input(im)
        enc_features = self.encoder(x)
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        strings_all = []
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                f_enc = enc_features[feature.shape[2]]
                feature, stats = block(feature, enc_feature=f_enc, mode='compress')
                strings_all.append(stats['strings'])
            else:
                feature = block(feature)
        strings_all.append((nB, nH, nW)) # smallest feature shape
        return strings_all

    @torch.no_grad()
    def decompress(self, compressed_object):
        nB, nH, nW = compressed_object[-1] # smallest feature shape

        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        str_i = 0
        for bi, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                strs_batch = compressed_object[str_i]
                feature, _ = block(feature, mode='decompress', strings=strs_batch)
                str_i += 1
            else:
                feature = block(feature)
        assert str_i == len(compressed_object) - 1, f'str_i={str_i}, len={len(compressed_object)}'
        im_hat = self.process_output(feature)
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

    @staticmethod
    def _log_channel_stats(channel_bpp_stats, log_dir, lmb):
        msg = '=' * 64 + '\n'
        msg += '---- row: latent blocks, colums: channels, avg over images ----\n'
        keys = sorted(channel_bpp_stats.keys())
        for k in keys:
            assert isinstance(channel_bpp_stats[k], AverageMeter)
            msg += ''.join([f'{a:<7.4f} ' for a in channel_bpp_stats[k].avg.tolist()]) + '\n'
        msg += '---- colums: latent blocks, avg over images ----\n'
        block_bpps = [channel_bpp_stats[k].avg.sum().item() for k in keys]
        msg += ''.join([f'{a:<7.4f} ' for a in block_bpps]) + '\n'
        with open(log_dir / f'channel-bppix-lmb{round(lmb)}.txt', mode='a') as f:
            print(msg, file=f)

    @torch.no_grad()
    def self_evaluate(self, img_dir, log_dir=None):
        if log_dir is not None:
            log_dir = Path(log_dir)
            channel_bpp_stats = defaultdict(AverageMeter)

        img_paths = list(Path(img_dir).rglob('*.*'))
        pbar = tqdm(img_paths, position=0, ascii=True)
        all_image_stats = defaultdict(float)
        for impath in pbar:
            img = Image.open(impath)
            img = crop_divisible_by(img, div=self.max_stride)
            im = tvf.to_tensor(img).unsqueeze_(0).to(device=self._dummy.device)
            x_hat, stats_all = self.forward_end2end(im)
            # compute bpp
            _, imC, imH, imW = im.shape
            kl = sum([stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]).mean(0) / (imC*imH*imW)
            bpp_estimated = kl.item() * self.log2_e * imC
            # compute psnr
            x_target = self.preprocess_target(im)
            distortion = self.distortion_func(x_hat, x_target).item()
            real = tvf.to_tensor(img)
            fake = self.process_output(x_hat).cpu().squeeze(0)
            mse = tnf.mse_loss(real, fake, reduction='mean').item()
            psnr = float(-10 * math.log10(mse))
            # accumulate results
            all_image_stats['count'] += 1
            all_image_stats['loss'] += float(kl.item() + self.distortion_lmb * distortion)
            all_image_stats['bpp']  += bpp_estimated
            all_image_stats['psnr'] += psnr
            # debugging
            if log_dir is not None:
                _to_bpp = lambda kl: kl.sum(dim=(2,3)).mean(0).cpu() / (imH*imW) * self.log2_e
                channel_bpps = [_to_bpp(stat['kl']) for stat in stats_all]
                for i, ch_bpp in enumerate(channel_bpps):
                    channel_bpp_stats[i].update(ch_bpp)
        # average over all images
        count = all_image_stats.pop('count')
        avg_stats = {k: v/count for k,v in all_image_stats.items()}
        # avg_stats['lambda'] = lmb
        if log_dir is not None:
            self._log_channel_stats(channel_bpp_stats, log_dir, self.distortion_lmb)

        return avg_stats


@register_model
def qarv_base_fr(lmb=2048, pretrained=False):
    cfg = dict()

    # variable rate
    cfg['distortion_lmb'] = float(lmb)

    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]
    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 8]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[ConvNeXtBlockFixRate(enc_dims[0], kernel_size=7) for _ in range(6)],
        ConvNeXtAdaLNPatchDown(enc_dims[0], enc_dims[1]),
        # 8x8
        *[ConvNeXtBlockFixRate(enc_dims[1], kernel_size=7) for _ in range(6)],
        ConvNeXtAdaLNPatchDown(enc_dims[1], enc_dims[2]),
        # 4x4
        *[ConvNeXtBlockFixRate(enc_dims[2], kernel_size=5) for _ in range(6)],
        ConvNeXtAdaLNPatchDown(enc_dims[2], enc_dims[3]),
        # 2x2
        *[ConvNeXtBlockFixRate(enc_dims[3], kernel_size=3) for _ in range(4)],
        ConvNeXtAdaLNPatchDown(enc_dims[3], enc_dims[3]),
        # 1x1
        *[ConvNeXtBlockFixRate(enc_dims[3], kernel_size=1) for _ in range(4)],
    ]

    cfg['dec_blocks'] = [
        # 1x1
        *[VRLatentBlockBase(dec_dims[0], z_dims[0], enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        ConvNeXtBlockFixRate(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        ConvNeXtBlockFixRate(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[VRLatentBlockBase(dec_dims[1], z_dims[1], enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(2)],
        ConvNeXtBlockFixRate(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        ConvNeXtBlockFixRate(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[VRLatentBlockBase(dec_dims[2], z_dims[2], enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(3)],
        ConvNeXtBlockFixRate(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        ConvNeXtBlockFixRate(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[VRLatentBlockBase(dec_dims[3], z_dims[3], enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(3)],
        ConvNeXtBlockFixRate(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[ConvNeXtBlockFixRate(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    cfg['max_stride'] = 64

    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    model = VariableRateLossyVAE(cfg)
    if pretrained is True:
        if lmb == 2048:
            wpath = 'runs/topic/qarv_base_fr_0_lmb2048/last_ema.pt'
        elif lmb == 512:
            wpath = 'runs/topic/qarv_base_fr_lmb512/last_ema.pt'
        elif lmb == 128:
            wpath = 'runs/topic/qarv_base_fr_lmb128/last_ema.pt'
        elif lmb == 16:
            wpath = 'runs/topic/qarv_base_fr_lmb16/last_ema.pt'
        else:
            raise NotImplementedError(f'{lmb=}')
        msd = torch.load(wpath)['model']
        model.load_state_dict(msd)
    elif isinstance(pretrained, str):
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    return model


@register_model
def qarvb_fr_qsf(lmb=2048, pretrained=False):
    cfg = dict()

    # variable rate
    cfg['distortion_lmb'] = float(lmb)

    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]
    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 8]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[ConvNeXtBlockFixRate(enc_dims[0], kernel_size=7) for _ in range(6)],
        ConvNeXtAdaLNPatchDown(enc_dims[0], enc_dims[1]),
        # 8x8
        *[ConvNeXtBlockFixRate(enc_dims[1], kernel_size=7) for _ in range(6)],
        ConvNeXtAdaLNPatchDown(enc_dims[1], enc_dims[2]),
        # 4x4
        *[ConvNeXtBlockFixRate(enc_dims[2], kernel_size=5) for _ in range(6)],
        ConvNeXtAdaLNPatchDown(enc_dims[2], enc_dims[3]),
        # 2x2
        *[ConvNeXtBlockFixRate(enc_dims[3], kernel_size=3) for _ in range(4)],
        ConvNeXtAdaLNPatchDown(enc_dims[3], enc_dims[3]),
        # 1x1
        *[ConvNeXtBlockFixRate(enc_dims[3], kernel_size=1) for _ in range(4)],
    ]

    cfg['dec_blocks'] = [
        # 1x1
        *[LatentBlockQSF(dec_dims[0], z_dims[0], enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        ConvNeXtBlockFixRate(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        ConvNeXtBlockFixRate(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[LatentBlockQSF(dec_dims[1], z_dims[1], enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(2)],
        ConvNeXtBlockFixRate(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        ConvNeXtBlockFixRate(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[LatentBlockQSF(dec_dims[2], z_dims[2], enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(3)],
        ConvNeXtBlockFixRate(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        ConvNeXtBlockFixRate(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[LatentBlockQSF(dec_dims[3], z_dims[3], enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(3)],
        ConvNeXtBlockFixRate(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[ConvNeXtBlockFixRate(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    cfg['max_stride'] = 64

    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    model = VariableRateLossyVAE(cfg)

    # load pre-trained weights
    url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/qarv_base-fr-lmb2048-bs16-500k.pt'
    msd = load_state_dict_from_url(url)['model']
    missing, unexpected = model.load_state_dict(msd, strict=False)
    assert len(unexpected) == 0

    _count = 0
    for n, p in model.named_parameters():
        if 'qsf_scaler' in n:
            _count += 1
        else:
            p.requires_grad_(False)
    assert len(missing) == _count

    if pretrained is True:
        raise NotImplementedError()
        msd = torch.load(wpath)['model']
        model.load_state_dict(msd)
    elif isinstance(pretrained, str):
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    return model


def main():
    model = qarvb_fr_qsf()
    debug = 1


if __name__ == '__main__':
    main()
