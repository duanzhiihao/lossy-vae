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
from timm.utils import AverageMeter

from lvae.utils.coding import crop_divisible_by, pad_divisible_by
import lvae.models.common as common
import lvae.models.entropy_coding as entropy_coding


MAX_LMB = 8192


def normed_cosine_decay(x: torch.Tensor):
    assert 0 <= x.min() <= x.max() <= 1
    y = 0.5 * (1 + torch.cos(x * math.pi))
    assert 0 <= y.min() <= y.max() <= 1
    return y


def sinusoidal_embedding(values: torch.Tensor, dim=256, max_period=128):
    assert values.dim() == 1 and (dim % 2) == 0
    exponents = torch.linspace(0, 1, steps=(dim // 2))
    freqs = torch.pow(max_period, -1.0 * exponents).to(device=values.device)
    args = values.view(-1, 1) * freqs.view(1, dim//2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class Identity2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input0, input1):
        return input0


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
        self.downsapmle = common.patch_downsample(in_ch, out_ch, rate=down_rate)

    def forward(self, x, emb):
        x = super().forward(x, emb)
        out = self.downsapmle(x)
        return out


class VRLatentBlockNoZ(nn.Module):
    def __init__(self, width, zdim, embed_dim, enc_width=None, kernel_size=7, mlp_ratio=2):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        self.resnet_front = MyConvNeXtBlockAdaLN(width, embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end   = MyConvNeXtBlockAdaLN(width, embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.requires_embedding = True

    def forward(self, feature, lmb_embedding):
        feature = self.resnet_front(feature, lmb_embedding)
        feature = self.resnet_end(feature, lmb_embedding)
        return feature


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
        self.is_latent_block = True

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
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        enc_feature = self.posterior0(enc_feature, lmb_embedding)
        feature = self.posterior1(feature, lmb_embedding)
        merged = torch.cat([feature, enc_feature], dim=1)
        merged = self.post_merge(merged)
        merged = self.posterior2(merged, lmb_embedding)
        qm = self.posterior(merged)
        return qm

    def fuse_feature_and_z(self, feature, z, lmb_embedding, log_lmb):
        # add the new information carried by z to the feature
        feature = feature + self.z_proj(z)
        return feature

    def forward(self, feature, lmb_embedding, enc_feature=None, mode='trainval',
                log_lmb=None, get_latent=False, latent=None, t=1.0, strings=None):
        """ a complicated forward function

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        feature, pm, pv = self.transform_prior(feature, lmb_embedding)

        additional = dict()
        if mode == 'trainval': # training or validation
            qm = self.transform_posterior(feature, enc_feature, lmb_embedding)
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
            qm = self.transform_posterior(feature, enc_feature, lmb_embedding)
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

        feature = self.fuse_feature_and_z(feature, z, lmb_embedding, log_lmb)
        feature = self.resnet_end(feature, lmb_embedding)
        if get_latent:
            additional['z'] = z.detach()
        return feature, additional

    def update(self):
        self.discrete_gaussian.update()


class FeatureExtractor(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x, emb=None):
        feature = x
        enc_features = OrderedDict()
        for i, block in enumerate(self.enc_blocks):
            if getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
            enc_features[int(feature.shape[2])] = feature
        return enc_features


def mse_loss(fake, real):
    assert fake.shape == real.shape
    return tnf.mse_loss(fake, real, reduction='none').mean(dim=(1,2,3))


class VariableRateLossyVAE(nn.Module):
    log2_e = math.log2(math.e)

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

        self._set_lmb_embedding(config)

        self.im_shift = float(config['im_shift'])
        self.im_scale = float(config['im_scale'])
        self.max_stride = config['max_stride']

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

        self.compressing = False
        # self._stats_log = dict()
        self._logging_images = config.get('log_images', None)
        self._logging_smpl_k = [1, 2]
        self._flops_mode = False

    def _set_lmb_embedding(self, config):
        assert len(config['log_lmb_range']) == 2
        self.log_lmb_range = (float(config['log_lmb_range'][0]), float(config['log_lmb_range'][1]))
        self.lmb_embed_dim = config['lmb_embed_dim']
        self.lmb_embedding = nn.Sequential(
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

    def sample_log_lmb(self, n):
        low, high = self.log_lmb_range
        if False:
            log_lmb = low + (high-low) * torch.rand(n, device=self._dummy.device)
        else:
            low, high = math.exp(low), math.exp(high) # lmb space
            p = 3.0
            low, high = math.pow(low, 1/p), math.pow(high, 1/p) # transformed space
            transformed_lmb = low + (high-low) * torch.rand(n, device=self._dummy.device)
            log_lmb = torch.log(transformed_lmb.pow(p))
        return log_lmb

    def expand_to_tensor(self, log_lmb, n):
        assert isinstance(log_lmb, (torch.Tensor, float, int)), f'type(log_lmb)={type(log_lmb)}'
        if isinstance(log_lmb, torch.Tensor) and (log_lmb.numel() == 1):
            log_lmb = log_lmb.item()
        if isinstance(log_lmb, (float, int)):
            log_lmb = torch.full(size=(n,), fill_value=float(log_lmb), device=self._dummy.device)
        assert log_lmb.shape == (n,), f'log_lmb={log_lmb}'
        return log_lmb

    def _get_lmb_embedding(self, log_lmb, n):
        log_lmb = self.expand_to_tensor(log_lmb, n=n)
        scaled = log_lmb * self.LOG_LMB_SCALE
        embedding = sinusoidal_embedding(scaled, dim=self.lmb_embed_dim[0], max_period=self._sin_period)
        embedding = self.lmb_embedding(embedding)
        return embedding

    def get_bias(self, bhw_repeat=(1,1,1)):
        nB, nH, nW = bhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        # feature = torch.zeros(nB, self.initial_width, nH, nW, device=self._dummy.device)
        return feature

    def forward_end2end(self, im: torch.Tensor, log_lmb: torch.Tensor, get_latents=False):
        x = self.preprocess_input(im)
        # ================ get lambda embedding ================
        emb = self._get_lmb_embedding(log_lmb, n=im.shape[0])
        # ================ Forward pass ================
        enc_features = self.encoder(x, emb)
        all_block_stats = []
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                key = int(feature.shape[2])
                f_enc = enc_features[key]
                feature, stats = block(feature, emb, enc_feature=f_enc, mode='trainval',
                                       log_lmb=log_lmb, get_latent=get_latents)
                all_block_stats.append(stats)
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
        return feature, all_block_stats

    def forward(self, batch, log_lmb=None, return_rec=False):
        if isinstance(batch, (tuple, list)):
            im, label = batch
        else:
            im = batch
        im = im.to(self._dummy.device)
        nB, imC, imH, imW = im.shape # batch, channel, height, width

        # ================ computing flops ================
        if self._flops_mode:
            raise NotImplementedError()
            return self._forward_flops(im, lmb_embedding)

        # ================ Forward pass ================
        if (log_lmb is None): # training
            log_lmb = self.sample_log_lmb(n=im.shape[0])
        assert isinstance(log_lmb, torch.Tensor) and log_lmb.shape == (nB,)
        x_hat, stats_all = self.forward_end2end(im, log_lmb)

        # ================ Compute Loss ================
        # rate
        kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        ndims = float(imC * imH * imW)
        kl = sum(kl_divergences) / ndims # nats per dimension
        # distortion
        x_target = self.preprocess_target(im)
        distortion = self.distortion_func(x_hat, x_target)
        # rate + distortion
        loss = kl + torch.exp(log_lmb) * distortion
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

    def conditional_sample(self, log_lmb, latents, emb=None, bhw_repeat=None, t=1.0):
        """ sampling, conditioned on (possibly a subset of) latents

        Args:
            latents (torch.Tensor): latent variables
            bhw_repeat (tuple): the bias constant will be repeated (batch, height, width) times
            t (float): temprature
        """
        # initialize latents variables
        if latents is None: # unconditional sampling
            latents = [None] * self.num_latents
            assert bhw_repeat is not None, f'bhw_repeat should be provided'
            nB, nH, nW = bhw_repeat
        else: # conditional sampling
            assert (bhw_repeat is None) and (len(latents) == self.num_latents)
            nB, _, nH, nW = latents[0].shape
        # initialize log_lmb and embedding
        log_lmb = self.expand_to_tensor(log_lmb, n=nB)
        if emb is None:
            emb = self._get_lmb_embedding(log_lmb, n=nB)
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        idx = 0
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                feature, _ = block(feature, emb, mode='sampling',
                                   log_lmb=log_lmb, latent=latents[idx], t=t)
                idx += 1
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
        assert idx == len(latents)
        im_samples = self.process_output(feature)
        return im_samples

    def unconditional_sample(self, log_lmb, bhw_repeat, t=1.0):
        """ unconditionally sample, ie, generate new images

        Args:
            bhw_repeat (tuple): repeat the initial constant feature n,h,w times
            t (float): temprature
        """
        return self.conditional_sample(log_lmb, latents=None, bhw_repeat=bhw_repeat, t=t)

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

        log_lmb = self.expand_to_tensor(self._default_log_lmb, n=1)
        # unconditional samples
        for k in self._logging_smpl_k:
            num = 6
            im_samples = self.unconditional_sample(log_lmb, bhw_repeat=(num,k,k))
            save_path = save_dir / f'samples_k{k}_hw{im_samples.shape[2]}.png'
            tv.utils.save_image(im_samples, fp=save_path, nrow=math.ceil(num**0.5))
        # reconstructions
        for imname in self._logging_images:
            impath = f'images/{imname}'
            im = tvf.to_tensor(Image.open(impath)).unsqueeze_(0).to(device=self._dummy.device)
            x_hat, _ = self.forward_end2end(im, log_lmb=log_lmb)
            im_hat = self.process_output(x_hat)
            tv.utils.save_image(torch.cat([im, im_hat], dim=0), fp=save_dir / imname)

    def compress_mode(self, mode=True):
        if mode:
            for block in self.dec_blocks:
                if hasattr(block, 'update'):
                    block.update()
        self.compressing = mode

    @torch.no_grad()
    def compress(self, im, log_lmb=None):
        if log_lmb is None: # use default log-lambda
            log_lmb = self._default_log_lmb
        log_lmb = self.expand_to_tensor(log_lmb, n=im.shape[0])
        lmb_embedding = self._get_lmb_embedding(log_lmb, n=im.shape[0])
        x = self.preprocess_input(im)
        enc_features = self.encoder(x, lmb_embedding)
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        strings_all = []
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                f_enc = enc_features[feature.shape[2]]
                feature, stats = block(feature, lmb_embedding, enc_feature=f_enc, mode='compress',
                                       log_lmb=log_lmb)
                strings_all.append(stats['strings'])
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, lmb_embedding)
            else:
                feature = block(feature)
        strings_all.append((nB, nH, nW)) # smallest feature shape
        strings_all.append(log_lmb) # log lambda
        return strings_all

    @torch.no_grad()
    def decompress(self, compressed_object):
        log_lmb = compressed_object[-1] # log lambda
        nB, nH, nW = compressed_object[-2] # smallest feature shape
        log_lmb = self.expand_to_tensor(log_lmb, n=nB)
        lmb_embedding = self._get_lmb_embedding(log_lmb, n=nB)

        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        str_i = 0
        for bi, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                strs_batch = compressed_object[str_i]
                feature, _ = block(feature, lmb_embedding, mode='decompress',
                                   log_lmb=log_lmb, strings=strs_batch)
                str_i += 1
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, lmb_embedding)
            else:
                feature = block(feature)
        assert str_i == len(compressed_object) - 2, f'str_i={str_i}, len={len(compressed_object)}'
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

    @torch.no_grad()
    def _self_evaluate(self, img_paths, log_lmb: float, pbar=False, log_dir=None):
        pbar = tqdm(img_paths) if pbar else img_paths
        all_image_stats = defaultdict(float)
        # self._stats_log = dict()
        if log_dir is not None:
            log_dir = Path(log_dir)
            channel_bpp_stats = defaultdict(AverageMeter)
        for impath in pbar:
            img = Image.open(impath)
            # imgh, imgw = img.height, img.width
            img = crop_divisible_by(img, div=self.max_stride)
            # img_padded = pad_divisible_by(img, div=self.max_stride)
            im = tvf.to_tensor(img).unsqueeze_(0).to(device=self._dummy.device)
            x_hat, stats_all = self.forward_end2end(im, log_lmb=self.expand_to_tensor(log_lmb,n=1))
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
            all_image_stats['loss'] += float(kl.item() + math.exp(log_lmb) * distortion)
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
        avg_stats['lambda'] = math.exp(log_lmb)
        if log_dir is not None:
            self._log_channel_stats(channel_bpp_stats, log_dir, log_lmb)

        return avg_stats

    @staticmethod
    def _log_channel_stats(channel_bpp_stats, log_dir, log_lmb):
        msg = '=' * 64 + '\n'
        msg += '---- row: latent blocks, colums: channels, avg over images ----\n'
        keys = sorted(channel_bpp_stats.keys())
        for k in keys:
            assert isinstance(channel_bpp_stats[k], AverageMeter)
            msg += ''.join([f'{a:<7.4f} ' for a in channel_bpp_stats[k].avg.tolist()]) + '\n'
        msg += '---- colums: latent blocks, avg over images ----\n'
        block_bpps = [channel_bpp_stats[k].avg.sum().item() for k in keys]
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
        if False: # uniform in log space
            log_lambdas = torch.linspace(start, end, steps=steps).tolist()
        else: # uniform in sqrt space
            start, end = math.exp(start), math.exp(end)
            p = 3.0
            lambdas = torch.linspace(math.pow(start,1/p), math.pow(end,1/p), steps=steps).pow(3)
            log_lambdas = torch.log(lambdas)
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
