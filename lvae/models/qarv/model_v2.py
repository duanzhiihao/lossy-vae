from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict, defaultdict
import math
import struct
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision as tv
import torchvision.transforms.functional as tvf
from timm.utils import AverageMeter

import lvae.utils.coding as coding
import lvae.models.common as common
import lvae.models.entropy_coding as entropy_coding


class LatentVariableBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.discrete_gaussian = entropy_coding.DiscretizedGaussian()
        self.requires_dict_input = True


class VRLVBlockV2(LatentVariableBlock):
    """ Vriable-Rate Latent Variable Block
    """
    default_embedding_dim = 256
    def __init__(self, width, zdim, enc_key, enc_width, embed_dim=None, kernel_size=7,
                 name=None):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width
        self.enc_key = enc_key
        self.out_feature_name = name

        block = common.ConvNeXtBlockAdaLN
        embed_dim = embed_dim or self.default_embedding_dim
        self.posterior0 = block(enc_width, embed_dim, kernel_size=kernel_size)
        self.posterior1 = block(width,     embed_dim, kernel_size=kernel_size)
        self.posterior2 = block(width,     embed_dim, kernel_size=kernel_size)
        self.post_merge = common.conv_k1s1(width + enc_width, width)
        self.posterior  = common.conv_k3s1(width, zdim)
        self.z_proj     = common.conv_k1s1(zdim, width)
        self.prior      = common.conv_k1s1(width, zdim*2)

    def transform_prior(self, feature):
        """ prior p(z_i | z_<i)

        Args:
            feature (torch.Tensor): feature map
        """
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        return pm, pv

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

    def fuse_feature_and_z(self, feature, z):
        # add the new information carried by z to the feature
        feature = feature + self.z_proj(z)
        return feature

    def forward(self, fdict):
        feature = fdict['feature']
        emb = fdict['lmb_emb']
        mode = fdict['mode']

        pm, pv = self.transform_prior(feature)

        if mode == 'trainval': # training or validation
            enc_feature = fdict['all_features'][self.enc_key]
            qm = self.transform_posterior(feature, enc_feature, emb)
            if self.training: # if training, use additive uniform noise
                z = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
                log_prob = entropy_coding.gaussian_log_prob_mass(pm, pv, x=z, bin_size=1.0, prob_clamp=1e-6)
                kl = -1.0 * log_prob
            else: # if evaluation, use residual quantization
                z, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
                kl = -1.0 * torch.log(probs)
            fdict['kl_divs'].append(kl)
        elif mode == 'sampling':
            latent = fdict['zs'].pop(0)
            t = fdict['temperature']
            if latent is None: # if z is not provided, sample it from the prior
                z = pm + pv * torch.randn_like(pm) * t + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
            else: # if `z` is provided, directly use it.
                assert pm.shape == latent.shape
                z = latent
        elif mode == 'compress': # encode z into bits
            enc_feature = fdict['all_features'][self.enc_key]
            qm = self.transform_posterior(feature, enc_feature, emb)
            indexes = self.discrete_gaussian.build_indexes(pv)
            strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
            z = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
            fdict['bit_strings'].append(strings)
        elif mode == 'decompress': # decode z from bits
            strings = fdict['bit_strings'].pop(0)
            indexes = self.discrete_gaussian.build_indexes(pv)
            z = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        else:
            raise ValueError(f'Unknown mode={mode}')

        feature = self.fuse_feature_and_z(feature, z)
        fdict['feature'] = feature
        fdict['zs'].append(z)
        if self.out_feature_name is not None:
            assert self.out_feature_name not in fdict['all_features']
            fdict['all_features'][self.out_feature_name] = feature
        return fdict

    def update(self):
        self.discrete_gaussian.update()


class CrossAttnTransformerNCHW(nn.Module):
    default_embedding_dim = 256
    def __init__(self, q_dim, kv_name, kv_dim, embed_dim=None):
        super().__init__()
        self.kv_name = kv_name

        embed_dim = embed_dim or self.default_embedding_dim
        # TODO: try standard nn.LayerNorm
        self.norm1_q = common.AdaptiveLayerNorm(q_dim, embed_dim)
        self.norm1_kv = common.AdaptiveLayerNorm(kv_dim, embed_dim)
        self.cross_attn = common.MultiheadAttention([q_dim, kv_dim, kv_dim], num_heads=8)
        self.layer_scale1 = nn.Parameter(torch.full(size=(1, 1, q_dim), fill_value=1e-5))

        self.norm2 = common.AdaptiveLayerNorm(q_dim, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(q_dim, q_dim*2),
            nn.GELU(),
            nn.Linear(q_dim*2, q_dim),
        )
        self.layer_scale2 = nn.Parameter(torch.full(size=(1, 1, q_dim), fill_value=1e-5))

        self.requires_dict_input = True

    def forward(self, fdict):
        x = fdict['feature'] # (B, C, H, W)
        B, C, H, W = x.shape
        kv = fdict['all_features'][self.kv_name] # (B, C, H, W)
        # to (B, H*W, C)
        x = x.flatten(2).transpose(1, 2) # (B, H*W, C)
        kv = kv.flatten(2).transpose(1, 2) # (B, H*W, C)
        emb = fdict['lmb_emb'] # (B, C)

        kv = self.norm1_kv(kv, emb)
        x = x + self.layer_scale1 * self.cross_attn(self.norm1_q(x, emb), kv, kv)
        x = x + self.layer_scale2 * self.mlp(self.norm2(x, emb))
         # (B, H*W, C) -> (B, C, H, W)
        x = x.transpose(1, 2).unflatten(2, sizes=[H, W])
        fdict['feature'] = x
        return fdict


class VariableRateLossyVAE(nn.Module):
    log2_e = math.log2(math.e)
    MAX_LOG_LMB = math.log(8192)

    def __init__(self, config: dict):
        super().__init__()
        # feature extractor (bottom-up path)
        self.encoder = common.FeatureExtractorWithEmbedding(config.pop('enc_blocks'))
        # latent variable blocks (top-down path)
        self.dec_blocks = nn.ModuleList(config.pop('dec_blocks'))
        # initial bias for the top-down path
        width = self.dec_blocks[0].in_channels
        self.dec_bias = nn.Parameter(torch.zeros(1, width, 1, 1))
        # lambda embedding layers
        self._setup_lmb_embedding(config)

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor
        self.num_latents = len([b for b in self.dec_blocks if isinstance(b, LatentVariableBlock)])
        self.max_stride = config['max_stride']
        self.compressing = False
        self._logging_images = config.get('log_images', [])
        self._flops_mode = False

    def _setup_lmb_embedding(self, config):
        _low, _high = config['lmb_range']
        self.lmb_range = (float(_low), float(_high))
        self.default_lmb = self.lmb_range[1]
        self.lmb_embed_dim = config['lmb_embed_dim']
        self.lmb_embedding = nn.Sequential(
            nn.Linear(self.lmb_embed_dim[0], self.lmb_embed_dim[1]),
            nn.GELU(),
            nn.Linear(self.lmb_embed_dim[1], self.lmb_embed_dim[1]),
        )
        self._sin_period = config['sin_period']

    def preprocess(self, im: torch.Tensor):
        # [0, 1] -> [-1, 1]
        assert (im.shape[2] % self.max_stride == 0) and (im.shape[3] % self.max_stride == 0)
        assert (im.dim() == 4) and (0 <= im.min() <= im.max() <= 1) and not im.requires_grad
        x = im.clone().add_(-0.5).mul_(2.0)
        return x

    def postprocess(self, x: torch.Tensor):
        # [-1, 1] -> [0, 1]
        im_hat = x.clone().clamp_(min=-1.0, max=1.0).mul_(0.5).add_(0.5)
        return im_hat

    def sample_lmb(self, n: int):
        low, high = self.lmb_range # original lmb space, 16 to 1024
        p = 3.0
        low, high = math.pow(low, 1/p), math.pow(high, 1/p) # transformed space
        transformed_lmb = low + (high-low) * torch.rand(n, device=self._dummy.device)
        lmb = torch.pow(transformed_lmb, exponent=p)
        assert isinstance(lmb, torch.Tensor) and lmb.shape == (n,)
        return lmb

    def get_lmb_embedding(self, lmb: torch.Tensor):
        assert isinstance(lmb, torch.Tensor) and lmb.dim() == 1
        scaled = torch.log(lmb) * self._sin_period / self.MAX_LOG_LMB
        embedding = common.sinusoidal_embedding(scaled, dim=self.lmb_embed_dim[0],
                                                max_period=self._sin_period)
        embedding = self.lmb_embedding(embedding)
        return embedding

    def get_initial_fdict(self, lmb, bias_bhw):
        """ Get an initial empty feature dictionary

        Args:
            bias_bhw (tuple): (batch, height, width) for the initial top-down feature
        """
        fdict = dict() # a feature dictionary containing all features
        fdict['lmb_emb'] = self.get_lmb_embedding(lmb) # lambda embedding
        # ======== for 'trainval' mode ========
        fdict['all_features'] = OrderedDict() # bottom-up encoder features
        nB, nH, nW = bias_bhw
        fdict['feature'] = self.dec_bias.expand(nB, -1, nH, nW) # main feature for the top-down path
        fdict['kl_divs'] = [] # kl (i.e., rate) for each latent variable
        # ======== for 'compress' and 'decompress' mode ========
        fdict['bit_strings'] = [] # compressed bit strings
        # ======== for 'sampling' mode ========
        fdict['zs'] = [] # latent variables
        fdict['temperature'] = 1.0 # temperature for sampling
        return fdict

    def forward_bottomup(self, im, lmb):
        bias_bhw = (im.shape[0], im.shape[2]//self.max_stride, im.shape[3]//self.max_stride)
        fdict = self.get_initial_fdict(lmb, bias_bhw)
        x = self.preprocess(im)
        fdict['all_features'] = self.encoder(x, fdict['lmb_emb'])
        return fdict, x

    def forward_topdown(self, fdict, mode='trainval'):
        fdict['mode'] = mode
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'requires_dict_input', False):
                fdict = block(fdict)
            elif getattr(block, 'requires_embedding', False):
                fdict['feature'] = block(fdict['feature'], fdict['lmb_emb'])
            elif isinstance(block, common.CompresionStopFlag) and (mode == 'compress'):
                # no need to execute remaining blocks when compressing
                return fdict
            else:
                fdict['feature'] = block(fdict['feature'])
        fdict['x_hat'] = fdict.pop('feature') # rename 'feature' to 'x_hat'
        return fdict

    def forward(self, im, lmb=None, return_fdict=False):
        im = im.to(self._dummy.device)
        B, imC, imH, imW = im.shape # batch, channel, height, width

        # ================ Forward pass ================
        lmb = lmb or self.sample_lmb(n=im.shape[0])
        assert lmb.shape == (B,)
        fdict, x = self.forward_bottomup(im, lmb)
        fdict = self.forward_topdown(fdict, mode='trainval')

        # ================ Compute Loss ================
        x_hat, kl_divs = fdict['x_hat'], fdict['kl_divs']
        # rate
        kl_divs = [kl.sum(dim=(1, 2, 3)) for kl in kl_divs]
        bpp = sum(kl_divs) * self.log2_e / float(imH * imW) # bits per pixel, shape (B,)
        # distortion
        mse = tnf.mse_loss(x_hat, x, reduction='none').mean(dim=(1,2,3))
        # rate + distortion
        loss = bpp + lmb * mse # (B,)

        metrics = OrderedDict()
        metrics['loss'] = loss.mean(0)

        # ================ Logging ================
        with torch.inference_mode(): # for training progress bar
            metrics['bpp'] = bpp.mean(0).item()
            metrics['mse'] = mse.mean(0).item()
            im_mse = tnf.mse_loss(self.postprocess(x_hat), im, reduction='mean')
            psnr = -10 * math.log10(im_mse.item())
            metrics['psnr'] = psnr
        if return_fdict:
            return metrics, fdict
        return metrics

    @torch.inference_mode()
    def _self_evaluate(self, img_paths, lmb, pbar=False, log_dir=None):
        lmb = torch.full((1,), lmb, device=self._dummy.device)
        pbar = tqdm(img_paths) if pbar else img_paths
        avg_meters = defaultdict(AverageMeter)
        for impath in pbar:
            im = tvf.to_tensor(Image.open(impath)).unsqueeze_(0).to(device=self._dummy.device)
            metrics = self.forward(im, lmb=lmb)
            for k,v in metrics.items():
                avg_meters[k].update(float(v))
        results = {k: v.avg for k,v in avg_meters.items()} # average over all images
        results['lambda'] = lmb.item()
        return results

    @torch.inference_mode()
    def self_evaluate(self, img_dir, lmb_range=None, steps=8, log_dir=None):
        img_paths = list(Path(img_dir).rglob('*.*'))
        start, end = self.lmb_range if (lmb_range is None) else lmb_range
        # uniform in cube root space
        lambdas = torch.linspace(math.log(start), math.log(end), steps=steps).exp()
        pbar = tqdm(lambdas, position=0, ascii=True)
        all_lmb_stats = defaultdict(list)
        for lmb in pbar:
            results = self._self_evaluate(img_paths, lmb, log_dir=log_dir)
            pbar.set_description(f'lmb={lmb.item():.3f}, {results=}')
            for k,v in results.items():
                all_lmb_stats[k].append(v)
        return all_lmb_stats

    def compress_mode(self, mode=True):
        if mode:
            for block in self.dec_blocks:
                if hasattr(block, 'update'):
                    block.update()
        self.compressing = mode

    @torch.inference_mode()
    def compress(self, im):
        assert im.shape[0] == 1, f'Right now only support a single image; got {im.shape=}'

        lmb = torch.full((1,), self.default_lmb, device=self._dummy.device) # use the default lambda
        fdict, _ = self.forward_bottomup(im, lmb)
        fdict = self.forward_topdown(fdict, mode='compress')

        assert len(fdict['bit_strings']) == self.num_latents
        all_lv_strings = [strings[0] for strings in fdict['bit_strings']]
        string = coding.pack_byte_strings(all_lv_strings)
        # encode lambda and image shape in the header
        nB, _, imH, imW = im.shape
        header1 = struct.pack('f', lmb)
        header2 = struct.pack('3H', nB, imH//self.max_stride, imW//self.max_stride)
        string = header1 + header2 + string
        return string

    @torch.inference_mode()
    def decompress(self, string):
        # extract lambda
        _len = 4
        lmb, string = struct.unpack('f', string[:_len])[0], string[_len:]
        # extract shape
        _len = 2 * 3
        (nB, nH, nW), string = struct.unpack('3H', string[:_len]), string[_len:]
        all_lv_strings = coding.unpack_byte_string(string)

        lmb = torch.full((1,), lmb, device=self._dummy.device) # use the default lambda
        fdict = self.get_initial_fdict(lmb, bias_bhw=(nB, nH, nW))
        fdict['bit_strings'] = [[s,] for s in all_lv_strings] # add batch dimension to each string
        fdict = self.forward_topdown(fdict, mode='decompress')
        assert len(fdict['bit_strings']) == 0
        im_hat = self.postprocess(fdict['x_hat'])
        return im_hat

    @torch.inference_mode()
    def compress_file(self, img_path, output_path):
        # read image
        img = Image.open(img_path)
        img_padded = coding.pad_divisible_by(img, div=self.max_stride)
        im = tvf.to_tensor(img_padded).unsqueeze_(0).to(device=self._dummy.device)
        # compress by model
        body_str = self.compress(im)
        header_str = struct.pack('2H', img.height, img.width)
        # save bits to file
        with open(output_path, 'wb') as f:
            f.write(header_str + body_str)

    @torch.inference_mode()
    def decompress_file(self, bits_path):
        # read from file
        with open(bits_path, 'rb') as f:
            header_str = f.read(4)
            body_str = f.read()
        img_h, img_w = struct.unpack('2H', header_str)
        # decompress by model
        im_hat = self.decompress(body_str)
        return im_hat[:, :, :img_h, :img_w]

    @torch.inference_mode()
    def conditional_sample(self, latents, bhw_repeat=None, t=1.0):
        """ sampling conditioned on a list of latents variables

        Args:
            latents (torch.Tensor): latent variables. If None, do unconditional sampling
            bhw_repeat (tuple): (batch, height, width) for the initial top-down feature
            t (float): temprature
        """
        if latents[0] is None:
            assert bhw_repeat is not None, f'bhw_repeat should be provided'
            nB, nH, nW = bhw_repeat
        else: # conditional sampling
            assert (len(latents) == self.num_latents)
            nB, _, nH, nW = latents[0].shape
        # initialize lmb and embedding
        lmb = torch.full((nB,), self.default_lmb, device=self._dummy.device) # use the default lambda
        fdict = self.get_initial_fdict(lmb, (nB, nH, nW))
        fdict['zs'] = latents
        fdict['temperature'] = t
        fdict = self.forward_topdown(fdict, mode='sampling')
        im_samples = self.postprocess(fdict['x_hat'])
        return im_samples

    @torch.inference_mode()
    def unconditional_sample(self, bhw_repeat, t=1.0):
        """ unconditionally sample, ie, generate new images

        Args:
            bhw_repeat (tuple): repeat the initial constant feature n,h,w times
            t (float): temprature
        """
        return self.conditional_sample([None]*self.num_latents, bhw_repeat=bhw_repeat, t=t)

    @torch.inference_mode()
    def study(self, save_dir, **kwargs):
        save_dir = Path(save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir(parents=False)

        # unconditional sampling
        for k in [1, 2]:
            num = 6
            im_samples = self.unconditional_sample(bhw_repeat=(num,k,k))
            save_path = save_dir / f'samples_k{k}_hw{im_samples.shape[2]}.png'
            tv.utils.save_image(im_samples, fp=save_path, nrow=math.ceil(num**0.5))

        # reconstructions
        lmb = torch.full((1,), self.default_lmb, device=self._dummy.device) # use the default lambda
        for imname in self._logging_images:
            impath = f'images/{imname}'
            im = tvf.to_tensor(Image.open(impath)).unsqueeze_(0).to(device=self._dummy.device)
            fdict, _ = self.forward_bottomup(im, lmb)
            fdict = self.forward_topdown(fdict, mode='trainval')
            im_hat = self.postprocess(fdict['x_hat'])
            tv.utils.save_image(torch.cat([im, im_hat], dim=0), fp=save_dir / imname)
