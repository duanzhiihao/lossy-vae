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


class VRLVBlockBase(nn.Module):
    """ Vriable-Rate Latent Variable Block
    """
    default_embedding_dim = 256
    def __init__(self, width, zdim, enc_key, enc_width, embed_dim=None, kernel_size=7, mlp_ratio=2):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width
        self.enc_key = enc_key

        block = common.ConvNeXtBlockAdaLN
        embed_dim = embed_dim or self.default_embedding_dim
        self.resnet_front = block(width,   embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end   = block(width,   embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.posterior0 = block(enc_width, embed_dim, kernel_size=kernel_size)
        self.posterior1 = block(width,     embed_dim, kernel_size=kernel_size)
        self.posterior2 = block(width,     embed_dim, kernel_size=kernel_size)
        self.post_merge = common.conv_k1s1(width + enc_width, width)
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

    def fuse_feature_and_z(self, feature, z):
        # add the new information carried by z to the feature
        feature = feature + self.z_proj(z)
        return feature

    def forward(self, fdict, mode='trainval', latent=None, t=1.0, strings=None):
        feature = fdict['feature']
        emb = fdict['lmb_emb']

        feature, pm, pv = self.transform_prior(feature, emb)

        if mode == 'trainval': # training or validation
            enc_feature = fdict['enc_features'][self.enc_key]
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
            if latent is None: # if z is not provided, sample it from the prior
                z = pm + pv * torch.randn_like(pm) * t + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
            else: # if `z` is provided, directly use it.
                assert pm.shape == latent.shape
                z = latent
        elif mode == 'compress': # encode z into bits
            enc_feature = fdict['enc_features'][self.enc_key]
            qm = self.transform_posterior(feature, enc_feature, emb)
            indexes = self.discrete_gaussian.build_indexes(pv)
            strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
            z = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
            fdict['bit_strings'].append(strings)
        elif mode == 'decompress': # decode z from bits
            assert strings is not None
            indexes = self.discrete_gaussian.build_indexes(pv)
            z = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        else:
            raise ValueError(f'Unknown mode={mode}')

        feature = self.fuse_feature_and_z(feature, z)
        feature = self.resnet_end(feature, emb)
        fdict['feature'] = feature
        fdict['zs'].append(z)
        return fdict

    def update(self):
        self.discrete_gaussian.update()


def mse_loss(fake, real):
    assert fake.shape == real.shape
    return tnf.mse_loss(fake, real, reduction='none').mean(dim=(1,2,3))


class VariableRateLossyVAE(nn.Module):
    log2_e = math.log2(math.e)
    MAX_LMB = 8192

    def __init__(self, config: dict):
        super().__init__()
        # feature extractor (bottom-up path)
        self.encoder = common.FeatureExtractorWithEmbedding(config.pop('enc_blocks'))
        # latent variable blocks (top-down path)
        self.dec_blocks = nn.ModuleList(config.pop('dec_blocks'))
        width = self.dec_blocks[0].in_channels
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))

        self.num_latents = len([b for b in self.dec_blocks if getattr(b, 'is_latent_block', False)])
        # loss function, for computing reconstruction loss
        self.distortion_name = 'mse'
        self.distortion_func = mse_loss

        self._setup_lmb_embedding(config)

        self.im_shift = float(config['im_shift'])
        self.im_scale = float(config['im_scale'])
        self.max_stride = config['max_stride']

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

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

    def preprocess_input(self, im: torch.Tensor):
        """ Shift and scale the input image

        Args:
            im (torch.Tensor): a batch of images, values should be between (0, 1)
        """
        assert (im.shape[2] % self.max_stride == 0) and (im.shape[3] % self.max_stride == 0)
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

    @torch.inference_mode()
    def _forward_flops(self, im):
        im = im.uniform_(0, 1)
        if self._flops_mode == 'compress':
            compressed_obj = self.compress(im)
        elif self._flops_mode == 'decompress':
            n, h, w = im.shape[0], im.shape[2]//self.max_stride, im.shape[3]//self.max_stride
            samples = self.unconditional_sample(bhw_repeat=(n,h,w))
        elif self._flops_mode == 'end-to-end':
            lmb = self.sample_lmb(n=im.shape[0])
            x_hat, stats_all = self.forward_end2end(im, lmb=lmb)
        else:
            raise ValueError(f'Unknown self._flops_mode: {self._flops_mode}')
        return

    def sample_lmb(self, n):
        low, high = self.lmb_range # original lmb space, 16 to 1024
        p = 3.0
        low, high = math.pow(low, 1/p), math.pow(high, 1/p) # transformed space
        transformed_lmb = low + (high-low) * torch.rand(n, device=self._dummy.device)
        lmb = torch.pow(transformed_lmb, exponent=p)
        assert isinstance(lmb, torch.Tensor) and lmb.shape == (n,)
        return lmb

    def expand_to_tensor(self, input_, n):
        assert isinstance(input_, (torch.Tensor, float, int)), f'{type(input_)=}'
        if isinstance(input_, torch.Tensor) and (input_.numel() == 1):
            input_ = input_.item()
        if isinstance(input_, (float, int)):
            input_ = torch.full(size=(n,), fill_value=float(input_), device=self._dummy.device)
        assert input_.shape == (n,), f'{input_=}, {input_.shape=}'
        return input_

    def _lmb_scaling(self, lmb: torch.Tensor):
        # p = 3.0
        # lmb_input = torch.pow(lmb / self.MAX_LMB, 1/p) * self._sin_period
        lmb_input = torch.log(lmb) * self._sin_period / math.log(self.MAX_LMB)
        return lmb_input

    def _get_lmb_embedding(self, lmb, n):
        lmb = self.expand_to_tensor(lmb, n=n)
        scaled = self._lmb_scaling(lmb)
        embedding = common.sinusoidal_embedding(scaled, dim=self.lmb_embed_dim[0],
                                                max_period=self._sin_period)
        embedding = self.lmb_embedding(embedding)
        return embedding

    def get_bias(self, bhw_repeat=(1,1,1)):
        nB, nH, nW = bhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        return feature

    def forward_end2end(self, im: torch.Tensor, lmb: torch.Tensor, mode='trainval'):
        x = self.preprocess_input(im)

        fdict = dict() # a feature dictionary containing all features
        fdict['lmb_emb'] = self._get_lmb_embedding(lmb, n=im.shape[0])
        fdict['enc_features'] = self.encoder(x, fdict['lmb_emb']) # bottom-up encoder features
        fdict['dec_features'] = [] # top-down decoder features
        fdict['zs'] = [] # latent variables
        fdict['kl_divs'] = [] # kl (i.e., rate) for each latent variable
        fdict['bit_strings'] = [] # compressed bit strings; only used in 'compress' mode
        nB, _, xH, xW = x.shape
        feature = self.get_bias(bhw_repeat=(nB, xH//self.max_stride, xW//self.max_stride))
        fdict['feature'] = feature # main feature; will be updated in the following loop
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                fdict = block(fdict, mode=mode)
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

        # ================ computing flops ================
        if self._flops_mode:
            return self._forward_flops(im)

        # ================ Forward pass ================
        lmb = self.sample_lmb(n=im.shape[0]) # (B,)
        fdict = self.forward_end2end(im, lmb)

        # ================ Compute Loss ================
        x_hat, kl_divs = fdict['x_hat'], fdict['kl_divs']
        # rate
        kl_divs = [kl.sum(dim=(1, 2, 3)) for kl in kl_divs]
        kl = sum(kl_divs) / float(imC * imH * imW) # nats per dimensionm, shape (B,)
        # distortion
        x_target = self.preprocess_target(im)
        distortion = self.distortion_func(x_hat, x_target) # (B,)
        # rate + distortion
        loss = kl + lmb * distortion # (B,)

        metrics = OrderedDict()
        metrics['loss'] = loss.mean(0)

        # ================ Logging ================
        with torch.inference_mode(): # for training progress bar
            metrics['bpp'] = kl.mean(0).item() * self.log2_e * imC
            metrics[self.distortion_name] = distortion.mean(0).item()
            im_hat = self.process_output(x_hat.detach())
            im_mse = tnf.mse_loss(im_hat, im, reduction='mean')
            psnr = -10 * math.log10(im_mse.item())
            metrics['psnr'] = psnr
        if return_fdict:
            return metrics, fdict
        return metrics

    @torch.inference_mode()
    def conditional_sample(self, lmb, latents, emb=None, bhw_repeat=None, t=1.0):
        """ sampling, conditioned on a list of latents variables

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
        lmb = self.expand_to_tensor(lmb, n=nB)
        if emb is None:
            emb = self._get_lmb_embedding(lmb, n=nB)

        fdict = dict() # a feature dictionary containing all features
        fdict['lmb_emb'] = emb
        fdict['dec_features'] = [] # top-down decoder features
        fdict['zs'] = [] # latent variables
        fdict['kl_divs'] = [] # kl (i.e., rate) for each latent variable
        fdict['bit_strings'] = [] # compressed bit strings; only used in 'compress' mode
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        fdict['feature'] = feature # main feature; will be updated in the following loop

        idx = 0
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                fdict = block(fdict, mode='sampling', latent=latents[idx], t=t)
                idx += 1
            elif getattr(block, 'requires_embedding', False):
                fdict['feature'] = block(fdict['feature'], emb)
            else:
                fdict['feature'] = block(fdict['feature'])
        assert idx == len(latents)
        im_samples = self.process_output(fdict['feature'])
        return im_samples

    @torch.inference_mode()
    def unconditional_sample(self, lmb, bhw_repeat, t=1.0):
        """ unconditionally sample, ie, generate new images

        Args:
            bhw_repeat (tuple): repeat the initial constant feature n,h,w times
            t (float): temprature
        """
        return self.conditional_sample(lmb, [None]*self.num_latents, bhw_repeat=bhw_repeat, t=t)

    @torch.inference_mode()
    def study(self, save_dir, **kwargs):
        save_dir = Path(save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir(parents=False)

        lmb = self.expand_to_tensor(self.default_lmb, n=1)
        # unconditional sampling
        for k in [1, 2]:
            num = 6
            im_samples = self.unconditional_sample(lmb, bhw_repeat=(num,k,k))
            save_path = save_dir / f'samples_k{k}_hw{im_samples.shape[2]}.png'
            tv.utils.save_image(im_samples, fp=save_path, nrow=math.ceil(num**0.5))
        # reconstructions
        for imname in self._logging_images:
            impath = f'images/{imname}'
            im = tvf.to_tensor(Image.open(impath)).unsqueeze_(0).to(device=self._dummy.device)
            fdict = self.forward_end2end(im, lmb=lmb)
            im_hat = self.process_output(fdict['x_hat'])
            tv.utils.save_image(torch.cat([im, im_hat], dim=0), fp=save_dir / imname)

    @torch.inference_mode()
    def _self_evaluate(self, img_paths, lmb: float, pbar=False, log_dir=None):
        pbar = tqdm(img_paths) if pbar else img_paths
        all_image_stats = defaultdict(AverageMeter)
        if log_dir is not None:
            log_dir = Path(log_dir)
            channel_bpp_stats = defaultdict(AverageMeter)
        for impath in pbar:
            img = Image.open(impath)
            imgh, imgw = img.height, img.width
            img_padded = coding.pad_divisible_by(img, div=self.max_stride)
            im = tvf.to_tensor(img_padded).unsqueeze_(0).to(device=self._dummy.device)

            fdict = self.forward_end2end(im, lmb=self.expand_to_tensor(lmb,n=1))
            x_hat = fdict['x_hat'][:, :, :imgh, :imgw]
            # compute bpp
            _, imC, imH, imW = im.shape
            kl = sum([kl.sum(dim=(1, 2, 3)) for kl in fdict['kl_divs']]).mean(0) / (imC*imgh*imgw)
            bpp_estimated = kl.item() * self.log2_e * imC
            # compute psnr
            im = tvf.to_tensor(img).unsqueeze_(0).to(device=self._dummy.device)
            x_target = self.preprocess_target(im)
            distortion = self.distortion_func(x_hat, x_target).item()
            real = tvf.to_tensor(img)
            fake = self.process_output(x_hat).cpu().squeeze(0)
            mse = tnf.mse_loss(real, fake, reduction='mean').item()
            psnr = float(-10 * math.log10(mse))
            # accumulate results
            all_image_stats['loss'].update(float(kl.item() + lmb * distortion))
            all_image_stats['bpp'].update(bpp_estimated)
            all_image_stats['psnr'].update(psnr)
            # debugging
            if log_dir is not None:
                _to_bpp = lambda kl: kl.sum(dim=(2,3)).mean(0).cpu() / (imH*imW) * self.log2_e
                channel_bpps = [_to_bpp(kl) for kl in fdict['kl_divs']]
                for i, ch_bpp in enumerate(channel_bpps):
                    channel_bpp_stats[i].update(ch_bpp)
        # average over all images
        avg_stats = {k: v.avg for k,v in all_image_stats.items()}
        avg_stats['lambda'] = lmb
        if log_dir is not None:
            self._log_channel_stats(channel_bpp_stats, log_dir, lmb)

        return avg_stats

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
        with open(log_dir / f'all_lmb_channel_stats.txt', mode='a') as f:
            print(msg, file=f)

    @torch.no_grad()
    def self_evaluate(self, img_dir, lmb_range=None, steps=8, log_dir=None):
        img_paths = list(Path(img_dir).rglob('*.*'))
        start, end = self.lmb_range if (lmb_range is None) else lmb_range
        # uniform in cube root space
        lambdas = torch.linspace(math.log(start), math.log(end), steps=steps).exp()
        pbar = tqdm(lambdas.tolist(), position=0, ascii=True)
        all_lmb_stats = defaultdict(list)
        if log_dir is not None:
            (Path(log_dir) / 'all_lmb_channel_stats.txt').unlink(missing_ok=True)
        for lmb in pbar:
            assert isinstance(lmb, float)
            results = self._self_evaluate(img_paths, lmb, log_dir=log_dir)
            pbar.set_description(f'{lmb=:.3f}, {results=}')
            for k,v in results.items():
                all_lmb_stats[k].append(v)
        return all_lmb_stats

    def compress_mode(self, mode=True):
        if mode:
            for block in self.dec_blocks:
                if hasattr(block, 'update'):
                    block.update()
        self.compressing = mode

    @torch.no_grad()
    def compress(self, im, lmb=None):
        lmb = lmb or self.default_lmb # if no lmb is provided, use the default one
        fdict = self.forward_end2end(im, lmb=lmb, mode='compress')
        assert len(fdict['bit_strings']) == self.num_latents
        assert im.shape[0] == 1, f'Right now only support a single image, got {im.shape=}'
        all_lv_strings = [strings[0] for strings in fdict['bit_strings']]
        string = coding.pack_byte_strings(all_lv_strings)
        # encode lambda and image shape in the header
        nB, _, imH, imW = im.shape
        header1 = struct.pack('f', lmb)
        header2 = struct.pack('3H', nB, imH//self.max_stride, imW//self.max_stride)
        string = header1 + header2 + string
        return string

    @torch.no_grad()
    def decompress(self, string):
        # extract lambda
        _len = 4
        lmb, string = struct.unpack('f', string[:_len])[0], string[_len:]
        # extract shape
        _len = 2 * 3
        (nB, nH, nW), string = struct.unpack('3H', string[:_len]), string[_len:]
        all_lv_strings = coding.unpack_byte_string(string)

        fdict = dict() # a feature dictionary containing all features
        lmb = self.expand_to_tensor(lmb, n=nB)
        fdict['lmb_emb'] = self._get_lmb_embedding(lmb, n=nB)
        fdict['dec_features'] = [] # top-down decoder features
        fdict['zs'] = [] # latent variables
        fdict['kl_divs'] = [] # kl (i.e., rate) for each latent variable
        fdict['bit_strings'] = [] # compressed bit strings; only used in 'compress' mode
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        fdict['feature'] = feature # main feature; will be updated in the following loop

        str_i = 0
        for bi, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                strs_batch = [all_lv_strings[str_i],]
                fdict = block(fdict, mode='decompress', strings=strs_batch)
                str_i += 1
            elif getattr(block, 'requires_embedding', False):
                fdict['feature'] = block(fdict['feature'], fdict['lmb_emb'])
            else:
                fdict['feature'] = block(fdict['feature'])
        assert str_i == len(all_lv_strings), f'str_i={str_i}, len={len(all_lv_strings)}'
        im_hat = self.process_output(fdict['feature'])
        return im_hat

    @torch.no_grad()
    def compress_file(self, img_path, output_path, lmb=None):
        # read image
        img = Image.open(img_path)
        img_padded = coding.pad_divisible_by(img, div=self.max_stride)
        im = tvf.to_tensor(img_padded).unsqueeze_(0).to(device=self._dummy.device)
        # compress by model
        body_str = self.compress(im, lmb=lmb)
        header_str = struct.pack('2H', img.height, img.width)
        # save bits to file
        with open(output_path, 'wb') as f:
            f.write(header_str + body_str)

    @torch.no_grad()
    def decompress_file(self, bits_path):
        # read from file
        with open(bits_path, 'rb') as f:
            header_str = f.read(4)
            body_str = f.read()
        img_h, img_w = struct.unpack('2H', header_str)
        # decompress by model
        im_hat = self.decompress(body_str)
        return im_hat[:, :, :img_h, :img_w]
