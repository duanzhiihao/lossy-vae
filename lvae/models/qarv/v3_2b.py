from PIL import Image
from pathlib import Path
from collections import OrderedDict
import math
import struct
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision as tv
import torchvision.transforms.functional as tvf

from lvae.models.registry import register_model
import lvae.models.common as cm
import lvae.utils.coding as coding
import lvae.models.entropy_coding as entropy_coding


class LatentVariableBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.requires_dict_input = True
        self.discrete_gaussian = entropy_coding.DiscretizedGaussian()
        self.name: str

    def update(self):
        self.discrete_gaussian.update()


class PriorBlock(LatentVariableBlock):
    def __init__(self, width, zdim, name=None):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width
        self.name = name

        self.prior = cm.conv_k1s1(width, zdim*2)
        self.z_proj = cm.conv_k1s1(zdim, width)

    def get_prior(self, feature):
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 # make logscale > -2.3
        pv = torch.exp(plogv)
        # pv = torch.exp(plogv) + 0.01
        return pm, pv

    def forward(self, fdict, qm=None):
        feature = fdict['em_feature']
        mode = fdict['mode']

        pm, pv = self.get_prior(feature)

        if mode == 'trainval': # training or validation
            assert qm is not None # posterior mean
            # qm = fdict['all_features'][f'{self.name}_qm']
            if self.training: # if training, then use additive uniform noise
                z = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
                log_prob = entropy_coding.gaussian_log_prob_mass(pm, pv, x=z, bin_size=1.0, prob_clamp=1e-6)
                kl = -1.0 * log_prob
            else: # if evaluation, then use residual quantization
                z, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
                kl = -1.0 * torch.log(probs)
            fdict['kl_divs'].append(kl)
        elif mode == 'sampling':
            z = fdict['zs'].pop(0)
            if z is None: # if z is not provided, sample it from the prior
                t = fdict['temperature']
                z = pm + pv * torch.randn_like(pm) * t + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
        elif mode == 'compress': # encode z into bits
            assert qm is not None # posterior mean
            # qm = fdict['all_features'][f'{self.name}_qm']
            indexes = self.discrete_gaussian.build_indexes(pv)
            strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
            z = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
            fdict['bit_strings'].append(strings)
        elif mode == 'decompress': # decode the bits
            strings = fdict['bit_strings'].pop(0)
            indexes = self.discrete_gaussian.build_indexes(pv)
            z = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        else:
            raise ValueError(f'Unknown mode={mode}')

        feature = feature + self.z_proj(z)

        fdict['em_feature'] = feature
        fdict['zs'].append(z)
        fdict['all_features'][f'{self.name}_z'] = z
        fdict['all_features'][f'{self.name}_out'] = feature
        return fdict


class PosteriorBlock(nn.Module):
    default_embedding_dim = 256
    def __init__(self, width, zdim, enc_key, enc_width, embed_dim=None, kernel_size=7):
        super().__init__()
        self.enc_key = enc_key

        block = cm.ConvNeXtBlockAdaLN
        embed_dim = embed_dim or self.default_embedding_dim
        self.posterior0 = block(enc_width, embed_dim, kernel_size=kernel_size)
        self.posterior1 = block(width,     embed_dim, kernel_size=kernel_size)
        self.posterior2 = block(width,     embed_dim, kernel_size=kernel_size)
        self.post_merge = cm.conv_k1s1(width + enc_width, width)
        self.posterior  = cm.conv_k3s1(width, zdim)

    def forward(self, fdict):
        f_enc = fdict['all_features'][self.enc_key]
        f_em = fdict['em_feature']
        lmb_emb = fdict['lmb_emb']

        assert f_em.shape[2:4] == f_enc.shape[2:4]
        f_enc = self.posterior0(f_enc, lmb_emb)
        f_em = self.posterior1(f_em, lmb_emb)
        merged = torch.cat([f_em, f_enc], dim=1)
        merged = self.post_merge(merged)
        merged = self.posterior2(merged, lmb_emb)
        qm = self.posterior(merged)
        return qm


class MergeFromEM(nn.Module):
    def __init__(self, key, in_dim, out_dim):
        super().__init__()
        self.key = key
        self.requires_dict_input = True
        self.merge = cm.conv_k1s1(in_dim, out_dim)

    def forward(self, fdict):
        feature = fdict['dec_feature']
        f_em_z = fdict['all_features'][f'{self.key}_z']
        f_em_out = fdict['all_features'][f'{self.key}_out']
        assert feature.shape[2:4] == f_em_z.shape[2:4] == f_em_out.shape[2:4]
        feature = self.merge(torch.cat([feature, f_em_z, f_em_out], dim=1))

        fdict['dec_feature'] = feature
        return fdict


def mlp(dims):
    return nn.Sequential(
        nn.Linear(dims[0], dims[1]),
        nn.GELU(),
        nn.Linear(dims[1], dims[1]),
    )

def freeze_modules(*modules):
    for m in modules:
        if isinstance(m, nn.Module):
            for p in m.parameters():
                p.requires_grad_(False)
        elif isinstance(m, nn.Parameter):
            m.requires_grad_(False)
        else:
            raise ValueError(f'Unknown module type: {type(m)}')


class VariableRateLossyVAE(nn.Module):
    log2_e = math.log2(math.e)
    MAX_LOG_LMB = math.log(8192)

    def __init__(self, config: dict):
        super().__init__()
        # Encoding
        self.enc_blocks = nn.ModuleList(config.pop('enc_blocks'))
        self.posteriors = nn.ModuleDict(config.pop('posteriors'))
        # Decoding (top-down path)
        self.em_blocks = nn.ModuleList(config.pop('em_blocks'))
        self.dec_blocks = nn.ModuleList(config.pop('dec_blocks'))
        # lambda embedding layers
        self._setup_lmb_embedding(config)

        # initial bias for the top-down path
        width = self.em_blocks[0].in_channels
        self.em_bias = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.dec_bias = nn.Parameter(torch.zeros(1, width, 1, 1))

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor
        self.num_latents = len([b for b in self.em_blocks if isinstance(b, LatentVariableBlock)])
        self.max_stride = config['max_stride']
        self._logging_images = config.get('log_images', [])
        self._flops_mode = False

    def _setup_lmb_embedding(self, config):
        _low, _high = config['lmb_range']
        self.lmb_range = (float(_low), float(_high))
        self.lmb_embed_dim = config['lmb_embed_dim']
        self._sin_period = config['sin_period']
        self.lmb_embedding = mlp(self.lmb_embed_dim)

        self.default_lmb = self.lmb_range[1]

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
        lmb_emb = cm.sinusoidal_embedding(scaled, self.lmb_embed_dim[0], max_period=self._sin_period)
        lmb_emb = self.lmb_embedding(lmb_emb)
        return lmb_emb

    def get_initial_fdict(self, lmb, bias_bhw):
        """ Get an initial empty feature dictionary

        Args:
            bias_bhw (tuple): (batch, height, width) for the initial top-down feature
        """
        fdict = dict() # a feature dictionary containing all features
        fdict['lmb_emb'] = self.get_lmb_embedding(lmb)
        # ======== for 'trainval' mode ========
        fdict['all_features'] = OrderedDict() # bottom-up encoder features
        nB, nH, nW = bias_bhw
        fdict['em_feature'] = self.em_bias.expand(nB, -1, nH, nW) # top-down entropy model feature
        fdict['dec_feature'] = self.dec_bias.expand(nB, -1, nH, nW) # top-down decoder feature
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
        x = self.preprocess(im) # normalized image

        f = x # feature
        enc_features = OrderedDict()
        for i, block in enumerate(self.enc_blocks):
            if isinstance(block, cm.SetKey):
                enc_features[block.key] = f
            elif getattr(block, 'requires_embedding', False):
                f = block(f, fdict['lmb_emb'])
            else:
                f = block(f)
        fdict['all_features'].update(enc_features)
        return fdict, x

    def forward_em(self, fdict, mode='trainval'): # top-down entropy model branch
        fdict['mode'] = mode

        for i, block in enumerate(self.em_blocks):
            if isinstance(block, LatentVariableBlock):
                qm = self.posteriors[block.name](fdict) if mode in ['trainval', 'compress'] else None
                fdict = block(fdict, qm=qm)
            elif getattr(block, 'requires_embedding', False):
                fdict['em_feature'] = block(fdict['em_feature'], fdict['lmb_emb'])
            else:
                fdict['em_feature'] = block(fdict['em_feature'])
        return fdict

    def forward_topdown(self, fdict, mode='trainval'):
        fdict = self.forward_em(fdict, mode)

        for i, block in enumerate(self.dec_blocks): # top-down decoder branch
            if getattr(block, 'requires_dict_input', False):
                fdict = block(fdict)
            elif getattr(block, 'requires_embedding', False):
                fdict['dec_feature'] = block(fdict['dec_feature'], fdict['lmb_emb'])
            else:
                fdict['dec_feature'] = block(fdict['dec_feature'])

        fdict['x_hat'] = fdict.pop('dec_feature') # rename 'dec_feature' to 'x_hat'
        return fdict

    def forward(self, im, lmb=None, return_fdict=False):
        im = im.to(self._dummy.device)
        B, imC, imH, imW = im.shape # batch, channel, height, width

        # ================ Forward pass ================
        if (lmb is None) and self.training:
            lmb = self.sample_lmb(n=im.shape[0])
        elif (lmb is None):
            lmb = torch.full((B,), self.default_lmb, device=self._dummy.device)
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

    def prepare_compression(self):
        for block in self.em_blocks:
            if hasattr(block, 'update'):
                block.update()

    @torch.inference_mode()
    def compress(self, im):
        assert im.shape[0] == 1, f'Right now only support a single image; got {im.shape=}'

        lmb = torch.full((1,), self.default_lmb, device=self._dummy.device) # use the default lambda
        fdict, _ = self.forward_bottomup(im, lmb)
        fdict = self.forward_em(fdict, mode='compress')

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
        # lmb = torch.full((1,), self.default_lmb, device=self._dummy.device) # use the default lambda
        # for imname in self._logging_images:
        #     impath = f'images/{imname}'
        #     im = tvf.to_tensor(Image.open(impath)).unsqueeze_(0).to(device=self._dummy.device)
        #     fdict, _ = self.forward_bottomup(im, lmb)
        #     fdict = self.forward_topdown(fdict, mode='trainval')
        #     im_hat = self.postprocess(fdict['x_hat'])
        #     tv.utils.save_image(torch.cat([im, im_hat], dim=0), fp=save_dir / imname)


@register_model
def q2b_4z(lmb_range=(64,8192), pretrained=False, sin_period=64):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = lmb_range
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = sin_period

    # model configuration
    ch = 128
    enc_dims = [ch*1, ch*2, ch*3, ch*2, ch*1]

    res_block = cm.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        cm.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0]) for _ in range(6)],
        cm.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1]) for _ in range(6)],
        cm.SetKey('enc_s8'),
        cm.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2]) for _ in range(6)],
        cm.SetKey('enc_s16'),
        cm.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=5) for _ in range(4)],
        cm.SetKey('enc_s32'),
        cm.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=3) for _ in range(4)],
        cm.SetKey('enc_s64'),
    ]

    dec_dims = [ch*1, ch*2, ch*3, ch*2, ch*1]
    z_dims = [128, 128, 256, 32]
    PosteriorBlock.default_embedding_dim = cfg['lmb_embed_dim'][1]
    cfg['posteriors'] = {
        'z1': PosteriorBlock(dec_dims[0], z_dims[0], 'enc_s64', enc_width=enc_dims[-1], kernel_size=3),
        'z2': PosteriorBlock(dec_dims[1], z_dims[1], 'enc_s32', enc_width=enc_dims[-2], kernel_size=5),
        'z3': PosteriorBlock(dec_dims[2], z_dims[2], 'enc_s16', enc_width=enc_dims[-3]),
        'z4': PosteriorBlock(dec_dims[3], z_dims[3], 'enc_s8',  enc_width=enc_dims[-4]),
    }
    cfg['em_blocks'] = [
        # 1x1
        PriorBlock(dec_dims[0], z_dims[0], name='z1'),
        res_block(dec_dims[0], kernel_size=3),
        cm.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=5),
        PriorBlock(dec_dims[1], z_dims[1], name='z2'),
        res_block(dec_dims[1], kernel_size=5),
        cm.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        res_block(dec_dims[2]),
        PriorBlock(dec_dims[2], z_dims[2], name='z3'),
        res_block(dec_dims[2]),
        cm.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        *[res_block(dec_dims[3]) for _ in range(2)],
        PriorBlock(dec_dims[3], z_dims[3], name='z4'),
    ]
    cfg['dec_blocks'] = [
        # 1x1
        MergeFromEM('z1', dec_dims[0]*2 + z_dims[0], dec_dims[0]),
        *[res_block(dec_dims[0], kernel_size=3) for _ in range(2)],
        cm.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        *[res_block(dec_dims[1], kernel_size=5) for _ in range(2)],
        MergeFromEM('z2', dec_dims[1]*2 + z_dims[1], dec_dims[1]),
        *[res_block(dec_dims[1], kernel_size=5) for _ in range(2)],
        cm.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        *[res_block(dec_dims[2]) for _ in range(3)],
        MergeFromEM('z3', dec_dims[2]*2 + z_dims[2], dec_dims[2]),
        *[res_block(dec_dims[2]) for _ in range(3)],
        cm.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        *[res_block(dec_dims[3]) for _ in range(3)],
        MergeFromEM('z4', dec_dims[3]*2 + z_dims[3], dec_dims[3]),
        *[res_block(dec_dims[3]) for _ in range(3)],
        cm.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4]) for _ in range(6)],
        cm.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    model = VariableRateLossyVAE(cfg)

    if pretrained is True:
        raise NotImplementedError()
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/cloc/qb4z_35m-sin64-500k.pt'
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    elif pretrained: # str or Path
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['model'])

    return model
