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
import lvae.models.qarv.model as qarv


@torch.no_grad()
def top1_acc(p: torch.Tensor, labels: torch.LongTensor):
    assert (p.device == labels.device) and (p.dim() == 2) and (p.shape[0] == labels.shape[0])
    _, p_cls = torch.max(p, dim=1)
    tp = (p_cls == labels)
    acc = float(tp.sum()) / len(tp)
    assert 0 <= acc <= 1
    return acc * 100.0


class ClassificationCutPoint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ClassificationLossyVAE(qarv.VariableRateLossyVAE):
    def __init__(self, config: dict):
        super().__init__(config)
        dim_cls = config['classification_channel']
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            # nn.LayerNorm(config['classification_channel']),
            nn.Flatten(1),
            # nn.Linear(dim_cls, dim_cls*2),
            # nn.GELU(),
            # nn.Linear(dim_cls*2, 1000)
            nn.Linear(dim_cls, 1000)
        )
        self.cls_only = bool(config['cls_only'])

    def forward_end2end(self, im: torch.Tensor, lmb: torch.Tensor, get_latents=False):
        x = self.preprocess_input(im)
        # ================ get lambda embedding ================
        emb = self._get_lmb_embedding(lmb, n=im.shape[0])
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
                                       get_latent=get_latents)
                all_block_stats.append(stats)
            elif isinstance(block, ClassificationCutPoint):
                y_logits = self.classifier(feature)
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
        return feature, all_block_stats, y_logits

    def forward(self, batch, lmb=None, return_rec=False):
        assert isinstance(batch, (tuple, list))
        im, y = batch
        im = im.to(self._dummy.device)
        y = y.to(self._dummy.device)
        nB, imC, imH, imW = im.shape # batch, channel, height, width

        # ================ computing flops ================
        if self._flops_mode:
            lmb = self.sample_lmb(n=im.shape[0])
            return self._forward_flops(im, lmb)

        # ================ Forward pass ================
        if (lmb is None): # training
            lmb = self.sample_lmb(n=im.shape[0])
        assert isinstance(lmb, torch.Tensor) and lmb.shape == (nB,)
        x_hat, stats_all, y_logits = self.forward_end2end(im, lmb)

        # ================ Compute Loss ================
        # rate
        kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        ndims = float(imC * imH * imW)
        kl = sum(kl_divergences) / ndims # nats per dimension
        # distortion
        x_target = self.preprocess_target(im)
        distortion = self.distortion_func(x_hat, x_target)
        # classification
        l_cls = tnf.cross_entropy(y_logits, y, reduction='mean')
        # rate + distortion
        if self.cls_only:
            loss = 0*(kl + lmb * distortion) + l_cls
        else:
            loss = kl + lmb * distortion + l_cls
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
            stats['top-1'] = top1_acc(y_logits, y)

        if return_rec:
            stats['im_hat'] = im_hat
        return stats

    def forward_cls(self, im):
        im = im.to(self._dummy.device)
        x = self.preprocess_input(im)
        # ================ get lambda embedding ================
        lmb = self.lmb_range[1]
        emb = self._get_lmb_embedding(lmb, n=im.shape[0])
        # ================ Forward pass ================
        enc_features = self.encoder(x, emb)
        y_logits = None
        all_block_stats = []
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                key = int(feature.shape[2])
                f_enc = enc_features[key]
                feature, stats = block(feature, emb, enc_feature=f_enc, mode='trainval')
                all_block_stats.append(stats)
            elif isinstance(block, ClassificationCutPoint):
                y_logits = self.classifier(feature)
                break
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
        assert y_logits is not None, 'ClassificationCutPoint not found'
        return y_logits, all_block_stats

    def conditional_sample(self, lmb, latents, emb=None, bhw_repeat=None, t=1.0):
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
        # initialize lmb and embedding
        lmb = self.expand_to_tensor(lmb, n=nB)
        if emb is None:
            emb = self._get_lmb_embedding(lmb, n=nB)
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        idx = 0
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                feature, _ = block(feature, emb, mode='sampling', latent=latents[idx], t=t)
                idx += 1
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, emb)
            else:
                feature = block(feature)
        assert idx == len(latents)
        im_samples = self.process_output(feature)
        return im_samples

    def unconditional_sample(self, lmb, bhw_repeat, t=1.0):
        """ unconditionally sample, ie, generate new images

        Args:
            bhw_repeat (tuple): repeat the initial constant feature n,h,w times
            t (float): temprature
        """
        return self.conditional_sample(lmb, latents=None, bhw_repeat=bhw_repeat, t=t)

    @torch.no_grad()
    def study(self, save_dir, **kwargs):
        save_dir = Path(save_dir)
        if not save_dir.is_dir():
            save_dir.mkdir(parents=False)

        lmb = self.expand_to_tensor(self.default_lmb, n=1)
        # unconditional samples
        for k in [2, 4]:
            num = 6
            im_samples = self.unconditional_sample(lmb, bhw_repeat=(num,k,k))
            save_path = save_dir / f'samples_k{k}_hw{im_samples.shape[2]}.png'
            tv.utils.save_image(im_samples, fp=save_path, nrow=math.ceil(num**0.5))
        # reconstructions
        for imname in self._logging_images:
            impath = f'images/{imname}'
            im = tvf.to_tensor(Image.open(impath)).unsqueeze_(0).to(device=self._dummy.device)
            x_hat, _, _ = self.forward_end2end(im, lmb=lmb)
            im_hat = self.process_output(x_hat)
            tv.utils.save_image(torch.cat([im, im_hat], dim=0), fp=save_dir / imname)

    def compress_mode(self, mode=True):
        if mode:
            for block in self.dec_blocks:
                if hasattr(block, 'update'):
                    block.update()
        self.compressing = mode

    @torch.no_grad()
    def compress(self, im, lmb=None):
        if lmb is None: # use default log-lambda
            lmb = self.default_lmb
        lmb = self.expand_to_tensor(lmb, n=im.shape[0])
        lmb_embedding = self._get_lmb_embedding(lmb, n=im.shape[0])
        x = self.preprocess_input(im)
        enc_features = self.encoder(x, lmb_embedding)
        nB, _, nH, nW = enc_features[min(enc_features.keys())].shape
        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        strings_all = []
        for i, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                f_enc = enc_features[feature.shape[2]]
                feature, stats = block(feature, lmb_embedding, enc_feature=f_enc, mode='compress')
                strings_all.append(stats['strings'])
            elif getattr(block, 'requires_embedding', False):
                feature = block(feature, lmb_embedding)
            else:
                feature = block(feature)
        strings_all.append((nB, nH, nW)) # smallest feature shape
        strings_all.append(lmb) # log lambda
        return strings_all

    @torch.no_grad()
    def decompress(self, compressed_object):
        lmb = compressed_object[-1] # log lambda
        nB, nH, nW = compressed_object[-2] # smallest feature shape
        lmb = self.expand_to_tensor(lmb, n=nB)
        lmb_embedding = self._get_lmb_embedding(lmb, n=nB)

        feature = self.get_bias(bhw_repeat=(nB, nH, nW))
        str_i = 0
        for bi, block in enumerate(self.dec_blocks):
            if getattr(block, 'is_latent_block', False):
                strs_batch = compressed_object[str_i]
                feature, _ = block(feature, lmb_embedding, mode='decompress', strings=strs_batch)
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
    def _self_evaluate(self, img_paths, lmb: float, pbar=False, log_dir=None):
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
            x_hat, stats_all, _ = self.forward_end2end(im, lmb=self.expand_to_tensor(lmb,n=1))
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
            all_image_stats['loss'] += float(kl.item() + lmb * distortion)
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
        p = 3.0
        lambdas = torch.linspace(math.pow(start,1/p), math.pow(end,1/p), steps=steps).pow(3)
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
