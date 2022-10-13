from pathlib import Path
from collections import OrderedDict, defaultdict
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision as tv

# from mycv.utils import AverageMeter, print_to_file
# from mycv.utils.opflow import forward_warp_torch, interpolate_and_scale
# import mycv.models.vae.blocks as vaeblocks
# import mycv.models.vae.qresvideo.video_model as qrvm
# import mycv.datasets.compression as mycpr
# from mycv.models.registry import register_video_model

from models.registry import register_model


class MEBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        # self.relu = nn.GELU()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


def myconvnext_down(dim, new_dim, kernel_size=7):
    module = nn.Sequential(
        vaeblocks.MyConvNeXtBlock(dim, kernel_size=kernel_size),
        vaeblocks.conv_k3s2(dim, new_dim),
        # nn.AvgPool2d(kernel_size=2, stride=2)
    )
    return module


class MyVideoModelSimple(nn.Module):
    log2_e = math.log2(math.e)

    def __init__(self, distortion_lmb):
        super().__init__()
        im_channels = 3
        # ================================ feature extractor ================================
        ch = 96
        enc_dims = (32, 64, ch*1, ch*2, ch*4, ch*4, ch*4)
        enc_nums     = (1, 2, 2, 2, 2, 2, 2)
        kernel_sizes = (7, 7, 7, 7, 7, 5, 3)
        enc_blocks = [vaeblocks.conv_k3s1(im_channels, 32),]
        for i, (dim, ks, num) in enumerate(zip(enc_dims, kernel_sizes, enc_nums,)):
            for _ in range(num):
                enc_blocks.append(vaeblocks.MyConvNeXtBlock(dim, kernel_size=ks))
            if i < len(enc_dims) - 1:
                new_dim = enc_dims[i+1]
                # enc_blocks.append(vaeblocks.MyConvNeXtPatchDown(dim, new_dim, kernel_size=ks))
                enc_blocks.append(myconvnext_down(dim, new_dim, kernel_size=ks))
        self.feature_extractor = vaeblocks.BottomUpEncoder(enc_blocks, dict_key='stride')

        self.strides_that_have_bits = set([4, 8, 16, 32, 64])
        # ================================ flow models ================================
        global_strides = (1, 2, 4, 8, 16, 32, 64)
        flow_dims = (None, None, 48, 72, 96, 128, 128)
        flow_zdims = (None, None, 2, 4, 4, 4, 4)
        kernel_sizes = (7, 7, 7, 7, 7, 5, 3)
        self.flow_blocks = nn.ModuleDict()
        for s, indim, dim, zdim, ks in zip(global_strides, enc_dims, flow_dims, flow_zdims, kernel_sizes):
            if s not in self.strides_that_have_bits:
                continue
            corr_dim, strided = (96, True) if (s == 4) else (128, False)
            module = qrvm.CorrelationFlowCodingBlock(indim, dim=dim, zdim=zdim, ks=ks,
                            corr_dim=corr_dim, strided_corr=strided)
            self.flow_blocks[f'stride{s}'] = module
        # ================================ p-frame models ================================
        dec_dims = enc_dims
        self.bias = nn.Parameter(torch.zeros(1, dec_dims[-1], 1, 1))
        self.dec_blocks = nn.ModuleDict()
        for s, dim, ks in zip(global_strides, dec_dims, kernel_sizes):
            if s in self.strides_that_have_bits:
                module = qrvm.SpyCodingFrameBlock(dim, zdim=8, kernel_size=ks)
            else:
                module = qrvm.ResConditional(dim, kernel_size=ks)
            self.dec_blocks[f'stride{s}'] = module
        # ================================ upsample layers ================================
        self.upsamples = nn.ModuleDict()
        for i, (s, dim) in enumerate(zip(global_strides, dec_dims)):
            if s == 1: # same as image resolution; no need to upsample
                conv = vaeblocks.conv_k3s1(dim, im_channels)
            else:
                conv = vaeblocks.patch_upsample(dim, dec_dims[i-1], rate=2)
            self.upsamples[f'stride{s}'] = conv

        self.global_strides = global_strides
        self.max_stride = max(global_strides)

        self.distortion_lmb = float(distortion_lmb)

        self.stop_monitoring()

    def stop_monitoring(self):
        self._monitor_flag = False
        self._bpp_stats = None

    def start_monitoring(self):
        self._monitor_flag = True
        self._bpp_stats = defaultdict(AverageMeter)

    def save_monitoring_results(self, log_dir):
        log_dir = Path(log_dir)
        if not log_dir.is_dir():
            log_dir.mkdir(parents=True)
        self._save_results_with_prefix('flow-', log_dir)
        self._save_results_with_prefix('frame-', log_dir)

    def _save_results_with_prefix(self, prefix, log_dir):
        keys = [k for k in sorted(self._bpp_stats.keys()) if k.startswith(prefix)]
        msg = '---- row: latent blocks, colums: channels, avg over images ----\n'
        for k in keys:
            msg += ''.join([f'{a:<7.4f} ' for a in self._bpp_stats[k].avg.tolist()]) + '\n'
        print_to_file(msg, log_dir/f'{prefix}bpp-channels.txt', mode='w')
        block_bpps = [self._bpp_stats[k].avg.sum(dim=0).item() for k in keys]
        msg = ''.join([f'{a:<7.4f} ' for a in block_bpps])
        print_to_file(msg, log_dir/f'{prefix}bpp-blocks.txt', mode='a')

    def get_bias(self, nhw_repeat=(1,1,1)):
        nB, nH, nW = nhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        return feature

    def forward_end2end(self, im, im_prev, get_intermediate=False):
        curr_features = self.feature_extractor(im)
        prev_features = self.feature_extractor(im_prev)
        flow = None
        nB, _, nH, nW = prev_features[self.max_stride].shape
        feature = self.get_bias(nhw_repeat=(nB, nH, nW))
        flow_stats = []
        frame_stats = []
        for s in sorted(self.global_strides, reverse=True): # from large stride to small stride
            # select features
            f_curr = curr_features[s]
            f_prev = prev_features[s]
            if flow is None: # the lowest resolution level
                nB, _, nH, nW = f_prev.shape
                flow = torch.zeros(nB, 2, nH, nW, device=im.device)
                f_warped = f_prev
            else: # bilinear upsampling of the flow
                flow = tnf.interpolate(flow, scale_factor=2, mode='bilinear') * 2.0
                # warp t-1 feature by the flow
                f_warped = forward_warp_torch(f_prev, flow)
            # TODO: use neighbourhood attention to implement flow estimation
            blk_key = f'stride{s}' # block key
            if s in self.strides_that_have_bits:
                flow, stats = self.flow_blocks[blk_key](flow, f_warped, f_curr)
                # warp t-1 feature again
                f_warped = forward_warp_torch(f_prev, flow)
            else:
                stats = dict()
            if get_intermediate:
                stats['flow'] = flow
            flow_stats.append(stats)
            # p-frame prediction
            if s in self.strides_that_have_bits:
                feature, stats = self.dec_blocks[blk_key](feature, f_warped, f_curr)
            else:
                feature = self.dec_blocks[blk_key]([feature, f_warped])
                stats = dict()
            if get_intermediate:
                stats['feature'] = feature
            frame_stats.append(stats)
            feature = self.upsamples[blk_key](feature)
        x_hat = feature
        return flow_stats, frame_stats, x_hat, flow

    @staticmethod
    def _sum_total_kl(list_of_kls):
        bpp = sum([kl.sum(dim=(1,2,3)) for kl in list_of_kls]).mean(0)
        return bpp

    def forward(self, im, im_prev):
        # ================ Forward pass ================
        flow_stats, frame_stats, x_hat, flow_hat = self.forward_end2end(im, im_prev)

        # ================ Compute loss ================
        num_pix = float(im.shape[2] * im.shape[3])
        # Rate
        flow_kls  = [stat['kl'] for stat in flow_stats  if ('kl' in stat)]
        frame_kls = [stat['kl'] for stat in frame_stats if ('kl' in stat)]
        # from total nats to bits-per-pixel
        flow_bpp  = self.log2_e * sum([kl.sum(dim=(1,2,3)) for kl in flow_kls]).mean(0) / num_pix
        frame_bpp = self.log2_e * sum([kl.sum(dim=(1,2,3)) for kl in frame_kls]).mean(0) / num_pix
        # Distortion
        mse = tnf.mse_loss(x_hat, im, reduction='mean')
        # Rate + lmb * Distortion
        loss = (flow_bpp + frame_bpp) + self.distortion_lmb * mse

        stats = OrderedDict()
        stats['loss'] = loss
        # ================ Logging ================
        with torch.no_grad():
            stats['bpp'] = (flow_bpp + frame_bpp).item()
            stats['psnr'] = -10 * math.log10(mse.item())
            stats['mv-bpp'] = flow_bpp.item()
            stats['fr-bpp'] = frame_bpp.item()
            warp_mse = tnf.mse_loss(forward_warp_torch(im_prev, flow_hat), im)
            stats['warp-psnr'] = -10 * math.log10(warp_mse.item())
            if self._monitor_flag:
                for i, kl in enumerate(flow_kls):
                    bpps = kl.cpu().sum(dim=(2,3)).mean(0).div_(num_pix).mul_(self.log2_e)
                    self._bpp_stats[f'flow-bpp-z{i}'].update(bpps)
                for i, kl in enumerate(frame_kls):
                    bpps = kl.cpu().sum(dim=(2,3)).mean(0).div_(num_pix).mul_(self.log2_e)
                    self._bpp_stats[f'frame-bpp-z{i}'].update(bpps)

        context = {'x_hat': x_hat}
        return stats, context

    def save_progressive(self, im, im_prev, save_dir: Path, prefix=''):
        flow_stats, frame_stats, _, _ = self.forward_end2end(im, im_prev, get_intermediate=True)
        # save flow
        flows = [interpolate_and_scale(st['flow'], im.shape[2:4], mode='nearest') for st in flow_stats]
        to_save = torch.stack(flows, dim=0).transpose_(0,1).flatten(0,1)
        to_save = tv.utils.flow_to_image(to_save).float().div(255.0)
        tv.utils.save_image(to_save, fp=save_dir/f'{prefix}prog-flow.png', nrow=len(self.global_strides))
        # save warped
        warped = [forward_warp_torch(im_prev, f) for f in flows]
        to_save = torch.stack(warped, dim=0).transpose_(0,1).flatten(0,1)
        tv.utils.save_image(to_save, fp=save_dir/f'{prefix}warped.png', nrow=len(self.global_strides))

        # save progressive image
        # progressive = []
        # L = len(stats_all)
        # for keep in range(1, L+1):
        #     latents = [stat['z'] if (i < keep) else None for (i,stat) in enumerate(stats_all)]
        #     # kl_divs = [stat['kl'] for (i,stat) in enumerate(stats_all) if (i < keep)]
        #     # kl = sum([kl.sum(dim=(1,2,3)) for kl in kl_divs]) / (imH * imW) * math.log2(math.e)
        #     sample = self.conditional_sample(latents, context, t=0)
        #     progressive.append(sample)
        # num = len(progressive) # number of progressive coding results for one single image
        # progressive = torch.stack(progressive, dim=0).transpose_(0,1).flatten(0,1)
        # tv.utils.save_image(progressive, fp=save_path, nrow=num)


class VideoCodingModel(nn.Module):
    def __init__(self, lmb=2048):
        super().__init__()

        # ================================ i-frame model ================================
        from mycv.models.vae.qres.library import qres34m
        i_lambda = lmb // 8
        self.i_model = qres34m(lmb=i_lambda)
        # # fix i-model parameters
        for p in self.i_model.parameters():
            p.requires_grad_(False)
        # load pre-trained parameters
        from mycv.paths import MYCV_DIR
        wpath = MYCV_DIR / f'weights/qres34m/lmb{i_lambda}/last_ema.pt'
        self.i_model.load_state_dict(torch.load(wpath)['model'])

        # ================================ p-frame model ================================
        self.p_model = MyVideoModelSimple(lmb)

        # self.testing_gop = 32 # group of pictures at test-time
        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

        self._stop_monitoring()

    def _start_monitoring(self):
        self.p_model.start_monitoring()

    def _stop_monitoring(self):
        self.p_model.stop_monitoring()

    def _save_monitoring_results(self, log_dir):
        self.p_model.save_monitoring_results(log_dir)

    def train(self, mode=True):
        super().train(mode)
        self.i_model.eval() # the i-frame model is always in eval mode
        return self

    def forward(self, frames, log_dir=None):
        assert isinstance(frames, list)
        assert all([(im.shape == frames[0].shape) for im in frames])
        frames = [f.to(device=self._dummy.device) for f in frames]

        # initialize statistics for training and logging
        stats = OrderedDict()
        stats['loss'] = 0.0
        stats['bpp']  = None
        stats['psnr'] = None

        # ---------------- i-frame ----------------
        assert not self.i_model.training
        with torch.no_grad():
            _stats_i = self.i_model(frames[0], return_rec=True)
            prev_frame = _stats_i['im_hat']
            # logging
            stats['i-bpp']  = _stats_i['bppix']
            stats['i-psnr'] = _stats_i['psnr']

        # ---------------- p-frames ----------------
        p_stats_keys = ['loss', 'mv-bpp', 'warp-psnr', 'p-bpp', 'p-psnr']
        for key in p_stats_keys:
            stats[key] = 0.0
        p_frames = frames[1:]
        for i, frame in enumerate(p_frames):
            # conditional coding of current frame
            _stats_p, context_p = self.p_model(frame, prev_frame)

            # logging
            stats['loss'] = stats['loss'] + _stats_p['loss']
            stats['mv-bpp']    += float(_stats_p['mv-bpp'])
            stats['warp-psnr'] += float(_stats_p['warp-psnr'])
            stats['p-bpp']     += float(_stats_p['fr-bpp'])
            stats['p-psnr']    += float(_stats_p['psnr'])
            if (log_dir is not None): # save results
                log_dir = Path(log_dir)
                save_images(log_dir, f'prev_xhat_cur-{i}.png',
                    [prev_frame, context_p['x_hat'], frame]
                )
                self.p_model.save_progressive(frame, prev_frame, save_dir=log_dir, prefix=f'p{i}-')
            prev_frame = context_p['x_hat']

        # all frames statictics
        stats['bpp'] = (stats['i-bpp'] + stats['mv-bpp'] + stats['p-bpp']) / len(frames)
        stats['psnr'] = (stats['i-psnr'] + stats['p-psnr']) / len(frames)
        # average over p-frames only
        for key in p_stats_keys:
            stats[key] = stats[key] / len(p_frames)

        return stats

    @torch.no_grad()
    def forward_eval(self, frames):
        return self.forward(frames)

    @torch.no_grad()
    def study(self, log_dir, **kwargs):
        # read frames
        frame_paths = [
            'images/horse1.png',
            'images/horse2.png',
            'images/horse3.png',
        ]
        frames = [read_image(fp) for fp in frame_paths]

        # log_dir = Path(log_dir)
        self.forward(frames, log_dir=log_dir)

    @torch.no_grad()
    def self_evaluate(self, dataset, max_frames, log_dir=None):
        self._start_monitoring()
        results = mycpr.video_fast_evaluate(self, dataset, max_frames)
        if (log_dir is not None):
            self._save_monitoring_results(log_dir)
        self._stop_monitoring()
        return results


@register_model
def spy_coding(lmb=2048):
    model = VideoCodingModel(lmb=lmb)
    return model


@torch.no_grad()
def save_images(log_dir: Path, fname, list_of_tensors):
    if not log_dir.is_dir():
        log_dir.mkdir()
    list_of_tensors = [im.detach().cpu() for im in list_of_tensors]
    num = len(list_of_tensors)
    B, _, H, W = list_of_tensors[0].shape
    grid = torch.stack(list_of_tensors, dim=0).transpose_(0, 1)
    assert grid.shape == (B, num, 3, H, W)
    grid = grid.reshape(B*num, 3, H, W)
    tv.utils.save_image(grid, fp=log_dir/fname, nrow=num)

def read_image(impath):
    return tv.io.read_image(impath).float().div_(255.0).unsqueeze_(0)


def main():
    from mycv.datasets.compression import video_fast_evaluate
    from mycv.datasets.imgen import simple_evaluate
    device = torch.device('cuda:0')

    model = spy_coding()

    wpath = 'd:/projects/_logs/last_ema.pt'
    msd = torch.load(wpath)['model']
    model.load_state_dict(msd)

    model = model.to(device=device)
    model.eval()

    if True: # True, False
        model.study('runs/debug')
        exit()

    if True: # True, False
        results = model.self_evaluate('uvg-1080p', max_frames=12, log_dir='runs/debug')
        print(results)

    debug = 1


if __name__ == '__main__':
    main()
