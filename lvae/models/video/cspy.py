from pathlib import Path
from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torchvision as tv

from lvae.models.registry import register_model
from lvae.evaluation.video import video_fast_evaluate
import lvae.models.common as common
import lvae.models.video.video_model as qrvm


backward_grid = [dict() for _ in range(9)] # 0~7 for GPU, -1 for CPU

def bilinear_warp(x: torch.Tensor, flow: torch.Tensor):
    """ `flow` is the "backward" flow.

    Args:
        x (torch.Tensor): previous frame
        flow (torch.Tensor): `position_x - position_future`
    """
    device_id = -1 if (x.device == torch.device('cpu')) else x.device.index
    shape_key = str(flow.shape)
    if shape_key not in backward_grid[device_id]:
        N, _, H, W = flow.shape
        tensor_hori = torch.linspace(-1.0, 1.0, W, device=x.device, dtype=x.dtype)
        tensor_vert = torch.linspace(-1.0, 1.0, H, device=x.device, dtype=x.dtype)
        tensor_hori = tensor_hori.view(1, 1, 1, W).expand(N, 1, H, -1)
        tensor_vert = tensor_vert.view(1, 1, H, 1).expand(N, 1, -1, W)
        backward_grid[device_id][shape_key] = torch.cat([tensor_hori, tensor_vert], dim=1)

    flow = torch.cat([
        flow[:, 0:1, :, :] / (x.shape[3] - 1.0) * 2.0,
        flow[:, 1:2, :, :] / (x.shape[2] - 1.0) * 2.0
    ], dim=1)

    grid = (backward_grid[device_id][shape_key] + flow)
    warped = tnf.grid_sample(
        input=x,
        grid=grid.permute(0, 2, 3, 1),
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return warped


def myconvnext_down(dim, new_dim, kernel_size=7):
    module = nn.Sequential(
        common.MyConvNeXtBlock(dim, kernel_size=kernel_size),
        common.conv_k3s2(dim, new_dim),
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
        enc_blocks = [common.conv_k3s1(im_channels, 32),]
        for i, (dim, ks, num) in enumerate(zip(enc_dims, kernel_sizes, enc_nums,)):
            for _ in range(num):
                enc_blocks.append(common.MyConvNeXtBlock(dim, kernel_size=ks))
            if i < len(enc_dims) - 1:
                new_dim = enc_dims[i+1]
                # enc_blocks.append(common.MyConvNeXtPatchDown(dim, new_dim, kernel_size=ks))
                enc_blocks.append(myconvnext_down(dim, new_dim, kernel_size=ks))
        self.feature_extractor = common.BottomUpEncoder(enc_blocks, dict_key='stride')

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
                conv = common.conv_k3s1(dim, im_channels)
            else:
                conv = common.patch_upsample(dim, dec_dims[i-1], rate=2)
            self.upsamples[f'stride{s}'] = conv

        self.global_strides = global_strides
        self.max_stride = max(global_strides)

        self.distortion_lmb = float(distortion_lmb)

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
                f_warped = bilinear_warp(f_prev, flow)
            # TODO: use neighbourhood attention to implement flow estimation
            blk_key = f'stride{s}' # block key
            if s in self.strides_that_have_bits:
                flow, stats = self.flow_blocks[blk_key](flow, f_warped, f_curr)
                # warp t-1 feature again
                f_warped = bilinear_warp(f_prev, flow)
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
            warp_mse = tnf.mse_loss(bilinear_warp(im_prev, flow_hat), im)
            stats['warp-psnr'] = -10 * math.log10(warp_mse.item())

        context = {'x_hat': x_hat}
        return stats, context


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
        frame_paths = [
            'images/horse1.png',
            'images/horse2.png',
            'images/horse3.png',
        ]
        frames = [read_image(fp) for fp in frame_paths]
        self.forward(frames, log_dir=log_dir)

    @torch.no_grad()
    def self_evaluate(self, dataset, max_frames, log_dir=None):
        results = video_fast_evaluate(self, dataset, max_frames)
        return results


@register_model
def cspy(lmb=2048):
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

    model = cspy()

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
