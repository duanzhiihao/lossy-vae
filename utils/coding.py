import os
import sys
import json
import math
import pickle
import numpy as np
import cv2
import torch
import torch.nn.functional as tnf
import torchvision.transforms.functional as tvf


def get_object_size(obj, unit='bits'):
    assert unit == 'bits'
    return sys.getsizeof(pickle.dumps(obj)) * 8


def pad_divisible_by(img, div=64):
    """ Pad a PIL.Image such that both its sides are divisible by `div`

    Args:
        img (PIL.Image): input image
        div (int, optional): denominator. Defaults to 64.

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


def crop_divisible_by(img, div=64):
    ''' Center crop a PIL.Image such that both its sides are divisible by `div`

    Args:
        img (PIL.Image): input image
        div (int, optional): denominator. Defaults to 64.

    Returns:
        PIL.Image: cropped image
    '''
    h_old, w_old = img.height, img.width
    if (h_old % div == 0) and (w_old % div == 0):
        return img
    h_new = div * (h_old // div)
    w_new = div * (w_old // div)
    cropped = tvf.center_crop(img, output_size=(h_new, w_new))
    return cropped


def compute_bpp(prob: torch.Tensor, num_pixels: int, batch_reduction='mean'):
    """ bits per pixel

    Args:
        prob (torch.Tensor): probabilities
        num_pixels (int): number of pixels
    """
    assert isinstance(prob, torch.Tensor) and prob.dim() == 4
    p_min = prob.detach().min()
    if p_min < 0: 
        print(f'Error: prob: {prob.shape}, min={p_min} is less than 0.')
    elif p_min == 0:
        print(f'Warning: prob: {prob.shape}, min={p_min} equals to 0.')
    elif torch.isnan(p_min):
        num = torch.isnan(prob).sum()
        print(f'Error: prob: {prob.shape}, {num} of it is nan.')

    nB = prob.shape[0]
    bpp = - torch.log2(prob).sum() / num_pixels
    if batch_reduction == 'mean':
        bpp = bpp / nB
    elif batch_reduction == 'sum':
        pass
    else:
        raise ValueError()
    return bpp


def _gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)]
    )
    return gauss/gauss.sum()


def _create_window(window_size, sigma, channel):
    _1D_window = _gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class MS_SSIM():
    ''' Adapted from https://github.com/lizhengwei1992/MS_SSIM_pytorch
    '''
    def __init__(self, max_val=1.0, reduction='mean'):
        # super().__init__()
        self.channel = 3
        self.max_val = max_val
        self.weight = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        assert reduction in {'mean'}, 'Invalid reduction'
        self.reduction = reduction

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor):
        _, c, w, h = img1.size()
        window_size = min(w, h, 11)
        sigma = 1.5 * window_size / 11

        window = _create_window(window_size, sigma, self.channel)
        window = window.to(device=img1.device, dtype=img1.dtype)

        mu1 = tnf.conv2d(img1, window, padding=window_size //
                       2, groups=self.channel)
        mu2 = tnf.conv2d(img2, window, padding=window_size //
                       2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = tnf.conv2d(
            img1*img1, window, padding=window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = tnf.conv2d(
            img2*img2, window, padding=window_size//2, groups=self.channel) - mu2_sq
        sigma12 = tnf.conv2d(img1*img2, window, padding=window_size //
                           2, groups=self.channel) - mu1_mu2

        C1 = (0.01*self.max_val)**2
        C2 = (0.03*self.max_val)**2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2*mu1_mu2 + C1)*V1)/((mu1_sq + mu2_sq + C1)*V2)
        mcs_map = V1 / V2
        if self.reduction == 'mean':
            ssim_map = ssim_map.mean()
            mcs_map = mcs_map.mean()
        return ssim_map, mcs_map

    def __call__(self, img1, img2):
        assert img1.shape == img2.shape and img1.device == img2.device
        self.weight = self.weight.to(device=img1.device)
        levels = 5

        if min(img1.shape[2:4]) < 2**levels:
            return torch.zeros(1)

        msssim = []
        mcs = []
        for i in range(levels):
            ssim_map, mcs_map = self._ssim(img1, img2)
            msssim.append(ssim_map)
            mcs.append(mcs_map)
            filtered_im1 = tnf.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = tnf.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2
        msssim = torch.stack(msssim)
        mcs = torch.stack(mcs)
        value = torch.prod(mcs[0:levels-1] ** self.weight[0:levels-1]) \
                * (msssim[levels-1] ** self.weight[levels-1])
        value: torch.Tensor
        return value

def compute_msssim(im1, im2):
    ms_ssim_func = MS_SSIM()
    return ms_ssim_func(im1, im2)


def bd_rate(r1, psnr1, r2, psnr2):
    """ Compute average bit rate saving of RD-2 over RD-1.

    Equivalent to the implementations in:
    https://github.com/Anserw/Bjontegaard_metric/blob/master/bjontegaard_metric.py
    https://github.com/google/compare-codecs/blob/master/lib/visual_metrics.py

    args:
        r1    (list, np.ndarray): baseline rate
        psnr1 (list, np.ndarray): baseline psnr
        r2    (list, np.ndarray): rate 2
        psnr2 (list, np.ndarray): psnr 2
    """
    lr1 = np.log(r1)
    lr2 = np.log(r2)

    # fit each curve by a polynomial
    degree = 3
    p1 = np.polyfit(psnr1, lr1, deg=degree)
    p2 = np.polyfit(psnr2, lr2, deg=degree)
    # compute integral of the polynomial
    p_int1 = np.polyint(p1)
    p_int2 = np.polyint(p2)
    # area under the curve = integral(max) - integral(min)
    min_psnr = max(min(psnr1), min(psnr2))
    max_psnr = min(max(psnr1), max(psnr2))
    auc1 = np.polyval(p_int1, max_psnr) - np.polyval(p_int1, min_psnr)
    auc2 = np.polyval(p_int2, max_psnr) - np.polyval(p_int2, min_psnr)

    # find avgerage difference
    avg_exp_diff = (auc2 - auc1) / (max_psnr - min_psnr)
    avg_diff = (np.exp(avg_exp_diff) - 1) * 100

    if False: # debug
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        l1 = plt.plot(psnr1, lr1, label='PSNR-logBPP 1',
                      marker='.', markersize=12, linestyle='None')
        l2 = plt.plot(psnr2, lr2, label='PSNR-logBPP 2',
                      marker='.', markersize=12, linestyle='None')
        x = np.linspace(min_psnr, max_psnr, num=100)
        # x = np.linspace(min_psnr-10, max_psnr+10, num=100)
        plt.plot(x, np.polyval(p1, x), label='polyfit 1',
                 linestyle='-', color=l1[0].get_color())
        plt.plot(x, np.polyval(p2, x), label='polyfit 2',
                 linestyle='-', color=l2[0].get_color())
        plt.legend(loc='lower right')
        plt.xlim(np.concatenate([psnr1,psnr2]).min()-1, np.concatenate([psnr1,psnr2]).max()+1)
        plt.ylim(np.concatenate([lr1, lr2]).min()-0.1, np.concatenate([lr1, lr2]).max()+0.1)
        # plt.ylim(np.concatenate(lr1, lr2).min(), max(np.concatenate(lr1, lr2)))
        plt.show()
    return avg_diff


class RDList():
    def __init__(self) -> None:
        self.stats_all = []
        self.bdrate_anchor = None

    def add_json(self, fpath, label='no label', **kwargs):
        with open(fpath, mode='r') as f:
            stat = json.load(f)
        if 'results' in stat:
            stat = stat['results']
        stat['label'] = label
        stat['kwargs'] = kwargs
        self.stats_all.append(stat)

    def set_video_info(self, fps, height, width):
        # self.video_info = (fps, height, width)
        self.norm_factor = float(fps * height * width)

    def add_json_bpms(self, fpath, label='no label', **kwargs):
        with open(fpath, mode='r') as f:
            stat = json.load(f)
        if 'results' in stat:
            stat = stat['results']
        # convert bits per second (bps) to bits per pixel (bpp)
        # fps, fh, fw = self.video_info
        # stat['bpp'] = stat['bps'] / (fps * fh * fw)
        stat['bpp'] = [r * 1000 / self.norm_factor for r in stat['bitrate']]
        stat['psnr'] = stat['psnr-rgb']
        stat['label'] = label
        stat['kwargs'] = kwargs
        self.stats_all.append(stat)

    def add_data(self, bpp=[], psnr=[], label='no label', **kwargs):
        stat = {
            'bpp': bpp,
            'psnr': psnr,
            'label': label
        }
        stat['kwargs'] = kwargs
        self.stats_all.append(stat)

    def set_bdrate_anchor(self, label=None):
        if label is None:
            anchor = self.stats_all[-1]
        else:
            anchor = [st for st in self.stats_all if (st['label'] == label)]
            assert len(anchor) == 1
            anchor = anchor[0]
        self.bdrate_anchor = anchor

    def compute_bdrate(self):
        if self.bdrate_anchor is None:
            return
        bd_anchor = self.bdrate_anchor
        print(f'BD-rate anchor = {bd_anchor["label"]}')
        for method in self.stats_all:
            if len(method['bpp']) == 0:
                continue
            bd = bd_rate(bd_anchor['bpp'], bd_anchor['psnr'],
                         method['bpp'], method['psnr'])
            print(method['label'], f'BD-rate = {bd}')
        print()

    def plot_all_stats(self, ax):
        for stat in self.stats_all:
            self._plot_stat(stat, ax=ax, **stat['kwargs'])

    @staticmethod
    def _plot_stat(stat, ax, ls='-', **kwargs):
        assert 'bpp' in stat, f'{stat}'
        x = stat['bpp']
        y = stat['psnr']
        label = stat['label']
        kwargs['marker'] = kwargs.get('marker', '.')
        kwargs['linewidth'] = kwargs.get('linewidth', 1.2)
        p = ax.plot(x, y, label=label, markersize=8, linestyle=ls, **kwargs)
        return p


@torch.no_grad()
def get_important_channels(model, topk=16):
    from mycv.paths import IMPROC_DIR
    device = next(model.parameters()).device
    img_dir = IMPROC_DIR / 'kodak'
    img_names = os.listdir(img_dir)
    # img_names = img_names[:1]

    entropy_sum = None
    value_max = None
    for imname in img_names:
        impath = img_dir / imname
        im = cv2.imread(str(impath))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        input_ = torch.from_numpy(im).permute(2,0,1).float() / 255.0
        input_ = input_.unsqueeze(0)
        input_ = input_.to(device=device)
        ih, iw = input_.shape[2:4]

        # encode
        outputs = model(input_)
        likelihoods = outputs['likelihoods']
        p_main = likelihoods['y']
        z = model.g_a(input_)
        zhat = torch.round(z)
        # rec = rec.clamp_(min=0, max=1)
        # rec = (rec.squeeze(0).cpu().permute(1,2,0) * 255).to(dtype=torch.uint8).numpy()
        # plt.imshow(rec); plt.show()

        # entropys
        assert p_main.shape[2:4] == (ih//16, iw//16)
        Hs = -torch.log2(p_main)
        Hs = Hs.sum(dim=3).sum(dim=2).squeeze(0).cpu()
        # plt.figure(); plt.bar(np.arange(len(Hs)), Hs)
        # plt.figure(); plt.bar(np.arange(len(Es)), Es); plt.show()

        # max values within channels
        assert zhat.shape[2:4] == (ih//16, iw//16)
        zmax = torch.amax(zhat, dim=(2,3)).squeeze(0).cpu()

        entropy_sum = entropy_sum + Hs if entropy_sum is not None else Hs
        value_max = torch.maximum(value_max, zmax) if value_max is not None else zmax

    entropy_mean = entropy_sum / len(img_names)
    # plt.bar(np.arange(len(entropy_mean)), entropy_mean); plt.show()
    entropys, indexs = torch.sort(entropy_mean, descending=True)
    indexs = indexs[:topk]
    debug = 1
    # plt.bar(np.arange(len(entropys)), entropys)
    # plt.xlabel('Rank')
    # plt.ylabel('Entropy')
    # plt.show()
    # exit()
    return indexs, value_max, zhat.shape[1]


@torch.no_grad()
def plot_response(model: torch.nn.Module, topk, save_path):
    assert not model.training
    device = next(model.parameters()).device

    if hasattr(model, 'decoder'):
        decode_func = model.decoder
    elif hasattr(model, 'g_s'):
        decode_func = model.g_s
    else:
        raise NotImplementedError()

    channel_indices, value_max, nC = get_important_channels(model, topk)

    images = []
    for k, chi in enumerate(channel_indices):
        # print(f'running channel {chi}')

        x = torch.zeros(1, nC, 3, 3, device=device)
        a = value_max[chi] * 1.2
        x[0, chi, 1, 1] = a
        rec1 = decode_func(x)
        # x[0, chi, 1, 1] = -a
        # rec2 = decode_func(x)
        # rec = torch.cat([rec1, rec2], dim=2)
        rec = rec1
        rec = rec.clamp_(min=0, max=1)
        rec = (rec.squeeze(0).cpu().permute(1,2,0) * 255).to(dtype=torch.uint8).numpy()
        # assert rec.shape == (48,48,3)
        h,w = rec.shape[0:2]
        # rec[h//2,:,:] = 255
        # resize images
        rec = cv2.resize(rec, (w*2,h*2), interpolation=cv2.INTER_NEAREST)
        rec = rec.copy()
        # label the image
        cv2.putText(rec, f'{k+1}', (4,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        # cv2.putText(rec, f'Rank {k}, channel {chi}', (0,16),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        # plt.imshow(rec); plt.show()
        images.append(rec)

    if len(images) == 8:
        h,w = images[0].shape[0:2]
        assert h == w
        border = (np.ones((h//16, w, 3)) * 255).astype(np.uint8)
        column1 = _insert_between(images[:4], border)
        column1 = np.concatenate(column1, axis=0)
        column2 = _insert_between(images[4:], border)
        column2 = np.concatenate(column2, axis=0)
        ch = column1.shape[0]
        middle = (np.ones((ch, w//16, 3)) * 255).astype(np.uint8)
        img = np.concatenate([column1, middle, column2], axis=1)
        # plt.imshow(img); plt.show()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), img)
    else:
        # add border
        border = (np.ones((h*2, w//8, 3)) * 255).astype(np.uint8)
        to_concat = _insert_between(images, border)
        # concat and save
        img = np.concatenate(to_concat, axis=1)
        # plt.imshow(img); plt.show()
        # convert color and save
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), img)

def _insert_between(l, item):
    new = [[element, item] for element in l]
    new = sum(new, [])[:-1]
    return new


def main():
    psnr = [23.115, 24.884, 26.792, 28.755, 30.811, 32.921, 35.151, 37.481, 40.061]
    bpp = [0.1636, 0.2469, 0.3631, 0.5205, 0.7203, 0.9811, 1.316, 1.705, 2.207]
    bpp2 = [0.01] * len(bpp)
    print(bd_rate(bpp, psnr, bpp2, psnr))
    debug = 1

if __name__ == '__main__':
    main()
