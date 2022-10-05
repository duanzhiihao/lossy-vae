from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import json
import math
import random
import itertools
import torch
import torchvision as tv
import torchvision.transforms.functional as tvf

from mycv.paths import MYCV_DIR, IMPROC_DIR, DATASETS_DIR
from mycv.utils.coding import bd_rate, crop_divisible_by
from timm.utils import AverageMeter


known_datasets = {
    'kodak': IMPROC_DIR / 'kodak',
    'clic2022-test': IMPROC_DIR / 'clic/test-2022',
    'uvg-1080p': DATASETS_DIR / 'video/uvg/1080p-frames'
}

anchor_stats = None


def _get_stats(json_path):
    with open(json_path, mode='r') as f:
        stats = json.load(fp=f)
    if 'results' in stats:
        stats = stats['results']
    return stats

def _set_anchor_stats():
    global anchor_stats
    anchor_stats = dict()
    anchor_stats['vtm-12.1'] = {
        'kodak':         _get_stats(json_path=MYCV_DIR / '../data/kodak-vtm-12.1-compressai.json'),
        'clic2022-test': _get_stats(json_path=MYCV_DIR / '../data/clic2022-test-vtm-12.1.json')
    }

def get_anchor_stats(dataset_name, anchor_name='vtm-12.1'):
    if anchor_stats is None:
        _set_anchor_stats()
    stats = anchor_stats[anchor_name][dataset_name]
    stats: dict
    return stats

def get_bd_rate_over_anchor(stats, dataset_name, anchor_name='vtm-12.1'):
    anchor_stats = get_anchor_stats(dataset_name, anchor_name)
    bdr = bd_rate(anchor_stats['bpp'], anchor_stats['psnr'], stats['bpp'], stats['psnr'])
    return bdr


@torch.no_grad()
def imcoding_evaluate(model: torch.nn.Module, dataset: str):
    tmp_bit_path = Path('tmp.bits')
    assert hasattr(model, 'compress_file')
    assert hasattr(model, 'decompress_file')

    root = known_datasets.get(dataset, Path(dataset))
    img_paths = list(root.rglob('*.*'))
    img_paths.sort()
    pbar = tqdm(img_paths)
    accumulated_stats = defaultdict(float)
    for impath in pbar:
        model.compress_file(impath, tmp_bit_path)
        num_bits = tmp_bit_path.stat().st_size * 8
        # model.decompress_file(tmp_bit_path, tmp_rec_path)
        fake = model.decompress_file(tmp_bit_path).squeeze(0).cpu()
        tmp_bit_path.unlink()

        # compute psnr
        real = tvf.to_tensor(Image.open(impath))
        # fake = tvf.to_tensor(Image.open(tmp_rec_path))
        mse = (real - fake).square().mean().item()
        psnr = -10 * math.log10(mse)
        # compute bpp
        bpp = num_bits / float(real.shape[1] * real.shape[2])
        stats = {
            'bpp':  float(bpp),
            'mse':  float(mse),
            'psnr': float(psnr)
        }

        # accumulate stats
        accumulated_stats['count'] += 1.0
        for k,v in stats.items():
            accumulated_stats[k] += v
        # logging
        msg = ', '.join([f'{k}={v:.3f}' for k,v in stats.items()])
        pbar.set_description(f'image {impath.stem}: {msg}')

    # average over all images
    count = accumulated_stats.pop('count')
    results = {k: v/count for k,v in accumulated_stats.items()}
    return results


class Vimeo90k(torch.utils.data.Dataset):
    def __init__(self, n_frames=3):
        self.root = DATASETS_DIR / 'vimeo-90k/sequences'
        self.sequence_dirs = list(tqdm(itertools.chain(*[d.iterdir() for d in self.root.iterdir()])))
        self.sequence_dirs.sort()

        self.transform = tv.transforms.Compose([
            tv.transforms.RandomCrop(256),
            tv.transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.n_frames = n_frames

    def __len__(self):
        return len(self.sequence_dirs)

    def __getitem__(self, index):
        sequence_dir = self.sequence_dirs[index]
        frame_paths = sorted(sequence_dir.rglob('*.*'))
        N = len(frame_paths)
        assert N == 7 # sanity check
        # randomly choose a subset of frames
        satrt_idx = random.randint(0, N - self.n_frames)
        frame_paths = frame_paths[satrt_idx:satrt_idx+self.n_frames]
        if random.random() < 0.5: # randomly reverse time
            frame_paths = frame_paths[::-1]

        frames = [tvf.to_tensor(Image.open(fp)) for fp in frame_paths]
        frames = self.transform(torch.stack(frames, dim=0))
        frames = torch.chunk(frames, chunks=self.n_frames, dim=0)
        frames = [f.squeeze_(0) for f in frames]

        return frames


@torch.no_grad()
def video_fast_evaluate(model: torch.nn.Module, dataset='uvg-1080p', max_frames=None):
    root = known_datasets.get(dataset, Path(dataset))
    assert root.is_dir(), f'cannot find {root} as a directory'
    sequence_paths = list(root.iterdir())

    pbar = tqdm(sequence_paths, position=0, ascii=True)
    accumulated_stats = defaultdict(float)
    for seq_path in pbar:
        # get all frame paths in the sequence folder
        frame_paths = sorted(seq_path.rglob('*.*'))
        # select only the first `max_frames` frames for fast evaluation
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]

        frames = []
        for fp in frame_paths:
            img = crop_divisible_by(Image.open(fp), div=64)
            frames.append(tvf.to_tensor(img).unsqueeze_(0))

        stats = model.forward_eval(frames)

        # accumulate stats
        accumulated_stats['count'] += 1.0
        for k,v in stats.items():
            accumulated_stats[k] += v
        # logging
        msg = ', '.join([f'{k}={v:.3f}' for k,v in stats.items()])
        pbar.set_description(f'sequence {seq_path.stem}: {msg}')

    # average over all images
    count = accumulated_stats.pop('count')
    results = {k: v/count for k,v in accumulated_stats.items()}
    return results


def _read_frame(fpath):
    img = crop_divisible_by(Image.open(fpath), div=64)
    return tvf.to_tensor(img).unsqueeze_(0)

@torch.no_grad()
def video_simple_evaluate(model: torch.nn.Module, dataset='uvg-1080p', gop=12, max_frames=None):
    root = known_datasets.get(dataset, Path(dataset))
    assert root.is_dir(), f'cannot find {root} as a directory'
    sequence_paths = sorted(root.iterdir())

    pbar = tqdm(sequence_paths, position=0, ascii=True)
    all_sequence_stats = defaultdict(AverageMeter)
    for seq_path in pbar:
        # get all frame paths in the sequence folder
        frame_paths = sorted(seq_path.rglob('*.*'))
        # select only the first `max_frames` frames for fast evaluation
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]

        seq_stats = defaultdict(AverageMeter)
        # split frames into groups (gop = group size)
        assert len(frame_paths) % gop == 0
        for i in range(len(frame_paths) // gop):
            # read from i*gop to (i+1)*gop
            frames = [_read_frame(fp) for fp in frame_paths[i*gop : (i+1)*gop]]
            assert len(frames) <= gop

            stats = model.forward_eval(frames)
            for k,v in stats.items():
                seq_stats[k].update(float(v), n=len(frames))

        # accumulate stats
        for k,meter in seq_stats.items():
            all_sequence_stats[k].update(meter.avg)
        # logging
        msg = ', '.join([f'{k}={meter.avg:.3f}' for k,meter in seq_stats.items()])
        pbar.set_description(f'sequence {seq_path.stem}: {msg}')
    # pbar.close()

    # average over all images
    results = {k: meter.avg for k,meter in all_sequence_stats.items()}
    return results
