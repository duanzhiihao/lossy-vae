from PIL import Image
from tqdm import tqdm
from pathlib import Path
from tempfile import gettempdir
from collections import defaultdict
import math
import torch
import torchvision.transforms.functional as tvf
from timm.utils import AverageMeter
from pytorch_msssim import ms_ssim

from lvae.paths import known_datasets
from lvae.utils.coding import crop_divisible_by


@torch.no_grad()
def imcoding_evaluate(model: torch.nn.Module, dataset: str):
    """ Evaluate image coding performance on a dataset, with entropy coding.

    Args:
        model (torch.nn.Module): pytorch model. \
            Need to have `compress_file` and `decompress_file` methods.
        dataset (str): dataset name or path to the dataset.

    Returns:
        dict[str -> float]: results, including bpp, mse, psnr.
    """
    assert hasattr(model, 'compress_file')
    assert hasattr(model, 'decompress_file')

    # find images
    root = known_datasets.get(dataset, Path(dataset))
    img_paths = list(root.rglob('*.*'))
    img_paths.sort()
    # temp folder to save bits
    tmp_bits_dir = Path(gettempdir())
    # start for-loop
    pbar = tqdm(img_paths, ascii=True)
    all_image_stats = defaultdict(AverageMeter)
    for impath in pbar:
        tmp_bits_path = tmp_bits_dir / f'{impath.stem}.bits'
        model.compress_file(impath, tmp_bits_path)
        num_bits = tmp_bits_path.stat().st_size * 8
        fake = model.decompress_file(tmp_bits_path).squeeze(0).cpu()
        tmp_bits_path.unlink()

        # compute psnr
        real = tvf.to_tensor(Image.open(impath))
        mse = (real - fake).square().mean().item()
        psnr = -10 * math.log10(mse)
        # compute ms-ssim
        mssm = ms_ssim(real.unsqueeze(0), fake.unsqueeze(0), data_range=1.0)
        # compute bpp
        bpp = num_bits / float(real.shape[1] * real.shape[2])
        stats = {
            'bpp':  float(bpp),
            'mse':  float(mse),
            'psnr': float(psnr),
            'ms-ssim': float(mssm),
        }

        # accumulate stats
        for k,v in stats.items():
            all_image_stats[k].update(v)
        # logging
        msg = ', '.join([f'{k}={v:.3f}' for k,v in stats.items()])
        pbar.set_description(f'image {impath.stem}: {msg}')

    # average over all images
    results = {k: meter.avg for k,meter in all_image_stats.items()}
    return results


@torch.no_grad()
def image_self_evaluate(model: torch.nn.Module, dataset: str, progress=True):
    """ Evaluate the model on a dataset with the model's `forward()` function.
    Typically, no entropy coding is used.

    Args:
        model (torch.nn.Module): pytorch model
        dataset (str): dataset name or path to the dataset.

    Returns:
        dict[str -> float]: results
    """
    device = next(model.parameters()).device
    # find images
    root = known_datasets.get(dataset, Path(dataset))
    img_paths = sorted(root.rglob('*.*'))
    # evaluate on all images and average the results
    pbar = tqdm(img_paths, ascii=True) if progress else img_paths
    all_image_stats = defaultdict(AverageMeter)
    for impath in pbar:
        img = Image.open(impath)
        if hasattr(model, 'max_stride'):
            img = crop_divisible_by(img, div=model.max_stride)
        im = tvf.to_tensor(img).unsqueeze_(0).to(device=device)
        stats = model(im)
        assert isinstance(stats, dict), f'{type(stats)=}. expected a dict.'

        # accumulate stats
        for k,v in stats.items():
            all_image_stats[k].update(v)
        # logging
        msg = ', '.join([f'{k}={v:.3f}' for k,v in stats.items()])
        if progress:
            pbar.set_description(f'image {impath.stem}: {msg}')

    # average over all images
    results = {k: meter.avg for k,meter in all_image_stats.items()}
    return results


@torch.no_grad()
def video_fast_evaluate(model: torch.nn.Module, dataset='uvg-1080p', max_frames=None):
    """ evaluate video compression performance (estimated, without actual entropy coding)

    Args:
        model (torch.nn.Module): pytorch model
        dataset (str): dataset name. Defaults to 'uvg-1080p'.
        max_frames (int): number of frames to evaluate. if None, evaluate all frames.
    """
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
