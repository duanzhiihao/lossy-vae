from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import math
import torch
import torchvision.transforms.functional as tvf
from timm.utils import AverageMeter

from lvae.paths import known_datasets


@torch.no_grad()
def imcoding_evaluate(model: torch.nn.Module, dataset: str):
    tmp_bit_path = Path('tmp.bits')
    assert hasattr(model, 'compress_file')
    assert hasattr(model, 'decompress_file')

    root = known_datasets.get(dataset, Path(dataset))
    img_paths = list(root.rglob('*.*'))
    img_paths.sort()
    pbar = tqdm(img_paths)
    all_image_stats = defaultdict(AverageMeter)
    for impath in pbar:
        model.compress_file(impath, tmp_bit_path)
        num_bits = tmp_bit_path.stat().st_size * 8
        fake = model.decompress_file(tmp_bit_path).squeeze(0).cpu()
        tmp_bit_path.unlink()

        # compute psnr
        real = tvf.to_tensor(Image.open(impath))
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
        for k,v in stats.items():
            all_image_stats[k].update(v)
        # logging
        msg = ', '.join([f'{k}={v:.3f}' for k,v in stats.items()])
        pbar.set_description(f'image {impath.stem}: {msg}')

    # average over all images
    results = {k: meter.avg for k,meter in all_image_stats.items()}
    return results
