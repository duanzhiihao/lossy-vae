from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import torch
import torchvision.transforms.functional as tvf

from lvae.paths import known_datasets
from lvae.utils.coding import crop_divisible_by


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
