'''
This is the global settings of dataset paths.
'''
from pathlib import Path


# The root directory of all datasets
_root = Path('d:/datasets')

known_datasets = {
    # Kodak images: http://www.cs.albany.edu/~xypan/research/snr/Kodak.html
    'kodak': _root / 'improcessing/kodak',

    # CLIC dataset: http://www.compression.cc
    'clic2022-test': _root / 'improcessing/clic/test-2022',

    # COCO dataset: http://cocodataset.org
    'coco-train2017': _root / 'coco/train2017',

    # ImageNet dataset: http://www.image-net.org
    'imagenet-train': _root / 'imagenet/train',
    'imagenet-val':   _root / 'imagenet/val',

    # Vimeo-90k dataset: http://toflow.csail.mit.edu/
    'vimeo-90k': _root / 'vimeo-90k/sequences',

    # UVG dataset: http://ultravideo.fi/#testsequences
    'uvg-1080p': _root / 'video/uvg/1080p-frames'
}
