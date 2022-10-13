'''
This is the global settings of dataset paths.
'''
from pathlib import Path


# The root directory of all datasets
_root = Path('d:/datasets')

known_datasets = {
    # Kodak: http://www.cs.albany.edu/~xypan/research/snr/Kodak.html
    'kodak': _root / 'improcesing/kodak',

    # CLIC dataset: http://www.compression.cc
    'clic2022-test': _root / 'improcessing/clic/test-2022',

    # COCO dataset: http://cocodataset.org
    'coco-train2017': _root / 'train2017',

    # ImageNet dataset: http://www.image-net.org
    'imagenet-train': _root / 'imagenet/train',
    'imagenet-val':   _root / 'imagenet/val',
}