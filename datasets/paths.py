'''
This is the global settings of dataset paths.
'''
from pathlib import Path


# Project root dir
MYCV_DIR = Path(__file__).parent

# The directory of all datasets
DATASETS_DIR = Path('d:/datasets')
assert DATASETS_DIR.is_dir(), f'{DATASETS_DIR} should exist.'

# datasets for image processing, e.g., coding, super resolution
# CLIC dataset: http://www.compression.cc
# Kodak: http://www.cs.albany.edu/~xypan/research/snr/Kodak.html
# Flickr2K (non-published): https://github.com/limbee/NTIRE2017/issues/25
# DIV2K dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K
IMPROC_DIR = DATASETS_DIR / 'improcessing'

# ImageNet dataset: http://www.image-net.org
IMAGENET_DIR = DATASETS_DIR / 'imagenet'

# Cityscapes dataset: https://www.cityscapes-dataset.com
CITYSCAPES_DIR = DATASETS_DIR / 'cityscapes'

# COCO dataset: http://cocodataset.org/#home
COCO_DIR = DATASETS_DIR / 'coco'

# MW-R, HABBOF, and CEPDOF dataset: http://vip.bu.edu/projects/vsns/cossy/datasets
COSSY_DIR = DATASETS_DIR / 'cossy'

# UCF-101 dataset: https://www.crcv.ucf.edu/data/UCF101.php
UCF101_DIR = DATASETS_DIR / 'ucf-101'

# DAVIS 2017 dataset: https://davischallenge.org/davis2017/code.html
DAVIS_DIR = DATASETS_DIR / 'davis'

# MPI Sintel dataset: http://sintel.is.tue.mpg.de
SINTEL_DIR = DATASETS_DIR / 'sintel'
