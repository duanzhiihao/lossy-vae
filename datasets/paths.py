'''
This is the global settings of dataset paths.
'''
from pathlib import Path


# The root directory containing all datasets
DATASETS_DIR = Path('d:/datasets')

# ========================================================================
# datasets for image processing, e.g., coding, super resolution
# ========================================================================
# Kodak: http://www.cs.albany.edu/~xypan/research/snr/Kodak.html
KODAK_DIR = DATASETS_DIR / 'improcesing/kodak'
# CLIC dataset: http://www.compression.cc
CLIC_DIR = DATASETS_DIR / 'improcessing/clic'

# COCO dataset: http://cocodataset.org/#home
COCO_DIR = DATASETS_DIR / 'coco'

# ImageNet dataset: http://www.image-net.org
IMAGENET_DIR = DATASETS_DIR / 'imagenet'
