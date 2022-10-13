from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import random
import logging
import torch
import torchvision as tv
import torchvision.transforms.functional as tvf
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .paths import DATASETS_DIR, IMAGENET_DIR, COCO_DIR, IMPROC_DIR


class ImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root # will be accessed by the training script
        self.img_paths = []
        self.transform = transform
        # scan and add images
        logging.info(f'Scanning through images in {root}...')
        pbar = Path(root).rglob('*.*')
        if logging.root.getEffectiveLevel() >= logging.INFO:
            pbar = tqdm(pbar)
        self.img_paths.extend(pbar)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        impath = self.img_paths[index]
        img = Image.open(impath).convert('RGB') 
        im = self.transform(img)
        return im


class FunctionRegistry(dict):
    def register(self, func):
        self[func.__name__] = func
        return func

_transforms = FunctionRegistry()

@_transforms.register
def pad8_affine_crop32():
    transforms = [
        tv.transforms.Pad(8, padding_mode='reflect'),
        # tv.transforms.RandomAffine(degrees=30, translate=(0.1,0.1), scale=(1.,1.25)),
        tv.transforms.RandomAffine(degrees=15, translate=(1/16,1/16), scale=(1.,1.2)),
        tv.transforms.CenterCrop(32),
    ]
    return transforms

class RandomDownScale():
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, img: Image.Image):
        shorter_side = min(img.height, img.width)
        if shorter_side <= self.min_size: # do not scale up
            return img
        new_size = random.randint(self.min_size, shorter_side)
        img = tvf.resize(img, size=new_size, interpolation=tvf.InterpolationMode.BICUBIC)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_size={self.min_size})"

@_transforms.register
def openimv6():
    transforms = [
        RandomDownScale(min_size=512),
        tv.transforms.RandomCrop(256, pad_if_needed=True, padding_mode='reflect'),
        tv.transforms.RandomHorizontalFlip(p=0.5)
    ]
    return transforms


_datasets = FunctionRegistry()

@_datasets.register
def cifar10train(transform):
    return tv.datasets.CIFAR10(DATASETS_DIR, train=True, transform=transform, download=True)

@_datasets.register
def cifar10test(transform):
    return tv.datasets.CIFAR10(DATASETS_DIR, train=False, transform=transform, download=True)

@_datasets.register
def imagenet64train(transform):
    return tv.datasets.ImageFolder(root=IMAGENET_DIR / 'train64', transform=transform)

@_datasets.register
def imagenet64val(transform):
    return tv.datasets.ImageFolder(root=IMAGENET_DIR / 'val64', transform=transform)

@_datasets.register
def vimeo90k(transform):
    return ImageDataset(root=DATASETS_DIR / 'vimeo-90k/sequences', transform=transform)

@_datasets.register
def coco2017train(transform):
    return ImageDataset(root=DATASETS_DIR / 'coco/train2017', transform=transform)

@_datasets.register
def openimv6train(transform):
    return ImageDataset(root=DATASETS_DIR / 'open-images-v6/train/data', transform=transform)

@_datasets.register
def kodak(transform):
    return ImageDataset(root=IMPROC_DIR / 'kodak', transform=transform)

def get_dateset(name: str, transform_cfg: str=None) -> Dataset:
    transform = []
    if transform_cfg is None:
        pass
    elif ('=' not in transform_cfg): # a pre-defined transform
        transform.extend(_transforms[transform_cfg]())
    else: # a dict
        transform_cfg: dict = eval(f'dict({transform_cfg})')
        if 'crop' in transform_cfg:
            t = tv.transforms.RandomCrop(transform_cfg['crop'], pad_if_needed=True, padding_mode='reflect')
            transform.append(t)
        if transform_cfg.get('hflip', False):
            t = tv.transforms.RandomHorizontalFlip(p=0.5)
            transform.append(t)
    transform.append(tv.transforms.ToTensor())
    transform = tv.transforms.Compose(transform)

    dataset = _datasets[name](transform=transform)
    return dataset

datasets_root = {
    'openim-v6-train':   DATASETS_DIR / 'open-images-v6/train/data',
    'celeba-trainval64': DATASETS_DIR / 'celeba/trainval64',
    'celeba-test64':     DATASETS_DIR / 'celeba/test64',
    'imagenet256train': IMAGENET_DIR / 'train256',
    'imagenet256val':   IMAGENET_DIR / 'val256',
    'imagenet128train': IMAGENET_DIR / 'train128',
    'imagenet128val':   IMAGENET_DIR / 'val128',
    # 'imagenet64train':  IMAGENET_DIR / 'train64',
    # 'imagenet64val':    IMAGENET_DIR / 'val64',
    'imagenet-val':     IMAGENET_DIR / 'val',
    'coco-train':   COCO_DIR / 'train2017',
    'coco-val-256': COCO_DIR / 'val2017_256',
    'clic-256':     IMPROC_DIR / 'clic/train_hw512s257',
    'div2k-256':    IMPROC_DIR / 'div2k/train_hr_hw512s257',
    'flickr2k-256': IMPROC_DIR / 'flickr2k/hr_hw512s257',
    'flickr2w':     IMPROC_DIR / 'flickr2w',
    'kodak':        IMPROC_DIR / 'kodak',
    'city-val':     DATASETS_DIR / 'cityscapes/leftImg8bit/val',
}


def get_dataloader(dataset_names, transform_cfg, batch_size, workers,
                   distributed=False, shuffle=True, drop_last=False):
    dataset = get_dateset(dataset_names, transform_cfg)
    sampler = DistributedSampler(dataset) if distributed else None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(False if distributed else shuffle),
        num_workers=workers, pin_memory=True, drop_last=drop_last, sampler=sampler
    )
    return dataloader


def get_train_generator(dataset_names, transform_cfg, batch_size, workers):
    dataset = get_dateset(dataset_names, transform_cfg)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=workers, pin_memory=True)
    while True:
        yield from dataloader
