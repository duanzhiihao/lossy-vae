from PIL import Image
from tqdm import tqdm
from pathlib import Path
import logging
import torchvision as tv
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from lvae.paths import known_datasets


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


_datasets = FunctionRegistry()

@_datasets.register
def coco_train2017(transform):
    return ImageDataset(root=known_datasets['coco-train2017'], transform=transform)

@_datasets.register
def vimeo90k(transform):
    return ImageDataset(root=known_datasets['vimeo-90k'], transform=transform)


def get_dateset(name: str, transform_cfg: str=None) -> Dataset:
    """ get image dataset from name

    Args:
        name (str): dataset name, see functions above
        transform_cfg (str, optional): config, example: 'crop=256,hflip=True'
    """
    transform = []
    if transform_cfg is not None:
        transform_cfg = eval(f'dict({transform_cfg})')
        assert isinstance(transform_cfg, dict)
        if 'crop' in transform_cfg:
            t = tv.transforms.RandomCrop(transform_cfg['crop'], pad_if_needed=True, padding_mode='reflect')
            transform.append(t)
        if transform_cfg.get('hflip', False):
            t = tv.transforms.RandomHorizontalFlip(p=0.5)
            transform.append(t)
    transform.append(tv.transforms.ToTensor())
    transform = tv.transforms.Compose(transform)

    dataset = _datasets[name](transform=transform)
    assert isinstance(dataset, Dataset)
    return dataset


def make_generator(dataset, batch_size, workers):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=workers, pin_memory=True)
    while True:
        yield from dataloader
