from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import torchvision as tv

from lvae.paths import known_datasets

__all__ = ['ImageDataset', 'get_image_dateset']


class ImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root # will be accessed by the training script
        self.transform = transform
        # scan and add images
        self.image_paths = sorted(Path(root).rglob('*.*'))
        assert len(self.image_paths) > 0, f'Found {len(self.image_paths)} images in {root}.'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        impath = self.image_paths[index]
        img = Image.open(impath).convert('RGB') 
        im = self.transform(img)
        return im


def get_image_dateset(name: str, transform_cfg: str=None) -> Dataset:
    """ get image dataset from name

    Args:
        name (str): dataset name, see functions above
        transform_cfg (str, optional): config, example: 'crop=256,hflip=True'
    """
    # make input transform
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

    # find dataset root, and initialize dataset
    dataset = ImageDataset(root=known_datasets.get(name, name), transform=transform)
    return dataset
