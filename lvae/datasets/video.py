from PIL import Image
from tqdm import tqdm
import random
import itertools
import torch
import torchvision as tv
import torchvision.transforms.functional as tvf

from lvae.paths import known_datasets


class Vimeo90k(torch.utils.data.Dataset):
    def __init__(self, n_frames=3):
        self.root = known_datasets['vimeo-90k']
        self.sequence_dirs = list(tqdm(itertools.chain(*[d.iterdir() for d in self.root.iterdir()])))
        self.sequence_dirs.sort()

        self.transform = tv.transforms.Compose([
            tv.transforms.RandomCrop(256),
            tv.transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.n_frames = n_frames

    def __len__(self):
        return len(self.sequence_dirs)

    def __getitem__(self, index):
        sequence_dir = self.sequence_dirs[index]
        frame_paths = sorted(sequence_dir.rglob('*.*'))
        N = len(frame_paths)
        assert N == 7 # sanity check
        # randomly choose a subset of frames
        satrt_idx = random.randint(0, N - self.n_frames)
        frame_paths = frame_paths[satrt_idx:satrt_idx+self.n_frames]
        if random.random() < 0.5: # randomly reverse time
            frame_paths = frame_paths[::-1]

        frames = [tvf.to_tensor(Image.open(fp)) for fp in frame_paths]
        frames = self.transform(torch.stack(frames, dim=0))
        frames = torch.chunk(frames, chunks=self.n_frames, dim=0)
        frames = [f.squeeze_(0) for f in frames]

        return frames
