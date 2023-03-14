import os
from torch.utils.data import DataLoader, DistributedSampler

__all__ = ['make_trainloader']


def _make_generator(dataloader: DataLoader):
    while True:
        yield from dataloader


def make_trainloader(dataset, batch_size: int, workers: int):
    """ Create training data loader.
    Note: in DDP mode, need to call `sampler.set_epoch(epoch)` before each epoch/iteration.

    Args:
        dataset (torch.utils.data.Dataset): PyTorch dataset
        batch_size (int): batch size on each GPU
        workers (int): number of CPU workers
    """
    world_size  = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1: # PyTorch DDP training
        sampler = DistributedSampler(dataset)
    else: # Single GPU training
        assert world_size == 1, f'Invalid {world_size=}'
        sampler = None
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(sampler is None), drop_last=True,
        num_workers=workers, pin_memory=True, sampler=sampler
    )
    generater = _make_generator(dataloader)
    return generater, sampler
