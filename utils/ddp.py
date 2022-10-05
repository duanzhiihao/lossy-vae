import os
import logging
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


class run_zero_first():
    """ Run on rank 0 first, followed by on other ranks.
    """
    def __init__(self):
        self.local_rank = int(os.environ.get('LOCAL_RANK', -1))

    def __enter__(self):
        if self.local_rank > 0:
            assert dist.is_initialized()
            dist.barrier(device_ids=[self.local_rank])

    def __exit__(self, type, value, traceback):
        _dist_debug(type, value, traceback)
        if self.local_rank == 0:
            dist.barrier(device_ids=[self.local_rank])
        return True


class run_sequentially():
    """ Run on rank 0, 1, 2, ... sequentially
    """
    def __init__(self):
        assert dist.is_initialized()
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        assert self.local_rank == dist.get_rank()
        assert self.world_size == dist.get_world_size()
        self.dummy = torch.ones(1, device=f'cuda:{self.local_rank}')

    def __enter__(self):
        if self.local_rank > 0:
            dist.recv(self.dummy, src=self.local_rank-1)
        # return self

    def __exit__(self, type, value, traceback):
        _dist_debug(type, value, traceback)
        if self.local_rank < self.world_size-1:
            dist.send(self.dummy, dst=self.local_rank+1)
        return True


def broadcast_object(obj, src=0, local_rank=None):
    if local_rank is None:
        local_rank = int(os.environ['LOCAL_RANK'])
    mail_box = [obj] if (local_rank == src) else [None]
    dist.broadcast_object_list(mail_box, src=src, device=torch.device(f'cuda:{local_rank}'))
    received = mail_box[0]
    if local_rank == src: # sanity check
        assert obj == received, f'local rank={local_rank}, old={obj}, new={received}'
    return received


@torch.no_grad()
def check_model_equivalence(model: DDP, log_path=None):
    assert isinstance(model, DDP)
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()
    logging.debug(f'Checking DDP equivalence of {type(model.module)}. Local rank = {local_rank}')

    msd = model.module.state_dict()
    success, failed, skipped = [], [], []
    for pname, p in msd.items():
        if not p.is_floating_point():
            skipped.append((pname, str(p.shape), str(p.dtype)))
            continue
        p_backup = p.detach().clone()
        dist.reduce(p, dst=0, op=dist.ReduceOp.SUM)
        if local_rank == 0:
            p.div_(float(world_size))
            flag = torch.allclose(p_backup, p)
            if flag:
                success.append(pname)
            else:
                failed.append((pname, str(p.shape), str(p.dtype)))
                # failed.append(f'{pname:<48s} {p.shape}')
            p.copy_(p_backup)

    logging.info(
        f'DDP parameters & buffers summary: '
        f'total {len(msd)}, same {len(success)}, different {len(failed)}, skipped {len(skipped)}'
    )
    if (local_rank == 0) and (log_path is not None):
        logging.info(f'Logging detailed information into {log_path}... (will overwrite the file!)')
        with open(log_path, 'w') as f:
            for pair in failed:
                print(f'Different: {pair[0]:<48s}{pair[1]:<24s}{pair[2]}', file=f)
            for pair in skipped:
                print(f'Skipped: {pair[0]:<48s}{pair[1]:<24s}{pair[2]}', file=f)


@torch.no_grad()
def sync_model_buffers(model: DDP, keys: list):
    """ Sync buffers which contain the specified keys.

    Args:
        model (DDP): DDP model.
        keys (list): list of keys. Example: `['running_mean', 'running_var']`
    """
    assert isinstance(model, DDP)
    world_size = dist.get_world_size()
    # make every node has the same running bn stats
    count = 0
    for name, buffer in model.module.named_buffers(recurse=True):
        if any([k in name for k in keys]):
            # average bn stats across whole group
            torch.distributed.all_reduce(buffer, op=dist.ReduceOp.SUM)
            buffer.div_(float(world_size))
            count += 1
    if dist.get_rank() == 0:
        print(f'Synced {count} buffers whose name contain any from {keys}.')


def _dist_debug(type, value, traceback):
    if type is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', -1))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        msg = f'local={local_rank}, world={world_size}'
        [logging.error(t) for t in (msg, type, value, traceback)]
        if local_rank >= 0:
            dist.destroy_process_group()
        exit()
