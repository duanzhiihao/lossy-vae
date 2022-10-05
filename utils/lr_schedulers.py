import math

from mycv.utils.general import warning2


def get_warmup_cosine_lrf(n, lrf_min, T_warmup, T):
    """ Linear warmup, then cosine learning rate factor

    Args:
        n (int): current epoch. 0, 1, 2, ..., T-1
        lrf_min (float): final (should also be minimum) learning rate factor
        T (int): total number of epochs

    >>> if n == 0: lrf = 1 / T_warmup
    >>> if n == 1: lrf = 2 / T_warmup
    ...
    >>> if n == T_warmup-1: lrf = 1.0
    >>> if n == T_warmup  : lrf = get_cosine_lrf(1, lrf_min, T-T_warmup)
    >>> if n == T_warmup+1: lrf = get_cosine_lrf(2, lrf_min, T-T_warmup)
    ...
    >>> if n == T         : lrf = lrf_min
    """
    assert T_warmup < T, f'warmup T_warmup={T_warmup}, total T={T}'
    assert 0 <= n <= T, f'n={n}, T={T}'
    if n < T_warmup:
        lrf = (n+1) / T_warmup
    else: # n = T_warmup, T_warmup+1, T_warmup+2, ..., T
        # lrf = lrf_min + 0.5 * (1 - lrf_min) * (1 + math.cos(_cur * math.pi / T))
        lrf = get_cosine_lrf(n-T_warmup+1, lrf_min, T-T_warmup+1)
    return lrf


def get_cosine_lrf(n, lrf_min, T):
    """ Cosine learning rate factor

    Args:
        n (int): current epoch. 0, 1, 2, ..., T
        lrf_min (float): final (should also be minimum) learning rate factor
        T (int): total number of epochs
    """
    assert 0 <= n <= T, f'n={n}, T={T}'
    lrf = lrf_min + 0.5 * (1 - lrf_min) * (1 + math.cos(n * math.pi / T))
    return lrf

def cosine_lr(n, lrf_min, T):
    """ deprecated
    """
    warning2('cosine_lr() is deprecated. Please use get_cosine_lrf() instead.')
    return get_cosine_lrf(n, lrf_min, T)


def twostep(n, T):
    assert T >= 2
    lrf = 1.0 if n < T/2.0 else 0.1
    return lrf


def threestep(n, T):
    assert T >= 3
    period = math.ceil(T / 3)
    lrf = 0.1 ** (n // period)
    return lrf


def get_step_lrf(n, decay, period):
    """ Decays the learning rate by `decay` every `period` epochs.

    Args:
        n (int): current epoch
        decay (float): 0.97
        period (int): 1
    """
    lrf = decay ** (n // period)
    return lrf


def adjust_lr_threestep(optimizer, cur_epoch, base_lr, total_epoch):
    """ Sets the learning rate to the initial LR decayed by 10 every total/3 epochs

    Args:
        optimizer (torch.optim.Optimizer): optimizer
        cur_epoch (int): current epoch
        base_lr (float): base learning rate
        total_epoch (int): total epoch
    """
    assert total_epoch >= 3
    period = math.ceil(total_epoch / 3)
    lr = base_lr * (0.1 ** (cur_epoch // period))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _adjust_lr_532(optimizer, cur_epoch, base_lr, total_epoch):
    assert total_epoch % 10 == 0
    if cur_epoch < round(total_epoch * 5/10):
        lrf = 1
    elif cur_epoch < round(total_epoch * 8/10):
        lrf = 0.1
    else:
        assert cur_epoch < total_epoch
        lrf = 0.01
    lr = base_lr * lrf
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
