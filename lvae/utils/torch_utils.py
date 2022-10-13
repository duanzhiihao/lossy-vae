import itertools
from pathlib import Path
from collections import OrderedDict, defaultdict
import logging
import random
import numpy as np
import torch
import torch.nn as nn


def set_random_seeds(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def num_params(model: nn.Module):
    """ Get the number of learnable parameters of a model.

    Args:
        model (nn.Module): pytorch model
    """    
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    return num


@torch.no_grad()
def flops_benchmark(model: nn.Module, input_shape=None, inputs=None, verbose=True):
    """ Floating point operations (FLOPs).
    Exactly one of the `input_shape` or `inputs` should be specified.

    Args:
        model (nn.Module): [description]
        input_shape (tuple): Example: (3, 224, 224)
        inputs (tuple): Example: (torch.randn(1,3,224,224), torch.randn(1,64,28,28))
        verbose (bool):
    """
    from mycv.utils.torch_utils import num_params
    from fvcore.nn import flop_count
    from thop import profile, clever_format
    from ptflops import get_model_complexity_info
    import torch.profiler as tp

    model.eval()

    if input_shape is not None:
        assert inputs is None
        inputs = (torch.randn(1, *input_shape), )
    else:
        assert inputs is not None

    print('========================================================================')
    # mycv
    mycv_params = num_params(model)
    with tp.profile(activities=[tp.ProfilerActivity.CPU], with_flops=True) as prof:
        # with tp.record_function("model_inference"):
        model(*inputs)
    tp_flops = sum([event.flops for event in prof.events()]) / 2
    if True:
        _debug = sum([event.flops for event in prof.key_averages()]) / 2
        assert tp_flops == _debug
    # print(kav.table(row_limit=-1, top_level_events_only=True))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))
    print('========================================================================')
    # thop
    thop_macs, thop_params = profile(model, inputs=inputs, verbose=verbose)
    # thop_macs, thop_params = clever_format([thop_macs, thop_params], "%.3f")
    print('========================================================================')
    # fvcore
    final_count, skipped_ops = flop_count(model, inputs=inputs)
    fv_gmacs = sum([v for k,v in final_count.items()])
    print('========================================================================')
    # ptflops
    if input_shape is not None:
        ptfl_macs, ptfl_params = get_model_complexity_info(
            model, input_shape, print_per_layer_stat=False, as_strings=False, verbose=verbose
        )
    else:
        ptfl_macs, ptfl_params = 0, 0

    print(f'===================={str(type(model)):^32}====================')
    template = '{:<8s}' + '{:^14s}' * 2
    lines = [
        template.format('', 'Parameters', 'FLOPs(MACs)'),
        template.format('torch:', f'{mycv_params/1e6:.3f}M', f'{tp_flops/1e9:.3g}B(G)'),
        template.format('thop:',  f'{thop_params/1e6:.3f}M', f'{thop_macs/1e9:.3g}B(G)'),
        template.format('ptfl:',  f'{ptfl_params/1e6:.3f}M', f'{ptfl_macs/1e9:.3g}B(G)'),
        template.format('fvcore:', '-', f'{fv_gmacs:.3g}B(G)'),
        f'fvcore details: count={final_count}, skipped_ops={skipped_ops}'
    ]
    for msg in lines:
        print(msg)
    print('========================================================================')


@torch.no_grad()
def speed_benchmark(model: nn.Module, input_shapes: list,
                    device: torch.device, bs:int=1, N:int=1000):
    """ Speed benchmark for a pytorch model

    Args:
        model (nn.Module): pytorch model
        input_shapes (list): exmaple: [(3,224,224), (64,28,28)]
        device (torch.device): device
        bs (int, optional): batch size. Defaults to 1.
        N (int, optional): number of iterations to run. Defaults to 1000.
    """
    import time
    from tqdm import tqdm

    model = model.to(device=device)
    model.eval()

    input_shapes = [(bs, *shape) for shape in input_shapes]

    print(f'Testing {type(model)} inference speed on {device}. Warming up first...')
    torch.backends.cudnn.benchmark = True
    for _ in range(max(10, N // 20)):
        inputs = [torch.randn(*shape, device=device) for shape in input_shapes]
        _ = model(*inputs)

    if not device == torch.device('cpu'):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    tic = time.time()
    for _ in tqdm(range(N)):
        inputs = [torch.randn(*shape, device=device) for shape in input_shapes]
        y = model(*inputs)
        a = y
    if not device == torch.device('cpu'):
        torch.cuda.synchronize()
    elapsed = time.time() - tic

    latency = elapsed / N
    if not device == torch.device('cpu'):
        mem = torch.cuda.max_memory_allocated(device) / 1e9
    else:
        mem = -1
    print(f'{device}, mem: {mem:.3g}G, {input_shapes}:',
          f'latency={latency:.3g}s={latency*1000:.2f}ms, FPS={1/latency:.2f}')


def model_benchmark(model: nn.Module, input_shape:tuple=(3,224,224), n_cpu=100, n_cuda=1000):
    """ Test and print the model parameter number, FLOPs, and speed

    Args:
        model (nn.Module): pytorch model
    """
    print()

    flops_benchmark(model, input_shape)
    print()

    device = torch.device('cpu')
    speed_benchmark(model, [input_shape], device, bs=1, N=n_cpu)
    print()

    device = torch.device('cuda:0')
    speed_benchmark(model, [input_shape], device, bs=1, N=n_cuda)
    print()


def model_profile_fvcore(model: nn.Module, input):
    """ get the model params and FLOPs

    Args:
        model (nn.Module): pytorch model
    """
    nparam = num_params(model)
    
    from fvcore.nn import flop_count
    flops_count, skipped_ops = flop_count(model, (input, )) 
    print(flops_count)

    return nparam, flops_count


def summary_weights(state_dict: OrderedDict, save_path='model.txt'):
    """ Summary the weights name and shape to a text file at save_path

    Args:
        state_dict (OrderedDict): model state dict
        save_path (str, optional): output save path. Defaults to 'model.txt'.
    """
    if not isinstance(state_dict, OrderedDict):
        print('Warning: state_dict is not a OrderedDict. keys may not be ordered.')
    if Path(save_path).exists():
        print(f'Warning: overwriting {save_path}')
    with open(save_path, 'w') as f:
        for k, v in state_dict.items():
            line = f'{k:<48s}{v.shape}'
            print(line, file=f)


def _get_model_weights(weights, verbose):
    """ Get model weights from a path str or a dict

    Args:
        weights (str, dict-like): a file path, or a dict, or the weights itself
    """
    if isinstance(weights, (str, Path)):
        logging.info(f'Loading weights from {weights}...')
        weights = torch.load(weights, map_location='cpu')
    if 'model' in weights:
        weights = weights['model']
    assert isinstance(weights, OrderedDict), 'model weights should be an OrderedDict'
    weights: OrderedDict
    return weights


def load_partial(model, weights, verbose=None):
    ''' Load weights that have the same name
    
    Args:
        model (torch.nn.Module): model
        weights (str or dict): weights
        verbose (bool, optional): deprecated.
    '''
    if verbose is not None:
        logging.warning(f'{__file__}.load_partial(): verbose parameter is deprecated')
    external_state = _get_model_weights(weights, verbose=verbose)

    self_state = model.state_dict()
    new_dic = OrderedDict()
    for k,v in external_state.items():
        if k in self_state and self_state[k].shape == v.shape:
            new_dic[k] = v
        else:
            debug = 1
    model.load_state_dict(new_dic, strict=False)
    # def _num(dic_):
    #     return sum([p.numel() for k,p in dic_.items()])

    msg = (f'{type(model).__name__}: {len(self_state)} layers,'
           f'saved: {len(external_state)} layers,'
           f'overlap & loaded: {len(new_dic)} layers.')
    logging.info(msg)


def weights_replace(weights, old, new, verbose=True):
    """ replace old with new

    Args:
        weights (str, dict-like): a file path, or a dict, or the weights itself
        verbose (bool, optional): print if True. Defaults to True.
    """
    weights = _get_model_weights(weights, verbose=verbose)

    count = 0
    new_dic = OrderedDict()
    for k,v in weights.items():
        if old in k:
            count += 1
        k = k.replace(old, new)
        new_dic[k] = v
    if verbose:
        print(f"Total {len(weights)}, renamed {count} '{old}' to '{new}'")
    return new_dic


def weights_rename(weights, new_list, verbose=True):
    """ rename the weights given a list of new names

    Args:
        weights (str, dict-like): a file path, or a dict, or the weights itself
        new_list (iterable): a list of new names
        verbose (bool, optional): print if True. Defaults to True.
    """
    weights = _get_model_weights(weights, verbose=verbose)
    new_list = list(new_list)

    wlen = len(weights.keys())
    assert wlen == len(new_list), f'Length mismatch: weights={wlen} vs. list={len(new_list)}'

    new_dic = OrderedDict()
    for (_,v), k in zip(weights.items(), new_list):
        new_dic[k] = v
    return new_dic


def weights_match(weights, model: nn.Module):
    """ Try to match the weights keys with the model.

    Args:
        weights (str, dict-like): a file path, or a dict, or the weights itself
        model (nn.Module): pytorch model

    ### Example:
        >>> model = resnet50()
        >>> weights = weights_match('/path/to/checkpoint.pt', model)
        >>> model.load_state_dict(weights)
    """
    weights = weights_rename(weights, model.state_dict().keys())
    return weights


def optimizer_named_states(optimizer: torch.optim.Optimizer, named_params):
    """ save optimizer state dict with named states

    Args:
        optimizer (torch.optim.Optimizer): pytorch optimizer
        named_params: the return of model.named_parameters()
    """
    param_to_name = {v:k for k,v in named_params}
    named_states = dict()
    for i, (p, pstate) in enumerate(optimizer.state.items()):
        if p not in param_to_name:
            raise ValueError(f'Optimizer parameter {i}, {p.shape} not in model')
        pname = param_to_name[p]
        named_states[pname] = pstate
    return {
        'state': named_states,
        'param_groups': None,
    }


def optimizer_load_partial(optimizer: torch.optim.Optimizer, named_params,
                           state_dict: dict, verbose=True):
    """ load optimizer state dict

    Args:
        optimizer (torch.optim.Optimizer): pytorch optimizer
        named_params: the return of model.named_parameters()
        state_dict (dict): saved state dict
        verbose (bool, optional): print info or not. Defaults to True.
    """
    param_to_name = {v:k for k,v in named_params}
    saved_named_states = state_dict['state']
    new_state = defaultdict(dict)
    count = 0

    # iterate through all parameters in the optimizer.param_groups
    parameters = [pg['params'] for pg in optimizer.param_groups]
    parameters = list(itertools.chain.from_iterable(parameters))
    for p in parameters:
        # parameter in optimizer should also be in the model
        if p not in param_to_name:
            raise ValueError(f'Optimizer parameter {p}, {p.shape} not in model')
        pname = param_to_name[p]
        if pname in saved_named_states: # if saved, load the state of it
            new_state[p] = saved_named_states[pname]
            count += 1
    # check other states
    cur_states = optimizer.state
    for p in cur_states:
        pname = param_to_name[p]
        if pname not in saved_named_states:
            new_state[p] = cur_states[p]

    if verbose:
        print(f'{type(optimizer).__name__}:',
              f'{len(parameters)} parameters, {len(cur_states)} states,',
              f'saved: {len(saved_named_states)} states,',
              f'overlap & loaded: {len(new_state)} states')

    optimizer.__setstate__({'state': new_state})


def _extract_parameters(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    return parameters


def get_grad_norm(parameters, norm_type=2.0) -> torch.Tensor:
    """ The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    This function is adapted from `torch.nn.utils.clip_grad_norm_()`.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = _extract_parameters(parameters)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def clip_grad_norm_(parameters, max_norm, norm_type=2.0, computed_norm=None) -> torch.Tensor:
    """ Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    This function is adapted from `torch.nn.utils.clip_grad_norm_()`

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. \
            only used when computed_norm=None.
        computed_norm (float or Tensor): if provided, use it instead of \
            computing the norm from the parameters.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = _extract_parameters(parameters)
    if computed_norm is None:
        computed_norm = get_grad_norm(parameters, norm_type=norm_type)
    clip_coef = max_norm / (computed_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for p in parameters:
        p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))
    return computed_norm


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/

    # init
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def reset_model_parameters(model: nn.Module, verbose=True):
    history = [set(), set()]
    for m in model.modules():
        mtype = type(m).__name__
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()
            history[0].add(mtype)
        else:
            history[1].add(mtype)
    if verbose:
        print(f'{type(model)}: reset parameters of {history[0]}; no effect on {history[1]}')


def is_parallel(model):
    '''
    Check if the model is DP or DDP
    '''
    flag = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    return flag


def de_parallel(model):
    m = model.module if is_parallel(model) else model
    return m


if __name__ == '__main__':
    # from torchvision.models import resnet50
    # model = resnet50()
    from timm.models.swin_transformer import swin_tiny_patch4_window7_224
    model = swin_tiny_patch4_window7_224()

    model_benchmark(model, (3,224,224), n_cpu=500, n_cuda=5000)
    device = torch.device('cuda:0')
    speed_benchmark(model, input_shapes=[(3,224,224)], device=device, bs=64, N=500)
    exit()

    # reset_model_parameters(model)
