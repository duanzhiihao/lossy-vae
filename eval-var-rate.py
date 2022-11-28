from pathlib import Path
from collections import defaultdict, OrderedDict
import json
import argparse
import math
import torch

from lvae.models.registry import get_model
from lvae.evaluation.image import imcoding_evaluate


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',        type=str,   default='qarv_base')
    parser.add_argument('-a', '--model_args',   type=str,   default='pretrained=True')
    parser.add_argument('-l', '--lmb_range',    type=float, default=[16, 2048], nargs='+')
    parser.add_argument('-s', '--steps',        type=int,   default=16)
    parser.add_argument('-n', '--dataset_name', type=str,   default='tecnick-rgb-1200')
    parser.add_argument('-d', '--device',       type=str,   default='cuda:0')
    args = parser.parse_args()

    kwargs = eval(f'dict({args.model_args})')
    model = get_model(args.model, **kwargs)

    model = model.to(device=torch.device(args.device))
    model.eval()
    model.compress_mode()

    start, end = args.lmb_range
    p = 3.0
    lambdas = torch.linspace(
        math.pow(start,1/p), math.pow(end,1/p), steps=args.steps
    ).pow(p).tolist()

    save_json_path = Path(f'runs/results/{args.dataset_name}-{args.model}.json')
    if not save_json_path.parent.is_dir():
        print(f'Creating {save_json_path.parent} ...')
        save_json_path.parent.mkdir(parents=True)

    all_lmb_stats = defaultdict(list)
    for lmb in lambdas:
        if hasattr(model, 'default_lmb'):
            model.default_lmb = lmb
        else:
            print(f'==== model {args.model} is deprecated. Please use new ones instead ====')
            model._default_log_lmb = math.log(lmb)
        results = imcoding_evaluate(model, args.dataset_name)
        print(results)

        for k,v in results.items():
            all_lmb_stats[k].append(v)
    # save to json
    json_data = OrderedDict()
    json_data['name'] = args.model
    json_data['lambdas'] = lambdas
    json_data['test-set'] = args.dataset_name
    with open(save_json_path, 'w') as f:
        json.dump(all_lmb_stats, fp=f, indent=4)
    # print results
    for k, vlist in all_lmb_stats.items():
        vlist_str = ', '.join([f'{v:.12f}'[:7] for v in vlist])
        print(f'{k:<6s} = [{vlist_str}]')


if __name__ == '__main__':
    main()
