from pathlib import Path
from collections import defaultdict
import json
import argparse
import math
import torch

from models.registry import get_model
from datasets.compression import imcoding_evaluate


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',        type=str, default='vr_small')
    parser.add_argument('-a', '--model_args',   type=str, default='pretrained=True')
    # parser.add_argument('-w', '--weights_path', type=str, default='runs/tpc_lm3pz_enc_0.pt')
    parser.add_argument('-n', '--dataset_name', type=str, default='kodak')
    # parser.add_argument('-n', '--dataset_name', type=str, default='clic2022-test')
    # parser.add_argument('-a', '--bd_anchor',    type=str, default='vtm-12.1')
    parser.add_argument('-d', '--device',       type=str, default='cuda:0')
    args = parser.parse_args()

    kwargs = eval(f'dict({args.model_args})')
    model = get_model(args.model, **kwargs)

    model = model.to(device=torch.device(args.device))
    model.eval()
    model.compress_mode()

    start, end = (16, 1024)
    log_lambdas = torch.linspace(math.log(start), math.log(end), steps=12).tolist()

    save_json_path = Path(f'runs/results/{args.dataset_name}-{args.model}.json')
    if not save_json_path.parent.is_dir():
        save_json_path.parent.mkdir(parents=True)
    all_lmb_stats = defaultdict(list)

    for log_lmb in log_lambdas:
        model._default_log_lmb = log_lmb
        results = imcoding_evaluate(model, args.dataset_name)
        print(results)

        for k,v in results.items():
            all_lmb_stats[k].append(v)
    # save to json
    with open(save_json_path, 'w') as f:
        json.dump(all_lmb_stats, fp=f, indent=4)
    # print results
    for k, vlist in all_lmb_stats.items():
        vlist_str = ', '.join([f'{v:.12f}'[:7] for v in vlist])
        print(f'{k:<6s} = [{vlist_str}]')


if __name__ == '__main__':
    main()
