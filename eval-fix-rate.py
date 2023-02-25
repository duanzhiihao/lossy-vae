from collections import defaultdict, OrderedDict
import json
import argparse
import torch

from lvae.models.registry import get_model
from lvae.evaluation import imcoding_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',        type=str,  default='qres34m')
    parser.add_argument('--lambdas',      type=int,  default=[16, 32, 64, 128, 256, 512, 1024, 2048], nargs='+')
    parser.add_argument('--dataset_name', type=str,  default='kodak')
    parser.add_argument('--device',       type=str,  default='cuda:0')
    args = parser.parse_args()

    save_json_path = args.save_json or f'results/{args.dataset}-{args.model}.json'
    if not save_json_path.parent.is_dir():
        print(f'Creating {save_json_path.parent} ...')
        save_json_path.parent.mkdir(parents=True)

    all_lmb_results = defaultdict(list)
    for lmb in args.lambdas:
        # initialize model
        model = get_model(args.model, lmb=lmb, pre_trained=True)

        print(f'Evaluating lmb={lmb} ...')
        model.compress_mode()
        model = model.to(device=torch.device(args.device))
        model.eval()

        # evaluate
        results = imcoding_evaluate(model, args.dataset)
        print('results:', results, '\n')

        # accumulate results
        for k,v in results.items():
            all_lmb_results[k].append(v)

    # save to json
    json_data = OrderedDict()
    json_data['name'] = args.model
    json_data['lambdas'] = args.lambdas
    json_data['test-set'] = args.dataset_name
    json_data['results'] = all_lmb_results
    with open(save_json_path, 'w') as f:
        json.dump(json_data, fp=f, indent=4)
    print(f'\nSaved results to {save_json_path} \n')

    # final print
    for k, vlist in all_lmb_results.items():
        vlist_str = ', '.join([f'{v:.12f}'[:8] for v in vlist])
        print(f'{k:<6s} = [{vlist_str}]')


if __name__ == '__main__':
    main()
