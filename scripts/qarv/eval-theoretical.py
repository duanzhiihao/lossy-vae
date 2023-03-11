import argparse
import torch

import lvae


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',      type=str,   default='qarv_base')
    parser.add_argument('-a', '--model_args', type=str,   default='pretrained=True')
    parser.add_argument('-l', '--lmb_range',  type=float, default=[16, 2048], nargs='+')
    parser.add_argument('-s', '--steps',      type=int,   default=8)
    parser.add_argument('-n', '--datasets',   type=str,   default=['kodak', 'tecnick-rgb-1200', 'clic2022-test'], nargs='+')
    parser.add_argument('-d', '--device',     type=str,   default='cuda:0')
    args = parser.parse_args()

    kwargs = eval(f'dict({args.model_args})')
    model = lvae.get_model(args.model, **kwargs)

    model = model.to(device=torch.device(args.device))
    model.eval()

    for name in args.datasets:
        img_dir = lvae.paths.known_datasets[name]
        stats = model.self_evaluate(img_dir, lmb_range=args.lmb_range, steps=args.steps)
        print(f'================ {name} ================')
        # print results
        for k, vlist in stats.items():
            vlist_str = ', '.join([f'{v:.12f}'[:7] for v in vlist])
            print(f'{k:<6s} = [{vlist_str}]')


if __name__ == '__main__':
    main()
