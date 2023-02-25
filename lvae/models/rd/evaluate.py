from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, OrderedDict
import json
import argparse
import math
import torch
import torch.nn.functional as tnf
import torchvision.transforms.functional as tvf
from timm.utils import AverageMeter

from lvae.paths import known_datasets
from lvae.models.registry import get_model
from lvae.utils.coding import pad_divisible_by


def evaluate_model(model, lmb, dataset_name):
    device = next(model.parameters()).device
    # get list of image paths
    img_dir = known_datasets[dataset_name]
    img_paths = list(Path(img_dir).rglob('*.*'))
    print(f'Evaluating {type(model)} on {len(img_paths)} images from {img_dir}')
    pbar = tqdm(img_paths)
    # evaluate model
    all_image_stats = defaultdict(AverageMeter)
    for impath in pbar:
        # read image
        img = Image.open(impath)
        imgh, imgw = img.height, img.width
        # pad image
        img_padded = pad_divisible_by(img, div=model.max_stride)

        # forward pass
        im = tvf.to_tensor(img_padded).unsqueeze_(0).to(device=device)
        x_hat, stats_all = model.forward_end2end(im, lmb=lmb)

        # compute bpp
        _, imC, imH, imW = im.shape
        kl = sum([stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]).mean(0) / (imH * imW)
        bpp_theoretical = kl.item() * math.log2(math.e)
        # compute psnr
        real = tvf.to_tensor(img)
        fake = model.process_output(x_hat).cpu().squeeze(0)[:, :imgh, :imgw]
        mse = tnf.mse_loss(real, fake, reduction='mean').item()
        psnr = float(-10 * math.log10(mse))
        # accumulate results
        all_image_stats['bpp'].update(bpp_theoretical)
        all_image_stats['psnr'].update(psnr)

    results = {k: meter.avg for k,meter in all_image_stats.items()}
    return results


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',        type=str,   default='rd_model_base')
    parser.add_argument('-a', '--model_args',   type=str,   default='pretrained=True')
    parser.add_argument('-l', '--lmb_range',    type=float, default=[4, 2048], nargs='+')
    parser.add_argument('-s', '--steps',        type=int,   default=16)
    parser.add_argument('-n', '--dataset_name', type=str,   default='tecnick-rgb-1200')
    parser.add_argument('-d', '--device',       type=str,   default='cuda:0')
    args = parser.parse_args()

    kwargs = eval(f'dict({args.model_args})')
    model = get_model(args.model, **kwargs)

    model = model.to(device=torch.device(args.device))
    model.eval()

    start, end = args.lmb_range
    lambdas = torch.linspace(
        math.log(start), math.log(end), steps=args.steps
    ).exp().tolist()

    save_json_path = Path(f'runs/results/{args.dataset_name}-{args.model}.json')
    if not save_json_path.parent.is_dir():
        print(f'Creating {save_json_path.parent} ...')
        save_json_path.parent.mkdir(parents=True)

    all_lmb_stats = defaultdict(list)
    for lmb in lambdas:
        results = evaluate_model(model, lmb, args.dataset_name)
        print(results)

        for k,v in results.items():
            all_lmb_stats[k].append(v)
    # save to json
    json_data = OrderedDict()
    json_data['name'] = args.model
    json_data['lambdas'] = lambdas
    json_data['test-set'] = args.dataset_name
    json_data['results'] = all_lmb_stats
    with open(save_json_path, 'w') as f:
        json.dump(json_data, fp=f, indent=4)
    print(f'Saved results to {save_json_path}.')

    # print results
    for k, vlist in all_lmb_stats.items():
        vlist_str = ', '.join([f'{v:.12f}'[:7] for v in vlist])
        print(f'{k:<6s} = [{vlist_str}]')


if __name__ == '__main__':
    main()
