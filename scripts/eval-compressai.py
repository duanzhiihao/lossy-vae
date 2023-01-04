from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from PIL import Image
import sys
import json
import pickle
import argparse
import math
import numpy as np
import torch
import torchvision.transforms.functional as tvf
import compressai.zoo.image as czi

from lvae.paths import known_datasets


def get_object_bits(obj):
    return sys.getsizeof(pickle.dumps(obj)) * 8


def pad_divisible_by(img, div=64):
    h_old, w_old = img.height, img.width
    if (h_old % div == 0) and (w_old % div == 0):
        return img
    h_tgt = round(div * math.ceil(h_old / div))
    w_tgt = round(div * math.ceil(w_old / div))
    # left, top, right, bottom
    padding = (0, 0, (w_tgt - w_old), (h_tgt - h_old))
    padded = tvf.pad(img, padding=padding, padding_mode='edge')
    if False:
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.show()
    # return padded, (top, left)
    return padded


@torch.no_grad()
def evaluate_model(model, dataset_root):
    device = next(model.parameters()).device
    img_paths = Path(dataset_root).rglob('*.*')

    all_image_stats = defaultdict(float)
    pbar = tqdm(img_paths)
    for impath in pbar:
        # read image
        img = Image.open(impath)
        imgh, imgw = img.height, img.width
        img_padded = pad_divisible_by(img, div=64)
        im = tvf.to_tensor(img_padded).unsqueeze_(0).to(device=device)

        output = model.forward(im)

        # psnr
        real = np.array(img).astype(np.float32) / 255.0
        fake = output['x_hat'].cpu().squeeze(0).permute(1,2,0)[:imgh, :imgw, :].numpy()
        mse = np.square(real - fake).mean()
        psnr = float(-10 * math.log10(mse))
        # bpp
        likelihoods = output['likelihoods']
        num_bits = -1.0 * torch.log2(likelihoods['y']).sum()
        if 'z' in likelihoods:
            bits2 = -1.0 * torch.log2(likelihoods['z']).sum()
            num_bits = num_bits + bits2
        bpp  = float(num_bits / (im.shape[2] * im.shape[3]))

        # logging
        pbar.set_description(f'image {impath.stem}: bpp={bpp:.5f}, psnr={psnr:.3f}')
        all_image_stats['bpp'] += bpp
        all_image_stats['psnr']  += psnr
        all_image_stats['count'] += 1

    # average over all images
    count = all_image_stats.pop('count')
    results = {k: v/count for k,v in all_image_stats.items()}
    return results


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',   type=str, default='mbt2018-mean')
    parser.add_argument('-t', '--testset', type=str, default='kodak')
    parser.add_argument('-d', '--device',  type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)
    model_name = args.model

    dataset_root = known_datasets[args.testset]
    save_json_path = f'runs/results/{args.testset}-{model_name}.json'

    all_lmb_stats = defaultdict(list)
    max_quality = max(list(czi.model_urls[model_name]['mse'].keys()))
    for q in range(1, max_quality+1):
        # initialize model
        model = czi._load_model(model_name, metric='mse', quality=q, pretrained=True)

        num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        print(f'Evaluating model {type(model)}, quality={q}, params={num_params/1e6:.3f} M')
        model = model.to(device=device)
        model.eval()

        results = evaluate_model(model, dataset_root)
        for k,v in results.items():
            all_lmb_stats[k].append(v)
        # save to json
        with open(save_json_path, 'w') as f:
            json.dump(all_lmb_stats, fp=f, indent=4)

    for k, vlist in all_lmb_stats.items():
        vlist_str = ', '.join([f'{v:.12f}'[:8] for v in vlist])
        print(f'{k:<6s} = [{vlist_str}]')


if __name__ == '__main__':
    main()