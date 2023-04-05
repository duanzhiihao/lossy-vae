from PIL import Image
from tqdm import tqdm
from pathlib import Path
import math
import argparse
import torch

import lvae



def compute_average(lmb_min, lmb_max):
    # compute the average of two numbers in log space
    return math.exp((math.log(lmb_min) + math.log(lmb_max)) / 2)


def binary_search_lmb(model, img_path, bits_path, tgt_bytes: int,
                      max_iter: int, tol: float):
    # find lambda that produces the target bytes. Search in log space
    bits_path = Path(bits_path)

    lmb_min, lmb_max = model.lmb_range
    lmb = compute_average(lmb_min, lmb_max)

    pbar = tqdm(range(max_iter))
    for i in pbar:
        img = Image.open(img_path)
        model.compress_file(img_path, bits_path, lmb=lmb)

        n_bytes = bits_path.stat().st_size
        if n_bytes > tgt_bytes:
            lmb_max = lmb
        else:
            lmb_min = lmb
        lmb = compute_average(lmb_min, lmb_max)

        bpp = n_bytes * 8 / (img.width * img.height)
        msg = f'{lmb=:.3f}, bytes={n_bytes}B, target={tgt_bytes}B, {bpp=:.3f}'
        if True: # debug: decompress and compute PSNR
            fake = model.decompress_file(bits_path).cpu()
            import torchvision.transforms.functional as tvf
            real = tvf.to_tensor(Image.open(img_path)).unsqueeze_(0)
            mse = torch.mean((fake - real) ** 2)
            psnr = -10 * math.log10(mse.item())
            msg += f', PSNR={psnr:.3f}'
            # save the reconstructed image
            fake = tvf.to_pil_image(fake.squeeze_(0))
            fake.save(f'runs/rec.png')
        pbar.set_description(msg)
        if abs(n_bytes - tgt_bytes) <= tol:
            break

    return lmb


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',        type=str, default='runs/lake720p.jpg')
    parser.add_argument('-b', '--bits',         type=str, default='runs/lake720p.bits')
    parser.add_argument('-m', '--model',        type=str, default='qarv_base')
    parser.add_argument('-t', '--target_bytes', type=int, default=1500)
    parser.add_argument('--search_device', type=str, default='cuda:0')
    parser.add_argument('--speed_device',  type=str, default='cpu')
    args = parser.parse_args()

    model = lvae.get_model(args.model, pretrained=True)
    model = model.to(device=torch.device(args.search_device))
    model.eval()
    model.compress_mode(True)

    lmb = binary_search_lmb(
        model, img_path=args.input, bits_path=args.bits, tgt_bytes=args.target_bytes,
        max_iter=50, tol=1
    )


if __name__ == '__main__':
    main()
