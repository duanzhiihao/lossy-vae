from tqdm import tqdm
from pathlib import Path
from PIL import Image
import argparse
import math
import torch
import torchvision as tv
import torchvision.transforms.functional as tvf

from lvae.models.qresvae.zoo import qres34m_lossless


@torch.inference_mode()
def evaluate_model(model, dataset_root):
    tmp_bit_path = Path('tmp.bits')

    img_paths = list(Path(dataset_root).rglob('*.*'))
    img_paths.sort()
    pbar = tqdm(img_paths)
    accumulated_bpp = 0.0
    for impath in pbar:
        model.compress_file(impath, tmp_bit_path)
        num_bits = tmp_bit_path.stat().st_size * 8
        fake = model.decompress_file(tmp_bit_path).squeeze(0).cpu()
        tmp_bit_path.unlink()

        # make sure the compression is lossless
        real = tvf.pil_to_tensor(Image.open(impath)) # uint8
        fake = torch.round_(fake * 255.0).to(dtype=torch.uint8) # uint8
        assert torch.equal(real, fake)
        # compute bpp
        bpp = num_bits / float(real.shape[1] * real.shape[2])
        # accumulate stats
        accumulated_bpp += float(bpp)

        # logging
        pbar.set_description(f'image {impath.stem}: bpp={bpp}')

    # average over all images
    avg_bpp = accumulated_bpp / len(img_paths)
    return avg_bpp


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',   type=str, default='d:/datasets/kodak')
    args = parser.parse_args()

    # initialize model
    model = qres34m_lossless(pretrained=True)

    model.compress_mode()
    model = model.cuda()
    model.eval()

    # evaluate
    avg_bpp = evaluate_model(model, args.root)
    print(f'Average bpp: {avg_bpp} \n')


if __name__ == '__main__':
    main()
