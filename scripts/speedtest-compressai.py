import argparse
from pathlib import Path
from time import time
from tqdm import tqdm
from PIL import Image
import torch
import torch.backends.cudnn
import torchvision.transforms.functional as tvf

from compressai.zoo.image import bmshj2018_factorized, mbt2018_mean, mbt2018, cheng2020_anchor


images_root = Path('/ssd0/datasets/kodak')


def speedtest(model, first=None, verbose=True):
    device = next(model.parameters()).device

    # find images
    img_paths = list(images_root.rglob('*.png'))
    img_paths = img_paths + img_paths # run over dataset two times
    if first is not None:
        img_paths = img_paths[:first]

    encode_time = 0
    decode_time = 0
    pbar = tqdm(img_paths) if verbose else img_paths
    for impath in pbar:
        im = tvf.to_tensor(Image.open(impath)).unsqueeze_(0).to(device=device)

        t_start = time()
        compressed_obj = model.compress(im)
        torch.cuda.synchronize()
        t_enc_finish = time()
        output = model.decompress(compressed_obj['strings'], compressed_obj['shape'])
        torch.cuda.synchronize()
        t_dec_finish = time()

        encode_time += (t_enc_finish - t_start)
        decode_time += (t_dec_finish - t_enc_finish)

    enc_time = encode_time / float(len(img_paths))
    dec_time = decode_time / float(len(img_paths))
    return enc_time, dec_time


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device',  type=str, default='cuda:0')
    parser.add_argument('-w', '--workers', type=int, default=None)
    args = parser.parse_args()

    print('================ version info ================')
    print(f'pytorch = {torch.__version__}')
    print(f'pytorch cuda = {torch.version.cuda}')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    print('================ device info ================')
    device = torch.device(args.device)
    if args.device == 'cpu':
        print(f'pytorch device = {device}')
    else:
        print(f'pytorch device = {torch.cuda.get_device_properties(device)}')
    if args.workers is not None:
        torch.set_num_threads(args.workers)
    print(f'pytorch uses {torch.get_num_threads()} CPU threads')
    print("================================")

    for model in [
        mbt2018_mean(1, metric='mse', pretrained=True),
        mbt2018_mean(8, metric='mse', pretrained=True),
        mbt2018(1, metric='mse', pretrained=True),
        mbt2018(8, metric='mse', pretrained=True),
        cheng2020_anchor(1, metric='mse', pretrained=True),
        cheng2020_anchor(6, metric='mse', pretrained=True),
    ]:
        num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        model = model.to(device=device)
        model.eval()
        model.update()

        _ = speedtest(model, first=2, verbose=False) # warm up
        enc_time, dec_time = speedtest(model)
        print(f'{type(model)}, device={device}')
        print(f'Parameters = {num_params} = {num_params/1e6:.3f} M')
        print(f'encode time={enc_time:.3f}s, decode time={dec_time:.3f}s \n')


if __name__ == '__main__':
    main()
