import argparse
from time import time
from tqdm import tqdm
from PIL import Image
import torch
import torch.backends.cudnn
import torchvision.transforms.functional as tvf

from lvae.paths import known_datasets
from lvae.models.registry import get_model


def speedtest(model, first=None, verbose=True):
    device = next(model.parameters()).device
    cuda_sync = torch.cuda.is_available()

    # find images
    image_root = known_datasets['kodak']
    img_paths = list(image_root.rglob('*.*'))
    if first is not None:
        img_paths = img_paths[:first]

    encode_time = 0
    decode_time = 0
    pbar = tqdm(img_paths) if verbose else img_paths
    for impath in pbar:
        im = tvf.to_tensor(Image.open(impath)).unsqueeze_(0).to(device=device)

        t_start = time()
        compressed_obj = model.compress(im)
        if cuda_sync:
            torch.cuda.synchronize()
        t_enc_finish = time()
        output = model.decompress(compressed_obj)
        if cuda_sync:
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
    parser.add_argument('-m', '--models',  type=str, default=['qarv_base'], nargs='+')
    parser.add_argument('-a', '--kwargs',  type=str, default='pretrained=True')
    parser.add_argument('-d', '--device',  type=str, default='cuda:0')
    parser.add_argument('-w', '--workers', type=int, default=None)
    args = parser.parse_args()

    print('---------------- version info ----------------')
    print(f'pytorch = {torch.__version__}')
    print(f'pytorch cuda = {torch.version.cuda}')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)

    print('---------------- device info ----------------')
    device = torch.device(args.device)
    if args.device == 'cpu':
        print(f'pytorch device = {device}')
    else:
        print(f'pytorch device = {torch.cuda.get_device_properties(device)}')
    if args.workers is not None:
        torch.set_num_threads(args.workers)
    print(f'pytorch uses {torch.get_num_threads()} CPU threads')
    print('--------------------------------')

    for name in args.models:
        kwargs = eval(f'dict({args.kwargs})')
        model = get_model(name, **kwargs)
        num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

        model = model.to(device=device)
        model.eval()
        model.compress_mode()

        print(f'{name}, {type(model)}, device={device}')
        print(f'Number of parameters: {num_params/1e6:.3f} M')
        _ = speedtest(model, first=4, verbose=False) # warm up
        enc_time, dec_time = speedtest(model)
        print(f'encode time={enc_time:.3f}s, decode time={dec_time:.3f}s')
        print()


if __name__ == '__main__':
    main()
