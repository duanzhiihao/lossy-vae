import argparse
from time import time
from tqdm import tqdm
import numpy as np
import cv2
from timm.utils import AverageMeter

from lvae.paths import known_datasets
from vvc import encode_intra, decode_to_yuv_file, get_temp_file_path


def encode_file(impath: str, output_path, qp=30):
    # convert to yuv file
    im = cv2.cvtColor(cv2.imread(str(impath)), cv2.COLOR_BGR2YUV)
    im = np.transpose(im, axes=(2, 0, 1))
    tmp_yuv_path = get_temp_file_path(suffix='.yuv')
    with open(tmp_yuv_path, "wb") as f:
        f.write(im.tobytes())

    imh, imw = im.shape[:2]
    t_start = time()
    msg, cmd = encode_intra(
        input_path=tmp_yuv_path,
        input_hw=(imh, imw),
        output_path=output_path,
        qp=qp
    )
    t_enc = time() - t_start

    tmp_yuv_path.unlink()
    return t_enc


def speedtest(qp=10):
    # find images
    image_root = known_datasets['kodak']
    img_paths = list(image_root.rglob('*.*'))
    tmp_bits_path = get_temp_file_path(suffix='.bits')
    tmp_rec_path = get_temp_file_path(suffix='.yuv')

    encode_meter = AverageMeter()
    decode_meter = AverageMeter()
    pbar = tqdm(img_paths)
    for impath in pbar:
        t_enc = encode_file(impath, tmp_bits_path, qp=qp)
        encode_meter.update(t_enc)

        t_start = time()
        decode_to_yuv_file(tmp_bits_path, tmp_rec_path)
        t_dec = time() - t_start
        decode_meter.update(t_dec)

        tmp_bits_path.unlink()
        tmp_rec_path.unlink()

    print(f'qp={qp}, encode time={encode_meter.avg:.3f}s, decode time={decode_meter.avg:.3f}s')
    return encode_meter.avg, decode_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quality', type=int, default=list(range(50,20,-1)), nargs='+')
    args = parser.parse_args()

    enc_time_all = []
    dec_time_all = []
    for q in args.quality:
        e, d = speedtest(q)
        enc_time_all.append(e)
        dec_time_all.append(d)

    for q, e, d in zip(args.quality, enc_time_all, dec_time_all):
        print(f'q={q},    enc={e:.4g}s,    dec={d:.4g}s')


if __name__ == '__main__':
    main()
