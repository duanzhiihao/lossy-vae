import time
import argparse
from multiprocessing.pool import ThreadPool
from pathlib import Path
from collections import OrderedDict
import numpy as np
import cv2

import mycv.utils.vvc as vvc
from mycv.utils import get_temp_file_path, json_load, json_dump

from mycv.paths import all_dataset_paths


def green_str(msg: str):
    return '\u001b[92m' + str(msg) + '\u001b[0m'


def evaluate_one_image(img_path: Path, q: int, result_path: Path):
    print(f'starting q={q}, image={img_path}, will save results to {result_path} ...')

    tic = time.time()
    im = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    img_hw = im.shape[:2]

    # compress to file and compute bpp
    tmp_bits_path = get_temp_file_path(suffix='.bits')
    _, cmd = vvc.encode_numpy_rgb(im, output_path=tmp_bits_path, quality=q)
    num_bits = Path(tmp_bits_path).stat().st_size * 8

    # decompression, and remove bits file
    im_hat = vvc.decode_to_numpy_rgb(tmp_bits_path, img_hw=img_hw)
    tmp_bits_path.unlink()

    # bits per pixel
    bpp  = float(num_bits / (img_hw[0] * img_hw[1]))
    # PSNR
    real = im.astype(np.float64) / 255.0
    fake = im_hat.astype(np.float64) / 255.0
    psnr = float(-10 * np.log10(np.square(fake - real).mean()))

    # save results
    stats = OrderedDict()
    stats['img_path'] = str(img_path)
    stats['command']  = str(cmd)
    stats['quality']  = q
    stats['bpp']      = bpp
    stats['psnr']     = psnr
    if result_path.is_file():
        all_images_results = json_load(result_path)
        assert isinstance(all_images_results, list)
        all_images_results.append(stats)
    else:
        all_images_results = [stats]
    json_dump(all_images_results, result_path)

    elapsed = time.time() - tic
    msg = f'quality={q}, image={img_path.name}, time={elapsed:.1f}s, bpp={bpp}, psnr={psnr}'
    print(green_str(msg))
    return bpp, psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--codec',   type=str, default='vtm18.0')
    parser.add_argument('-d', '--dataset', type=str, default='kodak')
    parser.add_argument('-q', '--quality', type=int, nargs='+', default=list(range(10,51)))
    parser.add_argument('-w', '--workers', type=int, default=2)
    args = parser.parse_args()

    # set VVC version
    vvc.version = args.codec
    # dataset root
    dataset_root = all_dataset_paths[args.dataset]
    assert dataset_root.is_dir(), f'{dataset_root=} does not exist.'
    image_paths = sorted(dataset_root.rglob('*.*'))
    print(f'Found {len(image_paths)} images in {dataset_root}.')
    # results saving directory
    results_save_dir = Path(f'results/{args.codec}-{args.dataset}')
    results_save_dir.mkdir(parents=True, exist_ok=False)

    # set up multiprocessing
    pool = ThreadPool(processes=args.workers)
    mp_results = []

    for q in args.quality:
        for impath in image_paths:
            mp_results.append(
                pool.apply_async(evaluate_one_image, args=(impath, q, results_save_dir/f'q{q}.json'))
            )
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
