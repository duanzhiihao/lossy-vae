from pathlib import Path
from tempfile import gettempdir
from collections import OrderedDict
from multiprocessing.pool import ThreadPool
import time
import json
import logging
import argparse
import random
import numpy as np
import cv2

import vvc


def green_str(msg: str):
    return '\u001b[92m' + str(msg) + '\u001b[0m'


def get_temp_file_path(suffix='.tmp'):
    dictionary = 'abcdefghijklmnopqrstuvwxyz0123456789'
    random_str = ''.join(random.choices(dictionary, k=16))
    return Path(gettempdir()) / f'{random_str}{suffix}'


def evaluate_one_image(img_path: Path, q: int, result_path: Path):
    logging.info(f'starting q={q}, image={img_path}, will save results to {result_path} ...')

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
        with open(result_path, mode='r') as f:
            all_images_results = json.load(fp=f)
        assert isinstance(all_images_results, list)
        all_images_results.append(stats)
    else:
        all_images_results = [stats]
    with open(result_path, mode='w') as f:
        json.dump(all_images_results, fp=f, indent=2)

    elapsed = time.time() - tic
    msg = f'quality={q}, image={img_path.name}, time={elapsed:.1f}s, bpp={bpp}, psnr={psnr}'
    logging.info(green_str(msg))
    return bpp, psnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--codec',        type=str, default='vtm18.0')
    parser.add_argument('-d', '--dataset_name', type=str, default='kodak')
    parser.add_argument('-p', '--dataset_path', type=str, default=None)
    parser.add_argument('-q', '--quality',      type=int, nargs='+', default=list(range(10,51)))
    parser.add_argument('-w', '--workers',      type=int, default=2)
    args = parser.parse_args()

    # set VVC version
    vvc.version = args.codec

    # init logging
    logging.basicConfig(
        level=logging.INFO, format= '[%(asctime)s] %(message)s', datefmt='%Y-%b-%d %H:%M:%S'
    )

    default_dataset_paths = {
        'kodak': 'd:/datasets/kodak',
    }
    # get dataset root
    if args.dataset_path is None:
        dataset_root = Path(default_dataset_paths[args.dataset_name])
    else:
        dataset_root = Path(args.dataset_path)
    assert dataset_root.is_dir(), f'{dataset_root=} does not exist.'
    logging.info('================================')
    logging.info(f'Data set name={args.dataset_name}, data path={args.dataset_path}')
    # find all images
    image_paths = sorted(dataset_root.rglob('*.*'))
    logging.info(f'Found {len(image_paths)} images in {dataset_root}.')
    # results saving directory
    _results_root = Path(f'runs/evaluation')
    all_quality_results_dir = _results_root / f'{args.codec}-{args.dataset_name}'
    all_quality_results_dir.mkdir(parents=True, exist_ok=False)
    final_result_path = _results_root / f'{args.codec}-{args.dataset_name}.json'
    logging.info(f'Will save individual results to {all_quality_results_dir},',
                 f'and will save overall results to {final_result_path}')
    logging.info('================================')

    # set up multiprocessing
    pool = ThreadPool(processes=args.workers)
    mp_results = []
    # run evaluation for all q, for all images
    for q in args.quality:
        for impath in image_paths:
            mp_results.append(
                pool.apply_async(evaluate_one_image, args=(impath, q, all_quality_results_dir/f'q{q}.json'))
            )
    pool.close()
    pool.join()

    # average results over all images
    all_file_paths = sorted(all_quality_results_dir.rglob('*.*'))
    final_json_data = {
        'name': args.codec,
        'dataset_name': args.dataset_name,
        'dataset_root': str(dataset_root),
        'quality': list(args.quality),
        'results': {
            'bpp': [],
            'psnr': []
        }
    }
    for fp in all_file_paths:
        with open(fp, mode='r') as f:
            list_of_dicts = json.load(fp=f)
        final_json_data['results']['bpp'].append(get_avg(list_of_dicts, 'bpp'))
        final_json_data['results']['psnr'].append(get_avg(list_of_dicts, 'psnr'))
    # save json data
    with open(final_result_path, mode='w') as f:
        json.dump(final_json_data, fp=f, indent=2)
    logging.info('================================')
    logging.info(green_str(f'Final results saved to {final_result_path}'))
    logging.info('================================')


def get_avg(list_of_dict, key):
    all_values = [d[key] for d in list_of_dict]
    avg = sum(all_values) / len(all_values)
    return avg


if __name__ == '__main__':
    main()
