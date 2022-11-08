from pathlib import Path
from tempfile import gettempdir
from collections import OrderedDict
from multiprocessing.pool import ThreadPool
import json
import time
import logging
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from timm.utils import AverageMeter

import vvc


def get_temp_file_path(suffix='.tmp'):
    dictionary = 'abcdefghijklmnopqrstuvwxyz0123456789'
    random_str = ''.join(random.choices(dictionary, k=16))
    return Path(gettempdir()) / f'{random_str}{suffix}'


def evaluate_one_video(frame_dir, args, quality, job_n=None):
    # sanity check
    results_dir = Path(args.results_dir)
    assert results_dir.is_dir(), f'results_dir={results_dir} does not exist'
    save_path = results_dir / f'q{quality}.json'

    logging.info(f'starting q={quality}, job{job_n}. frame_dir={frame_dir}, save_path={save_path}')
    tic = time.time()

    frame_dir = Path(frame_dir)
    _str = f'{args.name}-{args.codec}-q{quality}-gop{args.gop}-num{args.num_frames}'
    save_bit_path = Path(f'cache/{_str}/{frame_dir.stem}.bits')
    if not save_bit_path.parent.is_dir():
        save_bit_path.parent.mkdir(parents=True, exist_ok=True)
    rec_yuv_path = get_temp_file_path(suffix='.yuv')

    # encoding
    msg, cmd = vvc.encode_folder(
        frame_dir, save_bit_path, quality=quality, num_frames=args.num_frames,
        intra_period=args.gop
    )

    # decoding
    vvc.decode_to_yuv_file(save_bit_path, output_path=rec_yuv_path)

    # compute metrics
    ori_frame_paths = list(Path(frame_dir).glob('*.png'))
    ori_frame_paths.sort()
    img_height, img_width = cv2.imread(str(ori_frame_paths[0])).shape[:2]
    # compute psnr
    avg_psnr = AverageMeter()
    with open(rec_yuv_path, 'rb') as f:
        for fi, ori_fp in enumerate(ori_frame_paths):
            # read an original frame
            real = cv2.cvtColor(cv2.imread(str(ori_fp)), cv2.COLOR_BGR2RGB)
            assert real.shape == (img_height, img_width, 3)

            # read a reconstructed frame
            raw = f.read(img_height * img_width * 3)
            if raw == b'': # finish reading all frames
                break

            fake = np.frombuffer(raw, dtype=np.uint8).reshape(3, img_height, img_width).transpose(1, 2, 0)
            fake = cv2.cvtColor(fake, cv2.COLOR_YUV2RGB)
            if False:
                plt.figure(); plt.imshow(fake)
                plt.show()
            # compute psnr
            real = real.astype(np.float32) / 255.0
            fake = fake.astype(np.float32) / 255.0
            mse = np.square(real - fake).mean()
            psnr = -10 * np.log10(mse)
            avg_psnr.update(psnr)
        assert f.read(1) == b'' # make sure no bytes left
    # clean up
    rec_yuv_path.unlink()
    # sanity check
    num_frames = args.num_frames or len(ori_frame_paths)
    assert avg_psnr.count == fi == num_frames
    # compute bpp
    num_pixels = img_height * img_width * num_frames
    bpp = save_bit_path.stat().st_size * 8 / float(num_pixels)

    stats = OrderedDict()
    stats['video']   = str(frame_dir)
    stats['command'] = str(cmd)
    stats['quality'] = quality
    stats['bpp']     = bpp
    stats['psnr']    = avg_psnr.avg
    # save results
    if save_path.is_file():
        with open(save_path, mode='r') as f:
            all_seq_results = json.load(fp=f)
        assert isinstance(all_seq_results, list)
        all_seq_results.append(stats)
    else:
        all_seq_results = [stats]
    with open(save_path, mode='w') as f:
        json.dump(all_seq_results, fp=f, indent=2)

    elapsed = time.time() - tic
    msg = f'q={quality}, job{job_n}, time={elapsed:.1f}s. '
    msg += f'decoded {avg_psnr.count} out of {len(ori_frame_paths)} frames. '
    msg += f'frame_dir.stem={frame_dir.stem}, bpp={bpp}, psnr={avg_psnr.avg}'
    logging.info('\u001b[92m' + msg + '\u001b[0m')
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--codec',       type=str, default='vtm18.0')
    parser.add_argument('-n', '--name',        type=str, default='uvg-1080p')
    parser.add_argument('-q', '--quality',     type=int, default=[40], nargs='+')
    parser.add_argument('-g', '--gop',         type=int, default=12)
    parser.add_argument('-f', '--num_frames',  type=int, default=96)
    parser.add_argument('-w', '--workers',     type=int, default=2)
    args = parser.parse_args()

    # set VVC version
    vvc.version = args.codec

    # init logging
    logging.basicConfig(
        level=logging.INFO, format= '[%(asctime)s] %(message)s', datefmt='%Y-%b-%d %H:%M:%S'
    )

    frames_root = {
        'uvg-1080p': 'd:/datasets/video/uvg/1080p-frames',
        'mcl-jcv':   'd:/datasets/video/mcl-jcv/frames',
    }

    suffix = 'allframes' if (args.num_frames is None) else f'first{args.num_frames}'
    results_dir = Path(f'runs/results/{args.codec}-{args.name}-gop{args.gop}-{suffix}')
    if not results_dir.is_dir():
        results_dir.mkdir(parents=True)
    logging.info(f'Saving results to {results_dir}')
    args.results_dir = results_dir

    video_frame_dirs = list(Path(frames_root[args.name]).glob('*/'))
    video_frame_dirs.sort()
    logging.info(f'Total {len(video_frame_dirs)} sequences')

    # set up multiprocessing
    pool = ThreadPool(processes=args.workers)
    mp_results = []

    # enumerate all quality
    for q in args.quality:
        for i, vfd in enumerate(video_frame_dirs):
            mp_results.append(pool.apply_async(evaluate_one_video, args=(vfd, args, q, i)))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
