from sys import platform
from pathlib import Path
from tempfile import gettempdir
import subprocess
import random
import numpy as np
import cv2


version = 'vtm18.0'
current_file_folder = Path(__file__).parent


def _get_root():
    root = current_file_folder / str(version)
    return root

def get_enc_path():
    if platform == 'win32':
        fpath = _get_root() / f'EncoderApp.exe'
    elif platform == 'linux':
        fpath = _get_root() / f'EncoderApp'
    else:
        raise NotImplementedError(f'{platform=}')
    assert fpath.is_file()
    return fpath

def get_dec_path():
    if platform == 'win32':
        fpath = _get_root() / f'DecoderApp.exe'
    elif platform == 'linux':
        fpath = _get_root() / f'DecoderApp'
    else:
        raise NotImplementedError(f'{platform=}')
    assert fpath.is_file()
    return fpath


def get_temp_file_path(suffix='.tmp'):
    dictionary = 'abcdefghijklmnopqrstuvwxyz0123456789'
    random_str = ''.join(random.choices(dictionary, k=16))
    return Path(gettempdir()) / f'{random_str}{suffix}'


def run_command(cmd):
    if isinstance(cmd, list):
        cmd = ' '.join([str(c) for c in cmd])
    assert isinstance(cmd, str)
    # returned_obj = subprocess.run(cmd)
    # return returned_obj
    rv = subprocess.check_output(cmd, shell=True)
    msg = rv.decode("ascii")
    return msg, cmd


def encode_lowdelay(
        input_path, output_path, input_hw, qp, nframes,
        frame_rate=30, intra_period=-1,
        input_fmt='444'
    ):
    assert Path(input_path).suffix == '.yuv'
    cfg_path = _get_root() / 'encoder_lowdelay_vtm.cfg'
    img_height, img_width = input_hw
    cmd = [
        get_enc_path(),
        '-c', Path(cfg_path).resolve(),
        f'--InputFile={Path(input_path).resolve()}',
        f'--BitstreamFile={Path(output_path).resolve()}',
        f'--SourceWidth={input_hw[1]}',
        f'--SourceHeight={input_hw[0]}',
        '--InputBitDepth=8',
        '--OutputBitDepth=8',
        '--OutputBitDepthC=8',
        f'--InputChromaFormat={input_fmt}',
        f'--FrameRate={frame_rate}',
        f'--FramesToBeEncoded={nframes}',
        f'--IntraPeriod={intra_period}',
        f'--DecodingRefreshType=2',
        f'--QP={qp}',
        '--Level=6.2',
    ]
    msg, cmd = run_command(cmd)
    return msg, cmd


def encode_folder(input_dir, output_path, qp, num_frames=None, **kwargs):
    frame_paths = list(Path(input_dir).glob('*.png'))
    frame_paths.sort()
    if num_frames is None:
        num_frames = len(frame_paths)
    else:
        assert isinstance(num_frames, int)
        frame_paths = frame_paths[:num_frames]
    frames_yuv = [cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2YUV) for fp in frame_paths]
    fh, fw = frames_yuv[0].shape[:2]

    # write frames bytes to yuv file, frame by frame
    tmp_yuv_path = get_temp_file_path(suffix='.yuv')
    with open(tmp_yuv_path, 'wb') as f:
        for im in frames_yuv:
            assert im.shape[:2] == (fh, fw)
            im = np.transpose(im, axes=(2, 0, 1))
            f.write(im.tobytes())
    msg, cmd = encode_lowdelay(
        tmp_yuv_path, output_path, input_hw=(fh, fw), qp=qp, nframes=num_frames,
        **kwargs
    )
    tmp_yuv_path.unlink()
    return msg, cmd


def encode_intra(input_path, input_hw, output_path, qp, input_fmt=444):
    cfg_path = _get_root() / 'encoder_intra_vtm.cfg'
    cmd = [
        get_enc_path(),
        '-c', Path(cfg_path).resolve(),
        f'--InputFile={Path(input_path).resolve()}',
        f'--BitstreamFile={Path(output_path).resolve()}',
        f'--SourceWidth={input_hw[1]}',
        f'--SourceHeight={input_hw[0]}',
        f'--InputChromaFormat={input_fmt}',
        f'--FrameRate=1',
        f'--FramesToBeEncoded=1',
        f'--QP={qp}',
    ]
    msg, cmd = run_command(cmd)
    return msg, cmd


def encode_numpy_rgb(im: np.ndarray, output_path, qp=30):
    assert (im.dtype == np.uint8) and (im.ndim == 3) and (im.shape[2] == 3)
    # save to yuv file
    imh, imw = im.shape[:2]
    im = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
    im = np.transpose(im, axes=(2, 0, 1))

    tmp_yuv_path = get_temp_file_path(suffix='.yuv')
    with open(tmp_yuv_path, "wb") as f:
        f.write(im.tobytes())

    msg, cmd = encode_intra(
        input_path=tmp_yuv_path,
        input_hw=(imh, imw),
        output_path=output_path,
        qp=qp
    )

    tmp_yuv_path.unlink()
    return msg, cmd


def encode_png_file(input_path, output_path=None, qp=30):
    im = cv2.imread(str(input_path))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    encode_numpy_rgb(im, output_path=output_path, qp=qp)


def decode_to_yuv_file(bits_path, output_path=None):
    bits_path = Path(bits_path)
    if output_path is None:
        output_path = get_temp_file_path(suffix='.yuv')
    cmd = [
        get_dec_path(),
        f'--BitstreamFile={bits_path.resolve()}',
        f'--ReconFile={Path(output_path).resolve()}',
        f'--OutputBitDepth=8',
    ]
    msg, cmd = run_command(cmd)
    return msg, output_path


def decode_to_numpy_rgb(bits_path, img_hw):
    msg, output_yuv_path = decode_to_yuv_file(bits_path)

    rec = np.fromfile(output_yuv_path, dtype=np.uint8)
    rec = rec.reshape(3, img_hw[0], img_hw[1]).transpose((1,2,0))
    rec = cv2.cvtColor(rec, cv2.COLOR_YUV2RGB)

    output_yuv_path.unlink()
    return rec

