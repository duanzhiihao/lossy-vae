import argparse
import zipfile
from tqdm import tqdm
from pathlib import Path
from torch.hub import download_url_to_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='clic2022-test')
    parser.add_argument('-d', '--datasets_root', type=str, default='/ssd0/datasets')
    args = parser.parse_args()

    name = str(args.name)
    assert name in ['kodak', 'clic2022-test', 'tecnick']
    tgt_dir = Path(args.datasets_root) / name
    tgt_dir = tgt_dir.resolve()
    tgt_dir.mkdir(exist_ok=True, parents=True)

    print(f'Will download data to {tgt_dir} [yes/no]?')
    if input() != 'yes':
        print('Aborted')
        return

    if name == 'kodak':
        urls = [f'http://r0k.us/graphics/kodak/kodak/kodim{str(a).zfill(2)}.png' for a in range(1,25)]
        pbar = tqdm(urls)
        for url in pbar:
            download_url_to_file(url, dst=tgt_dir / Path(url).name, progress=False)
            pbar.set_description(f'Processing {url} ...')
    elif name == 'clic2022-test':
        url = 'https://storage.googleapis.com/clic2022_public/test_sets/image_test_30.zip'
        zip_path = tgt_dir / Path(url).name
        print(f'Downloading zip file to {zip_path}')
        download_url_to_file(url, dst=zip_path, progress=True)
        # unzip to the target dir
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tgt_dir)
        zip_path.unlink()
    elif name == 'tecnick':
        # download dataset zip file
        url = 'https://sourceforge.net/projects/testimages/files/OLD/OLD_SAMPLING/testimages.zip'
        zip_path = tgt_dir / Path(url).name
        print(f'Downloading zip file to {zip_path}')
        download_url_to_file(url, dst=zip_path, progress=True)
        # unzip to the target dir
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tgt_dir)
        zip_path.unlink()
    else:
        raise ValueError(f'{name=}')


if __name__ == '__main__':
    main()
