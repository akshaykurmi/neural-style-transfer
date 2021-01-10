import os
import zipfile
from urllib.request import urlopen

from tqdm import tqdm


def download_coco_2014_train(output_dir):
    url = "http://images.cocodataset.org/zips/train2014.zip"
    zip_fpath = os.path.join(output_dir, 'train2014.zip')
    extracted_fpath = os.path.join(output_dir, 'train2014')
    renamed_fpath = os.path.join(output_dir, 'coco_2014_train')

    if not os.path.exists(zip_fpath):
        response = urlopen(url)
        CHUNK = 16 * 1024
        with tqdm(desc=f'Downloading {url}', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            with open(zip_fpath, 'wb') as f:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(CHUNK)

    print(f'Extracting {zip_fpath}')
    with zipfile.ZipFile(zip_fpath, 'r') as zipf:
        zipf.extractall(output_dir)

    os.rename(extracted_fpath, renamed_fpath)
    os.remove(zip_fpath)


def train_model(args):
    pass


def stylize_image(args):
    pass
