import os
import zipfile
from urllib.request import urlopen

import tensorflow as tf
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


def load_coco_2014_train(images_dir, image_size, batch_size):
    dataset = tf.data.Dataset.list_files(os.path.join(images_dir, '*.jpg'))
    dataset = dataset.map(lambda image_fpath: load_image(image_fpath, image_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_image(image_fpath, image_size=None):
    image = tf.io.read_file(image_fpath)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    if image_size is not None:
        image_shape = tf.shape(image)
        height_and_width = tf.minimum(image_shape[0], image_shape[1])
        image = tf.image.resize_with_crop_or_pad(image, height_and_width, height_and_width)
        image = tf.image.resize(image, image_size, preserve_aspect_ratio=True)
    return image


def save_image(image, output_fpath):
    tf.keras.preprocessing.image.save_img(output_fpath, image, data_format='channels_last', scale=True)
