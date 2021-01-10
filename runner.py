import argparse
import os

from nst.download_train_data import download_coco_2014_train

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, type=str, choices=['download_data'])
args = parser.parse_args()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(DATA_DIR, exist_ok=True)

if __name__ == '__main__':
    if args.task == 'download_data':
        download_coco_2014_train(DATA_DIR)
