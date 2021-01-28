import argparse
import os
from datetime import datetime

from nst.data import download_coco_2014_train
from nst.eval import stylize_image
from nst.train import train_model

parser = argparse.ArgumentParser()

# Common Args
parser.add_argument('--task', required=True, type=str, choices=['download_data', 'train', 'stylize_image'],
                    help='The task to run')
parser.add_argument('--run_id', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"),
                    help='Unique identifier for the model')

# Task = train
parser.add_argument('--style_image_path', type=str, help='Path to the style image')

# Task = stylize_image
parser.add_argument('--content_image_path', type=str, help='Path to the image to stylize')
parser.add_argument('--output_image_path', type=str, help='Path to save the stylized image to')

args = parser.parse_args()

# Paths
args.base_dir = os.path.abspath(os.path.dirname(__file__))
args.data_dir = os.path.join(args.base_dir, 'data')
args.coco_img_dir = os.path.join(args.data_dir, 'coco_2014_train')
args.output_dir = os.path.join(args.base_dir, 'output', args.run_id)
args.ckpt_dir = os.path.join(args.output_dir, 'ckpt')
args.log_dir = os.path.join(args.output_dir, 'log')

# Training hyperparameters
args.epochs = 2
args.batch_size = 4
args.content_image_size = (256, 256)
args.style_image_size = None
args.content_weight = 1
args.style_weight = 1.5
args.learning_rate = 1e-3
args.log_interval = 100
args.ckpt_interval = 1000

os.makedirs(args.data_dir, exist_ok=True)
os.makedirs(args.output_dir, exist_ok=True)

if __name__ == '__main__':
    if args.task == 'download_data':
        download_coco_2014_train(args.data_dir)
    if args.task == 'train':
        train_model(args)
    if args.task == 'stylize_image':
        stylize_image(args)
