import tensorflow as tf

from nst.data import load_image, save_image
from nst.models import StyleTransferNet


def stylize_image(args):
    style_transfer_net = StyleTransferNet()
    checkpoint = tf.train.Checkpoint(style_transfer_net=style_transfer_net)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.ckpt_dir, max_to_keep=1)
    checkpoint_manager.restore_or_initialize()

    content_image = load_image(args.content_image_path, args.content_image_size)
    content_image = tf.expand_dims(content_image, axis=0)
    stylized_image = style_transfer_net(content_image)
    stylized_image = tf.squeeze(stylized_image)

    save_image(stylized_image, args.output_image_path)
