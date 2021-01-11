import tensorflow as tf


def normalize_image_batch(batch):
    # normalize using imagenet mean and std
    mean = tf.reshape(tf.constant([0.485, 0.456, 0.406]), (1, 1, -1))
    std = tf.reshape(tf.constant([0.229, 0.224, 0.225]), (1, 1, -1))
    batch /= 255.0
    batch -= mean
    batch /= std
    return batch


def gram_matrix(y):
    batch_size, height, width, channels = y.shape
    y = tf.transpose(y, [0, 3, 1, 2])
    features = tf.reshape(y, (batch_size, channels, height * width))
    features_t = tf.transpose(features, [0, 2, 1])
    gram_mat = tf.matmul(features, features_t)
    gram_mat /= channels * height * width
    return gram_mat
