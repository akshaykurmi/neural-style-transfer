import tensorflow as tf
from tqdm import tqdm

from nst.data import load_image, load_coco_2014_train
from nst.models import StyleTransferNet, VGG16
from nst.utils import normalize_image_batch, gram_matrix


def train_model(args):
    vgg = VGG16()
    style_transfer_net = StyleTransferNet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    content_mse = tf.keras.losses.MeanSquaredError()
    style_mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    checkpoint = tf.train.Checkpoint(style_transfer_net=style_transfer_net)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.ckpt_dir, max_to_keep=1)
    summary_writer = tf.summary.create_file_writer(args.log_dir)

    style_image = load_image(args.style_image_path, args.style_image_size)
    style_image = tf.expand_dims(style_image, axis=0)
    style_image = tf.repeat(style_image, repeats=args.batch_size, axis=0)
    style_image = normalize_image_batch(style_image)
    style_features = vgg(style_image)
    style_features_gram_matrix = {k: gram_matrix(v) for k, v in style_features.items()}

    train_dataset = load_coco_2014_train(args.coco_img_dir, args.content_image_size, args.batch_size)
    n_batches = tf.data.experimental.cardinality(train_dataset).numpy()

    for epoch_id in range(args.epochs):
        with tqdm(desc=f'Training Epoch {epoch_id + 1}', unit='batch', total=n_batches) as pbar:
            for batch_id, X_batch in enumerate(train_dataset):
                content_loss, style_loss, total_loss = train_step(vgg, style_transfer_net, optimizer, content_mse,
                                                                  style_mse, X_batch, style_features_gram_matrix,
                                                                  args.content_weight, args.style_weight)
                if batch_id % args.ckpt_interval == 0 or batch_id == n_batches - 1:
                    checkpoint_manager.save()
                if batch_id % args.log_interval == 0 or batch_id == n_batches - 1:
                    with summary_writer.as_default(), tf.name_scope('losses'):
                        tf.summary.scalar('content_loss', content_loss, step=(epoch_id * n_batches) + batch_id)
                        tf.summary.scalar('style_loss', style_loss, step=(epoch_id * n_batches) + batch_id)
                        tf.summary.scalar('total_loss', total_loss, step=(epoch_id * n_batches) + batch_id)
                pbar.update(1)


@tf.function(experimental_relax_shapes=True)
def train_step(vgg, style_transfer_net, optimizer, content_mse, style_mse, X_batch, style_features_gram_matrix,
               content_weight, style_weight):
    actual_batch_size = X_batch.shape[0]
    with tf.GradientTape() as tape:
        Y_batch = style_transfer_net(X_batch)
        X_batch = normalize_image_batch(X_batch)
        Y_batch = normalize_image_batch(Y_batch)
        features_X = vgg(X_batch)
        features_Y = vgg(Y_batch)
        features_Y_gram_matrix = {k1: gram_matrix(v1) for k1, v1 in features_Y.items()}

        content_loss = content_weight * content_mse(features_Y['relu2_2'], features_X['relu2_2'])
        style_loss = 0.0
        for k1 in ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']:
            style_loss += tf.reduce_mean(style_mse(
                features_Y_gram_matrix[k1], style_features_gram_matrix[k1][:actual_batch_size, :, :],
            ))
        style_loss *= style_weight
        total_loss = content_loss + style_loss

        gradients = tape.gradient(total_loss, style_transfer_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, style_transfer_net.trainable_variables))
    return content_loss, style_loss, total_loss
