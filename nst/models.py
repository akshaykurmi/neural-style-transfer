import tensorflow as tf
import tensorflow_addons as tfa


class VGG16(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vgg_layers = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=None).layers
        for layer in vgg_layers:
            layer.trainable = False
        self.relu1_2 = tf.keras.Sequential(vgg_layers[0:3])
        self.relu2_2 = tf.keras.Sequential(vgg_layers[3:6])
        self.relu3_3 = tf.keras.Sequential(vgg_layers[6:10])
        self.relu4_3 = tf.keras.Sequential(vgg_layers[10:14])

    def call(self, inputs, training=None, mask=None):
        inputs = tf.keras.applications.vgg16.preprocess_input(inputs, data_format='channels_last')
        r1 = self.relu1_2(inputs)
        r2 = self.relu2_2(r1)
        r3 = self.relu3_3(r2)
        r4 = self.relu4_3(r3)
        return {'relu1_2': r1, 'relu2_2': r2, 'relu3_3': r3, 'relu4_3': r4}

    def get_config(self):
        super().get_config()


class StyleTransferNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = tf.keras.layers.ReLU()

        # Section 1
        self.conv1 = PaddedConv(filters=32, kernel_size=(9, 9), strides=(1, 1))
        self.conv2 = PaddedConv(filters=64, kernel_size=(3, 3), strides=(2, 2))
        self.conv3 = PaddedConv(filters=128, kernel_size=(3, 3), strides=(2, 2))
        self.conv1_norm = create_instance_norm_layer()
        self.conv2_norm = create_instance_norm_layer()
        self.conv3_norm = create_instance_norm_layer()

        # Section 2
        self.residual1 = ResidualConvBlock(filters=128, kernel_size=(3, 3), strides=(1, 1))
        self.residual2 = ResidualConvBlock(filters=128, kernel_size=(3, 3), strides=(1, 1))
        self.residual3 = ResidualConvBlock(filters=128, kernel_size=(3, 3), strides=(1, 1))
        self.residual4 = ResidualConvBlock(filters=128, kernel_size=(3, 3), strides=(1, 1))
        self.residual5 = ResidualConvBlock(filters=128, kernel_size=(3, 3), strides=(1, 1))

        # Section 3
        self.upconv1 = UpsamplingConv(filters=64, kernel_size=(3, 3), strides=(1, 1), upsample_size=(2, 2))
        self.upconv2 = UpsamplingConv(filters=32, kernel_size=(3, 3), strides=(1, 1), upsample_size=(2, 2))
        self.upconv1_norm = create_instance_norm_layer()
        self.upconv2_norm = create_instance_norm_layer()
        self.conv4 = PaddedConv(filters=3, kernel_size=(9, 9), strides=(1, 1))

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv1_norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv2_norm(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.conv3_norm(x)
        x = self.relu(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.residual5(x)
        x = self.upconv1(x)
        x = self.upconv1_norm(x)
        x = self.relu(x)
        x = self.upconv2(x)
        x = self.upconv2_norm(x)
        x = self.relu(x)
        x = self.conv4(x)
        return x

    def get_config(self):
        super().get_config()


class PaddedConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, **kwargs):
        super().__init__(**kwargs)
        self.padding = [[0, 0]] + [[k // 2] * 2 for k in kernel_size] + [[0, 0]]
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                                           data_format='channels_last')

    def call(self, inputs, **kwargs):
        x = tf.pad(inputs, self.padding, mode='REFLECT')
        x = self.conv(x)
        return x


class UpsamplingConv(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, upsample_size, **kwargs):
        super().__init__(**kwargs)
        self.upsample = tf.keras.layers.UpSampling2D(size=upsample_size, interpolation='nearest',
                                                     data_format='channels_last')
        self.padded_conv = PaddedConv(filters=filters, kernel_size=kernel_size, strides=strides)

    def call(self, inputs, **kwargs):
        x = self.upsample(inputs)
        x = self.padded_conv(x)
        return x


class ResidualConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, **kwargs):
        super().__init__(**kwargs)
        self.padded_conv1 = PaddedConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.padded_conv2 = PaddedConv(filters=filters, kernel_size=kernel_size, strides=strides)
        self.norm1 = create_instance_norm_layer()
        self.norm2 = create_instance_norm_layer()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        x = self.padded_conv1(inputs)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.padded_conv2(x)
        x = self.norm2(x)
        x = x + inputs
        return x


def create_instance_norm_layer():
    return tfa.layers.InstanceNormalization(axis=-1, center=True, scale=True, beta_initializer='zeros',
                                            gamma_initializer='ones')
