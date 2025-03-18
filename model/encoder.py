import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell


from .components.positional import add_timing_signal_nd


class Encoder(object):
    """Class with a __call__ method that applies convolutions to an image"""

    def __init__(self, config):
        self._config = config


    def __call__(self, training, img, dropout):
        """Applies convolutions to the image

        Args:
            training: (tf.placeholder) tf.bool
            img: batch of img, shape = (?, height, width, channels), of type
                tf.uint8

        Returns:
            the encoded images, shape = (?, h', w', c')

        """
        img = tf.cast(img, tf.float32) / 255.

        with tf.variable_scope("convolutional_encoder"):
            # cnn 3x3 - 64 stride 1 with padding = 1
            out = tf.layers.conv2d(img, 64, 3, 1, "SAME",
                    activation=tf.nn.relu)
            
            # cnn 3x3 - 128 stride 1 with padding = 1
            out = tf.layers.conv2d(out, 128, 3, 1, "SAME",
                    activation=tf.nn.relu)
            
            # cnn 3x3 - 256 stride 1 with padding = 1
            out = tf.layers.conv2d(out, 256, 3, 1, "SAME",
                    activation=tf.nn.relu)

            # cnn 3x3 - 256 stride 1 with padding = 1
            out = tf.layers.conv2d(out, 256, 3, 1, "SAME",
                    activation=tf.nn.relu)
            
            # maxpool(2, 1)
            out = tf.layers.max_pooling2d(out, (2, 1), (2, 1), "SAME")

            # cnn 3x3 - 512 stride 1 with padding = 1
            out = tf.layers.conv2d(out, 512, 3, 1, "SAME",
                    activation=tf.nn.relu)
            
            # maxpool(2, 1)
            out = tf.layers.max_pooling2d(out, (2, 1), (2, 1), "SAME")

            # cnn 3x3 - 512 stride 1 with padding = 1
            out = tf.layers.conv2d(out, 512, 3, 1, "SAME",
                    activation=tf.nn.relu)

            if self._config.positional_embeddings:
                # from tensor2tensor lib - positional embeddings
                out = add_timing_signal_nd(out)

        return out
