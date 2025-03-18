import tensorflow as tf
from .components.positional import add_timing_signal_nd

class Encoder(object):
    def __init__(self, config):
        self._config = config

    def __call__(self, training, img, dropout):
        img = tf.cast(img, tf.float32) / 255.
        with tf.name_scope("convolutional_encoder"):
            out = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", activation="relu")(img)
            out = tf.keras.layers.Conv2D(128, 3, strides=1, padding="same", activation="relu")(out)
            out = tf.keras.layers.Conv2D(256, 3, strides=1, padding="same", activation="relu")(out)
            out = tf.keras.layers.Conv2D(256, 3, strides=1, padding="same", activation="relu")(out)
            out = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding="same")(out)
            out = tf.keras.layers.Conv2D(512, 3, strides=1, padding="same", activation="relu")(out)
            out = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding="same")(out)
            out = tf.keras.layers.Conv2D(512, 3, strides=1, padding="same", activation="relu")(out)
            if self._config.positional_embeddings:
                out = add_timing_signal_nd(out)
        return out