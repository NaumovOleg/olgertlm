import tensorflow as tf
import keras
import numpy as np

Layer = keras.layers.Layer


@keras.saving.register_keras_serializable()
class PositionalEmbedding(Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(max_len, d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def positional_encoding(self, max_len, d_model):
        angle_rates = 1 / np.power(
            10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model)
        )
        angle_rads = np.arange(max_len)[:, np.newaxis] * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]
