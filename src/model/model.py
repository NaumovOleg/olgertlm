import keras
import tensorflow as tf
from .layers import TransformerBlock, PositionalEmbedding


Embedding = keras.layers.Embedding
MultiHeadAttention = keras.layers.MultiHeadAttention
Dense = keras.layers.Dense
LayerNormalization = keras.layers.LayerNormalization
Dropout = keras.layers.Dropout
Model = keras.models.Model


class GPT(Model):
    """GPT"""

    def __init__(
        self, vocab_size, max_len, d_model, num_heads, dff, num_layers, rate=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embed = Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEmbedding(max_len, d_model)
        self.dropout = Dropout(rate)

        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.final_layer = Dense(vocab_size)

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]

        look_ahead_mask = self.create_look_ahead_mask(seq_len)
        padding_mask = self.create_padding_mask(x)
        combined_mask = tf.maximum(look_ahead_mask, padding_mask)

        x = self.token_embed(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_embed(x)
        x = self.dropout(x, training=training)

        for transformer in self.transformer_blocks:
            x = transformer(x, training=training, mask=combined_mask)

        x = self.final_layer(x)
        return x
