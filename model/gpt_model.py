import tensorflow as tf
from .transformer_block import TransformerBlock
import keras

Model = keras.Model
Embedding = keras.layers.Embedding
LayerNormalization = keras.layers.LayerNormalization
Dense = keras.layers.Dense


class GPTModel(Model):
    def __init__(self, vocab_size, maxlen, num_layers, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

        self.blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)
        ]

        self.norm = LayerNormalization(epsilon=1e-6)
        self.final_dense = Dense(vocab_size)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        x = x + positions
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        x = x[:, -1, :]  # Берём только последний токен
        logits = self.final_dense(x)
        return logits
