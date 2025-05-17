import tensorflow as tf
import keras

Sequential = keras.Sequential
Layer = keras.layers.Layer
MultiHeadAttention = keras.layers.MultiHeadAttention
Dense = keras.layers.Dense
LayerNormalization = keras.layers.LayerNormalization
Dropout = keras.layers.Dropout
Layer = keras.layers.Layer


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        # Многоголовое внимание
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Feed-forward network
        self.ffn = Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        # Нормализация
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        # Dropout для регуляризации
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        # inputs: (batch_size, seq_len, embed_dim)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        # Создаём causal mask (нижнетреугольная матрица)
        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)
        mask = i >= j - seq_len + seq_len  # True для j<=i
        mask = tf.cast(mask, tf.bool)
        mask = tf.reshape(mask, (1, seq_len, seq_len))
        mask = tf.tile(mask, (batch_size, 1, 1))  # (batch_size, seq_len, seq_len)

        # Self-attention
        attn_output = self.att(
            query=inputs, value=inputs, key=inputs, attention_mask=mask
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)
        return out2
