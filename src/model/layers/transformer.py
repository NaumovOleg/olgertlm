import keras

Layer = keras.layers.Layer
MultiHeadAttention = keras.layers.MultiHeadAttention
Dense = keras.layers.Dense
LayerNormalization = keras.layers.LayerNormalization
Sequential = keras.models.Sequential
Dropout = keras.layers.Dropout


@keras.saving.register_keras_serializable()
class TransformerBlock(Layer):
    """TransformerBlock"""

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = Sequential([Dense(dff, activation="relu"), Dense(d_model)])

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x, training, mask):
        # Multi-head attention
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
