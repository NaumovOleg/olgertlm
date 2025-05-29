import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from tensorflow.keras.layers import TextVectorization
import re
import string


# 1. Подготовка данных
def load_and_preprocess_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Простая предобработка текста
    text = text.lower().replace("\n", " ")
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text


# 2. Токенизация
def setup_tokenizer(text, vocab_size=20000):
    tokenizer = TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=None,  # Динамическая длина последовательности
    )
    tokenizer.adapt([text])
    return tokenizer


# 3. Создание датасета
def create_dataset(text, tokenizer, seq_length=100, batch_size=32):
    text_vec = tokenizer([text])[0]
    char_dataset = tf.data.Dataset.from_tensor_slices(text_vec)

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
    return (
        dataset.shuffle(10000)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )


# 4. Архитектура модели (как в предыдущем примере)
class PositionalEmbedding(layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(max_len, d_model)

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


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.ffn = keras.Sequential(
            [layers.Dense(dff, activation="relu"), layers.Dense(d_model)]
        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask=None):
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class GPT(keras.Model):
    def __init__(
        self, vocab_size, max_len, d_model, num_heads, dff, num_layers, rate=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embed = layers.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEmbedding(max_len, d_model)
        self.dropout = layers.Dropout(rate)

        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.final_layer = layers.Dense(vocab_size)

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


# 5. Функции для обучения
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_obj = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = loss_obj(real, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


# 6. Генерация текста
class TextGenerator:
    def __init__(self, model, tokenizer, temperature=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

    def generate_text(self, start_string, num_generate=100):
        input_indices = tokenizer([start_string])[0]
        input_indices = tf.expand_dims(input_indices, 0)

        text_generated = []

        for _ in range(num_generate):
            predictions = self.model(input_indices, training=False)
            predictions = predictions[:, -1, :] / self.temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)

            text_generated.append(predicted_id.numpy()[0][0])
            input_indices = tf.concat([input_indices, predicted_id], axis=-1)
            input_indices = input_indices[:, -100:]  # Ограничиваем длину контекста

        generated_text = tokenizer.sequences_to_texts([text_generated])[0]
        return start_string + generated_text


# 7. Основной цикл
if __name__ == "__main__":
    # Параметры
    FILE_PATH = "./datasets/small.txt"  # Замените на ваш файл с текстом
    VOCAB_SIZE = 20000
    MAX_LEN = 512
    D_MODEL = 256
    NUM_HEADS = 8
    DFF = 512
    NUM_LAYERS = 6
    DROPOUT_RATE = 0.1
    BATCH_SIZE = 64
    SEQ_LENGTH = 100
    EPOCHS = 10

    # Загрузка и подготовка данных
    text = load_and_preprocess_text(FILE_PATH)
    tokenizer = setup_tokenizer(text, VOCAB_SIZE)
    vocab_size = len(tokenizer.get_vocabulary())

    dataset = create_dataset(text, tokenizer, SEQ_LENGTH, BATCH_SIZE)

    # Создание модели
    model = GPT(
        vocab_size=vocab_size,
        max_len=MAX_LEN,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        num_layers=NUM_LAYERS,
        rate=DROPOUT_RATE,
    )

    # Компиляция
    learning_rate = CustomSchedule(D_MODEL)
    optimizer = keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    model.compile(optimizer=optimizer, loss=loss_function)

    # Обучение
    history = model.fit(dataset, epochs=EPOCHS)

    # Сохранение модели
    model.save("gpt_model.h5")

    # Генерация текста
    generator = TextGenerator(model, tokenizer, temperature=0.7)
    print(generator.generate_text("the meaning of life is", num_generate=100))
