import tensorflow as tf
import re
import string


def load_and_preprocess_text(text_path):
    """load_and_preprocess_text"""
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.lower().replace("\n", " ")
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text


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
