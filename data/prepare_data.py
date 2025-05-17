import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from config import Config


def load_data(file_path, maxlen):
    """Reads data from a file and prepares it for training."""

    print("LOading text", file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    text = " ".join([line.strip() for line in lines]).lower()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    sequence = tokenizer.texts_to_sequences([text])[0]
    vocab_size = len(tokenizer.word_index) + 1
    X, y = [], []
    for i in range(len(sequence) - maxlen):
        X.append(sequence[i : i + maxlen])
        y.append(sequence[i + maxlen])
    X = np.array(X)
    y = np.array(y)
    print(
        f"Последовательность токенов: {len(sequence)}, примеров: {len(X)}, vocab_size: {vocab_size}"
    )

    tokenizer_json = tokenizer.to_json()

    with open(
        f"{Config.SAVED_MODEL_PATH}/tokenizer.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(tokenizer_json)

    return X, y, tokenizer, vocab_size
