import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from config import Config

tokenizer_path = Path(f"{Config.TOKENIZER_DIR}/tokenizer.json")


def load_data(file_path, maxlen):
    """Reads data from a file and prepares it for training."""

    print("Loading text from:", file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    text = " ".join([line.strip() for line in lines]).lower()

    if tokenizer_path.exists():
        with tokenizer_path.open("r", encoding="utf-8") as f:
            tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
        print("--------------> Loading tokenizer from file")
    else:
        print("--------------> Creating new tokenizer from text")
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([text])
        with open(
            f"{Config.SAVE_TOKENIZER_DIR}/tokenizer.json",
            "w",
            encoding="utf-8",
            opener=None,
        ) as f:
            f.write(tokenizer.to_json())

    sequence = tokenizer.texts_to_sequences([text])[0]
    vocab_size = len(tokenizer.word_index) + 1
    X, y = [], []
    for i in range(len(sequence) - maxlen):
        X.append(sequence[i : i + maxlen])
        y.append(sequence[i + maxlen])
    X = np.array(X)
    y = np.array(y)
    print(
        f"Token sequences: {len(sequence)}, samples: {len(X)}, vocab_size: {vocab_size}"
    )

    return X, y, tokenizer, vocab_size
