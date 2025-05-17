import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


def load_data(file_path, maxlen):
    """Читает текст из файла и готовит X, y для языкового моделирования."""
    # Чтение данных
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    text = " ".join([line.strip() for line in lines]).lower()

    # Создаём токенизатор и преобразуем текст в последовательность индексов
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    sequence = tokenizer.texts_to_sequences([text])[0]
    vocab_size = len(tokenizer.word_index) + 1  # +1 для нулевого индекса

    # Строим выборку: для каждого фрагмента длины maxlen предсказываем следующий токен
    X, y = [], []
    for i in range(len(sequence) - maxlen):
        X.append(sequence[i : i + maxlen])
        y.append(sequence[i + maxlen])
    X = np.array(X)
    y = np.array(y)
    print(
        f"Последовательность токенов: {len(sequence)}, примеров: {len(X)}, vocab_size: {vocab_size}"
    )
    return X, y, tokenizer, vocab_size
