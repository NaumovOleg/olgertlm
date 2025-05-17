import json
from data import load_data
from model import GPTModel


import keras

SparseCategoricalCrossentropy = keras.losses.SparseCategoricalCrossentropy

# Параметры
DATA_PATH = "datasets/shakespere.txt"
MAXLEN = 20
NUM_LAYERS = 2
EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
EPOCHS = 1000
BATCH_SIZE = 4

# Загрузка и подготовка данных
X, y, tokenizer, vocab_size = load_data(DATA_PATH, maxlen=MAXLEN)
print("Пример преобразованной последовательности:", X[0], "->", y[0])


model = GPTModel(
    vocab_size=vocab_size,
    maxlen=MAXLEN,
    num_layers=NUM_LAYERS,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
)
loss_fn = SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer="adam",
    loss=loss_fn,
)

# Обучение модели
print("Начинаем обучение...")
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

# Сохраняем модель (веса) и конфигурацию
model.save_weights("./saved/gpt.weights.h5")
config = {
    "vocab_size": vocab_size,
    "maxlen": MAXLEN,
    "num_layers": NUM_LAYERS,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "ff_dim": FF_DIM,
}
with open("./saved/gpt_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f)
# Сохраняем токенизатор для генерации текста
tokenizer_json = tokenizer.to_json()
with open("./saved/tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_json)

print("Модель обучена и сохранена. Готова к генерации текста.")
