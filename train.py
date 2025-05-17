import json
from data import load_data
from model import GPTModel
from callbacks import HFPushCallback
import keras
import os
import re
import tensorflow as tf

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
DATASET_REPO_NAME = "keras-checkpoints-auto"
LOCAL_CHECKPOINT_DIR = "./checkpoints"

# Загрузка и подготовка данных
X, y, tokenizer, vocab_size = load_data(DATA_PATH, maxlen=MAXLEN)
print("Пример преобразованной последовательности:", X[0], "->", y[0])

os.makedirs(LOCAL_CHECKPOINT_DIR, exist_ok=True)


def find_latest_checkpoint():
    files = os.listdir(LOCAL_CHECKPOINT_DIR)
    checkpoint_files = [f for f in files if re.match(r"model_epoch_(\d+)\.h5", f)]
    if not checkpoint_files:
        return None, 0
    checkpoint_files.sort(key=lambda f: int(re.search(r"(\d+)", f).group(1)))
    last_file = checkpoint_files[-1]
    last_epoch = int(re.search(r"(\d+)", last_file).group(1))
    return os.path.join(LOCAL_CHECKPOINT_DIR, last_file), last_epoch


checkpoint_path, last_epoch = find_latest_checkpoint()

if checkpoint_path:
    print(f"[INFO] Загружаю модель из чекпоинта: {checkpoint_path}")
    model = tf.keras.models.load_model(checkpoint_path)
else:
    print("[INFO] Чекпоинт не найден. Начинаю с нуля.")
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


print("Начинаем обучение...")
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[HFPushCallback()])

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
