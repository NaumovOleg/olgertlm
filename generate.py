import json
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from model import GPTModel


config = json.load(open("./saved/gpt_config.json", "r"))
vocab_size = config["vocab_size"]
maxlen = config["maxlen"]
embed_dim = config["embed_dim"]
num_heads = config["num_heads"]
ff_dim = config["ff_dim"]
num_layers = config["num_layers"]

# Восстанавливаем токенизатор
with open("./saved/tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# Создаём модель и загружаем веса
model = GPTModel(
    vocab_size=vocab_size,
    maxlen=maxlen,
    num_layers=num_layers,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
)
model.build(input_shape=(None, maxlen))
model.load_weights("./saved/gpt.weights.h5")

# Генерация текста
prompt = "Hello how are"
print("Prompt:", prompt)
input_ids = tokenizer.texts_to_sequences([prompt.lower()])[0]
input_ids = [id for id in input_ids if id != 0]  # удаляем возможные паддинги

# Генерируем 20 токенов
for _ in range(20):
    seq = input_ids[-maxlen:]  # берём последние maxlen токенов
    seq_input = np.array([seq])
    logits = model.predict(seq_input)[0]  # (seq_len, vocab_size)
    next_id = int(
        np.argmax(logits[len(seq) - 1])
    )  # берём предсказание для последней позиции
    input_ids.append(next_id)

# Раскодируем сгенерированную последовательность
generated_text = " ".join(tokenizer.index_word.get(id, "") for id in input_ids)
print("Generated:", generated_text)
