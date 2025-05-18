from data import load_data
from model import ModelFactory
from config import Config
import numpy as np


_, _, tokenizer, vocab_size = load_data(Config.DATA_PATH, maxlen=Config.MAXLEN)

model_factory = ModelFactory(vocab_size)
model_factory.load()
model_factory.compile()


prompt = "Hello how are"
print("Prompt:", prompt)
input_ids = tokenizer.texts_to_sequences([prompt.lower()])[0]
input_ids = [id for id in input_ids if id != 0]

# Генерируем 20 токенов
for _ in range(20):
    seq = input_ids[-Config.MAXLEN :]
    seq_input = np.array([seq])
    logits = model_factory.model.predict(seq_input)[0]
    next_id = int(np.argmax(logits[len(seq) - 1]))
    input_ids.append(next_id)

generated_text = " ".join(tokenizer.index_word.get(id, "") for id in input_ids)
print("Generated:", generated_text)
