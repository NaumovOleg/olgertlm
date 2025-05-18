from data import load_data
from model import ModelFactory
from callbacks import HFPushCallback
from config import Config


X, y, tokenizer, vocab_size = load_data(Config.DATA_PATH, maxlen=Config.MAXLEN)

model_factory = ModelFactory(vocab_size)
model_factory.load()
model_factory.compile()

print("Start fitting------------------>...")
model_factory.model.fit(
    X,
    y,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=[HFPushCallback()],
    initial_epoch=model_factory.last_epoch,
)

model_factory.model.save("./saved/gpt.weights.keras")

print("Модель обучена и сохранена. Готова к генерации текста.")
