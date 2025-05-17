from data import load_data
from model import ModelFactory
from callbacks import HFPushCallback
from config import Config


X, y, tokenizer, vocab_size = load_data(Config.DATA_PATH, maxlen=Config.MAXLEN)

orcestrate = ModelFactory(vocab_size)
orcestrate.load()
orcestrate.compile()


print("Start fitting------------------>...")
orcestrate.model.fit(
    X,
    y,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=[HFPushCallback()],
    initial_epoch=orcestrate.last_epoch,
)

orcestrate.model.save("./saved/gpt.weights.keras")

print("Модель обучена и сохранена. Готова к генерации текста.")
