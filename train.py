import json
from data import load_data
from model import ModlelOrcestrate
from callbacks import HFPushCallback
from config import Config


X, y, tokenizer, vocab_size = load_data(Config.DATA_PATH, maxlen=Config.MAXLEN)


orcestrate = ModlelOrcestrate(vocab_size)
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
config = {
    "vocab_size": vocab_size,
    "maxlen": Config.MAXLEN,
    "num_layers": Config.NUM_LAYERS,
    "embed_dim": Config.EMBED_DIM,
    "num_heads": Config.NUM_HEADS,
    "ff_dim": Config.FF_DIM,
}

with open("./saved/gpt_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f)
tokenizer_json = tokenizer.to_json()
with open("./saved/tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_json)

print("Модель обучена и сохранена. Готова к генерации текста.")
