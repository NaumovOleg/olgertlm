from config import Config
import keras
from src.data import CustomTokenizer, create_dataset, load_and_preprocess_text
from src.model import GPT
from src.model.utils import CustomSchedule, loss_function, TextGenerator
import os

Adam = keras.optimizers.Adam

text = load_and_preprocess_text(Config.DATASET_PATH)
tokenizer = CustomTokenizer(Config.VOCAB_PATH, text, vocab_size=20000)
tokenized_text = tokenizer([text])[0]
num_tokens = len(tokenized_text)
total_sequences = len(tokenized_text) - Config.SEQ_LENGTH - 1
steps_per_epoch = max(1, total_sequences // Config.BATCH_SIZE)

dataset = create_dataset(text, tokenizer, Config.SEQ_LENGTH, Config.BATCH_SIZE)
vocab_size = tokenizer.get_vocab_size()


model = GPT(
    vocab_size=vocab_size,
    max_len=Config.MAX_LEN,
    d_model=Config.D_MODEL,
    num_heads=Config.NUM_HEADS,
    dff=Config.DFF,
    num_layers=Config.NUM_LAYERS,
    rate=Config.DROPOUT_RATE,
)


learning_rate = CustomSchedule(Config.D_MODEL)
optimizer = Adam(0.01, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss=loss_function)


history = model.fit(dataset.repeat(), verbose=1, epochs=Config.EPOCHS)

configs = {
    "raw token length": len(text),
    "first 100": text[:100],
    "token numbers": num_tokens,
    "text shape": tokenized_text.shape,
    "first 10 tokens": tokenized_text[:10],
    "tokenized text lenght": len(tokenized_text),
    "total sequences": total_sequences,
    "steps per epoch": steps_per_epoch,
    "vocab size": vocab_size,
}

print(configs)


generator = TextGenerator(model, tokenizer, temperature=0.7)
print(generator.generate_text("the meaning of life is", num_generate=100))
