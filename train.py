from config import Config
import keras
from src.data import CustomTokenizer, create_dataset, load_and_preprocess_text
from src.model import GPT
from src.model.utils import CustomSchedule, loss_function
import os

Adam = keras.optimizers.Adam

print(f"Dataset path: {Config.DATASET_PATH}")
print(f"File exists: {os.path.exists(Config.DATASET_PATH)}")

# Load and preprocess text
text = load_and_preprocess_text(Config.DATASET_PATH)
print(f"Raw text length: {len(text)}")
print(f"First 100 characters: {text[:100]}")

# Create tokenizer
tokenizer = CustomTokenizer(Config.VOCAB_PATH, text, vocab_size=20000)

# Tokenize text
tokenized_text = tokenizer([text])[0]
print(f"Tokenized text shape: {tokenized_text.shape}")
print(f"Tokenized text first 10 tokens: {tokenized_text[:10]}")

num_tokens = len(tokenized_text)
print(f"Number of tokens: {num_tokens}")

# Calculate steps
total_sequences = len(tokenized_text) - Config.SEQ_LENGTH - 1
steps_per_epoch = max(1, total_sequences // Config.BATCH_SIZE)

print(f"Text length: {len(tokenized_text)}")
print(f"Number of sequences: {total_sequences}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Batch size: {Config.BATCH_SIZE}")

# Create dataset
dataset = create_dataset(text, tokenizer, Config.SEQ_LENGTH, Config.BATCH_SIZE)
vocab_size = tokenizer.get_vocab_size()
print(f"Vocabulary size: {vocab_size}")


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
