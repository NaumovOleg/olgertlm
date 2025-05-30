from config import Config
import keras
from src.data import CustomTokenizer, create_dataset, load_and_preprocess_text
from src.model import GPT
from src.model.utils import CustomSchedule, loss_function, TextGenerator
import tensorflow as tf

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
optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])

configs = {
    "raw token length": len(text),
    "token numbers": num_tokens,
    "text shape": tokenized_text.shape,
    "tokenized text lenght": len(tokenized_text),
    "total sequences": total_sequences,
    "steps per epoch": steps_per_epoch,
    "vocab size": vocab_size,
}

print(configs)

print(Config.BATCH_SIZE, Config.SEQ_LENGTH)

history = model.fit(
    dataset.repeat(), epochs=Config.EPOCHS, steps_per_epoch=steps_per_epoch
)
model.save(f"{Config.SAVED_MODEL_DIR}/model.keras")

# dummy_input = tf.zeros(
#     (1, Config.SEQ_LENGTH)
# )  # Create a dummy batch with correct shape
# model(dummy_input, training=False)
# model.load_weights(f"{Config.SAVED_MODEL_DIR}/model.keras")

generator = TextGenerator(model, tokenizer, temperature=0.7)
print(generator.generate_text("I should tell", num_generate=100))

test_prompts = [
    "I should tell ",
    "Where senators shall mingle tears ",
    "Once upon a time ",
    "Hello ",
]

for temp in [0.5, 0.7, 1.0]:
    print(f"\nTemperature: {temp}")
    generator = TextGenerator(model, tokenizer, temperature=temp)
    for prompt in test_prompts:
        generated_text = generator.generate_text(prompt, num_generate=50)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated_text}'\n")
