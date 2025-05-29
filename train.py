from config import Config
from src.data import CustomTokenizer, create_dataset, load_and_preprocess_text

text = load_and_preprocess_text(Config.DATASET_PATH)

tokenizer = CustomTokenizer(
    Config.VOCAB_PATH, text, vocab_size=20000, sequence_length=100
)
tokenized_text = tokenizer([tokenizer.text])[0]

dataset = create_dataset(tokenizer.text, tokenizer)


print(dataset)
