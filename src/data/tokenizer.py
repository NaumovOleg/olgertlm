import keras
import os

TextVectorization = keras.layers.TextVectorization


class CustomTokenizer(TextVectorization):
    """CustomTokenizer"""

    def __init__(self, vocab_path, text, vocab_size=20000, sequence_length=None):
        super().__init__(
            max_tokens=vocab_size,
            output_mode="int",
            output_sequence_length=sequence_length,
        )
        self.vocab_path = f"{vocab_path}/vocab.txt"
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.text = text
        if os.path.exists(self.vocab_path):
            print(f"Loading from file:{self.vocab_path}")
            self.load_vocab(self.vocab_path)
        else:
            print(f"saving to file:{self.vocab_path}")
            self.adapt([text])
            self.save_vocab(self.vocab_path)

    def load_vocab(self, vocab_file_path):
        """load_vocab"""
        if not os.path.exists(vocab_file_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_file_path}")
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f if line.strip()]
        self.set_vocabulary(vocab)

    def save_vocab(self, vocab_file_path):
        """save_vocab"""
        vocab = self.get_vocabulary()
        with open(vocab_file_path, "w", encoding="utf-8") as f:
            for token in vocab:
                f.write(token + "\n")

    def get_vocab_size(self):
        return len(self.get_vocabulary())
