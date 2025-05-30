from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    DATASET_PATH = os.getenv("DATASET_PATH", "./datasets/sample_data.txt")
    VOCAB_PATH = os.getenv("VOCAB_PATH", "./artifacts/tokenizer")
    VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "20000"))
    MAX_LEN = int(os.getenv("MAX_LEN", "512"))
    D_MODEL = int(os.getenv("D_MODEL", "256"))
    NUM_HEADS = int(os.getenv("NUM_HEADS", "8"))
    DFF = int(os.getenv("DFF", "512"))
    NUM_LAYERS = int(os.getenv("NUM_LAYERS", "6"))
    DROPOUT_RATE = float(os.getenv("DROPOUT_RATE", "0.1"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
    SEQ_LENGTH = int(os.getenv("SEQ_LENGTH", "100"))
    EPOCHS = int(os.getenv("EPOCHS", "10"))
