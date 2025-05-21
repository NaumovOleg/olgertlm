from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    DATA_PATH = os.getenv("DATA_PATH", "datasets/big.txt")
    MAXLEN = int(os.getenv("MAXLEN", "20"))
    NUM_LAYERS = int(os.getenv("NUM_LAYERS", "4"))
    EMBED_DIM = int(os.getenv("EMBED_DIM", "64"))
    NUM_HEADS = int(os.getenv("NUM_HEADS", "4"))
    FF_DIM = int(os.getenv("FF_DIM", "128"))
    EPOCHS = int(os.getenv("EPOCHS", "500"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
    CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "/artifacts/model/checkpoints")
    SAVE_CHECKPOINT_DIR = os.getenv(
        "SAVE_CHECKPOINT_DIR", "/artifacts/model/checkpoints"
    )

    MODEL_DIR = os.getenv("MODEL_DIR", "/artifacts/model")
    SAVED_MODEL_DIR = os.getenv("SAVED_MODEL_DIR", "/artifacts/model")
    TOKENIZER_DIR = os.getenv("TOKENIZER_DIR", "/artifatcs/tokenizer")
    SAVE_TOKENIZER_DIR = os.getenv("SAVE_TOKENIZER_DIR", "/artifatcs/tokenizer")
