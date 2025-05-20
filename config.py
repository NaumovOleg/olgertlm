from dotenv import load_dotenv
import os

load_dotenv()


class Config:
    DATA_PATH = os.getenv("DATA_PATH", "datasets/big.txt")
    MAXLEN = int(os.getenv("MAXLEN", "20"))
    NUM_LAYERS = int(os.getenv("NUM_LAYERS", "1"))
    EMBED_DIM = int(os.getenv("EMBED_DIM", "4"))
    NUM_HEADS = int(os.getenv("NUM_HEADS", "NUM_HEADS"))
    FF_DIM = int(os.getenv("FF_DIM", "128"))
    EPOCHS = int(os.getenv("EPOCHS", "10"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
    LOCAL_CHECKPOINT_DIR = os.getenv(
        "LOCAL_CHECKPOINT_DIR", "/opt/ml/model/checkpoints"
    )
    SAVED_MODEL_PATH = os.getenv("SAVED_MODEL_PATH", "/opt/ml/model")
    SAVED_MODEL_PATH_FULL = os.getenv(
        "SAVED_MODEL_PATH_FULL", "/opt/ml/model/gpt.weights.keras"
    )
