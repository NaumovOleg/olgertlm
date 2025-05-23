from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()


IS_SAGEMAKER = bool("IS_SAGEMAKER" in os.environ)
BASE_DIR = Path("artifacts") if not IS_SAGEMAKER else Path("/opt/ml/processing")


class Config:
    DATA_PATH = os.getenv("DATA_PATH", "datasets/big.txt")
    MAXLEN = int(os.getenv("MAXLEN", "20"))
    NUM_LAYERS = int(os.getenv("NUM_LAYERS", "4"))
    EMBED_DIM = int(os.getenv("EMBED_DIM", "64"))
    NUM_HEADS = int(os.getenv("NUM_HEADS", "4"))
    FF_DIM = int(os.getenv("FF_DIM", "128"))
    EPOCHS = int(os.getenv("EPOCHS", "500"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
    CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", "/artifacts/model/checkpoints"))
    SAVE_CHECKPOINT_DIR = Path(
        os.getenv("SAVE_CHECKPOINT_DIR", "/artifacts/model/checkpoints")
    )

    MODEL_DIR = Path(os.getenv("MODEL_DIR", "/artifacts/model"))
    SAVED_MODEL_DIR = Path(os.getenv("SAVED_MODEL_DIR", "/artifacts/model"))
    TOKENIZER_DIR = Path(os.getenv("TOKENIZER_DIR", "/artifacts/tokenizer"))
    SAVE_TOKENIZER_DIR = Path(os.getenv("SAVE_TOKENIZER_DIR", "/artifacts/tokenizer"))
    IS_SAGEMAKER = IS_SAGEMAKER
    BASE_DIR = BASE_DIR
    TOKENIZER_DIR = BASE_DIR / "tokenizer"

    @staticmethod
    def ensure_dirs():
        """Ensure all necessary directories exist"""
        os.makedirs(Config.TOKENIZER_DIR, exist_ok=True)
