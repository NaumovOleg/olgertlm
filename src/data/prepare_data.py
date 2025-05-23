import numpy as np
from config import Config
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
import sagemaker
from pathlib import Path

tokenizer_path = Path(f"{Config.TOKENIZER_DIR}/tokenizer.json")

if Config.IS_SAGEMAKER:

    sagemaker_session = sagemaker.Session()
    bucket = sagemaker_session.default_bucket()
    prefix = "text-generation"


def save_tokenizer(tokenizer, filename="tokenizer.json"):
    """Save tokenizer locally and to S3 if in SageMaker"""
    Config.ensure_dirs()
    local_path = Config.TOKENIZER_DIR / filename

    if not Config.IS_SAGEMAKER:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(tokenizer.to_json())
            print(f"Saved tokenizer locally to: {local_path}")

    if Config.IS_SAGEMAKER:
        s3_key = f"{prefix}/tokenizer/{filename}"
        sagemaker_session.upload_data(str(local_path), bucket, s3_key)
        print(f"Saved tokenizer to S3: s3://{bucket}/{s3_key}")


def load_data(file_path, maxlen):
    """Reads data from a file and prepares it for training."""
    print("Loading text from:", file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    text = " ".join([line.strip() for line in lines]).lower()

    print("+++++++++++++++", tokenizer_path)

    if tokenizer_path.exists():
        with tokenizer_path.open("r", encoding="utf-8") as f:
            tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
        print("--------------> Loading tokenizer from file")
    else:
        print("--------------> Creating new tokenizer from text")
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([text])
        save_tokenizer(tokenizer)

    sequence = tokenizer.texts_to_sequences([text])[0]
    vocab_size = len(tokenizer.word_index) + 1
    X, y = [], []
    for i in range(len(sequence) - maxlen):
        X.append(sequence[i : i + maxlen])
        y.append(sequence[i + maxlen])
    X = np.array(X)
    y = np.array(y)
    print(
        f"Token sequences: {len(sequence)}, samples: {len(X)}, vocab_size: {vocab_size}"
    )

    return X, y, tokenizer, vocab_size
