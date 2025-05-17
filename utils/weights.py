import os
from config import Config
import re


os.makedirs(Config.LOCAL_CHECKPOINT_DIR, exist_ok=True)


def find_latest_checkpoint():
    files = os.listdir(Config.LOCAL_CHECKPOINT_DIR)
    checkpoint_files = [f for f in files if re.match(r"model_epoch_(\d+)\.keras", f)]
    if not checkpoint_files:
        return None, 0
    checkpoint_files.sort(key=lambda f: int(re.search(r"(\d+)", f).group(1)))
    last_file = checkpoint_files[-1]
    last_epoch = int(re.search(r"(\d+)", last_file).group(1))
    return os.path.join(Config.LOCAL_CHECKPOINT_DIR, last_file), last_epoch
