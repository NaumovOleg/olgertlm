import os
from huggingface_hub import Repository, HfApi, login
import keras

Callback = keras.callbacks.Callback
Sequential = keras.Sequential

HF_TOKEN = os.getenv("API_KEY") or ""
DATASET_REPO_NAME = "keras-checkpoints-example"
LOCAL_CHECKPOINT_DIR = "./checkpoints"

login(token=HF_TOKEN)

api = HfApi()
repo_url = api.create_repo(
    name=DATASET_REPO_NAME, token=HF_TOKEN, repo_type="dataset", exist_ok=True
)

repo = Repository(
    local_dir=LOCAL_CHECKPOINT_DIR,
    clone_from=repo_url,
    repo_type="dataset",
    token=HF_TOKEN,
)


class HFPushCallback(Callback):
    def __init__(self):
        self.repo = repo
        os.makedirs(repo.local_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        filename = f"model_epoch_{epoch+1}.h5"
        filepath = os.path.join(self.repo.local_dir, filename)
        self.model.save(filepath)
        print(f"[INFO] Сохраняю чекпоинт: {filepath}")
        self.repo.git_add(auto_lfs_track=True)
        self.repo.git_commit(f"Epoch {epoch+1} checkpoint")
        self.repo.git_push()
        print("[INFO] Чекпоинт отправлен на Hugging Face Hub ✅")
