import os
import keras
from config import Config


Callback = keras.callbacks.Callback
Sequential = keras.Sequential


class HFPushCallback(Callback):

    def __init__(self, save_dir=Config.CHECKPOINT_DIR):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        filename = f"model_epoch_{epoch+1}.keras"
        filepath = os.path.join(self.save_dir, filename)
        self.model.save(filepath)
        print(f"[INFO] Saved local checkpoint: {filepath}")
