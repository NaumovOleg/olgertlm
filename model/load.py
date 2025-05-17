import tensorflow as tf
from config import Config
from .gpt_model import GPTModel
import keras

from utils import find_latest_checkpoint


SparseCategoricalCrossentropy = keras.losses.SparseCategoricalCrossentropy


print(Config.DATA_PATH)


class ModlelOrcestrate:

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        checkpoint_path, last_epoch = find_latest_checkpoint()
        self.checkpoint_path = checkpoint_path
        self.last_epoch = last_epoch

    def compile(self):
        loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer="adam", loss=loss_fn)

    def load(self):
        if self.checkpoint_path:
            print(
                f"[INFO] loading from checkoint: {self.checkpoint_path} -  {self.last_epoch}"
            )
            self.model = tf.keras.models.load_model(
                self.checkpoint_path, custom_objects={"GPTModel": GPTModel}
            )
        else:
            print("[INFO] Чекпоинт не найден. Начинаю с нуля.")
            self.model = GPTModel(
                vocab_size=self.vocab_size,
                maxlen=Config.MAXLEN,
                num_layers=Config.NUM_LAYERS,
                embed_dim=Config.EMBED_DIM,
                num_heads=Config.NUM_HEADS,
                ff_dim=Config.FF_DIM,
            )
