from utils import find_latest_checkpoint
import keras
from config import Config
from .gpt_model import GPTModel


SparseCategoricalCrossentropy = keras.losses.SparseCategoricalCrossentropy
load_model = keras.models.load_model


class ModelFactory:

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
            self.model = load_model(
                self.checkpoint_path, custom_objects={"GPTModel": GPTModel}
            )
        else:
            print("[INFO] Checkpoint not  found.")
            self.model = GPTModel(
                vocab_size=self.vocab_size,
                maxlen=Config.MAXLEN,
                num_layers=Config.NUM_LAYERS,
                embed_dim=Config.EMBED_DIM,
                num_heads=Config.NUM_HEADS,
                ff_dim=Config.FF_DIM,
            )
