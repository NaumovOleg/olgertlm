from src.data import load_data
from src.model import ModelFactory, HFPushCallback
from config import Config
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU device name:", tf.test.gpu_device_name())
print("Available devices:", tf.config.list_physical_devices())

X, y, tokenizer, vocab_size = load_data(Config.DATA_PATH, maxlen=Config.MAXLEN)

model_factory = ModelFactory(vocab_size)
model_factory.load()
model_factory.compile()

print("Start fitting------------------>...")
history = model_factory.model.fit(
    X,
    y,
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=[HFPushCallback()],
    initial_epoch=model_factory.last_epoch,
    verbose=1,
)


model_factory.model.save(f"{Config.SAVED_MODEL_DIR}/gpt.model.keras")

print("Model saved --------------------------->")
