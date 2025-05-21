import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from config import Config

# Простейшая модель
model = keras.Sequential([layers.Dense(1, input_shape=(1,))])
model.compile(optimizer="sgd", loss="mean_squared_error")

# Примерные данные
x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
y = tf.constant([[2.0], [4.0], [6.0], [8.0]])

# Обучение
model.fit(x, y, epochs=5)

# Сохранение модели
save_path = f"{Config.SAVED_MODEL_DIR}/model.keras"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)

print(f"Model saved to {save_path}")
