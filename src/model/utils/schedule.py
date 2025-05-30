import tensorflow as tf
import keras

LearningRateSchedule = keras.optimizers.schedules.LearningRateSchedule


class CustomSchedule(LearningRateSchedule):
    """CustomSchedule"""

    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        """Required for serialization of the learning rate schedule"""
        return {"d_model": float(self.d_model), "warmup_steps": self.warmup_steps}

    @classmethod
    def from_config(cls, config):
        """Required for deserialization of the learning rate schedule"""
        return cls(**config)
