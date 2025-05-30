import tensorflow as tf
import keras

SparseCategoricalCrossentropy = keras.losses.SparseCategoricalCrossentropy


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_obj = SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = loss_obj(real, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_sum(loss) / tf.reduce_sum(mask)
