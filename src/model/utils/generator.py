import tensorflow as tf


class TextGenerator:
    def __init__(self, model, tokenizer, temperature=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

    def generate_text(self, start_string, num_generate=100):
        input_indices = self.tokenizer([start_string])[0]
        input_indices = tf.expand_dims(input_indices, 0)

        text_generated = []

        for _ in range(num_generate):
            predictions = self.model(input_indices, training=False)
            predictions = predictions[:, -1, :] / self.temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)

            text_generated.append(predicted_id.numpy()[0][0])
            input_indices = tf.concat([input_indices, predicted_id], axis=-1)
            input_indices = input_indices[:, -100:]  # Ограничиваем длину контекста

        generated_text = self.tokenizer.sequences_to_texts([text_generated])[0]
        return start_string + generated_text
