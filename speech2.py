import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import pyttsx3
import whisper
import torch
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Загрузка моделей
print("Загружаем Whisper для распознавания речи...")
model_whisper = whisper.load_model("base")  # можно "small", "medium", если есть ресурсы

print("Загружаем BlenderBot 3...")
tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-3B")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-3B")

# Инициализация TTS
tts = pyttsx3.init()
tts.setProperty("rate", 150)


def speak(text):
    print("Бот:", text)
    tts.say(text)
    tts.runAndWait()


def record_audio(duration=5, fs=16000):
    print("Говорите...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()
    return audio.flatten(), fs


def transcribe(audio, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        wav.write(f.name, fs, audio)
        result = model_whisper.transcribe(f.name, language="ru")
        return result["text"]


def generate_response(prompt, history=""):
    # Добавим историю, чтобы бот "помнил" контекст
    full_input = history + f" Пользователь: {prompt}\nБот:"
    inputs = tokenizer([full_input], return_tensors="pt")
    reply_ids = model.generate(**inputs, max_length=256)
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]

    # Обрезаем префикс из полного ввода
    response = response.replace(full_input, "").strip()
    return response, full_input + " " + response + "\n"


def main():
    print("Голосовой чат-бот с BlenderBot 3 запущен. Скажите 'выход' для завершения.")
    history = ""
    while True:
        audio, fs = record_audio(duration=5)
        text = transcribe(audio, fs)
        text = text.strip()
        if not text:
            print("Речь не распознана, попробуйте снова.")
            continue
        print("Вы:", text)
        if "выход" in text.lower():
            speak("До свидания!")
            break

        response, history = generate_response(text, history)
        speak(response)


if __name__ == "__main__":
    main()
