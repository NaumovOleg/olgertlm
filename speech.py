import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import pyttsx3
import whisper
import requests
import json
from gtts import gTTS
import playsound


OLLAMA_API_URL = (
    "http://localhost:11434/v1/chat/completions"  # стандартный локальный порт Ollama
)
MODEL_NAME = "openchat"

# Инициализация моделей
print("Загружаем Whisper для распознавания речи...")
model_whisper = whisper.load_model("small")  # или small/medium

tts = pyttsx3.init()
tts.setProperty("rate", 150)


def speak(text):
    print("Бот:", text)
    tts = gTTS(text=text, lang="ru")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
        tts.save(fp.name)
        playsound.playsound(fp.name)


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


def query_ollama(user_message) -> str:

    url = "http://localhost:11434/v1/chat/completions"
    user_message.insert(
        0,
        {
            "role": "system",
            "content": "Прими роль человека собеседника по имени Яванна. Никогда  не  говори  что  ты искусственный  интелект. Веди себя  естественно  для  человека",
        },
    )

    payload = {
        "model": "openchat",
        "messages": user_message,
    }

    response = requests.post(url, json=payload)
    print(response.json())
    return response.json()["choices"][0]["message"]["content"]


def main():
    print(
        "Голосовой чат-бот на OpenChat (Ollama) запущен. Скажите 'выход' чтобы закончить."
    )
    history = []
    while True:
        audio, fs = record_audio(duration=5)
        text = transcribe(audio, fs).strip()
        if not text:
            print("Речь не распознана, попробуйте снова.")
            continue
        print("Вы:", text)
        if "выход" in text.lower():
            speak("До свидания!")
            break

        history.append({"role": "user", "content": text})
        response: str = query_ollama(history)
        history.append({"role": "assistant", "content": response})

        speak(response)


if __name__ == "__main__":
    main()
