import tempfile
import pyttsx3
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import time
import queue

THRESHOLD = 0.01
SILENCE_DURATION = 2.0
SAMPLE_RATE = 16000

model_whisper = whisper.load_model("small")
tts = pyttsx3.init()
tts.setProperty("rate", 150)


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


def listen():

    audio, fs = record_audio(duration=5)
    text = transcribe(audio, fs).strip()
    if not text:
        print("Речь не распознана, попробуйте снова.")
        return None
    else:
        return text


def record_until_silence():
    print("Говорите (запись начнется при обнаружении речи)...")
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Статус ошибки записи: {status}")
        q.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    ):
        recording = []
        speaking = False
        silence_start = None

        while True:
            try:
                block = q.get(timeout=1)
            except queue.Empty:
                continue

            volume = np.linalg.norm(block)

            if volume > THRESHOLD:
                if not speaking:
                    print("Речь обнаружена...")
                    speaking = True
                silence_start = None
                recording.append(block)
            elif speaking:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("Речь завершена.")
                    break
                recording.append(block)

    audio_np = np.concatenate(recording, axis=0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16


def transcribe_audio(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        wav.write(f.name, SAMPLE_RATE, audio)
        result = model_whisper.transcribe(f.name, language="ru")
        return result["text"].strip()
