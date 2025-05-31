import tempfile
from gtts import gTTS
import playsound


def speak(text):
    print("Бот:", text)
    tts = gTTS(text=text, lang="ru")
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
        tts.save(fp.name)
        playsound.playsound(fp.name)
