from src.speech import (
    speak,
    generate_response,
    record_until_silence,
    transcribe_audio,
)


def main():
    while True:
        audio = record_until_silence()
        text = transcribe_audio(audio)
        if not text:
            print("Речь не распознана, попробуйте снова.")
            continue
        print("Вы:", text)
        if "выход" in text.lower():
            speak("До свидания!")
            break

        response = generate_response(text)

        speak(text=response)


if __name__ == "__main__":
    main()
