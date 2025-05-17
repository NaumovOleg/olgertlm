import gradio as gr
import subprocess


def run_training_live():
    process = subprocess.Popen(
        ["python", "train.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    logs = ""
    for line in process.stdout:
        logs += line
        yield logs  # обновляем логи постепенно

    process.stdout.close()
    process.wait()
    yield logs  # финальный вывод после завершения


with gr.Blocks() as demo:
    gr.Markdown("## Обучение GPT с логами в реальном времени")
    status = gr.Textbox(label="Лог обучения", lines=20)

    # Запускаем обучение и выводим логи при загрузке страницы
    demo.load(run_training_live, outputs=status)

# if __name__ == "__main__":
#     demo.launch()
