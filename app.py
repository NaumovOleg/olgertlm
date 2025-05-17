import gradio as gr
import subprocess


def run_training():
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
        yield logs

    process.stdout.close()
    process.wait()
    yield logs


with gr.Blocks() as demo:
    gr.Markdown("## Обучение GPT с логами в реальном времени")
    status = gr.Textbox(label="Лог обучения", lines=20, interactive=False)
    start_button = gr.Button("Начать обучение")
    start_button.click(fn=run_training, outputs=status)

demo.launch()
