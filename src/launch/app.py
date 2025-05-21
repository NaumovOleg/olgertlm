import gradio as gr
import subprocess
import os


def stream_logs():
    with open("training.log", "r", encoding="utf-8") as f:
        logs = ""
        for line in f:
            logs += line
            yield logs


def run_training():
    process = subprocess.Popen(
        ["python", "train.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    logs = ""
    for line in process.stdout:
        logs += line
        yield logs  # stream training logs

    process.stdout.close()
    process.wait()
    yield logs  # final output


def get_artifacts():
    return ["saved/gpt.weights.h5", "saved/gpt_config.json", "saved/tokenizer.json"]


with gr.Blocks() as demo:
    gr.Markdown("## 🚀 Train GPT Model and Download Files")

    logs_box = gr.Textbox(label="Training Logs", lines=20)
    train_btn = gr.Button("Start Training")

    gr.Markdown("### 📦 Download Trained Files")
    file_download = gr.File(
        label="Download Trained Files", interactive=True, file_types=[".json", ".h5"]
    )
    download_btn = gr.Button("Get Files")

    train_btn.click(fn=run_training, outputs=logs_box)
    download_btn.click(fn=get_artifacts, outputs=file_download)

demo.launch()
