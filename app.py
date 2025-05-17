import gradio as gr
import subprocess


def run_training():
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    return result.stdout + "\n" + result.stderr


with gr.Blocks() as demo:
    gr.Markdown("## Обучение GPT при старте Space")
    status = gr.Textbox(label="Лог обучения", lines=20)
    demo.load(run_training, outputs=status)

if __name__ == "__main__":
    demo.launch()
