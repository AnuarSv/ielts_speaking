import torch
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
import gradio as gr
from gemini import ask_gemini

# Проверка GPU
print("GPU available:", torch.cuda.is_available())
print("Current device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Загружаем модели
stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

def echo(audio):
    # 1️⃣ Распознаем речь
    transcript = stt_model.stt(audio)
    print(f"[STT] {transcript}")

    # 2️⃣ Ответ от Gemini
    response_text = ask_gemini(transcript)
    print(f"[Gemini] {response_text}")

    # 3️⃣ Синтез речи — потоковый режим
    for chunk in tts_model.stream_tts_sync(response_text):
        # Просто передаем chunk напрямую - fastrtc обработает его сам
        yield chunk

    # 5️⃣ Можно добавить текст для UI
    # yield response_text


# Создаем стрим с режимом "send-receive"
stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")

# Gradio UI
with gr.Blocks() as demo:
    stream.ui.render()

# Запуск
demo.launch()
