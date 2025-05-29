import queue
import sounddevice as sd
import soundfile as sf
import numpy as np  # make sure this is imported at the top
import vosk
import sys
import json
import pyttsx3
import requests
import threading
import tkinter as tk
import webbrowser
import datetime
import tempfile
import os

# CONFIG
VOSK_MODEL_PATH = "vosk-model-en-us-0.22-lgraph"
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# INIT
model = vosk.Model(VOSK_MODEL_PATH)
q = queue.Queue()
engine = pyttsx3.init()
last_transcription = ""
last_audio_data = bytearray()

# SETUP VOICE
def configure_voice():
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    voices = engine.getProperty('voices')
    if voices:
        engine.setProperty('voice', voices[0].id)

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def capture_speech():
    global last_transcription, last_audio_data
    update_status("🎤 Listening...")
    try:
        last_audio_data = bytearray()
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            rec = vosk.KaldiRecognizer(model, 16000)
            last_transcription = ""
            while True:
                try:
                    data = q.get(timeout=1.5)
                    last_audio_data.extend(data)
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        text = result.get("text", "").strip()
                        if text:
                            last_transcription = text
                            print(f"🗣 You said: {text}")
                            break
                except queue.Empty:
                    print("⌛ Timeout waiting for speech")
                    break
                except Exception as e:
                    print(f"❌ Error during speech capture: {e}")
                    break
    except Exception as e:
        print(f"❌ Microphone error: {e}")
    update_status("✅ Done")

def analyze_command(text):
    text = text.lower()
    if "open youtube" in text:
        webbrowser.open("https://youtube.com")
        return "Opening YouTube."
    elif "open google" in text:
        webbrowser.open("https://google.com")
        return "Opening Google."
    elif "what time is it" in text:
        now = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {now}."
    return None

def ask_llama(prompt):
    update_status("🧠 Thinking...")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=10)
        res.raise_for_status()
        return res.json()["response"]
    except requests.Timeout:
        return "Response took too long."
    except requests.RequestException as e:
        return f"Request error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
    finally:
        update_status("✅ Done")

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"🔊 TTS Error: {e}")

def on_process_speech():
    threading.Thread(target=capture_speech, daemon=True).start()

def on_respond_to_speech():
    if not last_transcription:
        speak("No speech captured yet.")
        return
    print(f"🔁 Processing captured speech: {last_transcription}")
    reply = analyze_command(last_transcription)
    if not reply:
        reply = ask_llama(last_transcription)
    if reply:
        print(f"🤖 LLM: {reply}")
        speak(reply)

def on_playback_audio():
    if not last_audio_data:
        speak("No audio captured yet.")
        return
    try:
        audio_np = np.frombuffer(last_audio_data, dtype=np.int16)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            file_path = f.name
            sf.write(file_path, audio_np, 16000, format='WAV', subtype='PCM_16')

        print(f"▶️ Playing back from: {file_path}")
        data, fs = sf.read(file_path, dtype='int16')
        sd.play(data, fs)
        sd.wait()
        os.remove(file_path)
    except Exception as e:
        print(f"❌ Error during playback: {e}")
        update_status("⚠️ Playback error")

# GUI
def update_status(text):
    status_label.config(text=text)

configure_voice()
window = tk.Tk()
window.title("Light Assist — GUI Assistant")
window.geometry("400x260")

btn_capture = tk.Button(window, text="🎤 Process Speech", font=("Arial", 14), command=on_process_speech)
btn_capture.pack(pady=10)

btn_respond = tk.Button(window, text="🧠 Respond to Captured Speech", font=("Arial", 14), command=on_respond_to_speech)
btn_respond.pack(pady=10)

btn_playback = tk.Button(window, text="▶️ Playback Captured Audio", font=("Arial", 14), command=on_playback_audio)
btn_playback.pack(pady=10)

status_label = tk.Label(window, text="Ready", font=("Arial", 12), fg="green")
status_label.pack(pady=10)

window.mainloop()
