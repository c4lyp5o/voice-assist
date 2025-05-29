import collections
from medium_assist_commands import process_command
import numpy as np
import os
# import pyttsx3
# import re
import requests
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import sounddevice as sd
import soundfile as sf
import tempfile
import threading
import time
import tkinter as tk
from tkinter import messagebox
import torch
# import webrtcvad
# import webbrowser
import whisper

# Configuration
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
CHANNELS = 1
# VAD_MODE = 3

# vad = webrtcvad.Vad(VAD_MODE)
model = load_silero_vad()
# engine = pyttsx3.init()
model = whisper.load_model("base")  # whisper CPU-only
last_audio_data = bytes()

# GUI
root = tk.Tk()
root.title("🧠 Talk To Me")
root.geometry("400x300")

status_label = tk.Label(root, text="Ready", font=("Arial", 12))
status_label.pack(pady=10)

spinner = tk.Label(root, text="", font=("Arial", 20))
spinner.pack()

spinner_running = False

def start_spinner():
    global spinner_running
    spinner_running = True
    spinner_chars = ["⏳", "🔄", "🔁", "🔃"]
    def animate(i=0):
        if spinner_running:
            spinner.config(text=spinner_chars[i % len(spinner_chars)])
            root.after(200, animate, i+1)
        else:
            spinner.config(text="")
    animate()

def stop_spinner():
    global spinner_running
    spinner_running = False

def update_status(text):
    root.after(0, lambda: status_label.config(text=text))

# Functions
# if you are using pyttsx3
# def speak(text):
#     try:
#         engine.say(text)
#         engine.runAndWait()
#     except Exception as e:
#         update_status(f"Error: {e}")
# if you are using coqui tts
def speak(text):
    try:
        response = requests.get(
            f"http://localhost:5002/api/tts?text={text}&speaker_id=&style_wav=&language_id=",
            headers={"accept": "audio/wav"},
        )
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(response.content)
            temp_path = f.name
        data, samplerate = sf.read(temp_path, dtype='int16')
        sd.play(data, samplerate)
        sd.wait()
        os.remove(temp_path)
    except Exception as e:
        update_status(f"❌ TTS error: {e}")

def record_voice():
    global last_audio_data
    start_spinner()
    ring_buffer = collections.deque(maxlen=10)
    speech_frames = []
    speech_started = False

    # Silero settings:
    max_silence_time = 1.0  # seconds of silence after speech to stop
    silence_window_frames = int(max_silence_time * 1000 / FRAME_DURATION_MS)
    silence_window = collections.deque(maxlen=silence_window_frames)

    def callback(indata, frames, time_info, status):
        nonlocal speech_started, speech_frames
        # Convert indata to numpy array (int16) for Silero
        audio_chunk = np.frombuffer(indata, dtype=np.int16)
        # Silero expects float32 PCM in -1..1, so convert
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        # Get speech probability from Silero
        speech_prob = silero_vad(audio_float, return_seconds=False)

        is_speech = speech_prob > 0.5

        if is_speech:
            if not speech_started:
                update_status("🎤 Speaking...")
                speech_started = True
            speech_frames.extend(ring_buffer)
            ring_buffer.clear()
            speech_frames.append(bytes(indata))
            silence_window.clear()
        elif speech_started:
            ring_buffer.append(bytes(indata))
            silence_window.append(False)
            if len(silence_window) == silence_window.maxlen:
                # If silence_window is all False, stop
                if all(not x for x in silence_window):
                    raise sd.CallbackStop()
        else:
            ring_buffer.append(bytes(indata))

    update_status("🎙 Listening...")
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                               dtype='int16', channels=CHANNELS, callback=callback):
            while True:
                time.sleep(0.1)
    except sd.CallbackStop:
        pass
    except Exception as e:
        update_status(f"⚠️ Recording error: {e}")
        return
    finally:
        stop_spinner()
        root.after(0, lambda: listen_button.config(state=tk.NORMAL))

    if speech_frames:
        last_audio_data = b''.join(speech_frames)
        update_status("✅ Speech captured")
    else:
        last_audio_data = bytes()
        update_status("❌ No speech detected")

def on_start_listening():
    listen_button.config(state=tk.DISABLED)
    threading.Thread(target=record_with_vad, daemon=True).start()

def on_playback_audio():
    def playback():        
        if not last_audio_data:
            speak("No audio captured yet.")
            return
        audio_np = np.frombuffer(last_audio_data, dtype=np.int16)
        try:
            playback_button.config(state=tk.DISABLED)
            start_spinner()
            audio_np = np.frombuffer(last_audio_data, dtype=np.int16)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                file_path = f.name
                sf.write(file_path, audio_np, SAMPLE_RATE, format='WAV', subtype='PCM_16')
            data, fs = sf.read(file_path, dtype='int16')
            sd.play(data, fs)
            sd.wait()
            os.remove(file_path)
        except Exception as e:
            update_status(f"⚠️ Playback error: {e}")
        finally:
            root.after(0, lambda: playback_button.config(state=tk.NORMAL))
            stop_spinner()
    threading.Thread(target=playback, daemon=True).start()

def on_transcribe():
    def transcribe():
        if not last_audio_data:
            speak("No audio to transcribe.")
            return
        audio_np = np.frombuffer(last_audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            transcribe_button.config(state=tk.DISABLED)
            start_spinner()
            update_status("🧠 Transcribing...")
            audio_np = np.frombuffer(last_audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            result = model.transcribe(audio_np, language="en")
            text = result["text"].strip()
            print("Recognized speech:", text)
            update_status(f"🗣 {text}")

            action_response = process_command(text)
            if action_response:
                speak(action_response)
            else:
                speak(text)
        except Exception as e:
            update_status(f"❌ Transcription error: {e}")
        finally:
            root.after(0, lambda: transcribe_button.config(state=tk.NORMAL))
            stop_spinner()
    threading.Thread(target=transcribe, daemon=True).start()

def on_closing():
    try:
        update_status("👋 Exiting...")
        speak("Goodbye!")
        # engine.stop()
    except Exception as e:
        print(f"Error during shutdown: {e}")
    finally:
        root.destroy()

# Buttons
listen_button = tk.Button(root, text="🎙 Start Listening", command=on_start_listening, font=("Arial", 12))
listen_button.pack(pady=5)
playback_button = tk.Button(root, text="▶ Playback", command=on_playback_audio, font=("Arial", 12))
playback_button.pack(pady=5)
transcribe_button = tk.Button(root, text="🧠 Transcribe", command=on_transcribe, font=("Arial", 12))
transcribe_button.pack(pady=5)


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
