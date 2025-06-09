import collections
import io
from medium_assist_commands import process_command
import numpy as np
import os
# import pyttsx3
import pyaudio
import requests
import sounddevice as sd
import soundfile as sf
import tempfile
import threading
import time
from silero_vad import load_silero_vad
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import torch
torch.set_num_threads(1)
import wave
import whisper

# Configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
NUM_SAMPLES = 512
CONFIDENCE_THRESHOLD = 0.5
GRACE_SECONDS = 1.5
MIN_FRAMES_BEFORE_CHECK = int(GRACE_SECONDS / (NUM_SAMPLES / SAMPLE_RATE))
MIN_FRAMES_BEFORE_TRANSCRIBE = int(GRACE_SECONDS / (NUM_SAMPLES / SAMPLE_RATE))
# OLLAMA_URL = "https://5dfd-183-171-102-196.ngrok-free.app/api/generate"
OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "deepseek-r1:8b"
OLLAMA_MODEL = "llama3.2:1b"

audio = pyaudio.PyAudio()
sileroModel = load_silero_vad()
whisperModel = whisper.load_model("base")  # whisper CPU-only
# if using pyttsx3
# engine = pyttsx3.init()
last_audio_data = []

# GUI
root = tk.Tk()
root.title("🧠 Talk To Me")
root.geometry("400x400")
style = ttk.Style()
style.theme_use('default')

style.configure("green.Horizontal.TProgressbar", troughcolor='gray', background='green')
style.configure("yellow.Horizontal.TProgressbar", troughcolor='gray', background='orange')
style.configure("red.Horizontal.TProgressbar", troughcolor='gray', background='red')

status_label = tk.Label(root, text="Ready", font=("Arial", 12))
status_label.pack(pady=10)

transcribe_label = tk.Label(root, text="", font=("Arial", 12))
transcribe_label.pack(pady=10)

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

def update_transcribed_text(text):
    root.after(0, lambda: transcribe_label.config(text=text))

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    return sound.squeeze()

# Functions
def speak(text):
# if you are using pyttsx3
    # try:
    #     engine.say(text)
    #     engine.runAndWait()
    # except Exception as e:
    #     update_status(f"TTS Error: {e}")
# if you are using coqui tts
    try:
        response = requests.get(
            f"http://localhost:5002/api/tts?text={text}&speaker_id=p228&style_wav=&language_id=",
            headers={"accept": "audio/wav"},
        )
        response.raise_for_status()
        # audio_data = np.frombuffer(response.content, dtype=np.int16)
        # play_stream = audio.open(format=FORMAT,
        #                        channels=CHANNELS,
        #                        rate=SAMPLE_RATE,
        #                        output=True)
        with io.BytesIO(response.content) as buffer:
            with wave.open(buffer, 'rb') as wf:
                channels = wf.getnchannels()
                rate = wf.getframerate()
                sampwidth = wf.getsampwidth()
                audio_data = wf.readframes(wf.getnframes())

        play_stream = audio.open(format=audio.get_format_from_width(sampwidth),
                                channels=channels,
                                rate=rate,
                                output=True)
        play_stream.write(audio_data)
        play_stream.stop_stream()
        play_stream.close()
    except Exception as e:
        update_status(f"❌ TTS error: {e}")

def update_meters(confidence, volume_rms):
    def set_vad_style(c):
        if c > 0.75:
            vad_bar.config(style="green.Horizontal.TProgressbar")
        elif c > 0.5:
            vad_bar.config(style="yellow.Horizontal.TProgressbar")
        else:
            vad_bar.config(style="red.Horizontal.TProgressbar")

    root.after(0, lambda: confidence_label.config(text=f"VAD Confidence: {confidence:.2f}"))
    root.after(0, lambda c=confidence: vad_bar.config(value=c))
    set_vad_style(confidence)

    db_val = min(100, 20 * np.log10(volume_rms + 1e-6) + 60)
    root.after(0, lambda: volume_label.config(text=f"Volume: {db_val:.2f} dB"))
    root.after(0, lambda v=db_val: volume_bar.config(value=v))

def listen():
    global is_listening
    is_listening = True
    last_audio_data.clear()
    start_spinner()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    update_status("🎙 Listening...")
    print("✅ Started Listening")

    frame_count = 0
    voiced_confidences = []
    low_confidence_streak = 0
    confidence = 0
    db_val = 0

    try:
        while is_listening:
            audio_chunk = stream.read(NUM_SAMPLES, exception_on_overflow=False)
            last_audio_data.append(audio_chunk)
            audio_int16 = np.frombuffer(audio_chunk, np.int16)
            audio_float32 = int2float(audio_int16)
            confidence = sileroModel(torch.from_numpy(audio_float32), SAMPLE_RATE).item()
            voiced_confidences.append(confidence)
            volume_rms = np.sqrt(np.mean(audio_float32**2))
            db_val = min(100, 20 * np.log10(volume_rms + 1e-6) + 60)
            update_meters(confidence, volume_rms)
            print(f"[Frame {frame_count}] Confidence: {confidence:.2f}")
            frame_count += 1

            if frame_count >= MIN_FRAMES_BEFORE_CHECK:
                if confidence < CONFIDENCE_THRESHOLD:
                    low_confidence_streak += 1
                else:
                    low_confidence_streak = 0

                if low_confidence_streak >= MIN_FRAMES_BEFORE_TRANSCRIBE:
                    print(f"🛑 Confidence < {CONFIDENCE_THRESHOLD} for {GRACE_SECONDS}s, stopping...")
                    break

    except Exception as e:
        update_status(f"⚠️ Recording error: {e}")

    finally:
        stop_spinner()
        print("✅ Stopped Listening")
        stream.stop_stream()
        stream.close()
        is_listening = False
        confidence = 0
        db_val = 0
        root.after(0, lambda c=confidence: vad_bar.config(value=c))
        root.after(0, lambda v=db_val: volume_bar.config(value=v))
        root.after(0, lambda: confidence_label.config(text="VAD Confidence: 0.00"))
        root.after(0, lambda: volume_label.config(text="Volume: 0.00 dB"))
        root.after(0, lambda: listen_button.config(state=tk.NORMAL))
    
    if last_audio_data:
        toTranscribe = b''.join(last_audio_data)
        audio_np = np.frombuffer(toTranscribe, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            update_status("🧠 Transcribing...")
            transcribeResult = whisperModel.transcribe(audio_np, language="en")
            transcribeText = transcribeResult["text"].strip()
            print("Recognized speech:", transcribeText)
            update_transcribed_text(f"📝 {transcribeText}")
            action_response = process_command(transcribeText)
            if not action_response:
                speak('No action set for this command yet.')
                # action_response = ask_llama(transcribeText)
                return
            speak(action_response)
        except Exception as e:
            update_status(f"❌ Transcription error: {e}")
        finally:
            root.after(0, lambda: playback_button.config(state=tk.NORMAL))    

def ask_llama(prompt):
    update_status("🧠 Thinking...")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
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

def on_start_listening():
    listen_button.config(state=tk.DISABLED)
    threading.Thread(target=listen, daemon=True).start()

def playback():
    if not last_audio_data:
        speak("No audio captured yet.")
        return
    try:
        playback_button.config(state=tk.DISABLED)
        start_spinner()
        play_stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        output=True)
        for frame in last_audio_data:
            play_stream.write(frame)
        play_stream.stop_stream()
        play_stream.close()
        update_status("✅ Playback completed")
    except Exception as e:
        update_status(f"⚠️ Playback error: {e}")
    finally:
        root.after(0, lambda: playback_button.config(state=tk.NORMAL))
        stop_spinner()

def on_start_playback():
    playback_button.config(state=tk.DISABLED)
    threading.Thread(target=playback, daemon=True).start()

def on_closing():
    try:
        update_status("👋 Exiting...")
        # speak("Goodbye!")
        # engine.stop()
    except Exception as e:
        print(f"Error during shutdown: {e}")
    finally:
        root.destroy()

def open_settings():
    settings_window = tk.Toplevel(root)
    settings_window.title("Settings")
    settings_window.geometry("300x200")

    # Model name
    tk.Label(settings_window, text="LLM Model:").pack()
    model_entry = tk.Entry(settings_window)
    model_entry.insert(0, OLLAMA_MODEL)
    model_entry.pack()

    # Grace seconds
    tk.Label(settings_window, text="Grace Seconds:").pack()
    grace_entry = tk.Entry(settings_window)
    grace_entry.insert(0, str(GRACE_SECONDS))
    grace_entry.pack()

    # Confidence threshold
    tk.Label(settings_window, text="Confidence Threshold:").pack()
    threshold_entry = tk.Entry(settings_window)
    threshold_entry.insert(0, str(CONFIDENCE_THRESHOLD))
    threshold_entry.pack()

    def save_settings():
        global OLLAMA_MODEL, GRACE_SECONDS, CONFIDENCE_THRESHOLD
        OLLAMA_MODEL = model_entry.get()
        GRACE_SECONDS = float(grace_entry.get())
        CONFIDENCE_THRESHOLD = float(threshold_entry.get())
        messagebox.showinfo("Settings", "Changes saved!")
        settings_window.destroy()

    tk.Button(settings_window, text="Save", command=save_settings).pack(pady=10)

# VAD Confidence
confidence_label = tk.Label(root, text="VAD Confidence: 0.00", font=("Arial", 10))
confidence_label.pack()
vad_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", maximum=1.0)
vad_bar.pack(pady=(0, 10))

# Volume Level
volume_label = tk.Label(root, text="Volume: 0.00 dB", font=("Arial", 10))
volume_label.pack()
volume_bar = ttk.Progressbar(root, orient="horizontal", length=200, mode="determinate", maximum=100)
volume_bar.pack(pady=(0, 10))

# Buttons
listen_button = tk.Button(root, text="🎙 Start Listening", command=on_start_listening, font=("Arial", 12))
listen_button.pack(pady=5)
playback_button = tk.Button(root, text="▶ Playback", command=on_start_playback, font=("Arial", 12))
playback_button.pack(pady=5)
tk.Button(root, text="⚙ Settings", command=open_settings, font=("Arial", 12)).pack(pady=5)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
