import numpy as np
import pyaudio
from silero_vad import load_silero_vad
import torch
torch.set_num_threads(1)
import whisper

# Config
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
NUM_SAMPLES = 512
MIN_FRAMES_BEFORE_CHECK = 47
CONFIDENCE_THRESHOLD = 0.5
GRACE_SECONDS = 1.5
GRACE_FRAMES = int(GRACE_SECONDS / (NUM_SAMPLES / SAMPLE_RATE))  # ~47 frames

# Init PyAudio and VAD and Whisper
audio = pyaudio.PyAudio()
sileroVadModel = load_silero_vad()
whisperModel = whisper.load_model("base")

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    return sound.squeeze()

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

data = []
voiced_confidences = []

print("🎙️ Started Recording")

frame_count = 0
low_confidence_streak = 0

while True:
    audio_chunk = stream.read(NUM_SAMPLES, exception_on_overflow=False)
    data.append(audio_chunk)

    audio_int16 = np.frombuffer(audio_chunk, np.int16)
    audio_float32 = int2float(audio_int16)
    confidence = sileroVadModel(torch.from_numpy(audio_float32), SAMPLE_RATE).item()
    voiced_confidences.append(confidence)

    print(f"[Frame {frame_count}] Confidence: {confidence:.2f}")

    frame_count += 1

    # After warm-up, start evaluating confidence
    if frame_count >= MIN_FRAMES_BEFORE_CHECK:
        if confidence < CONFIDENCE_THRESHOLD:
            low_confidence_streak += 1
        else:
            low_confidence_streak = 0  # reset streak

        if low_confidence_streak >= GRACE_FRAMES:
            print(f"🛑 Confidence < {CONFIDENCE_THRESHOLD} for {GRACE_SECONDS}s, stopping...")
            break

print("✅ Stopped Recording")
stream.stop_stream()
stream.close()

# ---- Playback ----
play_stream = audio.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=SAMPLE_RATE,
                         output=True)

print("🔊 Playing back...")
for frame in data:
    play_stream.write(frame)

play_stream.stop_stream()
play_stream.close()
audio.terminate()
raw_audio = b''.join(data)
audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
result = whisperModel.transcribe(audio_np, language="en")
text = result["text"].strip()
print("Recognized speech:", text)
print("🎉 Done.")
