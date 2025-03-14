import keyboard
import numpy as np
import pyaudio
import torch

from transformers import pipeline


def on_press(event):
    global audio_frames, listening
    if not listening:
        audio_frames = []
        listening = True
        print("->> Listening...")
        

def on_release(event):
    global audio_frames, listening
    if listening:
        listening = False
        print("<<- Assessing...")
        data = b''.join(audio_frames)
            
        audio_data = np.frombuffer(b''.join(audio_frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        
        # Transcribe using Whisper
        result = asr_pipeline(
            {"array": audio_data, "sampling_rate": RATE}
        )

        print(f"Transcription: {result['text']}")
        
        do_things(result['text'])
    return True


def do_things(text):
    if "hello" in text.lower():
        print("Hello, sir. I already hate this world.") # TODO: make tts speak it

def audio_callback(in_data, frame_count, time_info, status):
    global audio_frames, listening
    if listening:
        audio_frames.append(in_data)
    return (None, pyaudio.paContinue)


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 4096

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base.en",
    device_map="auto"
)

listening = False
audio_frames = []

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)

print("[*] Avatar online")

keyboard.on_press_key('right shift', on_press)
keyboard.on_release_key('right shift', on_release)
keyboard.wait()