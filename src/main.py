import pyaudio
import numpy as np
from pynput import keyboard
from transformers import pipeline

def main():
    # Initialize the Whisper model
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base.en",
        device=-1  # Change to 0 for GPU acceleration if available
    )

    # Audio configuration
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 4096

    # Recording control variables
    is_recording = False
    audio_frames = []

    # Initialize PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Keyboard event handlers
    def on_press(key):
        nonlocal is_recording, audio_frames
        if key == keyboard.Key.space and not is_recording:
            audio_frames = []
            is_recording = True
            print("Recording... (hold spacebar)")

    def on_release(key):
        nonlocal is_recording
        if key == keyboard.Key.space and is_recording:
            is_recording = False
            print("Processing...")
            
            # Convert and normalize audio data
            audio_array = np.frombuffer(b''.join(audio_frames), dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0

            # Perform speech recognition
            result = asr_pipeline(
                {"raw": audio_array, "sampling_rate": RATE},
                generate_kwargs={"task": "transcribe"}
            )
            
            print(f"\nTranscription: {result['text']}\n")
        return True

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        print("Press and hold SPACE to record, release to transcribe. Press Ctrl+C to exit.")
        while True:
            if is_recording:
                # Continuously read audio data while recording
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_frames.append(data)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Cleanup resources
        stream.stop_stream()
        stream.close()
        p.terminate()
        listener.stop()

if __name__ == "__main__":
    main()