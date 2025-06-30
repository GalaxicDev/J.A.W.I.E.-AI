# main.py

from wakeWordEngine import OpenWakewordEngine
from transcriber import Transcriber
from jawieVoice import OrionVoice

import os
os.add_dll_directory(r"C:\Program Files\NVIDIA\CUDNN\v9.10\bin\12.9")

print("[MAIN] Booting Jowie Voice Assistant...")
transcriber = Transcriber()
OrionVoice = OrionVoice()

# Optional: Initial greeting
OrionVoice.speak("Hello, I am Jowie. Please let me know if I can assist you with anything.")

def on_wakeword_detected():
    print("[MAIN] Wakeword detected! Activating transcriber...")
    try:
        audio = transcriber.record_audio(duration=5.0)  # Record 5 seconds of audio
        transcription = transcriber.transcribe(audio)
        print(f"[MAIN] Transcription: {transcription}")
        OrionVoice.speak(f"You said: {transcription}")
    except Exception as e:
        print(f"[MAIN] Error during transcription: {e}")

# Start the always-on assistant
listener = OpenWakewordEngine(on_wakeword=on_wakeword_detected)
listener.start()