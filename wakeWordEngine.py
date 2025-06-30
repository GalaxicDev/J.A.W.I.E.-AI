import sounddevice as sd
import queue
import numpy as np
import os
import json
import sys
import time
from openwakeword.model import Model
from openwakeword.utils import download_models
from jawieVoice import OrionVoice

# Constants
SETTINGS_FILE = "settings.json"
DEFAULT_MODEL_NAME = r"C:\Users\woutv\PycharmProjects\orion\.venv\Lib\site-packages\openwakeword\resources\models\hey_jowiiee.tflite"
THRESHOLD = 0.7

# Download OpenWakeWord models if needed
download_models()

class OpenWakewordEngine:
    def __init__(self, on_wakeword=None):
        self.settings = self.load_settings()
        self.device = self.settings.get("input_device", None)
        self.fs = 16000
        self.q = queue.Queue()
        self.tts = OrionVoice()
        self.running = True
        self.on_wakeword = on_wakeword

        print("[ORION] Loading OpenWakeWord model...")
        self.model = Model(wakeword_models=[DEFAULT_MODEL_NAME])
        print("[ORION] Model loaded.")

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            return json.load(open(SETTINGS_FILE))
        return {"wakeword_enabled": True, "input_device": None}

    def callback(self, indata, frames, time_, status):
        if status:
            print(f"[VAD] Status: {status}")
        self.q.put(np.frombuffer(indata, dtype=np.int16))

    def listen_loop(self):
        with sd.RawInputStream(samplerate=self.fs, blocksize=8000, device=self.device,
                               dtype='int16', channels=1, callback=self.callback):
            print("[ORION] Listening for wakeword...")
            while self.running:
                data = self.q.get()
                if data is not None:
                    # Normalize audio to float32 in range -1.0 to 1.0
                    audio = data.astype(np.float32) / 32768.0

                    # Save raw audio to a file for debugging
                    with open("debug_audio.raw", "ab") as f:
                        f.write(data.tobytes())

                    # Log the normalized audio data
                    print(f"[DEBUG] Normalized audio: {audio[:10]}... (showing first 10 samples)")

                    # Get predictions from the model
                    scores = self.model.predict(audio)

                    # Log the raw predictions
                    print(f"[DEBUG] Model predictions: {scores}")

                    for name, score in scores.items():
                        print(f"[WAKEWORD] {name} score: {score:.2f}")
                        if score > THRESHOLD:
                            print("[ORION] Wakeword detected!")
                            self.tts.speak("I'm listening.")
                            if self.on_wakeword:
                                self.on_wakeword()

    def start(self):
        if not self.settings.get("wakeword_enabled", True):
            print("[ORION] Wakeword disabled in settings.")
            return
        try:
            self.listen_loop()
        except KeyboardInterrupt:
            self.running = False
            print("\n[ORION] Wakeword listener stopped.")


if __name__ == "__main__":
    engine = OpenWakewordEngine()
    engine.start()
