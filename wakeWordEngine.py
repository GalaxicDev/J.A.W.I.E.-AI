import sounddevice as sd
import queue
import json
import os
import sys
import time
from vosk import Model, KaldiRecognizer
from difflib import SequenceMatcher
from jawieVoice import JawieVoice

WAKEWORDS = [
    "hey orion", "hi orion", "hello orion", "joey", "jowie", "orion",
    "jowey", "jowy", "jowey voice assistant", "hey jowie", "hi jowie", "hello jowie"
]

SETTINGS_FILE = "settings.json"
MODEL_PATH = "models/vosk-model-small-en-us-0.15"

class VoskWakewordEngine:
    def __init__(self, on_wakeword=None):
        self.settings = self.load_settings()
        self.device = self.settings.get("input_device", None)
        self.fs = 16000
        self.q = queue.Queue()
        self.tts = OrionVoice()
        self.rec = None
        self.running = True
        self.on_wakeword = on_wakeword # Callback for wakeword detection

        if not os.path.exists(MODEL_PATH):
            print(f"Vosk model not found at {MODEL_PATH}")
            sys.exit(1)

        print("[ORION] Loading Vosk model...")
        model = Model(MODEL_PATH)
        self.rec = KaldiRecognizer(model, self.fs)
        self.rec.SetWords(True)

    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            return json.load(open(SETTINGS_FILE))
        return {"wakeword_enabled": True, "input_device": None}

    def callback(self, indata, frames, time_, status):
        if status:
            print(f"[VAD] Status: {status}")
        self.q.put(bytes(indata))

    def is_wakeword(self, text):
        text = text.lower().strip()
        for w in WAKEWORDS:
            score = SequenceMatcher(None, text, w).ratio()
            if score > 0.75:
                return True
        return False

    def listen_loop(self):
        with sd.RawInputStream(samplerate=self.fs, blocksize=8000, device=self.device,
                               dtype='int16', channels=1, callback=self.callback):
            print("[ORION] Listening for wakeword...")
            while self.running:
                data = self.q.get()
                if self.rec.AcceptWaveform(data):
                    result = json.loads(self.rec.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(f"[VOSK] Heard: {text}")
                        if self.is_wakeword(text):
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
    engine = VoskWakewordEngine()
    engine.start()