import numpy as np
import sounddevice as sd
import json
from pathlib import Path
from faster_whisper import WhisperModel

SETTINGS_FILE = Path("settings.json")

class Transcriber:
    def __init__(self, model_size="large-v3", model_path="models/", device_idx=None):
        self.fs = 16000
        self.model = WhisperModel(
            model_size,
            compute_type="int8",          # Faster + lower RAM
            download_root=model_path
        )
        self.device = device_idx or self.load_device()

    def list_devices(self):
        print("\n🎙 Available Input Devices:\n")
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                print(f"{i}: {dev['name']}")

    def select_device(self):
        self.list_devices()
        choice = int(input("\n🔧 Enter input device index to use: "))
        self.device = choice
        self.save_device(choice)

    def save_device(self, index: int):
        SETTINGS_FILE.write_text(json.dumps({"input_device": index}))

    def load_device(self):
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text()).get("input_device", None)
        return None

    def record_audio(self, duration: float = 5.0) -> np.ndarray:
        if self.device is None:
            raise Exception("No input device selected. Please call `select_device()` first.")
        print(f"[STT] Recording {duration}s of audio from device {self.device}...")
        audio = sd.rec(int(duration * self.fs), samplerate=self.fs, channels=1, dtype='int16', device=self.device)
        sd.wait()
        return audio.flatten().astype(np.float32) / 32768.0

    def transcribe(self, audio: np.ndarray) -> str:
        print("[STT] Transcribing...")
        segments, _ = self.model.transcribe(audio)
        full_text = " ".join([seg.text for seg in segments])
        print(f"[STT] Transcribed: {full_text.strip()}")
        return full_text.strip()
