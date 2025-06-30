import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from vad import VoiceActivityDetector  # Optional: see note below
import time
import json
import re
from pathlib import Path
from jawieVoice import JawieVoice


INTENT_KEYWORDS = [
    r"^(hey|hi|hello)?\s*(jowie|orion)[,]?",
    r"^(what|how|could you|can you|tell me|do you|please)\b"
]

SETTINGS_FILE = "settings.json"

class SmartListener:
    def __init__(self, model_size="base", model_path="models/", device_idx=None, use_vad=False):
        self.fs = 16000
        self.buffer_duration = 3.0
        self.model = WhisperModel(model_size, compute_type="int8", download_root=model_path)
        self.device = device_idx or self.load_device()
        self.tts = JawieVoice()
        self.use_vad = use_vad
        if self.use_vad:
            self.vad = VoiceActivityDetector(sample_rate=self.fs)

    def load_device(self):
        if not Path(SETTINGS_FILE).exists():
            return None
        with open(SETTINGS_FILE) as f:
            return json.load(f).get("input_device")

    def listen(self):
        print("[SMART] Starting intelligent listener...")
        with sd.InputStream(samplerate=self.fs, channels=1, dtype='int16', device=self.device) as stream:
            while True:
                audio = stream.read(int(self.fs * self.buffer_duration))[0].flatten().astype(np.float32) / 32768.0

                if self.use_vad and not self.vad.is_speech(audio):
                    print("[SMART] Skipped: no voice activity detected.")
                    continue

                transcription = self.transcribe(audio)
                print(f"[SMART] Heard: {transcription}")
                if self.is_intended_for_assistant(transcription):
                    print(f"[SMART] User spoke to Jowie: {transcription}")
                    self.tts.speak("Sure. Let me help with that.")
                else:
                    print(f"[SMART] Ignored: {transcription}")

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(audio)
        return " ".join([seg.text for seg in segments]).strip()

    def is_intended_for_assistant(self, text):
        text = text.lower().strip()
        for pattern in INTENT_KEYWORDS:
            if re.search(pattern, text):
                return True
        return False


if __name__ == "__main__":
    listener = SmartListener(model_size="small", use_vad=True)  # Set to False to disable VAD
    listener.listen()
