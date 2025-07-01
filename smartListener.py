import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from vad import VoiceActivityDetector  # Optional: see note below
import json
import re
from pathlib import Path
from jawieVoice import JawieVoice

INTENT_KEYWORDS = [
    r"^(hey|hi|hello|hallo|hei)?\s*(jowie|joey|jowy|jowey|jowee|jerry|jawie|joby|joe)[,\s]",
    r"\b(jowie|joey|jowy|jowey|jowee|jerry|jawie|joby|joe)\b.*(can you|could you|would you|please|tell me|what|how|do you|show me)"
    r"^(can you|could you|would you|please|tell me|what|how|do you|show me)\s*(jowie|joey|jowy|jowey|jowee|jerry|jawie|joby|joe)[,\s]",
]

SETTINGS_FILE = "settings.json"

class SmartListener:
    def __init__(self, model_size="base", model_path="models/", device_idx=None, use_vad=False, questionCallback=None):
        self.fs = 16000
        self.chunk_size = int(self.fs * 0.5)  # 0.5s chunks
        self.max_silence_duration = 1.2
        self.min_command_duration = 1.0
        self.model = WhisperModel(model_size, compute_type="int8", download_root=model_path)
        self.device = device_idx or self.load_device()
        self.tts = JawieVoice()
        self.use_vad = use_vad
        self.callback = questionCallback
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
            buffer = np.zeros((0,), dtype=np.float32)
            silence_chunks = 0
            while True:
                chunk = stream.read(self.chunk_size)[0].flatten().astype(np.float32) / 32768.0
                buffer = np.concatenate((buffer, chunk))

                is_speech = True
                if self.use_vad:
                    is_speech = self.vad.is_speech(chunk)

                if is_speech:
                    silence_chunks = 0
                else:
                    silence_chunks += 1

                silence_duration = silence_chunks * (self.chunk_size / self.fs)
                buffer_duration = len(buffer) / self.fs

                if silence_duration >= self.max_silence_duration and buffer_duration >= self.min_command_duration:
                    print("[SMART] Detected silence, processing command...")
                    transcription = self.transcribe(buffer)
                    if self.is_intended_for_assistant(transcription):
                        print(f"[SMART] User spoke to Jowie: {transcription}")
                        self.tts.speak("Sure. Let me help with that.")
                        self.callback(transcription)
                    else:
                        print(f"[SMART] Ignored: {transcription}")
                    buffer = np.zeros((0,), dtype=np.float32)
                    silence_chunks = 0

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
    listener = SmartListener(model_size="large-v3", use_vad=True)

    # Example callback function to handle user commands
    def on_user_spoke_to_assistant(transcript):
        print(f"[MAIN] User spoke to Jowie: {transcript}")
        # Here you can handle the transcript, e.g. pass it to AIEngine for processing
        # response = ai.ask(transcript)
        # if response:
        #     OrionVoice.speak(response)
        # else:
        #     OrionVoice.speak("Sorry, I didn't understand that.")

    listener.callback = on_user_spoke_to_assistant

    # Start continuous listening
    listener.listen()