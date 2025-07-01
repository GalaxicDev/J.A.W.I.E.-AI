import requests
import soundfile as sf
import sounddevice as sd
import io
import threading
import queue

class JawieVoice:
    def __init__(self):
        print("[ORION TTS] Kokoro streaming voice active")
        self.q = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        while True:
            text = self.q.get()
            if not text.strip():
                continue
            try:
                payload = {
                    "model": "kokoro",
                    "input": text,
                    "voice": "af_heart",
                    "response_format": "wav",
                    "download_format": "mp3",
                    "speed": 1.0,
                    "stream": True,
                    "return_download_link": False,
                    "lang_code": "a",
                    "volume_multiplier": 1.0,
                    "normalization_options": {
                        "normalize": True,
                        "unit_normalization": False,
                        "url_normalization": True,
                        "email_normalization": True,
                        "optional_pluralization_normalization": True,
                        "phone_normalization": True,
                        "replace_remaining_symbols": True
                    }
                }

                response = requests.post(
                    "http://localhost:8880/v1/audio/speech", json=payload, stream=True)
                response.raise_for_status()

                buf = io.BytesIO()
                for chunk in response.iter_content(chunk_size=8192):
                    buf.write(chunk)
                buf.seek(0)

                data, fs = sf.read(buf, dtype="float32")
                sd.play(data, fs)
                sd.wait()

            except Exception as e:
                print(f"[TTS Error] {e}")

    def speak(self, text: str):
        self.q.put(text)


