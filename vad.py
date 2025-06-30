import webrtcvad
import numpy as np

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, aggressiveness=2):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration = 30  # ms
        self.frame_size = int(sample_rate * self.frame_duration / 1000)

    def is_speech(self, audio):
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # mono only

        # Convert to 16-bit PCM
        int16_audio = (audio * 32768).astype(np.int16)
        raw_bytes = int16_audio.tobytes()

        # Break into 30ms chunks and test each
        for i in range(0, len(raw_bytes) - self.frame_size * 2, self.frame_size * 2):
            frame = raw_bytes[i:i + self.frame_size * 2]
            if self.vad.is_speech(frame, self.sample_rate):
                return True
        return False
