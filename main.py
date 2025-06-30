from smartListener import SmartListener
from transcriber import Transcriber
from jawieVoice import JawieVoice

import os
os.add_dll_directory(r"C:\\Program Files\\NVIDIA\\CUDNN\\v9.10\\bin\\12.9")

print("[MAIN] Booting Jowie Voice Assistant...")

transcriber = Transcriber()
OrionVoice = JawieVoice()

# Optional: Initial greeting
OrionVoice.speak("Hello, I am Jowie. Please let me know if I can assist you with anything.")

# Optional: handler to execute full assistant logic after voice intent is confirmed
def on_user_spoke_to_assistant(transcript):
    print(f"[MAIN] Assistant invoked with: {transcript}")
    OrionVoice.speak(f"You said: {transcript}")
    # Expand: add further command parsing or function execution

# Initialize the smart listener
listener = SmartListener(model_size="large-v3", use_vad=True)

# Start continuous listening
listener.listen()
