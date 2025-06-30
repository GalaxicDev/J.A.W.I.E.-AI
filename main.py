from smartListener import SmartListener
from transcriber import Transcriber
from jawieVoice import JawieVoice
#from AIEngine import AIEngine

import os
os.add_dll_directory(r"C:\\Program Files\\NVIDIA\\CUDNN\\v9.10\\bin\\12.9")

print("[MAIN] Booting Jowie Voice Assistant...")

transcriber = Transcriber()
OrionVoice = JawieVoice()
#AIEngine = AIEngine()

# Optional: handler to execute full assistant logic after voice intent is confirmed
def on_user_spoke_to_assistant(transcript):
    print(f"[MAIN] Assistant invoked with: {transcript}")
    OrionVoice.speak(f"You said: {transcript}")

# Initialize the smart listener
listener = SmartListener(model_size="large-v3-turbo", use_vad=True,
                         questionCallback=on_user_spoke_to_assistant)

# Optional: Initial greeting
OrionVoice.speak("Hello, I am Joey. Please let me know if I can assist you with anything.")

# Start continuous listening
listener.listen()
