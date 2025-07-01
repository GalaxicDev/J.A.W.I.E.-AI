from smartListener import SmartListener
from transcriber import Transcriber
from jawieVoice import JawieVoice
from AIEngine import AIEngine

import os
os.add_dll_directory(r"C:\\Program Files\\NVIDIA\\CUDNN\\v9.10\\bin\\12.9")

print("[MAIN] Booting Jowie Voice Assistant...")

transcriber = Transcriber()
OrionVoice = JawieVoice()
ai = AIEngine("llama3")  # or "mistral"

# list devices debug
transcriber.select_device()

# Optional: handler to execute full assistant logic after voice intent is confirmed
def on_user_spoke_to_assistant(transcript):
    print(f"[MAIN] User spoke to Jowie: {transcript}")
    # Here you can handle the transcript, e.g. pass it to AIEngine for processing
    response = ai.ask(transcript)
    if response:
        OrionVoice.speak(response)
    else:
        OrionVoice.speak("Sorry, I didn't understand that.")
    # Optionally reset the AI engine state if needed
    ai.reset()

# Initialize the smart listener
listener = SmartListener(model_size="medium.en", use_vad=True,
                         questionCallback=on_user_spoke_to_assistant)

# Optional: Initial greeting
OrionVoice.speak("Hello, I am Joey. Please let me know if I can assist you with anything.")

# Start continuous listening
listener.listen()
