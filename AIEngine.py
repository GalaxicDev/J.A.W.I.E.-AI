import requests
import soundfile as sf
import sounddevice as sd
import io
import json
import time
from datetime import datetime
from threading import Thread
from jawieVoice import JawieVoice
from actions.tool_weather import get_weather_report

class AIEngine:

    # TODO: see what model is best (llama3 or hermes2-mistral or openchat or mistral)

    def __init__(self, model="mistral"):
        self.model = model
        self.system_prompt  = (
            "You are Jowie, a helpful and smart AI assistant developed to help the user with anything, via voice, "
            "desktop automation, and internet tools. You speak conversationally and in a"
            "• Multi-turn chat memory. "
            "• Fetch real-time data (weather, news). "
            "• Interact with desktop apps (Spotify, Word, file system) – only after asking permission. "
            "• Read screen contents via OCR tools. "
            "• Always ask before destructive actions (deleting, uninstalling, formatting). "
            "• Occasionally suggest next steps (“Would you like me to open Word?”). "
            "• Speak short confirmation phrases (“On it, sir”) when executing tasks. "
            "• End interface politely when idle (e.g., “I’m here if you need me”). "
            "When you want to trigger an action, respond ONLY with a valid JSON object using: {\"action\": \"name\", \"params\": {...}}"
            "Current actions are: play_spotify, search_web, get_weather, get_date. "

        )
        self.chat_history = [{"role":"system","content": self.system_prompt}]
        self.tts = JawieVoice()

    def ask(self, user_input: str):
        self.chat_history.append({"role":"user","content": user_input})
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": self.model, "messages": self.chat_history, "stream": True},
            stream=True,
        )
        reply = ""
        sentence_buffer = ""
        json_triggered = False

        print("Jowie:", end=" ", flush=True)

        for line in resp.iter_lines():
            if not line:
                continue

            chunk = line.decode().removeprefix("data: ").strip()
            if chunk == "[DONE]":
                break

            try:
                content = json.loads(chunk).get("message", {}).get("content", "")
            except:
                continue

            if json_triggered:
                continue

            reply += content
            print(content, end="", flush=True)
            sentence_buffer += content

            if any(p in sentence_buffer for p in [". ", "! ", "? "]):
                sentence = sentence_buffer.strip()
                if sentence:
                    Thread(target=self.tts.speak, args=(sentence,), daemon=True).start()
                sentence_buffer = ""

            if '"action"' in sentence_buffer or '{"action":' in sentence_buffer:
                try:
                    obj = json.loads(sentence_buffer[sentence_buffer.index('{'):])
                    action = obj.get("action")
                    params = obj.get("params", {})
                    print(f"\n[Tool] Detected action: {action} Params: {params}")
                    json_triggered = True
                    self.handle_action(action, params)
                    break
                except Exception as e:
                    continue

        if sentence_buffer.strip() and not json_triggered:
            Thread(target=self.tts.speak, args=(sentence_buffer.strip(),), daemon=True).start()

        print()
        self.chat_history.append({"role":"assistant","content": reply})
        return reply

    def handle_action(self, action: str, params: dict):
        if action == "play_spotify":
            self.tts.speak("Playing Spotify track")
        elif action == "search_web":
            self.tts.speak("Searching the web now")
        elif action == "get_weather":
            city = params.get("city", "Brussels")
            self.tts.speak("One moment, fetching the latest weather...")
            weather = get_weather_report(city)
            self.tts.speak(weather)
        elif action == "get_date":
            current_date = datetime.now().strftime("%A, %B %d %Y")
            self.tts.speak(f"Today is {current_date}.")
        else:
            self.tts.speak(f"Sorry, I don't know how to '{action}' yet.")

    def reset(self):
        self.chat_history = [{"role":"system","content": self.system_prompt}]

if __name__ == "__main__":
    ai = AIEngine("llama3")

    print(r"""
                 ██╗ █████╗ ██╗    ██╗██╗███████╗
                 ██║██╔══██╗██║    ██║██║██╔════╝
                 ██║███████║██║ █╗ ██║██║███████╗
            ██   ██║██╔══██║██║███╗██║██║██╔════╝
            ╚█████╔╝██║  ██║╚███╔███╔╝██║███████║
             ╚════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝╚══════╝
        """)
    print("Welcome to J.A.W.I.E. AI Assistant!")

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Jowie: Goodbye!")
            break

        ai.ask(user_input)
