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
    def __init__(self, model="mistral"):
        self.model = model
        self.system_prompt = """
        You are Jowie, a helpful and friendly voice assistant. You reply briefly, speak naturally, and act when needed.

        - Always reply out loud first before executing an action. For example: "Okay, let me check." or "Sure, give me a second."
        - Then return an action using this format only:
          {"action": "action_name", "params": { ... }}

        Current supported actions:
        - get_weather — params: { "city": string }
        - get_date — no params

        Don't describe your reasoning or explain how you're thinking — just speak like a real assistant helping a user.
        Don't explain how you did your action, keep a natural flow and conversation like between a human and an assistant.

        If you're idle, say something like "I'm here if you need me." 
        Always keep things short and human-like.
        Do not include any internal <think> tags or commentary — just reply plainly.
        """

        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        self.tts = JawieVoice()

    def ask(self, user_input: str):
        self.chat_history.append({"role": "user", "content": user_input})
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
                    json_start = sentence_buffer.index("{")
                    obj = json.loads(sentence_buffer[json_start:])
                    action = obj.get("action")
                    params = obj.get("params", {})
                    print(f"\n[Tool] Detected action: {action} Params: {params}")
                    json_triggered = True
                    self.handle_action(action, params)
                    return  # Stop ask() flow; continuation happens in handle_action
                except Exception as e:
                    continue

        if sentence_buffer.strip() and not json_triggered:
            Thread(target=self.tts.speak, args=(sentence_buffer.strip(),), daemon=True).start()

        print()
        self.chat_history.append({"role": "assistant", "content": reply})
        return reply

    def handle_action(self, action: str, params: dict):
        result = ""
        if action == "get_weather":
            city = params.get("city", "Brussels")
            self.tts.speak("One moment, fetching the latest weather...")
            result = get_weather_report(city)
        elif action == "get_date":
            result = datetime.now().strftime("%A, %B %d %Y")
        else:
            result = f"Sorry, I don't know how to '{action}' yet."
            self.tts.speak(result)
            return

        # Append function result to conversation
        self.chat_history.append({
            "role": "function",
            "name": action,
            "content": result
        })

        reply = self.continue_conversation()
        if reply:
            Thread(target=self.tts.speak, args=(reply,), daemon=True).start()

    def continue_conversation(self):
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": self.model, "messages": self.chat_history, "stream": False}
        )

        try:
            reply = resp.json()["message"]["content"]
        except Exception as e:
            print("[Error] Failed to get reply after function call:", e)
            return ""

        self.chat_history.append({"role": "assistant", "content": reply})
        print(f"\nJowie: {reply}")
        return reply

    def reset(self):
        self.chat_history = [{"role": "system", "content": self.system_prompt}]


if __name__ == "__main__":
    ai = AIEngine("mistral")

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
