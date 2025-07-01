import requests
import soundfile as sf
import sounddevice as sd
import io
import json
import time
from datetime import datetime
from jawieVoice import JawieVoice
from actions.tool_weather import get_weather_report


class AIEngine:
    def __init__(self, model="mistral"):
        self.model = model
        self.system_prompt = """
        You are Jowie, a helpful and friendly voice assistant. You reply briefly, speak naturally, and act when needed.
        You are very knowledgeable. An expert. Think and respond with confidence.  
        
        - Always reply out loud first before executing an action. For example: "Okay, let me check." or "Sure, give me a second."
        - Then return an action using this format only:
          {"action": "action_name", "params": { ... }}
        - never think for more than 2 seconds before replying.
        - don't doubt your response too long, or think its a test. Use your instincts and knowledge to answer.

        Current supported actions:
        - get_weather — params: { "city": string }
        - get_date — no params

        Don't describe your reasoning or explain how you're thinking — just speak like a real assistant helping a user.
        Don't explain how you did your action, keep a natural flow and conversation like between a human and an assistant.

        If you're idle, say something like "I'm here if you need me." Always keep things short and human-like. Do not 
        include any internal <think> tags or commentary — just reply plainly. If you don't know how to do something, 
        say "Sorry, I don't know how to do that yet." If the user says your name wrong (e.g. "Joey" instead of 
        "Jowie"), don't correct them, just respond normally. This is because of a error with the voice recognition 
        model." If the user asks to send an email reply with the text from the email and don't actually try to send 
        it with actions. Your actions are only used to fetch information that you can use to assist via voice, 
        not to perform tasks like sending emails or making calls. Your replies are always read by a text-to-speech 
        system, so keep them short and natural."""

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
                    self.tts.speak(sentence.strip())
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
            self.tts.speak(sentence_buffer.strip())

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

        self.chat_history.append({
            "role": "assistant",
            "content": result
        })

        reply = self.continue_conversation()
        if reply:
            self.tts.speak(reply)

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
    ai = AIEngine("neural-chat:7b")

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
