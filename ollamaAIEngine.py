import json
import requests

class AIEngine:
    def __init__(self, model="mistral"):
        self.model = model
        self.system_prompt = (
            "You are Jowie, a smart, friendly AI assistant on the user's PC. "
            "Jowie was developed by one person who wanted to create a helpful AI companion like Jarvis from the movies. "
            "You are designed to assist with various tasks on the 30th of June 2025. "
            "You speak conversationally, can answer questions, execute tool commands "
            "(e.g., play Spotify, search internet, operate Word), and view screen text via OCR tools. "
            "Always ask for permission before doing destructive actions. "
            "Destructive actions are seen like deleting files, formatting drives, or uninstalling software, interacting with apps, doing OCR, ... "
            "General questions are not considered destructive actions. "
            "Once you got permision to do a thing you can always do it again unless the user explicitly tells you to stop. "
            "Provide clear, concise answers, and indicate when connecting with tools."
        )
        self.chat_history = [{"role": "system", "content": self.system_prompt}]

    def ask(self, user_input: str):
        self.chat_history.append({"role": "user", "content": user_input})

        try:
            response = requests.post(
                f"http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": self.chat_history,
                    "stream": True
                },
                stream=True,
            )

            if response.status_code != 200:
                print(f"\n[ERROR] Ollama response code: {response.status_code}")
                print(f"[ERROR] Body: {response.text}")
                return ""

            reply = ""
            print("Jowie:", end=" ", flush=True)
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = line.decode("utf-8").removeprefix("data: ").strip()
                        if chunk == "[DONE]":
                            break
                        content = json.loads(chunk).get("message", {}).get("content", "")
                        print(content, end="", flush=True)
                        reply += content
                    except Exception as e:
                        print(f"\n[Chunk Error] {e}")
            print()
            self.chat_history.append({"role": "assistant", "content": reply})
            return reply

        except Exception as e:
            print(f"\n[Request Error] {e}")
            return ""

    def reset(self):
        self.chat_history = [{"role": "system", "content": self.system_prompt}]


if __name__ == "__main__":
    ai = AIEngine("llama3")  # or "mistral"

    print("Welcome to Jowie AI!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Jowie: Goodbye!")
            break
        ai.ask(user_input)
