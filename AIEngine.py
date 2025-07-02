import json
from datetime import datetime
import ollama
from jawieVoice import JawieVoice
from actions.tool_weather import get_weather_report

class AIEngine:
    def __init__(self, model="mistral"):
        self.model = model
        self.tts = JawieVoice()

        self.system_prompt = """
        You are Jowie, a helpful and friendly voice assistant. You reply briefly, speak naturally, and act when needed.
        You are very knowledgeable. An expert. Think and respond with confidence.

        When a tool is used, always incorporate the tool's result into your response naturally. Don't just say "I 
        used a tool", but rather include the information it provides in your reply. The tool result isn't shared with 
        the user, you are to interpret it and pass it to the user as part of your response. You can use tools to get 
        information like the weather or current date, but you can also answer questions directly if you know the 
        answer. Never give the user a JSON response to use a tool. You don't need to explain how you work or what 
        tools you have, just use them when needed. You don't need to use a function/tool every time, 
        only when necessary. When you know the answer, just give it directly and don't explain that you use your 
        knowledge for this and don't have a tool. If you don't know how to do something, just say so. Keep it short 
        and natural.
        If the user asks question that are related to the weather, but you don't need the weather tool to use it, then don't use it.
        Never use [insert_temperature] or some things like this when the user is expecting an answer. Use to tools to get those variables or say you don't know.
        Don't hallucinate or make up answers, always provide accurate information.
        """

        self.chat_history = [{"role": "system", "content": self.system_prompt}]

        # Define tool schema (OpenAI style)
        self.tools = [{
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'description': 'Get the current weather in a specific city.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'city': {
                            'type': 'string',
                            'description': 'Name of the city to get the weather for'
                        }
                    },
                    'required': ['city']
                }
            }
        }, {
            'type': 'function',
            'function': {
                'name': 'get_date',
                'description': 'Get the current date.',
                'parameters': {
                    'type': 'object',
                    'properties': {}
                }
            }
        }]

        self.available_functions = {
            'get_weather': self.call_get_weather,
            'get_date': self.call_get_date
        }

    def call_get_weather(self, city: str) -> str:
        self.tts.speak("Let me check the weather...")
        return get_weather_report(city)

    def call_get_date(self) -> str:
        return datetime.now().strftime("%A, %B %d %Y")

    def ask(self, user_input: str):
        self.chat_history.append({"role": "user", "content": user_input})

        print("Jowie:", end=" ", flush=True)

        # Step 1: Get initial reply (and possible tool_call)
        response = ollama.chat(
            model=self.model,
            messages=self.chat_history,
            tools=self.tools
        )

        assistant_msg = response['message']['content']
        tool_calls = response['message'].get('tool_calls') or []
        print(response)

        if assistant_msg:
            print(assistant_msg)
            self.tts.speak(assistant_msg)
            self.chat_history.append({"role": "assistant", "content": assistant_msg})

        # Step 2: Handle any tool calls
        for call in tool_calls:
            func_name = call['function']['name']
            args = call['function']['arguments']

            tool_func = self.available_functions.get(func_name)
            if not tool_func:
                print(f"[Error] Unknown function: {func_name}")
                continue

            print(f"\n[Tool] Calling {func_name} with {args}")
            result = tool_func(**args)
            print(f"[Tool] Result: {result}")

            # Step 3: Feed result back into chat
            self.chat_history.append({
                "role": "tool",
                "name": func_name,
                "content": f"[TOOL RESULT] {result}"
            })

            print("[CHAT HISTORY] Updated with tool result.")
            print("Chat History:", json.dumps(self.chat_history, indent=2))
            # Step 4: Let model respond after tool result
            followup = ollama.chat(
                model=self.model,
                messages=self.chat_history
            )

            final_reply = followup['message']['content']
            print("Jowie:", final_reply)
            self.tts.speak(final_reply)
            self.chat_history.append({"role": "assistant", "content": final_reply})

    def reset(self):
        self.chat_history = [{"role": "system", "content": self.system_prompt}]


if __name__ == "__main__":
    ai = AIEngine("qwen2.5:7b")

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
