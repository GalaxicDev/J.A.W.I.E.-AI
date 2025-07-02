import json
from datetime import datetime
import ollama
from jawieVoice import JawieVoice
from actions.tool_weather import get_weather_report
from actions.search_internet import search_internet
import re

class AIEngine:
    def __init__(self, model="mistral"):
        self.model = model
        self.tts = JawieVoice()

        self.system_prompt = """
            # Overview
            You are Jowie, a smart, friendly, and helpful voice assistant.
            Your job is to help the user by either using your built-in knowledge or by calling a tool to get real-time information.

            ## Tools Available
            You can use these tools when needed:
            - `get_weather(city: str)`: Fetches the real-time weather in a specific city.
            - `get_date()`: Gets the current date.
            - `search_internet(query: str)`: Searches the internet for up-to-date or factual information.

            ## Rules
            - Do NOT guess. If a user asks for information that may change over time (e.g. weather, rankings, current events), always use a tool.
            - If you're unsure of a fact or it's outside your training, use the `search_internet` tool to verify.
            - Do NOT say you're just an AI that can't browse. You CAN call tools.
            - DO speak naturally. Don’t say you used a tool — just answer like you knew the answer.
            - DO NOT return code, JSON, or internal function call info to the user.
            - DO call tools automatically when useful. The user doesn't need to ask you to.
            - Only skip a tool if you are absolutely certain of your answer.
            - Don't say you are an AI or a chatbot. Just be Jowie, a helpful assistant.
            - DO NOT say you don't have access to real-time information. You can use search_internet or any other tool to get the latest data.
            - DO NOT say you are a language model or that you can't browse the internet. You can use tools to get information.
            - It is 2025 so don't give any data of years that passed if you got this in your trained memory. Use the tools to get the latest information.

            ## Style
            - Speak briefly and clearly.
            - Act confident, even if you're using tools behind the scenes.
            - Never over-explain how you work. Just help the user like a human assistant would.

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
            },
        }, {
            'type': 'function',
            'function': {
                'name': 'search_internet',
                'description': 'Search the internet for information.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'The search query to look up'
                        }
                    },
                    'required': ['query']
                }
            }
        }]

        self.available_functions = {
            'get_weather': self.call_get_weather,
            'get_date': self.call_get_date,
            'search_internet': self.call_search_internet
        }

    def call_get_weather(self, city: str) -> str:
        self.tts.speak("Let me check the weather...")
        return get_weather_report(city)

    def call_get_date(self) -> str:
        return datetime.now().strftime("%A, %B %d %Y")

    def call_search_internet(self, query: str) -> str:
        self.tts.speak("Give me a moment to fetch that information...")
        return search_internet(query)

    def ask(self, user_input: str):
        self.chat_history.append({"role": "user", "content": user_input})

        print("Jowie:", end=" ", flush=True)

        # Step 1: Get initial reply (and possible tool_call)
        response = ollama.chat(
            model=self.model,
            messages=self.chat_history,
            tools=self.tools,
            #think=False,
        )

        assistant_msg = response['message']['content']
        tool_calls = response['message'].get('tool_calls') or []
        print("[DEBUG] response:", response)

        if assistant_msg:
            print(assistant_msg)
            assistant_tts = self.clean_tts(assistant_msg)

            self.tts.speak(assistant_tts)
            self.chat_history.append({"role": "assistant", "content": assistant_msg})

        # Step 2: Handle any tool calls
        for call in tool_calls:
            func_name = call['function']['name']
            args = call['function']['arguments']

            tool_func = self.available_functions.get(func_name)
            if not tool_func:
                print(f"[Error] Unknown function: {func_name}")
                continue

            result = tool_func(**args)

            # Step 3: Feed result back into chat
            self.chat_history.append({
                "role": "tool",
                "name": func_name,
                "content": f"[TOOL RESULT] {result}"
            })

            # Step 4: Let model respond after tool result
            followup = ollama.chat(
                model=self.model,
                messages=self.chat_history
            )

            final_reply = followup['message']['content']
            tts_reply = self.clean_tts(final_reply)
            print("Jowie:", final_reply)
            self.tts.speak(tts_reply)
            self.chat_history.append({"role": "assistant", "content": final_reply})

    def clean_tts(self, text):
        # remove all thinking content
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove emojis using a regex pattern
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
            "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002700-\U000027BF"  # Dingbats
            "\U000024C2-\U0001F251"  # Enclosed Characters
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub("", text)
        return text

    def reset(self):
        self.chat_history = [{"role": "system", "content": self.system_prompt}]


if __name__ == "__main__":
    ai = AIEngine("hermes3:8b") # hermes3:3b-llama3.2-q4_K_M | hermes3:8b (1) | llama3-groq-tool-use:8b-q4_K_M

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
