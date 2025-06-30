import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import threading
import queue

class AIEngine:
    SYSTEM_PROMPT = (
        "You are Jowie, a smart, friendly AI assistant on the user's PC. "
        "Jowie was developed by one person who wanted to create a helpful AI companion like Jarvis from the movies."
        "When asked about the history of Jowie, you can use these facts but rephrase them to be better suited for a conversational context"
        "You are designed to assist with various tasks on the 30th of June 2025 "
        "You speak conversationally, can answer questions, execute tool commands "
        "(e.g., play Spotify, search internet, operate Word), and view screen text via OCR tools. "
        "Always ask for permission before doing destructive actions. "
        "Provide clear, concise answers, and indicate when connecting with tools."
    )

    def __init__(self,
                 model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", # nvidia/Llama3-ChatQA-1.5-8B
                 device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AIEngine] Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": self.device},
        )
        self.chat_history = [{"role": "system", "content": self.SYSTEM_PROMPT}]

    def ask(self, user_input: str,
            max_initial_tokens=64,
            max_total_tokens=256,
            temperature=0.7):
        # Append user's message
        self.chat_history.append({"role": "user", "content": user_input})

        # Build conversation text
        conv = ""
        for msg in self.chat_history:
            tag = msg["role"]
            conv += f"<{tag}>: {msg['content']}\n"
        conv += "<assistant>:"

        inputs = self.tokenizer(conv, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)


        # prepare the streamer for real-time output
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        output_queue = queue.Queue()

        def generate_loop(max_tokens):
            self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id
            )

        threading.Thread(target=generate_loop, args=(max_total_tokens,), daemon=True).start()

        # Collect full response
        reply = ""
        for token in streamer:
            output_queue.put(token)
            reply += token

        self.chat_history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        self.chat_history = [{"role": "system", "content": self.SYSTEM_PROMPT}]

if __name__ == "__main__":
        ai = AIEngine()

        print(r"""
                 ██╗ █████╗ ██╗    ██╗██╗███████╗
                 ██║██╔══██╗██║    ██║██║██╔════╝
                 ██║███████║██║ █╗ ██║██║███████╗
            ██   ██║██╔══██║██║███╗██║██║██╔════╝
            ╚█████╔╝██║  ██║╚███╔███╔╝██║███████║
             ╚════╝ ╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝╚══════╝
                """)
        print("Welcome to J.A.W.I.E. AI Assistant!")
        print("Please type your question or command.")

        while True:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "quit"}:
                print("Jowie: Goodbye!")
                break
            print("Jowie:", ai.ask(user_input))