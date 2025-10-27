# src/llm_io/hf_client_local.py
from transformers import pipeline

class HFClientLocal:
    def __init__(self, model_name: str, max_new_tokens: int = 200, temperature: float = 0.7):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.gen = pipeline("text-generation", model=self.model_name, return_full_text=False)

    def generate(self, prompt: str) -> str:
        out = self.gen(prompt, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=self.temperature)
        return out[0]["generated_text"].strip()
