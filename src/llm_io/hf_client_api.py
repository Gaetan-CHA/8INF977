from huggingface_hub import InferenceClient

class HFClientAPI:
    def __init__(self, model_name: str, token: str, max_new_tokens: int = 200, temperature: float = 0.7):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # Ne lie pas de provider; passe le model à chaque appel.
        self.client = InferenceClient(api_key=token)

    def generate(self, prompt: str) -> str:
        # 1) Essaye le mode "chat" (conversational)
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e_chat:
            # 2) Fallback "text-generation"
            try:
                out = self.client.text_generation(
                    prompt,
                    model=self.model_name,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                return out.strip()
            except Exception as e_text:
                raise RuntimeError(
                    f"HF Inference indisponible pour '{self.model_name}'. "
                    f"chat_error={e_chat}; text_error={e_text}"
                )
