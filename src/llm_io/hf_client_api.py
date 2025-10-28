from huggingface_hub import InferenceClient

class HFClientAPI:
    def __init__(self, model_name, token, max_new_tokens, temperature, endpoint_url=None):
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.client = (InferenceClient(endpoint_url=endpoint_url, token=token, timeout=30)
                       if endpoint_url else InferenceClient(model=model_name, token=token, timeout=30))

    def generate(self, prompt: str) -> str:
        # 1) conversational
        try:
            resp = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
            # objets récents: choices[0].message["content"] ou .message.content
            choice = resp.choices[0].message
            return choice["content"] if isinstance(choice, dict) else choice.content
        except AttributeError:
            # vieille lib: pas de chat_completion
            pass
        except Exception as e_chat:
            # si provider n’expose pas conversational on tombera sur text_generation
            msg = str(e_chat).lower()
            if "rate limit" in msg or "429" in msg:
                raise RuntimeError(f"HF chat rate-limited pour '{self.model_name}': {e_chat}") from e_chat

        # 2) fallback text-generation
        try:
            return self.client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        except Exception as e_tg:
            # message clair quand le provider exige conversational
            emsg = str(e_tg)
            if "Supported task: conversational" in emsg:
                raise RuntimeError(f"Provider exige conversational pour '{self.model_name}'. Mets à jour huggingface_hub ou utilise chat_completion.") from e_tg
            raise RuntimeError(f"HF indisponible pour '{self.model_name}': {e_tg}") from e_tg
