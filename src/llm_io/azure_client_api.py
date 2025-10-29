class AzureClientAPI:
    """
    Adaptateur simple pour garder A.generate(text) -> str
    Utilise chat.completions avec un seul message user.
    """
    def __init__(self, *, endpoint: str, api_key: str, api_version: str,
                 deployment: str, temperature: float = 1.0,
                 max_completion_tokens: int = 512, top_p: float = 1.0,
                 frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
                 model_name: str | None = None):
        self.model_name = model_name or deployment
        self.deployment = deployment
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

        self._client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )

    def generate(self, user_text: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.deployment,  # IMPORTANT: c’est le nom du déploiement Azure
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_text},
            ],
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return ""
