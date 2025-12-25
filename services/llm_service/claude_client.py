import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()


class ClaudeClient:
    """
    LLM клиент с поддержкой Claude API и OpenRouter.

    Провайдер выбирается через LLM_PROVIDER в .env:
    - "claude" (по умолчанию) — прямой Claude API
    - "openrouter" — через OpenRouter (дешевле, больше моделей)
    """

    def __init__(self, api_key: str = None, provider: str = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "openrouter")

        if self.provider == "openrouter":
            self._init_openrouter()
        else:
            self._init_claude(api_key)

    def _init_claude(self, api_key: str = None):
        from anthropic import Anthropic
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-haiku-4-5-20251001"

    def _init_openrouter(self):
        from openai import OpenAI
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    def generate_answer(
        self,
        query: str,
        context: str,
        sources: List[Dict],
        max_tokens: int = 1024
    ) -> str:
        chapters = sorted(set(s['chapter'] for s in sources))
        chapters_str = ", ".join(str(ch) for ch in chapters)

        prompt = f"""Ты - система для ответов на вопросы по роману "Мастер и Маргарита".

ВАЖНО: Ты должен отвечать ТОЛЬКО на основе предоставленного контекста ниже. НЕ используй свои знания о романе. Если ответа нет в контексте - так и скажи.

Контекст из книги:
{context}

Вопрос: {query}

Требования к ответу:
1. Используй ТОЛЬКО информацию из контекста выше
2. НЕ добавляй факты из своих знаний о романе
3. Если в контексте нет ответа, напиши: "В предоставленных фрагментах нет информации для ответа на этот вопрос"
4. Укажи главы источников: {chapters_str}
5. Приведи краткую цитату из контекста"""

        if self.provider == "openrouter":
            return self._generate_openrouter(prompt, max_tokens)
        else:
            return self._generate_claude(prompt, max_tokens)

    def _generate_claude(self, prompt: str, max_tokens: int) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    def _generate_openrouter(self, prompt: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
