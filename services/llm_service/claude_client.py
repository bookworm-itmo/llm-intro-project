import os
from typing import List, Dict
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class ClaudeClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-haiku-4-5-20251001"

    def generate_answer(
        self,
        query: str,
        context: str,
        sources: List[Dict],
        max_tokens: int = 1024
    ) -> str:
        chapters = sorted(set(s['chapter'] for s in sources))
        chapters_str = ", ".join(str(ch) for ch in chapters)

        prompt = f"""Ты - эксперт по роману "Мастер и Маргарита" Михаила Булгакова.

На основе предоставленного контекста ответь на вопрос пользователя.

Контекст из книги:
{context}

Вопрос: {query}

Требования к ответу:
1. Отвечай только на основе предоставленного контекста
2. Если информации недостаточно, так и скажи
3. Укажи главы, на основе которых дан ответ
4. Приведи краткую цитату в поддержку ответа

Главы, использованные для ответа: {chapters_str}"""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text
