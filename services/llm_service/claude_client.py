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

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text
