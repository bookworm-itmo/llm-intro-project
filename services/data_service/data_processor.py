import re
import json
import xml.etree.ElementTree as ET
from typing import List, Dict
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BookProcessor:
    def __init__(self, book_path: str):
        self.book_path = Path(book_path)
        self.chapters = []
        self.chunks = []

    def load_book(self) -> str:
        """Загрузка книги из FB2 или TXT формата"""
        if self.book_path.suffix.lower() == '.fb2':
            return self._load_fb2()
        else:
            with open(self.book_path, 'r', encoding='utf-8') as f:
                return f.read()

    def _load_fb2(self) -> str:
        """Парсинг FB2 файла и извлечение текста"""
        tree = ET.parse(self.book_path)
        root = tree.getroot()

        # Ищем body с любым namespace
        body = None
        for elem in root.iter():
            if elem.tag.endswith('}body') or elem.tag == 'body':
                body = elem
                break

        if body is not None:
            return ''.join(body.itertext())
        else:
            # Fallback: весь текст
            return ''.join(root.itertext())

    def split_into_chapters(self, text: str) -> List[Dict[str, str]]:
        # Паттерн для поиска глав
        chapter_pattern = r'Глава\s+(\d+)'
        splits = re.split(chapter_pattern, text)

        chapters = []
        for i in range(1, len(splits), 2):
            if i + 1 < len(splits):
                chapter_num = splits[i]
                chapter_text = splits[i + 1].strip()
                # Очищаем текст от лишних символов
                chapter_text = re.sub(r'\n+', ' ', chapter_text)
                chapter_text = re.sub(r'\s+', ' ', chapter_text)
                chapters.append({
                    'chapter_number': int(chapter_num),
                    'text': chapter_text
                })

        self.chapters = chapters
        return chapters

    def create_chunks(self, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Разбивает текст на чанки с помощью RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", "!", "?", " "],
        )

        chunks = []
        chunk_id = 0

        for chapter in self.chapters:
            text = chapter['text']
            chapter_num = chapter['chapter_number']

            chapter_chunks = splitter.split_text(text)

            for chunk_text in chapter_chunks:
                chunk_text = chunk_text.strip()
                if len(chunk_text) >= 50:  # минимальная длина чанка
                    chunks.append({
                        'id': chunk_id,
                        'text': chunk_text,
                        'chapter': chapter_num,
                        'length': len(chunk_text)
                    })
                    chunk_id += 1

        self.chunks = chunks
        return chunks

    def save_chunks(self, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def process(self, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        text = self.load_book()
        self.split_into_chapters(text)
        chunks = self.create_chunks(chunk_size, overlap)
        return chunks
