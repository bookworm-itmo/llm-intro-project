"""
Модуль для работы с исходным файлом книги.

Книга "Мастер и Маргарита" скачивается вручную с Флибусты:
https://flibusta.is/b/813954

Файл должен быть сохранён как: data/master_and_margarita.fb2
"""
from pathlib import Path

BOOK_URL = "https://flibusta.is/b/813954"
BOOK_PATH = Path("data/master_and_margarita.fb2")


def check_book_exists():
    """Проверяет наличие файла книги."""
    if not BOOK_PATH.exists():
        raise FileNotFoundError(
            f"Файл книги не найден: {BOOK_PATH}\n"
            f"Скачайте книгу вручную с {BOOK_URL} и сохраните как {BOOK_PATH}"
        )
    size_mb = BOOK_PATH.stat().st_size / (1024 * 1024)
    print(f"[OK] Книга найдена: {BOOK_PATH} ({size_mb:.1f} MB)")
    return BOOK_PATH
