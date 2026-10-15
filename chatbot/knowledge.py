"""
knowledge.py — загрузка базы знаний чат-бота из JSON-файла.

База знаний хранится отдельно от кода (в папке data/), что позволяет
менять формулировки ответов и набор ключевых слов, не трогая программу.
"""

import json
import os


# Папка data/ лежит рядом с пакетом chatbot/, на уровень выше этого файла.
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")

# Соответствие "тон -> имя файла базы знаний".
_TONE_FILES = {
    "academic": "knowledge_academic.json",
    "friendly": "knowledge_friendly.json",
}


def load_knowledge(tone="academic"):
    """
    Загружает базу знаний для выбранного тона.

    :param tone: "academic" или "friendly"
    :return: словарь с ключами "categories" (список категорий) и
             "fallback" (ответ по умолчанию).
    """
    if tone not in _TONE_FILES:
        raise ValueError(
            f"Неизвестный тон: {tone!r}. Допустимые значения: "
            f"{', '.join(_TONE_FILES)}."
        )

    path = os.path.join(_DATA_DIR, _TONE_FILES[tone])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл базы знаний не найден: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return data
