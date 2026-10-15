"""
conftest.py — служебный файл pytest.

Размещён в корне проекта намеренно. При запуске команды `pytest` его
наличие заставляет pytest добавить корневую папку проекта в пути поиска
модулей Python (sys.path). Без этого тесты из папки tests/ не находят
пакет `chatbot`, который лежит в корне, и падают с ошибкой
ModuleNotFoundError: No module named 'chatbot'.

Явная вставка пути ниже делает запуск независимым от способа вызова
(`pytest` или `python -m pytest`) и от версии pytest.
"""

import os
import sys

# Абсолютный путь к папке, где лежит этот файл (корень проекта).
_ROOT = os.path.dirname(os.path.abspath(__file__))

# Добавляем корень в начало списка путей, если его там ещё нет.
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
