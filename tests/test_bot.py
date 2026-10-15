"""
test_bot.py — тесты чат-бота (уровень 1).

Запуск:
    pytest

Проверяется:
  - предобработка фразы (лемматизация, удаление стоп-слов);
  - корректность определения категории на наборе типовых вопросов;
  - поведение на нераспознаваемом вводе (fallback);
  - идентичность распознавания для обоих тонов (тон влияет только
    на текст ответа, но не на логику).
"""

import pytest

from chatbot.knowledge import load_knowledge
from chatbot.preprocess import preprocess
from chatbot.matcher_level1 import MatcherLevel1


# Набор «фраза -> ожидаемая категория».
CASES = [
    ("Привет", "greeting"),
    ("Здравствуйте", "greeting"),
    ("Сколько стоит обучение на программиста?", "cost"),
    ("Какая цена за семестр?", "cost"),
    ("Какие у вас есть программы?", "programs"),
    ("Сколько факультетов в университете?", "faculties"),
    ("Где находится приёмная комиссия?", "admission_contacts"),
    ("Дайте телефон приёмной", "admission_contacts"),
    ("Можно ли учиться заочно?", "forms"),
    ("Какие документы нужны для поступления?", "documents"),
    ("Как поступить к вам?", "how_to_enroll"),
    ("Можно оформить кредит на обучение?", "loan"),
    ("Есть бюджетные места?", "budget"),
    ("А общежитие дают?", "dormitory"),
    ("Я иностранец, могу ли поступить?", "foreign"),
    ("Можно учиться онлайн?", "distance"),
    ("Хочу на искусственный интеллект", "it_faculties"),
    ("Как войти в личный кабинет?", "lms"),
    ("Расскажи про университет", "about"),
    ("Спасибо большое!", "thanks"),
    ("абракадабра непонятный запрос", "fallback"),
]


@pytest.fixture(params=["academic", "friendly"])
def matcher(request):
    """Сопоставитель уровня 1 для обоих тонов."""
    return MatcherLevel1(load_knowledge(request.param))


@pytest.mark.parametrize("phrase,expected", CASES)
def test_category(matcher, phrase, expected):
    """Каждая типовая фраза распознаётся в ожидаемую категорию."""
    code, answer, score = matcher.match(preprocess(phrase))
    assert code == expected, f"Фраза {phrase!r}: получили {code}, ждали {expected}"
    assert isinstance(answer, str) and answer, "Ответ должен быть непустой строкой"


def test_preprocess_lemmatization():
    """Слова приводятся к начальной форме."""
    lemmas = preprocess("обучающихся студентов")
    assert "обучаться" in lemmas
    assert "студент" in lemmas


def test_preprocess_removes_stopwords():
    """Стоп-слова отбрасываются."""
    lemmas = preprocess("а в и не на же бы")
    assert lemmas == set()


def test_empty_input_gives_fallback():
    """Пустая по смыслу фраза приводит к fallback."""
    m = MatcherLevel1(load_knowledge("academic"))
    code, answer, score = m.match(preprocess("и в на же бы"))
    assert code == "fallback"
    assert score == 0
