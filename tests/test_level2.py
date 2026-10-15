"""
test_level2.py — тесты уровня 2 (синонимы, приоритет намерения, память).

Запуск:
    pytest tests/test_level2.py
"""

import pytest

from chatbot.knowledge import load_knowledge
from chatbot.preprocess2 import preprocess2
from chatbot.matcher_level2 import MatcherLevel2


BASE_CASES = [
    ("Привет", "greeting"),
    ("Сколько стоит обучение на программиста?", "cost"),
    ("Какие у вас есть программы?", "programs"),
    ("Сколько факультетов?", "faculties"),
    ("Где приёмная комиссия?", "admission_contacts"),
    ("Можно ли учиться заочно?", "forms"),
    ("Какие документы нужны?", "documents"),
    ("Как поступить?", "how_to_enroll"),
    ("Можно оформить кредит?", "loan"),
    ("Есть бюджетные места?", "budget"),
    ("А общежитие дают?", "dormitory"),
    ("Я иностранец, могу поступить?", "foreign"),
    ("Можно учиться онлайн?", "distance"),
    ("Хочу на искусственный интеллект", "it_faculties"),
    ("Как войти в личный кабинет?", "lms"),
    ("Расскажи про университет", "about"),
    ("Спасибо!", "thanks"),
    # Случай, который уровень 1 решал неверно:
    ("Сколько стоит магистратура?", "cost"),
]

SYNONYM_CASES = [
    ("Сколько стоит общага?", "cost"),
    ("Почём обучение?", "cost"),
    ("Дайте ваши контакты", "admission_contacts"),
    ("Хочу учиться на айти", "it_faculties"),
    ("Есть рассрочка?", "loan"),
    ("Можно бесплатно учиться?", "budget"),
    ("Где общага?", "dormitory"),
]


@pytest.fixture(params=["academic", "friendly"])
def matcher(request):
    return MatcherLevel2(load_knowledge(request.param))


@pytest.mark.parametrize("phrase,expected", BASE_CASES)
def test_base(matcher, phrase, expected):
    code, answer, score = matcher.match(preprocess2(phrase))
    assert code == expected, f"{phrase!r}: получили {code}, ждали {expected}"


@pytest.mark.parametrize("phrase,expected", SYNONYM_CASES)
def test_synonyms(matcher, phrase, expected):
    code, answer, score = matcher.match(preprocess2(phrase))
    assert code == expected, f"{phrase!r}: получили {code}, ждали {expected}"


def test_intent_priority():
    """При сочетании намерения и темы побеждает намерение."""
    m = MatcherLevel2(load_knowledge("academic"))
    code, _a, _s = m.match(preprocess2("сколько стоит магистратура"))
    assert code == "cost"


def test_dialogue_memory_inherits_intent():
    """Уточняющий вопрос наследует предыдущее намерение."""
    m = MatcherLevel2(load_knowledge("academic"))
    assert m.match(preprocess2("Сколько стоит магистратура?"))[0] == "cost"
    assert m.match(preprocess2("А бакалавриат?"))[0] == "cost"
    assert m.match(preprocess2("А колледж?"))[0] == "cost"


def test_dialogue_memory_switches_intent():
    """Новое намерение перекрывает старое в памяти."""
    m = MatcherLevel2(load_knowledge("academic"))
    m.match(preprocess2("Сколько стоит магистратура?"))   # cost
    assert m.match(preprocess2("Как туда поступить?"))[0] == "how_to_enroll"
    assert m.match(preprocess2("А в магистратуру?"))[0] == "how_to_enroll"
