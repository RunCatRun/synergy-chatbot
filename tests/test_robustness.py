"""
test_robustness.py — тесты устойчивости: справка о возможностях,
исправление опечаток, распознавание команды выхода с опечаткой.
"""

import pytest

from chatbot.knowledge import load_knowledge
from chatbot.preprocess2 import preprocess2
from chatbot.matcher_level2 import MatcherLevel2
from bot import _is_exit


@pytest.fixture(params=["academic", "friendly"])
def matcher(request):
    return MatcherLevel2(load_knowledge(request.param))


HELP_CASES = [
    "Что ты умеешь?",
    "Что ты знаешь?",
    "Чем можешь помочь?",
    "Какие темы можно спросить?",
    "помощь",
    "справка",
    "хелп",
]

TYPO_CASES = [
    ("Где общжитие?", "dormitory"),
    ("Сколько стоит магистратра?", "cost"),
    ("Какие програмы есть?", "programs"),
    ("нужен ли пасспорт?", "documents"),
]


@pytest.mark.parametrize("phrase", HELP_CASES)
def test_help_category(matcher, phrase):
    """Вопросы о возможностях бота распознаются как справка."""
    code, answer, score = matcher.match(preprocess2(phrase))
    assert code == "help", f"{phrase!r}: получили {code}"


@pytest.mark.parametrize("phrase,expected", TYPO_CASES)
def test_typos_corrected(matcher, phrase, expected):
    """Запросы с опечатками распознаются верно."""
    code, answer, score = matcher.match(preprocess2(phrase))
    assert code == expected, f"{phrase!r}: получили {code}, ждали {expected}"


def test_greeting_lists_topics():
    """Ответ-приветствие перечисляет доступные темы."""
    for tone in ("academic", "friendly"):
        kb = load_knowledge(tone)
        greeting = next(c["answer"] for c in kb["categories"]
                        if c["code"] == "greeting")
        assert "программ" in greeting.lower()
        assert "стоимост" in greeting.lower()


def test_exit_with_typo():
    """Команда выхода распознаётся с лёгкой опечаткой."""
    vocab = {"вход", "логин"}  # имитация словаря ключевых слов
    assert _is_exit("выход", vocab)
    assert _is_exit("выхд", vocab)        # опечатка
    assert not _is_exit("вход", vocab)    # ключевое слово, не выход
    assert not _is_exit("стоимость", vocab)
