"""
test_level3.py — тесты уровня 3 (каскад уровня 2 и семантики).

Ветка TF-IDF и каскадная логика проверяются всегда. Векторная ветка
требует модель spaCy с векторами (lg или md); при её отсутствии
соответствующие тесты пропускаются.
"""

import pytest

# Уровень 3 опирается на numpy и scikit-learn со скомпилированными
# модулями. Если они не загружаются в текущем окружении (частая причина —
# запуск с сетевого/UNC-пути в Windows), весь файл аккуратно пропускается,
# не прерывая прогон тестов уровней 1 и 2.
pytest.importorskip("numpy", reason="numpy не загружается (возможно, сетевой путь)")
pytest.importorskip("sklearn", reason="scikit-learn не загружается")

from chatbot.knowledge import load_knowledge
from chatbot.preprocess2 import preprocess2
from chatbot.matcher_level3 import MatcherLevel3
from chatbot.semantic import TfidfBackend, VectorBackend
from chatbot.nlp_spacy import get_engine_info


HAS_VECTORS = bool(get_engine_info().get("has_vectors"))


# Базовые случаи: каскад обязан сохранять поведение уровня 2.
BASE_CASES = [
    ("Сколько стоит магистратура?", "cost"),
    ("Где общага?", "dormitory"),
    ("Что ты умеешь?", "help"),
    ("Какие документы нужны?", "documents"),
    ("Как поступить?", "how_to_enroll"),
    ("справка", "help"),
    ("абракадабра рандомный набор", "fallback"),
]


@pytest.fixture(params=["academic", "friendly"])
def matcher_tfidf(request):
    return MatcherLevel3(load_knowledge(request.param), method="tfidf")


@pytest.mark.parametrize("phrase,expected", BASE_CASES)
def test_cascade_preserves_level2(matcher_tfidf, phrase, expected):
    code, answer, measure = matcher_tfidf.match(preprocess2(phrase), phrase)
    assert code == expected, f"{phrase!r}: получили {code}, ждали {expected}"


def test_tfidf_paraphrase_with_word_overlap():
    """TF-IDF распознаёт перефразировку при совпадении слов."""
    kb = load_knowledge("academic")
    tf = TfidfBackend(kb)
    code, sim = tf.match("как подать документы на поступление")
    assert code == "documents"
    assert sim > 0.12


def test_tfidf_empty_query():
    """Пустой запрос не даёт ложного совпадения."""
    kb = load_knowledge("academic")
    tf = TfidfBackend(kb)
    code, sim = tf.match("   ")
    assert code is None
    assert sim == 0.0


@pytest.mark.skipif(not HAS_VECTORS, reason="нужна модель spaCy с векторами (lg/md)")
def test_vector_backend_available():
    """Векторный движок доступен и возвращает категорию при векторной модели."""
    kb = load_knowledge("academic")
    vb = VectorBackend(kb)
    assert vb.available
    code, sim = vb.match("сколько стоит обучение")
    assert code is not None
    assert 0.0 <= sim <= 1.0


@pytest.mark.skipif(not HAS_VECTORS, reason="нужна модель spaCy с векторами (lg/md)")
def test_vector_paraphrase_without_overlap():
    """Векторное сходство распознаёт перефразировку без общих слов."""
    kb = load_knowledge("academic")
    m = MatcherLevel3(kb, method="vectors")
    # У фразы нет общих слов с ключевыми словами категории «общежитие».
    code, answer, measure = m.match(preprocess2("куда селят приезжих студентов"),
                                    "куда селят приезжих студентов")
    # Ожидаем осмысленную категорию, не fallback (точное значение зависит
    # от модели; проверяем, что семантика хоть что-то распознала).
    assert code != "fallback"
