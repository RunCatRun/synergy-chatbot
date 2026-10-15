"""
matcher_level3.py — сопоставление уровня 3 (продвинутый).

Каскадная стратегия:

  1. Сначала отрабатывает уровень 2 (точное сопоставление по ключевым
     словам, синонимы, исправление опечаток, приоритет намерения,
     память диалога). Если он дал уверенный результат — он принимается.

  2. Если уровень 2 не уверен (служебный ответ либо слабое совпадение
     веса 1), подключается семантическое сходство — векторное (spaCy)
     или TF-IDF. При сходстве выше порога берётся его результат.

  3. Если семантика тоже ниже порога, но уровень 2 дал хоть какое-то
     совпадение, возвращается результат уровня 2; иначе — служебный
     ответ (fallback).

Такой порядок не замедляет частые точные совпадения и одновременно
позволяет распознавать перефразировки без общих слов, на которых
уровни 1 и 2 уходят в fallback.
"""

from chatbot.matcher_level2 import MatcherLevel2
from chatbot.semantic import build_backend


# Уровень 2 считается уверенным при сумме весов не ниже этого значения
# (то есть совпало хотя бы одно специфичное ключевое слово).
_L2_TRUST = 2

# Порог семантического сходства. Значения подобраны под усреднённые
# векторы spaCy и TF-IDF; при необходимости настраиваются.
_SEM_THRESHOLD = {
    "vectors": 0.40,
    "tfidf": 0.12,
}


class MatcherLevel3:
    """Сопоставитель продвинутого уровня (каскад уровня 2 и семантики)."""

    def __init__(self, knowledge, method="vectors"):
        """
        :param knowledge: база знаний
        :param method: семантический движок — "vectors" или "tfidf"
        """
        self.method = method
        self.threshold = _SEM_THRESHOLD.get(method, 0.5)
        self.level2 = MatcherLevel2(knowledge)
        self.backend = build_backend(method, knowledge)
        self.vocab = self.level2.vocab
        self._answer_by_code = {
            c["code"]: c["answer"] for c in
            [{"code": cc["code"], "answer": cc["answer"]}
             for cc in knowledge["categories"]]
        }
        self.fallback = knowledge["fallback"]

    def match(self, lemmas, text=""):
        """
        :param lemmas: множество лемм (из preprocess2)
        :param text: исходная фраза пользователя (нужна семантике)
        :return: (код категории, ответ, мера) — мера это вес уровня 2
                 либо сходство семантики
        """
        # Шаг 1: уровень 2.
        code, answer, score = self.level2.match(lemmas)
        if code != "fallback" and score >= _L2_TRUST:
            return code, answer, score

        # Шаг 2: семантическое сходство добивает слабые случаи.
        sem_code, sim = self.backend.match(text)
        if sem_code is not None and sim >= self.threshold:
            return sem_code, self._answer_by_code.get(sem_code, answer), sim

        # Шаг 3: результат уровня 2, если он хоть что-то нашёл; иначе fallback.
        if code != "fallback":
            return code, answer, score
        return "fallback", self.fallback, 0
