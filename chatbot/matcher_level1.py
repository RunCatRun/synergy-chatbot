"""
matcher_level1.py — сопоставление уровня 1 (базовый).

Логика:
  1. ключевые слова каждой категории заранее приводятся к леммам;
  2. для входной фразы (множества лемм) считается СУММА ВЕСОВ
     совпавших ключевых слов;
  3. побеждает категория с наибольшей суммой весов;
  4. при равенстве сумм предпочтение отдаётся категории с более
     специфичным совпадением (большим максимальным весом слова);
  5. если совпадений нет — возвращается ответ по умолчанию (fallback).

Веса нужны, чтобы общие слова («обучение», «программа», «учиться»)
не перетягивали на себя запрос, когда в нём есть более точное слово
(«стоимость», «заочный», «онлайн»). Общие слова имеют вес 1,
специфичные — вес 2.
"""

import re

import pymorphy3


_morph = pymorphy3.MorphAnalyzer()

# Общие, «слабые» слова: встречаются во многих вопросах и не должны
# решать исход в одиночку. Им присваивается вес 1. Остальные ключевые
# слова считаются специфичными (вес 2).
_WEAK_LEMMAS = {
    "обучение", "учиться", "программа", "уровень", "направление",
    "система", "информация",
}


def _lemmatize_keyword(word):
    """Приводит одно ключевое слово к лемме (латиницу/цифры не трогает)."""
    word = word.lower()
    if re.fullmatch(r"[a-z0-9]+", word):
        return word
    return _morph.parse(word)[0].normal_form


class MatcherLevel1:
    """Сопоставитель базового уровня с весами ключевых слов."""

    def __init__(self, knowledge):
        self.fallback = knowledge["fallback"]
        self.categories = []
        for cat in knowledge["categories"]:
            weighted = {}
            for k in cat["keywords"]:
                lemma = _lemmatize_keyword(k)
                weighted[lemma] = 1 if lemma in _WEAK_LEMMAS else 2
            self.categories.append({
                "code": cat["code"],
                "keywords": weighted,
                "answer": cat["answer"],
            })
        # Полный словарь ключевых слов (нужен внешним проверкам).
        self.vocab = set()
        for cat in self.categories:
            self.vocab |= set(cat["keywords"])

    def match(self, lemmas, text=""):
        """
        Определяет наиболее подходящую категорию для множества лемм.

        :param lemmas: set лемм входной фразы
        :return: (код категории, ответ, суммарный вес совпадений).
                 Если совпадений нет — ("fallback", текст fallback, 0).
        """
        best_code = "fallback"
        best_answer = self.fallback
        best_score = 0
        best_peak = 0

        for cat in self.categories:
            matched = lemmas & set(cat["keywords"])
            if not matched:
                continue
            score = sum(cat["keywords"][w] for w in matched)
            peak = max(cat["keywords"][w] for w in matched)

            if score > best_score or (score == best_score and peak > best_peak):
                best_score = score
                best_peak = peak
                best_code = cat["code"]
                best_answer = cat["answer"]

        return best_code, best_answer, best_score
