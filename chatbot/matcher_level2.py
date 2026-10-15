"""
matcher_level2.py — сопоставление уровня 2 (средний).

Надстраивается над уровнем 1 и добавляет три механизма:

  1. Синонимы и spaCy-лемматизация — выполняются в предобработке
     (preprocess2), сюда уже поступает множество канонических лемм.

  2. Приоритет категорий-намерений. Категории делятся на «намерения»
     (спрашивают о свойстве: стоимость, поступление, документы, формы,
     кредит, бюджет) и «темы» (называют объект: программы, факультеты).
     При равенстве весов побеждает намерение: вопрос «сколько стоит
     магистратура» относится к стоимости, а не к программам.

  3. Память диалога. Сопоставитель хранит последнюю распознанную
     категорию-намерение. Короткий уточняющий вопрос, состоящий только
     из названия темы («а бакалавриат?»), наследует это намерение:
     после вопроса о стоимости «а бакалавриат?» снова даёт стоимость.

При отсутствии совпадений возвращается служебный ответ (fallback).
"""

from chatbot.fuzzy import correct_lemmas

# Категории-намерения: спрашивают о характеристике объекта.
_INTENT_CATEGORIES = {
    "cost", "how_to_enroll", "documents", "forms",
    "loan", "budget", "distance",
}

# Категории-темы: называют объект (используются в памяти диалога).
_TOPIC_CATEGORIES = {"programs", "faculties", "it_faculties"}

# Леммы названий уровней обучения — признак уточняющего вопроса.
_LEVEL_LEMMAS = {
    "бакалавриат", "магистратура", "специалитет", "колледж",
    "аспирантура", "ординатура", "школа",
}

# «Слабые» (общие) ключевые слова имеют вес 1, остальные — вес 2.
_WEAK_LEMMAS = {
    "обучение", "учиться", "программа", "уровень", "направление",
    "система", "информация",
}


class MatcherLevel2:
    """Сопоставитель среднего уровня с приоритетом намерения и памятью."""

    def __init__(self, knowledge):
        self.fallback = knowledge["fallback"]
        self.categories = []
        for cat in knowledge["categories"]:
            weighted = {}
            for k in cat["keywords"]:
                # Ключевые слова в базе уже близки к леммам; считаем,
                # что предобработка приведёт вход к тем же формам.
                lemma = k.lower()
                weighted[lemma] = 1 if lemma in _WEAK_LEMMAS else 2
            self.categories.append({
                "code": cat["code"],
                "keywords": weighted,
                "answer": cat["answer"],
            })
        self._answer_by_code = {c["code"]: c["answer"] for c in self.categories}
        # Полный словарь ключевых слов (для исправления опечаток и защиты выхода).
        self.vocab = set()
        for cat in self.categories:
            self.vocab |= set(cat["keywords"])
        # Память диалога: последняя категория-намерение.
        self.last_intent = None

    def _score_all(self, lemmas):
        """Возвращает список (код, ответ, сумма весов, пиковый вес)."""
        results = []
        for cat in self.categories:
            matched = lemmas & set(cat["keywords"])
            if not matched:
                continue
            score = sum(cat["keywords"][w] for w in matched)
            peak = max(cat["keywords"][w] for w in matched)
            results.append((cat["code"], cat["answer"], score, peak))
        return results

    def match(self, lemmas, text=""):
        """
        Определяет категорию для множества лемм входной фразы.

        :param lemmas: set канонических лемм (из preprocess2)
        :return: (код категории, ответ, суммарный вес)
        """
        # Исправление опечаток: близкие к ключевым словам формы
        # подменяются на канонические.
        lemmas = correct_lemmas(lemmas, self.vocab)

        results = self._score_all(lemmas)

        if not results:
            return "fallback", self.fallback, 0

        # Память диалога: если запрос — только название уровня обучения
        # (тема), а ранее звучало намерение, наследуем это намерение.
        only_level = lemmas and lemmas <= _LEVEL_LEMMAS
        if only_level and self.last_intent is not None:
            answer = self._answer_by_code[self.last_intent]
            return self.last_intent, answer, 0

        # Выбор лучшей категории с приоритетом намерения при равенстве.
        def sort_key(item):
            code, _ans, score, peak = item
            is_intent = 1 if code in _INTENT_CATEGORIES else 0
            return (score, is_intent, peak)

        best = max(results, key=sort_key)
        best_code, best_answer, best_score, _peak = best

        # Обновляем память, если распознали намерение.
        if best_code in _INTENT_CATEGORIES:
            self.last_intent = best_code

        return best_code, best_answer, best_score
