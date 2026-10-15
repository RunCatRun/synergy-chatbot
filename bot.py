"""
bot.py — запуск консольного чат-бота Университета «Синергия».

НАСТРОЙКА (строки ниже):
  TONE            — тон ответов:    "academic"  или  "friendly"
  LEVEL           — уровень логики:  1, 2 или 3
  SEMANTIC_METHOD — движок уровня 3: "vectors" (spaCy) или "tfidf"

Уровни:
  1 — взвешенное пересечение ключевых слов (pymorphy3, NLTK);
  2 — + синонимы, spaCy-леммы, приоритет намерения, память диалога,
      исправление опечаток, справка;
  3 — каскад: уровень 2, затем семантическое сходство (векторное или
      TF-IDF) для перефразировок без общих слов.

Запуск:
  python bot.py

Для выхода: выход, пока, exit или Ctrl+C.
"""

import difflib

from chatbot.knowledge import load_knowledge


# ───────────────────────── НАСТРОЙКИ ─────────────────────────
TONE = "academic"        # "academic" или "friendly"
LEVEL = 3               # 1, 2 или 3
SEMANTIC_METHOD = "vectors"  # для уровня 3: "vectors" или "tfidf"
# ─────────────────────────────────────────────────────────────

_EXIT_WORDS = {"выход", "выйти", "пока", "exit", "quit", "стоп"}


def _is_exit(text, vocab):
    """
    Распознаёт команду выхода, в том числе с лёгкой опечаткой («выхд»).

    Опечатка засчитывается только для одиночного слова, которого нет
    среди ключевых слов базы знаний, чтобы слово «вход» (личный кабинет)
    не принималось за «выход».
    """
    t = text.lower().strip()
    if t in _EXIT_WORDS:
        return True
    if len(t.split()) == 1 and len(t) >= 4 and t not in vocab:
        if difflib.get_close_matches(t, list(_EXIT_WORDS), n=1, cutoff=0.8):
            return True
    return False


def build(tone, level, method):
    """Возвращает (функция предобработки, объект сопоставителя)."""
    knowledge = load_knowledge(tone)
    if level == 1:
        from chatbot.preprocess import preprocess
        from chatbot.matcher_level1 import MatcherLevel1
        return preprocess, MatcherLevel1(knowledge)
    if level == 2:
        from chatbot.preprocess2 import preprocess2
        from chatbot.matcher_level2 import MatcherLevel2
        return preprocess2, MatcherLevel2(knowledge)
    if level == 3:
        try:
            from chatbot.preprocess2 import preprocess2
            from chatbot.matcher_level3 import MatcherLevel3
            return preprocess2, MatcherLevel3(knowledge, method=method)
        except ImportError as exc:
            print(
                "[!] Уровень 3 недоступен: не загрузились numpy/scikit-learn\n"
                f"    ({exc}).\n"
                "[!] Частая причина — запуск с сетевого (UNC) пути в Windows.\n"
                "[!] Переключаюсь на уровень 2.\n"
            )
            from chatbot.preprocess2 import preprocess2
            from chatbot.matcher_level2 import MatcherLevel2
            return preprocess2, MatcherLevel2(knowledge)
    raise ValueError(f"Уровень {level} не реализован.")


def main():
    preprocess, matcher = build(TONE, LEVEL, SEMANTIC_METHOD)
    vocab = getattr(matcher, "vocab", set())

    # Сведения о движке обработки текста.
    engine_note = ""
    if LEVEL >= 2:
        try:
            from chatbot.nlp_spacy import get_engine_info
            info = get_engine_info()
            if info["engine"] == "spacy":
                engine_note = f", NLP: spaCy/{info['model']}"
            else:
                engine_note = ", NLP: pymorphy3 (spaCy недоступен)"
        except Exception:
            pass
    if LEVEL == 3:
        engine_note += f", семантика: {SEMANTIC_METHOD}"

    print(
        f"Информационный помощник Университета «Синергия» "
        f"(уровень {LEVEL}, тон: {TONE}{engine_note}).\n"
        f"Задайте вопрос или введите «выход» для завершения.\n"
    )

    farewell = None
    for cat in load_knowledge(TONE)["categories"]:
        if cat["code"] == "farewell":
            farewell = cat["answer"]
            break

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nБот: Завершение работы.")
            break

        if not user_input:
            continue
        if _is_exit(user_input, vocab):
            print(f"Бот: {farewell}")
            break

        lemmas = preprocess(user_input)
        code, answer, score = matcher.match(lemmas, user_input)
        print(f"Бот: {answer}")


if __name__ == "__main__":
    main()
