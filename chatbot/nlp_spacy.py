"""
nlp_spacy.py — обработка текста средствами spaCy (уровни 2 и 3).

Модуль загружает русскую модель spaCy и предоставляет лемматизацию
с учётом контекста предложения. Модель ищется по убыванию размера:
ru_core_news_lg -> ru_core_news_md -> ru_core_news_sm. Это позволяет
проекту работать на любой из установленных моделей.

Если ни одна модель spaCy не найдена, модуль переключается на
pymorphy3, чтобы программа сохранила работоспособность. Для уровня 3
(векторное сходство) требуется модель md или lg, содержащая векторы;
модель sm векторов не имеет.
"""

import re

# Список моделей в порядке предпочтения (от точной к лёгкой).
_MODEL_NAMES = ["ru_core_news_lg", "ru_core_news_md", "ru_core_news_sm"]

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

_nlp = None          # загруженная модель spaCy (или None)
_model_name = None   # имя загруженной модели
_has_vectors = False # есть ли у модели векторы (нужно для уровня 3)

# Запасной морфоанализатор на случай отсутствия spaCy.
_morph = None


def _try_load_spacy():
    """Пытается загрузить первую доступную модель spaCy."""
    global _nlp, _model_name, _has_vectors
    try:
        import spacy
    except ImportError:
        return False
    for name in _MODEL_NAMES:
        try:
            _nlp = spacy.load(name)
            _model_name = name
            _has_vectors = _nlp.vocab.vectors_length > 0
            return True
        except OSError:
            continue
    return False


def _init():
    """Инициализирует NLP-движок: сначала spaCy, иначе pymorphy3."""
    global _morph
    if _try_load_spacy():
        return
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()


# Инициализация при импорте модуля.
_init()


def get_engine_info():
    """Возвращает сведения о текущем движке (для журнала и отчёта)."""
    if _nlp is not None:
        return {
            "engine": "spacy",
            "model": _model_name,
            "has_vectors": _has_vectors,
        }
    return {"engine": "pymorphy3", "model": None, "has_vectors": False}


def lemmatize(text):
    """
    Приводит фразу к списку лемм с учётом контекста (если доступен spaCy).

    Слова латиницей и цифры сохраняются без изменений.

    :param text: входная строка
    :return: список лемм (с возможными повторами, порядок сохранён)
    """
    text = text.lower()
    if _nlp is not None:
        lemmas = []
        for token in _nlp(text):
            if token.is_space or token.is_punct:
                continue
            t = token.text
            if re.fullmatch(r"[a-z0-9]+", t):
                lemmas.append(t)
            else:
                lemmas.append(token.lemma_)
        return lemmas
    # Запасной путь: pymorphy3.
    lemmas = []
    for tok in _TOKEN_RE.findall(text):
        if re.fullmatch(r"[a-z0-9]+", tok):
            lemmas.append(tok)
        else:
            lemmas.append(_morph.parse(tok)[0].normal_form)
    return lemmas


def get_doc(text):
    """
    Возвращает обработанный объект spaCy Doc для входного текста
    (нужен уровню 3 для доступа к векторам). Если spaCy недоступен,
    возвращает None.
    """
    if _nlp is None:
        return None
    return _nlp(text.lower())
