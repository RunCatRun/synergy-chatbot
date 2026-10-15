"""
semantic.py — семантическое сходство фразы с категориями (уровень 3).

Два сменных движка с единым интерфейсом match(text) -> (код, сходство):

  - VectorBackend  — векторное сходство на модели spaCy ru_core_news_lg.
    Для каждой категории строится эталонный вектор как среднее векторов
    её ключевых слов. Входная фраза переводится в вектор, сравнение —
    косинусным сходством. Понимает перефразировки без общих слов.
    Требует модель с векторами (lg или md); на модели sm недоступен.

  - TfidfBackend   — статистическое сходство TF-IDF (scikit-learn).
    Категория представляется «документом» из лемм её ключевых слов и
    ответа; запрос сравнивается с документами категорий в пространстве
    TF-IDF. Метод лексический: улавливает совпадение слов с поправкой
    на их распространённость, но не понимает смысл незнакомых слов.
    Не требует векторной модели.

Оба движка работают по леммам, что уравнивает словоформы русского языка.
"""

import numpy as np

from chatbot.nlp_spacy import lemmatize, get_doc, get_engine_info


def _cosine(a, b):
    """Косинусное сходство двух векторов numpy (0, если вектор нулевой)."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _lemma_text(words):
    """Превращает список слов/фраз в строку лемм (для TF-IDF и эталонов)."""
    parts = []
    for w in words:
        parts.extend(lemmatize(w))
    return " ".join(parts)


class VectorBackend:
    """Векторное сходство на модели spaCy с векторами."""

    def __init__(self, knowledge):
        info = get_engine_info()
        self.available = bool(info.get("has_vectors"))
        self.codes = []
        self.protos = []
        if not self.available:
            return
        # Эталонный вектор категории — вектор строки её ключевых слов.
        for cat in knowledge["categories"]:
            text = " ".join(cat["keywords"])
            doc = get_doc(text)
            if doc is None:
                continue
            self.codes.append(cat["code"])
            self.protos.append(doc.vector)
        self.protos = np.array(self.protos) if self.protos else np.array([])

    def match(self, text):
        """Возвращает (код категории, сходство 0..1) или (None, 0.0)."""
        if not self.available or len(self.protos) == 0:
            return None, 0.0
        doc = get_doc(text)
        if doc is None:
            return None, 0.0
        vec = doc.vector
        best_code, best_sim = None, 0.0
        for code, proto in zip(self.codes, self.protos):
            sim = _cosine(vec, proto)
            if sim > best_sim:
                best_sim, best_code = sim, code
        return best_code, best_sim


class TfidfBackend:
    """Статистическое сходство TF-IDF на леммах (scikit-learn)."""

    def __init__(self, knowledge):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.available = True
        self.codes = []
        corpus = []
        # Документ категории = леммы ключевых слов и текста ответа.
        for cat in knowledge["categories"]:
            doc_text = _lemma_text(cat["keywords"]) + " " + _lemma_text([cat["answer"]])
            self.codes.append(cat["code"])
            corpus.append(doc_text)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(corpus).toarray()

    def match(self, text):
        """Возвращает (код категории, сходство 0..1) или (None, 0.0)."""
        query = _lemma_text([text])
        if not query.strip():
            return None, 0.0
        qvec = self.vectorizer.transform([query]).toarray()[0]
        best_code, best_sim = None, 0.0
        for code, row in zip(self.codes, self.matrix):
            sim = _cosine(qvec, row)
            if sim > best_sim:
                best_sim, best_code = sim, code
        return best_code, best_sim


def build_backend(method, knowledge):
    """
    Создаёт семантический движок по имени метода.

    :param method: "vectors" или "tfidf"
    :param knowledge: база знаний
    """
    if method == "vectors":
        return VectorBackend(knowledge)
    if method == "tfidf":
        return TfidfBackend(knowledge)
    raise ValueError(f"Неизвестный метод: {method!r}. Допустимо: vectors, tfidf.")
