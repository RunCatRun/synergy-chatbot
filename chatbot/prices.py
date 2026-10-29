"""
prices.py — ответ о стоимости по набору данных data/prices.

Если в вопросе названо направление подготовки или программа из набора,
общий ответ о программах или стоимости заменяется перечнем найденных
программ с ценами. Набор — выборка из приказов Университета о стоимости
на 2025/2026 учебный год (см. data/prices/README.md).
"""

import glob
import json
import os
import re

from chatbot.nlp_spacy import lemmatize
from chatbot.preprocess2 import _STOPWORDS

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PRICES_DIR = os.path.join(_BASE_DIR, "data", "prices")

# Категории, общий ответ которых уточняется ценами, если названа программа.
# Узкие намерения (бюджет, поступление, документы и др.) не трогаются.
_REFINED = {"programs", "cost", "about", "help", "fallback",
            "faculties", "it_faculties"}
_LEVEL_ORDER = ["общее образование", "среднее профессиональное образование",
                "бакалавриат", "специалитет", "магистратура",
                "аспирантура", "ординатура"]
_MAX_ITEMS = 8


def _key(name):
    """Множество значимых лемм названия; уточнения в скобках не учитываются."""
    if not name:
        return frozenset()
    name = re.sub(r"\([^)]*\)", " ", name)
    return frozenset(l for l in lemmatize(name)
                     if l not in _STOPWORDS and len(l) > 1)


def _load(folder=_PRICES_DIR):
    records = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data["meta"]
        for item in data["items"]:
            record = dict(item)
            record["level"] = meta["level"]
            record["applies_to"] = meta["applies_to"]
            record["academic_year"] = meta["academic_year"]
            record["_dkey"] = _key(item["direction"])
            record["_pkey"] = _key(item.get("program"))
            records.append(record)
    return records


_RECORDS = _load()


def _money(n):
    return f"{n:,}".replace(",", " ")


def _cost(p):
    text = f"{_money(p['price_year'])} ₽ в год"
    if p.get("price_total"):
        text += f", {_money(p['price_total'])} ₽ за весь период"
    return text


def _line(r):
    parts = [r["level"]]
    if r.get("program"):
        parts.append(f"«{r['program']}»")
    parts.append(f"{r['form']} форма")
    if "by_base" in r:
        bases = [f"после {b} класса — {_cost(r['by_base'][b])}"
                 for b in ("9", "11") if b in r["by_base"]]
        price = "; ".join(bases)
    else:
        price = _cost(r)
    return "• " + ", ".join(parts) + ": " + price


def find(text):
    """Записи набора, направление или программа которых названы в вопросе."""
    q = {l for l in lemmatize(text) if l not in _STOPWORDS}
    found, names = [], []
    for r in _RECORDS:
        hit = None
        if r["_dkey"] and r["_dkey"] <= q:
            hit = r["direction"]
        elif r["_pkey"] and r["_pkey"] <= q:
            hit = r["program"]
        if hit:
            found.append(r)
            name = re.sub(r"\s*\([^)]*\)", "", hit)
            if name not in names:
                names.append(name)
    found.sort(key=lambda r: _LEVEL_ORDER.index(r["level"])
               if r["level"] in _LEVEL_ORDER else len(_LEVEL_ORDER))
    return found, names


def refine(code, answer, text):
    """Возвращает ответ с ценами, если в вопросе названа программа из набора."""
    if code not in _REFINED:
        return answer
    found, names = find(text)
    if not found:
        return answer
    year = found[0]["academic_year"]
    lines = [f"{', '.join(names)} — стоимость обучения "
             f"в {year} учебном году:"]
    lines += [_line(r) for r in found[:_MAX_ITEMS]]
    if len(found) > _MAX_ITEMS:
        lines.append(f"…и ещё {len(found) - _MAX_ITEMS} программ.")
    applies = {r["applies_to"] for r in found}
    who = f"для {applies.pop()}" if len(applies) == 1 else \
        "для зачисленных в 2025/2026 учебном году"
    lines.append(f"Базовая стоимость по приказам Университета {who}; "
                 f"варианты с особыми условиями не указаны. Актуальные цены "
                 f"и полный перечень программ — на synergyuniversity.ru.")
    return "\n".join(lines)
