"""
bench_speed.py — замер скорости работы чат-бота.

Кладётся в корень проекта (рядом с bot.py).

Запуск:
    python bench_speed.py               # вывод только в консоль
    python bench_speed.py --html        # + HTML-отчёт с меткой времени
    python bench_speed.py --html --out report.html   # своё имя файла

Ничего в проекте не меняет: только импортирует готовые модули,
строит распознаватель уровня 3 и измеряет время.
"""

import os
import sys
import time
import html
import platform
import argparse
import datetime
import statistics

# чтобы импорт пакета chatbot работал при запуске из корня проекта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPS = 50           # повторов на фразу для усреднения
TONE = "academic"
METHOD = "vectors"  # семантический движок уровня 3

# фразы подобраны так, чтобы задействовать разные ветви распознавания
PHRASES = [
    ("Сколько стоит обучение?",            "ключевые слова",                  False),
    ("Где общага?",                        "синоним",                         False),
    ("Где общжитие?",                      "опечатка",                        False),
    ("реально ли вообще не платить",       "ключевое слово в перефразировке", False),
    ("куда селят приезжих студентов",      "перефразировка -> векторы",       True),
    ("можно грызть гранит науки из дома",  "перефразировка -> векторы",       True),
]


def measure():
    """Выполняет замер и возвращает словарь с результатами."""
    t0 = time.perf_counter()
    from chatbot.knowledge import load_knowledge
    from chatbot.nlp_spacy import get_engine_info
    from chatbot.preprocess2 import preprocess2
    from chatbot.matcher_level3 import MatcherLevel3
    knowledge = load_knowledge(TONE)
    matcher = MatcherLevel3(knowledge, method=METHOD)
    startup_ms = (time.perf_counter() - t0) * 1000.0

    info = get_engine_info()

    # прогрев: первый вызов всегда дороже, его в замер не берём
    for text, _, _ in PHRASES:
        matcher.last_intent = None
        matcher.match(preprocess2(text), text)

    rows = []
    for text, kind, is_vec in PHRASES:
        pre_t, mat_t = [], []
        for _ in range(REPS):
            matcher.last_intent = None
            a = time.perf_counter()
            lemmas = preprocess2(text)
            b = time.perf_counter()
            matcher.match(lemmas, text)
            c = time.perf_counter()
            pre_t.append((b - a) * 1000.0)
            mat_t.append((c - b) * 1000.0)
        pre, mat = statistics.mean(pre_t), statistics.mean(mat_t)
        rows.append({"text": text, "kind": kind, "vec": is_vec,
                     "pre": pre, "mat": mat, "tot": pre + mat})

    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "engine": info.get("engine"),
        "model": info.get("model", "-"),
        "has_vectors": info.get("has_vectors"),
        "startup_ms": startup_ms,
        "reps": REPS,
        "rows": rows,
        "avg": statistics.mean(r["tot"] for r in rows),
        "stamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def print_console(d):
    print()
    print("Окружение:")
    print(f"  Python   : {d['python']} ({d['system']})")
    print(f"  Движок   : {d['engine']} / модель {d['model']} / "
          f"has_vectors={d['has_vectors']}")
    print()
    print("Однократная загрузка при старте:")
    print(f"  модель + построение эталонов категорий: {d['startup_ms']:9.1f} мс")
    print()
    if not d["has_vectors"]:
        print("  ВНИМАНИЕ: векторов нет (нужна модель ru_core_news_lg).")
        print("  Времена ветви <векторы> в этом окружении не показательны.")
        print()

    header = (f"{'Фраза':38} | {'тип':32} | "
              f"{'предобр., мс':>13} | {'распозн., мс':>13} | {'всего, мс':>10}")
    print(f"Время распознавания (среднее по {d['reps']} повторам):")
    print(header)
    print("-" * len(header))
    for r in d["rows"]:
        show = r["text"] if len(r["text"]) <= 37 else r["text"][:36] + "."
        print(f"{show:38} | {r['kind']:32} | "
              f"{r['pre']:13.2f} | {r['mat']:13.2f} | {r['tot']:10.2f}")
    print("-" * len(header))
    print(f"{'Среднее по всем фразам':38} | {'':32} | "
          f"{'':>13} | {'':>13} | {d['avg']:10.2f}")
    print()


def write_html(d, path):
    e = html.escape
    warn = ""
    if not d["has_vectors"]:
        warn = ('<p class="warn">Внимание: запуск без модели ru_core_news_lg '
                '(has_vectors=False). Времена векторной ветви не показательны.</p>')

    trs = []
    for r in d["rows"]:
        cls = ' class="vec"' if r["vec"] else ""
        trs.append(
            f"<tr{cls}><td>{e(r['text'])}</td><td>{e(r['kind'])}</td>"
            f"<td class='n'>{r['pre']:.2f}</td><td class='n'>{r['mat']:.2f}</td>"
            f"<td class='n b'>{r['tot']:.2f}</td></tr>"
        )
    rows_html = "\n".join(trs)

    doc = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Отчёт о скорости работы чат-бота</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
         color: #1f2430; max-width: 880px; margin: 32px auto; padding: 0 18px;
         line-height: 1.5; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h2 {{ font-size: 17px; margin-top: 26px; }}
  .sub {{ color: #6b7280; font-size: 13px; margin-bottom: 22px; }}
  .env {{ background: #f6f7f9; border: 1px solid #e3e6ea; border-radius: 8px;
         padding: 12px 16px; font-size: 14px; margin-bottom: 18px; }}
  .env div {{ margin: 2px 0; }}
  .k {{ color: #6b7280; display: inline-block; width: 150px; }}
  .startup {{ font-size: 16px; margin: 14px 0 8px; }}
  .startup b {{ font-size: 20px; color: #1d4ed8; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ border: 1px solid #e3e6ea; padding: 7px 10px; text-align: left; }}
  th {{ background: #eef2f7; font-weight: 600; }}
  td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.b {{ font-weight: 600; }}
  tr.vec {{ background: #eff5ff; }}
  tfoot td {{ font-weight: 600; background: #f6f7f9; }}
  .warn {{ background: #fff4e5; border: 1px solid #ffd9a0; border-radius: 6px;
          padding: 8px 12px; font-size: 13px; }}
  .note {{ color: #6b7280; font-size: 13px; margin-top: 14px; }}
  .leg {{ display: inline-block; width: 12px; height: 12px; background: #eff5ff;
         border: 1px solid #cfe0ff; vertical-align: middle; margin-right: 6px; }}
</style></head><body>
<h1>Отчёт о скорости работы чат-бота</h1>
<div class="sub">Информационный помощник абитуриентов Университета «Синергия». Сформирован {e(d['stamp'])}.</div>

<div class="env">
  <div><span class="k">Python</span> {e(d['python'])} ({e(d['system'])})</div>
  <div><span class="k">Движок NLP</span> {e(str(d['engine']))}</div>
  <div><span class="k">Модель</span> {e(str(d['model']))}</div>
  <div><span class="k">Векторы</span> has_vectors = {e(str(d['has_vectors']))}</div>
  <div><span class="k">Повторов на фразу</span> {d['reps']}</div>
</div>
{warn}

<div class="startup">Однократная загрузка при старте
  (модель и построение эталонов категорий): <b>{d['startup_ms']:.0f} мс</b></div>

<h2>Время распознавания одной фразы</h2>
<table>
<thead><tr><th>Фраза</th><th>Тип запроса</th>
<th style="text-align:right">Предобработка, мс</th>
<th style="text-align:right">Распознавание, мс</th>
<th style="text-align:right">Всего, мс</th></tr></thead>
<tbody>
{rows_html}
</tbody>
<tfoot><tr><td colspan="4">Среднее по всем фразам</td>
<td class="n">{d['avg']:.2f}</td></tr></tfoot>
</table>

<p class="note"><span class="leg"></span>Голубым выделены фразы-перефразировки
без общих слов с базой знаний — они доходят до семантической (векторной)
ветви распознавания, наиболее затратной по времени.</p>
<p class="note">Каждое значение — среднее по {d['reps']} повторам после прогрева.
Замер выполнен на указанном окружении; на другой машине абсолютные значения
будут отличаться, соотношение между ветвями сохраняется.</p>
</body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


def main():
    ap = argparse.ArgumentParser(description="Замер скорости работы чат-бота.")
    ap.add_argument("--html", action="store_true",
                    help="дополнительно сохранить HTML-отчёт")
    ap.add_argument("--out", metavar="ФАЙЛ", default=None,
                    help="имя HTML-файла (по умолчанию с меткой времени)")
    args = ap.parse_args()

    data = measure()
    print_console(data)

    if args.html or args.out:
        out = args.out
        if not out:
            stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M")
            out = f"bench_report_{stamp}.html"
        path = write_html(data, out)
        print(f"HTML-отчёт сохранён: {os.path.abspath(path)}")
        print()


if __name__ == "__main__":
    main()
