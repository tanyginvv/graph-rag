"""
Скрипт структурного деления для статьи one-for-all.
Разбивает markdown по заголовкам ##/###, выделяет таблицы,
формирует первичные чанки с оценкой token_count.

Использование:
    python chunks 2/docs/one-for-all/structural_division.py

Вход:  articles v2/mds/One-for-All A Lightweight Stabilized and Parameter-Efficient Pre-trained LLM for Time Series Forecasting.md
Выход: stdout — таблица структурных блоков с номерами строк и оценкой токенов.
"""

import re
import os

# Путь к исходной статье (от корня проекта)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARTICLE_PATH = os.path.join(
    PROJECT_ROOT,
    'articles v2', 'mds',
    'One-for-All A Lightweight Stabilized and Parameter-Efficient Pre-trained LLM for Time Series Forecasting.md'
)

def estimate_tokens(text):
    """Грубая оценка: слова + LaTeX-формулы (~1 токен на формулу)."""
    words = len(re.findall(r'\w+', text))
    latex = len(re.findall(r'\$', text)) // 2
    return words + latex

def has_table(text):
    """Проверяет, содержит ли блок markdown-таблицу."""
    return bool(re.search(r'\|[-:| ]+\|', text))

def split_by_headers(filepath):
    """Разбивает markdown-файл на структурные блоки по заголовкам ##, ###, ####."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sections = []
    current = {'level': 0, 'title': 'Preamble', 'lines': [], 'start': 1}

    for i, line in enumerate(lines, 1):
        m2 = re.match(r'^## (.+)', line)
        m3 = re.match(r'^### (.+)', line)
        m4 = re.match(r'^#### (.+)', line)

        if m2:
            if current['lines']:
                sections.append(current)
            current = {'level': 2, 'title': m2.group(1).strip(), 'lines': [line], 'start': i}
        elif m3:
            if current['lines']:
                sections.append(current)
            current = {'level': 3, 'title': m3.group(1).strip(), 'lines': [line], 'start': i}
        elif m4:
            if current['lines']:
                sections.append(current)
            current = {'level': 4, 'title': m4.group(1).strip(), 'lines': [line], 'start': i}
        else:
            current['lines'].append(line)

    if current['lines']:
        sections.append(current)

    return sections

def main():
    sections = split_by_headers(ARTICLE_PATH)

    print(f"{'Line':>5s} | Lvl | {'Section':<75s} | {'Lines':>5s} | {'Tokens':>6s} | Note")
    print("-" * 120)

    for s in sections:
        text = ''.join(s['lines'])
        tok = estimate_tokens(text)
        marker = " [TABLE]" if has_table(text) else ""
        # Сокращаем заголовок если длинный
        title = s['title'][:75]
        end_line = s['start'] + len(s['lines']) - 1
        line_range = f"{s['start']}-{end_line}"
        print(f"{line_range:>5s} | {s['level']:3d} | {title:<75s} | {len(s['lines']):5d} | {tok:6d} |{marker}")

    # Сводка по подразделам (###)
    print("\n--- Подразделы (###) с оценкой токенов ---")
    for s in sections:
        if s['level'] == 3:
            text = ''.join(s['lines'])
            tok = estimate_tokens(text)
            end_line = s['start'] + len(s['lines']) - 1
            print(f"  L{s['start']:4d}-L{end_line:4d} | {s['title'][:70]:70s} | ~{tok:5d} tok")

    # Итог
    total_tok = sum(estimate_tokens(''.join(s['lines'])) for s in sections)
    print(f"\nВсего блоков: {len(sections)}, суммарно ~{total_tok} токенов")

if __name__ == '__main__':
    main()
