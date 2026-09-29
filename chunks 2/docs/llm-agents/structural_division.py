"""
Скрипт структурного деления для статьи llm-agents (DCATS).
Разбивает markdown по заголовкам ##/###, выделяет таблицы,
формирует первичные чанки с оценкой token_count.

Использование:
    python chunks 2/docs/llm-agents/structural_division.py
"""

import re
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARTICLE_PATH = os.path.join(
    PROJECT_ROOT, 'articles v2', 'mds',
    'Empowering Time Series Forecasting with LLM-Agents.md'
)

def estimate_tokens(text):
    words = len(re.findall(r'\w+', text))
    latex = len(re.findall(r'\$', text)) // 2
    return words + latex

def has_table(text):
    return bool(re.search(r'\|[-:| ]+\|', text))

def split_by_headers(filepath):
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
    print(f"{'Line':>5s} | Lvl | {'Section':<65s} | {'Lines':>5s} | {'Tokens':>6s} | Note")
    print("-" * 115)
    for s in sections:
        text = ''.join(s['lines'])
        tok = estimate_tokens(text)
        marker = " [TABLE]" if has_table(text) else ""
        title = s['title'][:65]
        end_line = s['start'] + len(s['lines']) - 1
        print(f"{s['start']:4d}-{end_line:<4d} | {s['level']:3d} | {title:<65s} | {len(s['lines']):5d} | {tok:6d} |{marker}")

    # Subsection summary
    print("\n--- Подразделы (###) ---")
    for s in sections:
        if s['level'] == 3:
            text = ''.join(s['lines'])
            tok = estimate_tokens(text)
            end_line = s['start'] + len(s['lines']) - 1
            print(f"  L{s['start']:4d}-L{end_line:4d} | {s['title'][:65]:65s} | ~{tok:5d} tok")

    total_tok = sum(estimate_tokens(''.join(s['lines'])) for s in sections)
    print(f"\nВсего блоков: {len(sections)}, суммарно ~{total_tok} токенов")

if __name__ == '__main__':
    main()
