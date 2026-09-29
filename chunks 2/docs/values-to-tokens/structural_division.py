"""
Скрипт структурного деления для статьи values-to-tokens (TokenCast).
"""
import re, os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARTICLE_PATH = os.path.join(PROJECT_ROOT, 'articles v2', 'mds',
    'FROM VALUES TO TOKENS AN LLM-DRIVEN FRAMEWORK FOR CONTEXT-AWARE TIME SERIES FORECASTING VIA SYMBOLIC DISCRETIZATION.md')

def tok_est(text):
    return len(re.findall(r'\w+', text)) + len(re.findall(r'\$', text))//2

def has_tbl(text):
    return bool(re.search(r'\|[-:| ]+\|', text))

def split(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    secs = []; cur = {'level':0,'title':'Preamble','lines':[],'start':1}
    for i,ln in enumerate(lines,1):
        m2=re.match(r'^## (.+)',ln); m3=re.match(r'^### (.+)',ln); m4=re.match(r'^#### (.+)',ln)
        if m2:
            if cur['lines']: secs.append(cur)
            cur = {'level':2,'title':m2.group(1).strip(),'lines':[ln],'start':i}
        elif m3:
            if cur['lines']: secs.append(cur)
            cur = {'level':3,'title':m3.group(1).strip(),'lines':[ln],'start':i}
        elif m4:
            if cur['lines']: secs.append(cur)
            cur = {'level':4,'title':m4.group(1).strip(),'lines':[ln],'start':i}
        else: cur['lines'].append(ln)
    if cur['lines']: secs.append(cur)
    return secs

secs = split(ARTICLE_PATH)
print(f"{'Line':>5s} | Lvl | {'Section':<65s} | {'Lines':>5s} | {'Tok':>6s}")
print("-"*100)
for s in secs:
    t=''.join(s['lines']); tok=tok_est(t)
    e=s['start']+len(s['lines'])-1
    print(f"{s['start']:4d}-{e:<4d} | {s['level']:3d} | {s['title'][:65]:65s} | {len(s['lines']):5d} | {tok:6d}")
print(f"\nTotal blocks: {len(secs)}, ~{sum(tok_est(''.join(s['lines'])) for s in secs)} tokens")
