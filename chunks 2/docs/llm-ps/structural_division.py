import re, os, glob, json
root = os.getcwd()
fpath = glob.glob(os.path.join(root, 'articles v2/mds/*LLM-PS*'))[0]
with open(fpath, 'r', encoding='utf-8-sig') as fh: lines = fh.readlines()
print(f'Lines: {len(lines)}')
secs=[]; cur={'level':0,'title':'Preamble','lines':[],'start':1}
for i,ln in enumerate(lines,1):
    m2=re.match(r'^## (.+)',ln); m3=re.match(r'^### (.+)',ln); m4=re.match(r'^#### (.+)',ln)
    if m2:
        if cur['lines']: secs.append(cur)
        cur={'level':2,'title':m2.group(1).strip(),'lines':[ln],'start':i}
    elif m3:
        if cur['lines']: secs.append(cur)
        cur={'level':3,'title':m3.group(1).strip(),'lines':[ln],'start':i}
    elif m4:
        if cur['lines']: secs.append(cur)
        cur={'level':4,'title':m4.group(1).strip(),'lines':[ln],'start':i}
    else: cur['lines'].append(ln)
if cur['lines']: secs.append(cur)
def tok(t): return len(re.findall(r'\w+',t))+len(re.findall(r'\$',t))//2
for s in secs:
    e=s['start']+len(s['lines'])-1
    print(f'{s["start"]:4d}-{e:<4d} | {s["level"]:3d} | {s["title"][:70]:70s} | {tok("".join(s["lines"])):5d}')
