"""
THE DECODED-KEY JOIN. The zodiac centers are the ONE decoded header (the Occitan
month-names). Test whether the ~30 day-records (star-labels) around each medallion
carry structure RELATIVE to the known month:
  A. Are months distinct? within-medallion vs across-medallion label overlap.
  B. KEY JOIN: does label morphology vary with the month-ordinal (the known key)?
     If yes => the records 'know' their month (referential). If no => month-blind fill.
  C. ORDINAL: within a ring (labels in transcription/ring order), do consecutive
     labels progress (day 1..30) — decay of similarity with ring-distance?
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(23)
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph']
SUF=['eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y']
def morphs(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s

def fk(f): m=re.match(r'f(\d+)([rv])(\d*)',f); return (int(m.group(1)),0 if m.group(2)=='r' else 1,int(m.group(3) or 0))
# zodiac label-folios in sequence (proxy for month-ordinal)
zfol=[f for f in sorted(C['folios'],key=fk) if re.match(r'f7[0-3][rv]',f)]
medallions=[]
for f in zfol:
    labs=[w for ln in C['folios'][f] if ltype(ln['locus'])=='L' for w in ln['words'] if '?' not in w and len(w)>=2]
    if len(labs)>=10: medallions.append((f,labs))
for i,(f,labs) in enumerate(medallions):
    print(f"  ordinal {i:2d}  {f:7s}  {len(labs)} day-labels")
print(f"\n{len(medallions)} medallions")

def coreset(labs): return set(morphs(w)[1] for w in labs)
# A. within vs across medallion overlap
def jac(a,b): u=a|b; return len(a&b)/len(u) if u else 0
cs=[coreset(l) for _,l in medallions]
across=np.mean([jac(cs[i],cs[j]) for i in range(len(cs)) for j in range(i+1,len(cs))])
print(f"\nA. label core-overlap across different medallions: {across:.3f}")
print(f"   (if months were distinct topics this would be LOW; high = shared day-vocab)")

# B. KEY JOIN: does label morphology correlate with month-ordinal?
ordn=np.arange(len(medallions))
feats={'mean_len':[],'q_rate':[],'suf_y':[],'pre_ok':[]}
for _,labs in medallions:
    feats['mean_len'].append(np.mean([len(w) for w in labs]))
    feats['q_rate'].append(np.mean([morphs(w)[0].startswith('q') for w in labs]))
    feats['suf_y'].append(np.mean([morphs(w)[2]=='y' for w in labs]))
    feats['pre_ok'].append(np.mean([morphs(w)[0] in ('ok','ot','o') for w in labs]))
print(f"\nB. correlation of label features with month-ordinal (the known key):")
for k,v in feats.items():
    r=np.corrcoef(ordn,v)[0,1]
    print(f"   {k:9s}: r={r:+.2f}")
print("   (|r|~0 across the board => the day-records are month-BLIND: no referential join)")

# C. ORDINAL within ring: consecutive vs distant label similarity
def ring_decay(labs):
    cs=[set(morphs(w)) for w in labs]
    adj=np.mean([jac(cs[i],cs[i+1]) for i in range(len(cs)-1)])
    far=np.mean([jac(cs[i],cs[j]) for i in range(len(cs)) for j in range(i+3,len(cs))])
    return adj,far
adjs=[];fars=[]
for _,labs in medallions:
    a,f=ring_decay(labs); adjs.append(a);fars.append(f)
print(f"\nC. within-ring progression (labels in ring order):")
print(f"   adjacent-label similarity {np.mean(adjs):.3f}  vs distant {np.mean(fars):.3f}")
print(f"   gap {np.mean(adjs)-np.mean(fars):+.3f}  (positive = a day-by-day progression)")
