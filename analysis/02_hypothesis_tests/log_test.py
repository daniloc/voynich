"""
MANIFEST vs TRANSITION-LOG.
Manifest = a set of items, EXCHANGEABLE: reorder the entries and nothing is lost.
Transition-log = a TRAJECTORY: same ingredients, but priority/proportion drift from
entry to entry, so ORDER carries process information even without grammar.

Discriminators (within paragraphs = within one 'entry sequence'):
  T1 LAG DECAY: in a log, nearby lines are more similar than distant lines (a drifting
     state); shuffle line order and that distance-dependence vanishes. In a manifest/bag
     the lag curve is FLAT.
  T2 DIRECTIONAL DRIFT: in a log the distribution moves AWAY from where it started
     (monotone dissimilarity vs distance); a bag fluctuates with no net direction.
  T3 PROPORTION SHIFT: morpheme proportions change systematically from start→end of a
     paragraph (priority/proportion evolving through the entry).
We control for topic by working WITHIN paragraphs and comparing to within-paragraph
shuffles, so any signal is sequence/order, not section vocabulary.
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(3)
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
def folio_key(f):
    m=re.match(r'f(\d+)([rv])(\d*)',f); return (int(m.group(1)),0 if m.group(2)=='r' else 1,int(m.group(3) or 0))
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph']
SUF=['eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y']
def morphs(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    c=r[:len(r)-len(s)] if s else r
    return [m for m in (p,'#'+c,s) if m and m!='#']

# build paragraphs as ordered lists of line morpheme-bags
paras=[]; cur=[]
for f in sorted(C['folios'],key=folio_key):
    for ln in C['folios'][f]:
        if ltype(ln['locus'])!='P': continue
        new = '@' in ln['locus']
        ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
        if not ws: continue
        bag=Counter(m for w in ws for m in morphs(w))
        if new and cur: paras.append(cur); cur=[]
        cur.append(bag)
    # folio break also ends a paragraph
    if cur: paras.append(cur); cur=[]
paras=[p for p in paras if len(p)>=6]
print(f"paragraphs (>=6 lines): {len(paras)}  total lines: {sum(len(p) for p in paras)}")

def cos(a,b):
    keys=set(a)|set(b); na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
    return sum(a[k]*b[k] for k in keys)/(na*nb) if na and nb else 0

# ---- T1 lag decay ------------------------------------------------------------
def lag_curve(plist, maxlag=6):
    sums=defaultdict(list)
    for p in plist:
        for i in range(len(p)):
            for d in range(1,maxlag+1):
                if i+d<len(p): sums[d].append(cos(p[i],p[i+d]))
    return {d:np.mean(v) for d,v in sums.items()}
real=lag_curve(paras)
def shuffle_para(p):
    idx=rng.permutation(len(p)); return [p[i] for i in idx]
shuf=lag_curve([shuffle_para(p) for p in paras])
print("\nT1  similarity vs lag (within paragraph):")
print("   lag :  "+"  ".join(f"{d}" for d in range(1,7)))
print("   real:  "+"  ".join(f"{real[d]:.3f}" for d in range(1,7)))
print("   shuf:  "+"  ".join(f"{shuf[d]:.3f}" for d in range(1,7)))
print(f"   adjacent(lag1) real {real[1]:.3f} vs shuf {shuf[1]:.3f};  decay real {real[1]-real[6]:+.3f} vs shuf {shuf[1]-shuf[6]:+.3f}")

# permutation test on adjacent-vs-distant gap (real) and real-vs-shuffle
def adj_gap(plist):
    a=[];f=[]
    for p in plist:
        for i in range(len(p)-1): a.append(cos(p[i],p[i+1]))
        a.append(0)  # placeholder noop
    # adjacent mean minus far mean
    adj=np.mean([cos(p[i],p[i+1]) for p in plist for i in range(len(p)-1)])
    far=np.mean([cos(p[i],p[j]) for p in plist for i in range(len(p)) for j in range(i+3,len(p))])
    return adj,far
adj,far=adj_gap(paras)
nulls=[]
for _ in range(400):
    sp=[shuffle_para(p) for p in paras]
    a,f=adj_gap(sp); nulls.append(a-f)
obs=adj-far; nulls=np.array(nulls); p1=(np.sum(nulls>=obs)+1)/(len(nulls)+1)
print(f"   adjacent−distant similarity gap: real {obs:+.3f}  vs shuffled {nulls.mean():+.3f}  p={p1:.4f}")

# ---- T2 directional drift ----------------------------------------------------
# does dissimilarity from the FIRST line grow monotonically with position?
drift=[]
for p in paras:
    d0=[1-cos(p[0],p[i]) for i in range(len(p))]
    # slope of dissimilarity vs index
    x=np.arange(len(p)); drift.append(np.polyfit(x,d0,1)[0])
drift=np.array(drift)
# null: same but shuffled order
drift_sh=[]
for p in paras:
    sp=shuffle_para(p); d0=[1-cos(sp[0],sp[i]) for i in range(len(sp))]
    drift_sh.append(np.polyfit(np.arange(len(sp)),d0,1)[0])
drift_sh=np.array(drift_sh)
print(f"\nT2  drift-away-from-start slope: real {drift.mean():+.4f}  vs shuffled {drift_sh.mean():+.4f}")
print(f"    fraction of paragraphs with positive drift: real {100*np.mean(drift>0):.0f}%  shuf {100*np.mean(drift_sh>0):.0f}%")

# ---- T3 proportion shift start->end ------------------------------------------
firsts=Counter(); lasts=Counter()
for p in paras:
    h=len(p)//3 or 1
    for b in p[:h]: firsts+=b
    for b in p[-h:]: lasts+=b
def dist(c): n=sum(c.values()); return {k:v/n for k,v in c.items()}
fd,ld=dist(firsts),dist(lasts)
shift=sorted(set(fd)|set(ld), key=lambda m: (ld.get(m,0)-fd.get(m,0)))
print("\nT3  morphemes that RISE start→end (proportion grows through the entry):")
print("   ", [(m,round(ld.get(m,0)-fd.get(m,0),3)) for m in shift[-6:]][::-1])
print("    morphemes that FALL start→end:")
print("   ", [(m,round(fd.get(m,0)-ld.get(m,0),3)) for m in shift[:6]])
print("\nverdict cue: real>shuffle on T1/T2 and a systematic T3 shift = TRANSITION LOG")
print("            (ordered trajectory). flat curves + no drift = exchangeable manifest.")
