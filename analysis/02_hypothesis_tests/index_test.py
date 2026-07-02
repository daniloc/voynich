"""
IS THE FOLIO-OPENER A REAL INDEX (points at the page's content/type)?
Test: does the opener-key predict the folio's SECTION/picture-type?
CONTROL: compare the opener to a RANDOM word from the same folio. If the opener is a
genuine index entry it should predict section BETTER than a random word; if it predicts
only as well as any word, it's just sharing the page's dialect (not a special index).
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(17)
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
def section(f):
    n=int(re.match(r'f(\d+)',f).group(1))
    return ('herbal' if n<=66 else 'astro' if n<=73 else 'bio' if n<=84 else 'cosmo' if n<=86 else 'pharma' if n<=102 else 'recipes')
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph','cfh']
SUF=['eedy','eody','edy','aiin','aiir','ain','iin','dy','ol','or','ar','al','am','dam','ey','eey','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s
def feats(w):
    p,c,s=decomp(w); return {'first':w[0],'prefix':p,'suffix':s,'len':min(8,len(w))}

def fk(f): m=re.match(r'f(\d+)([rv])(\d*)',f); return (int(m.group(1)),0 if m.group(2)=='r' else 1,int(m.group(3) or 0))
folios=[]
for f in sorted(C['folios'],key=fk):
    words=[w for ln in C['folios'][f] if ltype(ln['locus'])=='P' for w in ln['words'] if '?' not in w and len(w)>=2]
    if len(words)<8: continue
    opener=words[0]
    folios.append({'f':f,'sec':section(f),'opener':opener,'words':words})
secs=[d['sec'] for d in folios]
print(f"folios: {len(folios)}  sections: {Counter(secs)}")

def H(xs):
    c=Counter(xs); n=len(xs); return -sum(v/n*math.log2(v/n) for v in c.values())
def MI(xs,ys):
    n=len(xs); jx=Counter(xs);jy=Counter(ys);jj=Counter(zip(xs,ys))
    return sum(c/n*math.log2((c/n)/((jx[x]/n)*(jy[y]/n))) for (x,y),c in jj.items())
Hsec=H(secs)
print(f"H(section) = {Hsec:.2f} bits\n")

# MI(section ; feature) for opener vs random-word-from-folio
print(f"{'feature':9s}{'MI opener':>11}{'MI rand-word':>14}{'ratio':>8}")
for feat in ['first','prefix','suffix','len']:
    op=[feats(d['opener'])[feat] for d in folios]
    mi_op=MI(secs,op)
    mis=[]
    for _ in range(30):
        rw=[feats(rng.choice(d['words']))[feat] for d in folios]
        mis.append(MI(secs,rw))
    mi_rw=np.mean(mis)
    print(f"{feat:9s}{mi_op:>11.3f}{mi_rw:>14.3f}{mi_op/max(mi_rw,1e-6):>8.2f}")

# nearest-centroid section prediction (leave-one-out) from opener feature vector
def vec(w):
    fp=feats(w); v=[]
    v+= [1 if fp['prefix']==p else 0 for p in PRE+['']]
    v+= [1 if fp['suffix']==s else 0 for s in SUF+['']]
    v+= [fp['len']/8]
    return np.array(v,float)
def loo_acc(getword):
    X=np.array([vec(getword(d)) for d in folios]); y=np.array(secs)
    correct=0
    for i in range(len(folios)):
        mask=np.arange(len(folios))!=i
        cents={s:X[mask][y[mask]==s].mean(0) for s in set(y)}
        pred=min(cents, key=lambda s: np.sum((X[i]-cents[s])**2))
        correct+= (pred==y[i])
    return correct/len(folios)
maj=Counter(secs).most_common(1)[0][1]/len(secs)
acc_op=loo_acc(lambda d: d['opener'])
acc_rw=np.mean([loo_acc(lambda d: rng.choice(d['words'])) for _ in range(5)])
print(f"\nsection prediction accuracy (leave-one-out, nearest-centroid):")
print(f"   from OPENER     : {100*acc_op:.0f}%")
print(f"   from RANDOM word: {100*acc_rw:.0f}%")
print(f"   majority baseline: {100*maj:.0f}%")
print("   => opener >> random word means the opener is a SPECIAL index (points at content).")
print("      opener ~ random word means it only carries the page's dialect (not an index).")

# ordinal drift: does opener length / q-rate trend across the book?
oplen=np.array([len(d['opener']) for d in folios]); idx=np.arange(len(folios))
print(f"\nordinal check: corr(folio-order, opener-length) = {np.corrcoef(idx,oplen)[0,1]:+.2f} "
      f"(near 0 = not a counting index)")
