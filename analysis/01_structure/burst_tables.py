"""
PART 1 — BURSTINESS: do words CLUMP topically (real-language content words do) or
spread evenly (a topic-free generator)? Burstiness coefficient B=(σ-μ)/(σ+μ) on the
inter-occurrence gaps. B>0 bursty/clumped, B~0 Poisson/random. Compare Voynich to
English & Latin, and to its own global- and within-section-shuffles (to see if any
clumping is finer than the section level).

PART 2 — SINGLE TABLE vs COLLECTION: one table = stable schema + stable column-domains
throughout; a collection = same record format but DIFFERENT domains per section.
Tests: (a) schema invariance (field-independence the same everywhere?), (b) domain
divergence (do sections use different cores/affix-values?), (c) discreteness.
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(29)
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
def section(f):
    n=int(re.match(r'f(\d+)',f).group(1))
    return ('herbal' if n<=66 else 'astro' if n<=73 else 'bio' if n<=84 else 'cosmo' if n<=86 else 'pharma' if n<=102 else 'recipes')
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph']
SUF=['eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s

# ordered token stream + per-token section
stream=[]; sec_of=[]
def fk(f): m=re.match(r'f(\d+)([rv])(\d*)',f); return (int(m.group(1)),0 if m.group(2)=='r' else 1,int(m.group(3) or 0))
for f in sorted(C['folios'],key=fk):
    for ln in C['folios'][f]:
        if ltype(ln['locus'])!='P': continue
        for w in ln['words']:
            if '?' not in w and len(w)>=2: stream.append(w); sec_of.append(section(f))

def burstiness(tokens, mink=10):
    pos=defaultdict(list)
    for i,w in enumerate(tokens): pos[w].append(i)
    Bs=[]
    for w,ps in pos.items():
        if len(ps)<mink: continue
        g=np.diff(ps); m,s=g.mean(),g.std()
        if m+s>0: Bs.append((s-m)/(s+m))
    return np.mean(Bs), len(Bs)

def words_from(path,keep=120000):
    t=re.sub(r'[^a-z\s]',' ',open(path,encoding='utf-8',errors='ignore').read().lower())
    return t.split()[:keep]

print("="*68); print("PART 1 — BURSTINESS (clumping; >0 = topical, ~0 = random)"); print("="*68)
realB,n=burstiness(stream)
gsh=stream[:]; rng.shuffle(gsh); gB,_=burstiness(gsh)
# within-section shuffle
bysec=defaultdict(list)
for w,s in zip(stream,sec_of): bysec[s].append(w)
for s in bysec: rng.shuffle(bysec[s])
wsh=[]; ptr=defaultdict(int)
for s in sec_of: wsh.append(bysec[s][ptr[s]]); ptr[s]+=1
wB,_=burstiness(wsh)
enB,_=burstiness(words_from('data/controls/english.txt'))
laB,_=burstiness(words_from('data/controls/latin.txt'))
print(f"  Voynich (real)            B = {realB:+.3f}  (n={n} words)")
print(f"  Voynich global-shuffle    B = {gB:+.3f}   <- random baseline")
print(f"  Voynich within-section-shuf B={wB:+.3f}   <- section-level clumping only")
print(f"  English (KJV)             B = {enB:+.3f}")
print(f"  Latin (Cicero)            B = {laB:+.3f}")
print("  reading: real>>within-section => FINE topical clumping (pro-language).")
print("           real~within-section => clumping is ONLY the section dialect.")

print("\n"+"="*68); print("PART 2 — SINGLE TABLE vs COLLECTION OF TABLES"); print("="*68)
secs=['herbal','astro','bio','pharma','recipes']
toks_by_sec=defaultdict(list)
for w,s in zip(stream,sec_of): toks_by_sec[s].append(w)
def MI(xs,ys):
    n=len(xs);jx=Counter(xs);jy=Counter(ys);jj=Counter(zip(xs,ys))
    return sum(c/n*math.log2((c/n)/((jx[x]/n)*(jy[y]/n))) for (x,y),c in jj.items())
print("  (a) SCHEMA INVARIANCE — field-independence I(prefix;suffix) per section:")
for s in secs:
    D=[decomp(w) for w in toks_by_sec[s]]
    if len(D)<200: continue
    mi=MI([p for p,_,_ in D],[x for _,_,x in D])
    print(f"      {s:8s}: I(pre;suf)={mi:.2f} bits  (similar across sections = SHARED schema)")
print("\n  (b) DOMAIN DIVERGENCE — core(entity) overlap between section pairs (Jaccard):")
cores={s:set(decomp(w)[1] for w in toks_by_sec[s] if len(toks_by_sec[s])>200) for s in secs}
cores={s:set(decomp(w)[1] for w in toks_by_sec[s]) for s in secs if len(toks_by_sec[s])>200}
sl=list(cores)
for i in range(len(sl)):
    for j in range(i+1,len(sl)):
        a,b=cores[sl[i]],cores[sl[j]]; print(f"      {sl[i]:8s} vs {sl[j]:8s}: {len(a&b)/len(a|b):.2f}")
print("      (low overlap = sections draw from DIFFERENT entity domains = different tables)")

print("\n  (c) DISCRETENESS — per-folio affix-profile: within- vs between-section distance:")
def fvec(f):
    D=[decomp(w) for ln in C['folios'][f] if ltype(ln['locus'])=='P' for w in ln['words'] if '?' not in w and len(w)>=2]
    if len(D)<15: return None
    pc=Counter(p for p,_,_ in D); sc=Counter(s for _,_,s in D)
    v=np.array([pc.get(p,0) for p in PRE+['']]+[sc.get(s,0) for s in SUF+['']],float); v/=v.sum() or 1; return v
fol={f:fvec(f) for f in C['folios']}; fol={f:v for f,v in fol.items() if v is not None}
fsec={f:section(f) for f in fol}
def d(a,b): return np.sqrt(((a-b)**2).sum())
within=[];between=[]
fl=list(fol)
for i in range(len(fl)):
    for j in range(i+1,len(fl)):
        dist=d(fol[fl[i]],fol[fl[j]])
        (within if fsec[fl[i]]==fsec[fl[j]] else between).append(dist)
print(f"      within-section folio distance : {np.mean(within):.3f}")
print(f"      between-section folio distance: {np.mean(between):.3f}")
print(f"      ratio {np.mean(within)/np.mean(between):.2f}  (<1 = sections are discrete tables; ~1 = one continuum)")
