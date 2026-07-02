"""
THE LIBRARY'S SHADOW. The corpus encodes real source-works (sections = disjoint entity
domains). Even without decoding, the imprint preserves the library's ARCHITECTURE.
Recover, from the shadow alone:
  1. CATALOG: how many works, their relative SIZES.
  2. CROSS-REFERENCE NETWORK: which works share content (materia-medica -> preparation
     -> application?), controlling for the common 'glue' stratum and section size.
  3. THE SHARED STRATUM: the lingua-franca cores across all works (the compiler's
     invariant vocabulary vs. genuinely shared content).
  4. THE FUNCTIONAL MAP: is one section a HUB (integrates others) and one ISOLATED?
"""
import json, re, numpy as np
from collections import Counter, defaultdict
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
def section(f):
    n=int(re.match(r'f(\d+)',f).group(1))
    return ('herbal' if n<=66 else 'astro' if n<=73 else 'bio' if n<=84 else 'cosmo' if n<=86 else 'pharma' if n<=102 else 'recipes')
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph']
SUF=['eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y']
def core(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return r[:len(r)-len(s)] if s else r

secs=['herbal','astro','bio','cosmo','pharma','recipes']
sec_cores=defaultdict(Counter); sec_tokens=Counter()
globalcore=Counter()
for f,L in C['folios'].items():
    s=section(f)
    for ln in L:
        if ltype(ln['locus']) not in ('P','L'): continue
        for w in ln['words']:
            if '?' in w or len(w)<2: continue
            c=core(w); sec_cores[s][c]+=1; sec_tokens[s]+=1; globalcore[c]+=1

# 1. CATALOG
print("="*66); print("1. CATALOG (works recovered from the shadow)"); print("="*66)
for s in secs:
    print(f"  {s:8s}: {sec_tokens[s]:6,} tokens   {len(sec_cores[s]):4d} distinct entities")

# content cores only: drop the glue (top 25 global) and hapax
glue=set(c for c,_ in globalcore.most_common(25))
def content_set(s): return set(c for c,n in sec_cores[s].items() if c not in glue and n>=2)
cs={s:content_set(s) for s in secs}

# 2/4. cross-reference network: enrichment over chance (controls for vocab size)
print("\n"+"="*66); print("2. CROSS-REFERENCE NETWORK (content cores; enrichment vs chance)"); print("="*66)
allcontent=set().union(*cs.values()); U=len(allcontent)
def enrich(a,b):
    A,B=cs[a],cs[b]; obs=len(A&B); exp=len(A)*len(B)/U
    return obs/exp if exp>0 else 0
print("        "+" ".join(f"{s[:4]:>6}" for s in secs))
for a in secs:
    row=" ".join(f"{enrich(a,b):6.2f}" if a!=b else f"{'--':>6}" for b in secs)
    print(f"  {a:6s}{row}")
print("  (>1 = sections share content MORE than chance = a real cross-reference)")
# hub score = mean enrichment with others
print("\n  integration (mean cross-ref enrichment per work):")
for s in sorted(secs, key=lambda s:-np.mean([enrich(s,t) for t in secs if t!=s])):
    print(f"    {s:8s}: {np.mean([enrich(s,t) for t in secs if t!=s]):.2f}")

# 3. shared stratum
print("\n"+"="*66); print("3. SHARED STRATUM (cores spanning the most works)"); print("="*66)
spread=Counter()
for c in allcontent:
    spread[c]=sum(1 for s in secs if c in cs[s])
multi=[c for c,n in spread.items() if n>=4]
print(f"  content cores in >=4 of 6 works: {len(multi)}  e.g. {multi[:12]}")
print(f"  content cores in exactly 1 work (work-specific): {sum(1 for n in spread.values() if n==1)}")
print(f"  => the multi-work cores are the library's lingua franca (shared materia / glue);")
print(f"     the 1-work cores are each source's proper vocabulary.")

# 5. directional: which PAIR dominates among 2-section cores (the main cross-ref edge)?
pairs=Counter()
for c in allcontent:
    inn=[s for s in secs if c in cs[s]]
    if len(inn)==2: pairs[tuple(sorted(inn))]+=1
print("\n  top cross-reference EDGES (cores shared by exactly two works):")
for (a,b),n in pairs.most_common(6):
    print(f"    {a:8s} <-> {b:8s}: {n}")
