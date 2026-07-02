"""
ROSTER / ORG-CHART-OVER-TIME hypothesis.
A roster = a persistent POPULATION of named entities recurring across snapshots,
each carrying an ATTRIBUTE that can change while the entity persists. Mapped onto
the morphology: CORE = entity (the name), AFFIX = its attribute/status this entry.
Predictions distinct from manifest/bag:
  A. PERSISTENT MEMBERSHIP — consecutive entries share their entity (core) set far
     more than distant entries; a stable membership with gradual turnover.
  B. SMALL CORE POPULATION + LONG TAIL — a few hundred cores recur in many entries
     (the 'members'); most cores are transient.
  C. ENTITY TRACKED WITH CHANGING ATTRIBUTE — the SAME core appears with DIFFERENT
     affixes across the document (an entity whose status varies), and that variation
     is structured, not random.
  D. HIERARCHY (org chart) — nested/stacked prefixes = an attribute hierarchy.
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(5)
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
def folio_key(f):
    m=re.match(r'f(\d+)([rv])(\d*)',f); return (int(m.group(1)),0 if m.group(2)=='r' else 1,int(m.group(3) or 0))
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph']
SUF=['eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s

# paragraphs as ordered list of (core-set, words)
paras=[]; cur_words=[]
for f in sorted(C['folios'],key=folio_key):
    for ln in C['folios'][f]:
        if ltype(ln['locus'])!='P': continue
        new='@' in ln['locus']
        ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
        if not ws: continue
        if new and cur_words: paras.append(cur_words); cur_words=[]
        cur_words+=ws
    if cur_words: paras.append(cur_words); cur_words=[]
paras=[p for p in paras if len(p)>=5]
para_cores=[set(decomp(w)[1] for w in p) for p in paras]
print(f"paragraphs: {len(paras)}")

# ---- A. membership persistence (core-set Jaccard vs lag) --------------------
def jac(a,b): u=a|b; return len(a&b)/len(u) if u else 0
def lagjac(order):
    sums=defaultdict(list)
    for i in range(len(order)):
        for d in range(1,7):
            if i+d<len(order): sums[d].append(jac(para_cores[order[i]],para_cores[order[i+d]]))
    return {d:np.mean(v) for d,v in sums.items()}
real=lagjac(list(range(len(paras))))
sh=lagjac(list(rng.permutation(len(paras))))
print("\nA. entity(core)-set overlap vs lag between paragraphs:")
print("   lag :  "+"  ".join(f"{d}" for d in range(1,7)))
print("   real:  "+"  ".join(f"{real[d]:.3f}" for d in range(1,7)))
print("   shuf:  "+"  ".join(f"{sh[d]:.3f}" for d in range(1,7)))
print(f"   adjacent overlap real {real[1]:.3f} vs shuffled {sh[1]:.3f}  (higher real = persistent membership)")

# ---- B. core population: recurrence distribution ----------------------------
core_in_paras=Counter()
for cs in para_cores:
    for c in cs: core_in_paras[c]+=1
vals=np.array(list(core_in_paras.values()))
members=np.sum(vals>=10)
print(f"\nB. distinct cores: {len(core_in_paras)}")
print(f"   cores in >=10 paragraphs ('persistent members'): {members}")
print(f"   cores in exactly 1 paragraph (transient): {np.sum(vals==1)} ({100*np.mean(vals==1):.0f}%)")
top=core_in_paras.most_common(12)
print(f"   the persistent population (top cores): {[c for c,_ in top]}")
print(f"   their reach: {[n for _,n in top]} of {len(paras)} paragraphs")

# ---- C. same entity, changing attribute -------------------------------------
# for persistent cores, how many DISTINCT affix-dressings do they take? and does the
# attribute (q-prefix rate) drift across the document (entity status over time)?
word_seq=[]  # (paraindex, word)
for pi,p in enumerate(paras):
    for w in p: word_seq.append((pi,w))
by_core=defaultdict(list)
for pi,w in word_seq:
    by_core[decomp(w)[1]].append((pi,w))
persistent=[c for c,n in core_in_paras.items() if n>=10]
multi=np.mean([len(set((decomp(w)[0],decomp(w)[2]) for _,w in by_core[c])) for c in persistent])
print(f"\nC. persistent entities wear on average {multi:.1f} distinct affix-dressings each")
# attribute drift: correlation of q-prefix rate with paragraph index, per persistent core
drifts=[]
for c in persistent:
    occ=by_core[c]
    if len(occ)<20: continue
    x=np.array([pi for pi,_ in occ]); y=np.array([1 if decomp(w)[0].startswith('q') else 0 for _,w in occ])
    if y.std()>0 and x.std()>0: drifts.append(np.corrcoef(x,y)[0,1])
drifts=np.array(drifts)
print(f"   q-attribute vs document-position correlation (per entity): mean {np.nanmean(drifts):+.3f}")
print(f"   fraction of entities with |corr|>0.2 (status tracks position): {100*np.mean(np.abs(drifts)>0.2):.0f}%")

# ---- D. hierarchy: stacked prefixes -----------------------------------------
allw=[w for p in paras for w in p]
def nlayers(w):
    n=0; r=w
    for _ in range(3):
        p=next((x for x in sorted(PRE,key=len,reverse=True) if r.startswith(x) and len(r)>len(x)),'')
        if not p: break
        n+=1; r=r[len(p):]
    return n
layers=Counter(nlayers(w) for w in allw)
tot=sum(layers.values())
print(f"\nD. prefix-stacking (attribute hierarchy):")
for k in sorted(layers): print(f"   {k} prefix-layer(s): {100*layers[k]/tot:.0f}%")
