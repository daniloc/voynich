"""
THE 'CONTACT' MOVE: stop reading rows; pivot the table along the ENTITY axis.
Collapse 38k word-rows by their core (entity key). Each entity becomes a point whose
coordinates are its ATTRIBUTE PROFILE (how it is declined: prefix dist + suffix dist).
Then look for the structure that's invisible in the stream:
  1. DIMENSIONALITY — is the entity-attribute space low-rank (few entity TYPES)?
  2. ENTITY CLASSES — cluster entities; do they form coherent declension-classes?
  3. THE JOIN — do classes map to SECTIONS and to LABEL-vs-PROSE (the picture keys)?
  4. KEY-LEVEL CROSS-MODAL — at the entity level (not whole-word), do label-entities
     concentrate in particular classes/sections (a refined, re-keyed crib test)?
Plus the literal depth axis: the folio-opener key sequence as a candidate index column.
"""
import json, re, numpy as np
from collections import Counter, defaultdict
from scipy.cluster.vq import kmeans2, whiten
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
def section(f):
    n=int(re.match(r'f(\d+)',f).group(1))
    return ('herbal' if n<=66 else 'astro' if n<=73 else 'bio' if n<=84 else 'cosmo' if n<=86 else 'pharma' if n<=102 else 'recipes')
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph','cfh']
SUF=['eedy','eody','edy','aiin','aiir','ain','iin','dy','ol','or','ar','al','am','dam','ey','eey','y']
PI={p:i for i,p in enumerate(PRE+[''])}; SI={s:i for i,s in enumerate(SUF+[''])}
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s

# pivot: entity -> attribute counts + context
ent=defaultdict(lambda:{'pre':Counter(),'suf':Counter(),'sec':Counter(),'lab':0,'tot':0})
for f,L in C['folios'].items():
    for ln in L:
        reg=ltype(ln['locus'])
        for w in ln['words']:
            if '?' in w or len(w)<2: continue
            p,c,s=decomp(w); e=ent[c]
            e['pre'][p]+=1; e['suf'][s]+=1; e['sec'][section(f)]+=1; e['tot']+=1
            if reg=='L': e['lab']+=1
ents=[c for c,d in ent.items() if d['tot']>=12]
print(f"entities (cores, freq>=12): {len(ents)}  covering "
      f"{100*sum(ent[c]['tot'] for c in ents)/sum(d['tot'] for d in ent.values()):.0f}% of tokens")

# feature matrix: normalized prefix(15)+suffix(18) profiles
def vec(c):
    d=ent[c]; pv=np.zeros(len(PI)); sv=np.zeros(len(SI))
    for k,n in d['pre'].items(): pv[PI[k]]=n
    for k,n in d['suf'].items(): sv[SI[k]]=n
    pv/=pv.sum() or 1; sv/=sv.sum() or 1
    return np.concatenate([pv,sv])
M=np.array([vec(c) for c in ents])

# 1. dimensionality (PCA via SVD)
Mc=M-M.mean(0); U,S,Vt=np.linalg.svd(Mc,full_matrices=False)
ev=(S**2)/(S**2).sum()
print(f"\n1. DIMENSIONALITY of entity-attribute space:")
print(f"   variance explained by top 3 comps: {100*ev[:3].sum():.0f}%  | top 6: {100*ev[:6].sum():.0f}%")
print(f"   (low-rank => entities fall into few declension TYPES)")

# 2. cluster entities
k=6
cent,lab=kmeans2(whiten(M), k, seed=3, minit='++')
print(f"\n2. ENTITY CLASSES (k={k}):")
for ci in range(k):
    idx=[i for i in range(len(ents)) if lab[i]==ci]
    if not idx: continue
    # dominant attributes of the class
    pre=Counter(); suf=Counter(); sec=Counter(); labrate=[]
    for i in idx:
        c=ents[i]; pre+=ent[c]['pre']; suf+=ent[c]['suf']; sec+=ent[c]['sec']
        labrate.append(ent[c]['lab']/ent[c]['tot'])
    topsec=sec.most_common(1)[0]
    print(f"   class {ci}: {len(idx):4d} entities | pre={[p for p,_ in pre.most_common(3)]} "
          f"suf={[s for s,_ in suf.most_common(3)]} | top-section {topsec[0]} "
          f"({100*topsec[1]//sum(sec.values())}%) | label-rate {100*np.mean(labrate):.0f}%")
    print(f"             examples: {[ents[i] for i in idx[:8]]}")

# 3+4. the JOIN: are label-entities a distinct class/section?
labent=[c for c in ents if ent[c]['lab']/ent[c]['tot']>0.3]
proent=[c for c in ents if ent[c]['lab']==0]
def secmix(cs):
    s=Counter()
    for c in cs: s+=ent[c]['sec']
    n=sum(s.values()); return {k:round(v/n,2) for k,v in s.most_common(4)}
print(f"\n3. THE JOIN — label-keys vs prose-keys:")
print(f"   entities used mainly as LABELS: {len(labent)}  section mix {secmix(labent)}")
print(f"   entities never used as labels : {len(proent)}  section mix {secmix(proent)}")

# depth axis: folio-opener key sequence
def fk(f): m=re.match(r'f(\d+)([rv])(\d*)',f); return (int(m.group(1)),0 if m.group(2)=='r' else 1,int(m.group(3) or 0))
openers=[]
for f in sorted(C['folios'],key=fk):
    for ln in C['folios'][f]:
        if ltype(ln['locus'])=='P':
            ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
            if ws: openers.append(decomp(ws[0])[1]); break
print(f"\n5. DEPTH AXIS — folio-opener KEY sequence ({len(openers)} folios):")
print(f"   distinct opener-keys: {len(set(openers))} ({100*len(set(openers))/len(openers):.0f}% unique) "
      f"-> {'index-like (mostly distinct)' if len(set(openers))/len(openers)>0.7 else 'repetitive (not an index)'}")
print(f"   first 20: {openers[:20]}")
