"""
POWERED CROSS-MODAL CRIB SEARCH.
Visual tags below were read from the plant images by a vision model (structural
morphology, not pixel color). For a visual attribute (e.g. flower colour) we
pool the TEXT of all folios that share each value and ask:

  is there any glyph-substring (n-gram) over-represented in one group vs another,
  by MORE than you'd get from random regrouping of the same folios?

If a substring robustly separates 'red-flower' text from 'blue-flower' text, it is
a candidate cipher-morpheme for that visual attribute -- the pictures acting as
known-plaintext. A null result is reported honestly (it is the likely outcome and
still informative: it bounds how descriptive the text can be).
"""
import json, re, math, numpy as np
from collections import Counter
from itertools import combinations

# ---- vision-read structural tags (folio -> attributes) ---------------------
TAGS = {
 'f4r':  dict(flower='red',  root='fibrous',  leaf='pinnate'),
 'f6r':  dict(flower='red',  root='fibrous',  leaf='fern'),
 'f8r':  dict(flower='none', root='rhizome',  leaf='lobed'),
 'f9r':  dict(flower='none', root='taproot',  leaf='curly'),
 'f11r': dict(flower='blue', root='fibrous',  leaf='frilly'),
 'f14r': dict(flower='red',  root='rhizome',  leaf='sword'),
 'f15r': dict(flower='white',root='rhizome',  leaf='toothed'),
 'f18r': dict(flower='blue', root='fibrous',  leaf='oval'),
 'f20r': dict(flower='red',  root='fibrous',  leaf='pinnate'),
 'f23r': dict(flower='blue', root='rhizome',  leaf='palmate'),
 'f26r': dict(flower='blue', root='taproot',  leaf='scalloped'),
 'f31r': dict(flower='white',root='taproot',  leaf='oval'),
 'f35r': dict(flower='blue', root='bulb',     leaf='thin'),
 'f37r': dict(flower='red',  root='fibrous',  leaf='lance'),
 'f43r': dict(flower='white',root='fibrous',  leaf='small'),
 'f52r': dict(flower='blue', root='taproot',  leaf='lance'),
 'f55r': dict(flower='red',  root='bulb',     leaf='palmate'),
}

C = json.load(open('data/corpus/corpus.json'))
def folio_words(fol):
    if fol not in C['folios']: return []
    return [w for ln in C['folios'][fol] for w in ln['words'] if '?' not in w]

# attach text; keep only tagged folios that actually have text
DATA = {f:folio_words(f) for f in TAGS if folio_words(f)}
print("tagged folios with text:", len(DATA),
      " total tokens:", sum(len(v) for v in DATA.values()))

def group(attr, val):
    return [f for f in DATA if TAGS[f][attr]==val]

def ngram_counts(folios, k):
    c = Counter()
    for f in folios:
        for w in DATA[f]:
            for i in range(len(w)-k+1):
                c[w[i:i+k]] += 1
    tot = sum(c.values())
    return c, tot

def enrichment(attr, valA, valB, k=3, topn=8):
    A, B = group(attr,valA), group(attr,valB)
    cA,tA = ngram_counts(A,k); cB,tB = ngram_counts(B,k)
    grams = [g for g in set(cA)|set(cB) if cA[g]+cB[g] >= 12]   # ignore rare noise
    rows=[]
    for g in grams:
        a,b = cA[g], cB[g]
        # log2 ratio of rates, with smoothing
        rate = ((a+1)/(tA+1))/((b+1)/(tB+1))
        rows.append((g, math.log2(rate), a, b))
    rows.sort(key=lambda r:-r[1])
    return A,B,rows

def perm_test(attr, valA, valB, k=3, nperm=4000, seed=1989):
    """Family-wise control: is the BEST separating n-gram better than under random
    relabelling of the SAME folios? Returns observed max|log2ratio| and its p."""
    A,B,rows = enrichment(attr,valA,valB,k)
    allf = A+B; nA=len(A)
    obs = max(abs(r[1]) for r in rows)
    rng = np.random.default_rng(seed)
    # precompute per-folio ngram vectors over the union vocabulary
    vocab = [r[0] for r in rows]
    vidx={g:i for i,g in enumerate(vocab)}
    fvec={}
    for f in allf:
        v=np.zeros(len(vocab))
        for w in DATA[f]:
            for i in range(len(w)-k+1):
                g=w[i:i+k]
                if g in vidx: v[vidx[g]]+=1
        fvec[f]=v
    M=np.array([fvec[f] for f in allf])
    worse=0
    for _ in range(nperm):
        idx=rng.permutation(len(allf))
        gA=M[idx[:nA]].sum(0); gB=M[idx[nA:]].sum(0)
        tA=gA.sum()+1; tB=gB.sum()+1
        lr=np.abs(np.log2(((gA+1)/tA)/((gB+1)/tB)))
        if lr.max()>=obs: worse+=1
    return obs,(worse+1)/(nperm+1),rows,A,B

print("\n" + "="*72)
print("CRIB SEARCH 1: flower colour  RED vs BLUE  (trigrams)")
print("="*72)
obs,p,rows,A,B = perm_test('flower','red','blue',k=3)
print(f"  red folios : {A}")
print(f"  blue folios: {B}")
print(f"  most RED-enriched trigrams : {[(g,round(lr,2)) for g,lr,a,b in rows[:6]]}")
print(f"  most BLUE-enriched trigrams: {[(g,round(lr,2)) for g,lr,a,b in rows[-6:]]}")
print(f"  best separation |log2ratio|={obs:.2f}   permutation p={p:.4f}")
print("  ->", "CANDIDATE CRIB: a substring tracks flower colour" if p<0.05
      else "NULL: no substring separates colour better than random regrouping")

print("\n" + "="*72)
print("CRIB SEARCH 2: root type  RED-TAPROOT vs RHIZOME  (trigrams)")
print("="*72)
obs,p,rows,A,B = perm_test('root','taproot','rhizome',k=3)
print(f"  taproot folios: {A}")
print(f"  rhizome folios: {B}")
print(f"  taproot-enriched: {[(g,round(lr,2)) for g,lr,a,b in rows[:6]]}")
print(f"  rhizome-enriched: {[(g,round(lr,2)) for g,lr,a,b in rows[-6:]]}")
print(f"  best separation |log2ratio|={obs:.2f}   permutation p={p:.4f}")
print("  ->", "CANDIDATE CRIB" if p<0.05 else "NULL: no root-type morpheme detected")

print("\n(Interpretation guard: with ~6 folios/group this is exploratory. A real")
print(" crib must survive on held-out folios tagged the same way.)")
