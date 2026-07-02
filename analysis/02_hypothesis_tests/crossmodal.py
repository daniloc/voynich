"""
THE NEWLY-POSSIBLE TEST: does the PICTURE predict the TEXT on the same folio?
Never feasible before at scale: you'd need a botanist to score every plant AND
an aligned transcription. Now: extract pigment features from each herbal page
automatically, pull that folio's text, and ask whether visually-similar plants
carry more-similar text than random folio pairs would.

Honest framing: alignment (folio<->PDF page) is approximate and color is a crude
visual proxy. This demonstrates the METHOD and gives a real, falsifiable number.
A null result here is itself important: it would mean text does not track imagery.
"""
import json, numpy as np
from collections import Counter
from itertools import combinations

feats = {i:f for i,*f in json.load(open('data/corpus/page_features.json'))}  # page -> [ink,green,red,blue,edge]
C = json.load(open('data/corpus/corpus.json'))
META = C['meta']

# folio recto page model (verified on anchors f14->26, f26->50, f37->72, f55->108)
def recto_page(n): return 2*n - 2

# build per-folio record for HERBAL-section folios that actually look like plants
recs = []
for fol, lines in C['folios'].items():
    if not fol.startswith('f'): continue
    m = __import__('re').match(r'f(\d+)([rv])', fol)
    if not m: continue
    n, side = int(m.group(1)), m.group(2)
    if n > 57: continue                      # herbal section (Currier-A heavy)
    pg = recto_page(n) + (1 if side=='v' else 0)
    if pg not in feats: continue
    ink,green,red,blue,edge = feats[pg]
    if green < 0.04: continue                # keep only pages that really show foliage
    words = [w for ln in lines for w in ln['words'] if '?' not in w]
    if len(words) < 12: continue
    recs.append(dict(folio=fol, green=green, red=red, blue=blue,
                     vocab=set(words), words=words))

print(f"aligned herbal folios with foliage + text: {len(recs)}")

# visual distance = euclidean in (green,red,blue); text similarity = vocab Jaccard
def vdist(a,b):
    return sum((a[k]-b[k])**2 for k in ('green','red','blue'))**0.5
def jac(a,b):
    u=a|b; return len(a&b)/len(u) if u else 0

pairs = list(combinations(range(len(recs)),2))
vd = np.array([vdist(recs[i],recs[j]) for i,j in pairs])
tj = np.array([jac(recs[i]['vocab'],recs[j]['vocab']) for i,j in pairs])

# split pairs into visually-SIMILAR (closest 25%) vs visually-DIFFERENT (farthest 25%)
lo, hi = np.quantile(vd,0.25), np.quantile(vd,0.75)
sim_tj = tj[vd<=lo]; dif_tj = tj[vd>=hi]
print(f"\nvocabulary overlap (Jaccard) between folio pairs:")
print(f"  visually-SIMILAR plants : {sim_tj.mean():.4f}  (n={len(sim_tj)})")
print(f"  visually-DIFFERENT plants: {dif_tj.mean():.4f}  (n={len(dif_tj)})")
print(f"  ratio sim/diff          : {sim_tj.mean()/dif_tj.mean():.3f}")

# Mantel-style permutation test: is the vdist<->tj correlation more than chance?
import math
def pearson(x,y):
    x=np.asarray(x);y=np.asarray(y)
    return float(((x-x.mean())*(y-y.mean())).mean()/(x.std()*y.std()+1e-12))
obs = pearson(vd, tj)
rng = np.random.default_rng(1989)
perm = []
for _ in range(2000):
    perm.append(pearson(vd, rng.permutation(tj)))
perm = np.array(perm)
p = (np.sum(np.abs(perm) >= abs(obs))+1)/(len(perm)+1)
print(f"\nMantel test  (visual distance vs vocabulary overlap):")
print(f"  observed correlation r = {obs:+.4f}   (negative = closer-looking => more shared words)")
print(f"  permutation p-value    = {p:.4f}")
print("  interpretation:", "SIGNAL: imagery tracks text" if p<0.05 else "no detectable coupling at this resolution")

# whichever way it goes, show the per-folio dominant suffix vs color (descriptive)
print("\nper-folio glance (green, red, top-3 suffixes):")
def suf(words):
    return [s for s,_ in Counter(w[-2:] for w in words if len(w)>=2).most_common(3)]
for r in sorted(recs, key=lambda r:-r['green'])[:10]:
    print(f"  {r['folio']:6s} g={r['green']:.3f} r={r['red']:.3f}  suff={suf(r['words'])}")
