"""
LABEL COLLISION / DETERMINISM TEST.
A function (hash in either direction) demands: same input -> same output, and
distinct inputs -> distinct outputs (collision resistance). We can't pixel-align
every label, but we can test the statistical fingerprints that distinguish:
  - HASH / NAMING code  : labels near-unique (each object its own label)
  - DESCRIPTIVE vocab   : labels heavily REUSED (shared category words: 'root','red')
  - INDEPENDENT / filler : labels follow the ordinary corpus word distribution
Test 1: are labels more collision-resistant than size-matched running prose?
Test 2 (determinism): do REPEATED labels land in visually-consistent contexts
        (low feature spread) more than random word groups would?
"""
import json, re, numpy as np
from collections import Counter, defaultdict
rng = np.random.default_rng(1989)

C=json.load(open('data/corpus/corpus.json')); META=C['meta']
feats={i:f for i,*f in json.load(open('data/corpus/page_features.json'))}
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'

labels=[]; prose=[]
label_folios=defaultdict(list)
for f,lines in C['folios'].items():
    for ln in lines:
        t=ltype(ln['locus'])
        ws=[w for w in ln['words'] if '?' not in w]
        if t=='L':
            labels+=ws
            for w in ws: label_folios[w].append(f)
        elif t=='P':
            prose+=ws

def uniq_frac(tokens):
    c=Counter(tokens); return sum(1 for v in c.values() if v==1)/len(c)

print("="*70)
print("TEST 1 — COLLISION RESISTANCE: labels vs size-matched prose")
print("="*70)
L_uniq=uniq_frac(labels)
n=len(labels)
samp=[uniq_frac(rng.choice(prose,size=n,replace=False)) for _ in range(2000)]
samp=np.array(samp)
p=(np.sum(samp>=L_uniq)+1)/(len(samp)+1)
print(f"  label tokens={n}, distinct={len(set(labels))}")
print(f"  unique-word fraction:")
print(f"    LABELS                : {L_uniq:.3f}")
print(f"    prose (size-matched)  : {samp.mean():.3f}  (95% range {np.quantile(samp,.025):.3f}-{np.quantile(samp,.975):.3f})")
print(f"  permutation p          : {p:.4f}")
print("  =>", "labels are SIGNIFICANTLY more collision-resistant than prose"
      if p<0.05 else "labels no more unique than ordinary text")
print("     hash/naming predicts MORE unique; descriptive vocab predicts LESS unique.")

print("\n" + "="*70)
print("TEST 2 — DETERMINISM: do repeated labels share a visual context?")
print("="*70)
# map label -> page features, only folios we can align to a page with features
def recto_page(n): return 2*n if n<12 else 2*n-2
def folio_page(f):
    m=re.match(r'f(\d+)([rv])$',f)            # skip foldout sub-pages (f67r2 etc.)
    if not m: return None
    n=int(m.group(1))
    if n>57: return None                       # alignment unreliable past herbal
    pg=recto_page(n)+(1 if m.group(2)=='v' else 0)
    return pg if pg in feats else None

# build per-label list of feature vectors (green,red,blue) for aligned occurrences
vecs=defaultdict(list)
for w,fs in label_folios.items():
    for f in fs:
        pg=folio_page(f)
        if pg is not None:
            ink,g,r,b,e=feats[pg]; vecs[w].append([g,r,b])
testable={w:np.array(v) for w,v in vecs.items() if len(v)>=2}
print(f"  labels with >=2 aligned occurrences: {len(testable)}")
if testable:
    # observed: mean within-label spread (avg std across the 3 features)
    def spread(arr): return arr.std(axis=0).mean()
    obs=np.mean([spread(a) for a in testable.values()])
    # null: regroup the SAME occurrence-vectors into random groups of same sizes
    allvecs=np.array([v for a in testable.values() for v in a])
    sizes=[len(a) for a in testable.values()]
    nulls=[]
    for _ in range(2000):
        idx=rng.permutation(len(allvecs)); k=0; sp=[]
        for s in sizes:
            sp.append(spread(allvecs[idx[k:k+s]])); k+=s
        nulls.append(np.mean(sp))
    nulls=np.array(nulls)
    p2=(np.sum(nulls<=obs)+1)/(len(nulls)+1)
    print(f"  mean within-label visual spread : {obs:.4f}")
    print(f"  null (random regrouping)        : {nulls.mean():.4f}")
    print(f"  permutation p (lower=consistent): {p2:.4f}")
    print("  =>", "repeated labels DO share visual context (determinism signal)"
          if p2<0.05 else "no visual consistency: repeated labels land on unrelated-looking pages")
else:
    print("  too few aligned repeated labels in herbal section to test here.")
