"""
THE MANIFEST HYPOTHESIS.
The prose has no word-order syntax (within-line shuffle = identical bigram entropy):
a positional BAG. So stop reading it as prose and read it as an INVENTORY — a list of
items. Then the question is segmentation: does the text break into discrete manifests
with internally-coherent vocabulary and detectable BOUNDARIES, and do those boundaries
fall on the physical paragraph breaks — or somewhere else (hidden manifest structure)?

Method: TextTiling on morpheme bags. Slide along the line sequence; at each gap measure
vocabulary dissimilarity between the block before and the block after. Deep valleys of
similarity = candidate manifest boundaries. Validate vs (1) random segmentation and
(2) the physical paragraph marks (locus '@' = new paragraph, '+' = continuation).
Plus an INVENTORY-COMPLETION test: within a true manifest, new item-types accumulate
then saturate; a boundary resets the accumulation.
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(11)
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

# ordered prose lines with paragraph-start flag
seq=[]
for f in sorted(C['folios'],key=folio_key):
    for ln in C['folios'][f]:
        if ltype(ln['locus'])!='P': continue
        ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
        if not ws: continue
        para_start = ln['locus'].strip().startswith('@') or ln['locus'].strip().startswith('<@')
        # IVTFF: '@P0' new paragraph, '+P0' continuation
        para_start = '@' in ln['locus']
        feats=Counter(m for w in ws for m in morphs(w))
        seq.append({'f':f,'para':para_start,'feat':feats,'ws':ws})
print(f"prose lines in sequence: {len(seq)}")

# ---- TextTiling similarity curve --------------------------------------------
K=6  # block size in lines
def blockvec(a,b):
    c=Counter()
    for i in range(a,b):
        if 0<=i<len(seq): c+=seq[i]['feat']
    return c
def cos(a,b):
    keys=set(a)|set(b); na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
    if na==0 or nb==0: return 0
    return sum(a[k]*b[k] for k in keys)/(na*nb)
sim=[cos(blockvec(i-K+1,i+1), blockvec(i+1,i+K+1)) for i in range(len(seq)-1)]
sim=np.array(sim)
# depth score at each gap: how deep a valley relative to neighbouring peaks
depth=np.zeros(len(sim))
for i in range(len(sim)):
    l=sim[max(0,i-K):i+1].max(); r=sim[i:i+K+1].max()
    depth[i]=(l-sim[i])+(r-sim[i])
thr=depth.mean()+depth.std()
bound=set(i for i in range(1,len(depth)-1) if depth[i]>thr and depth[i]>=depth[i-1] and depth[i]>=depth[i+1])
print(f"detected manifest boundaries: {len(bound)}  (avg segment = {len(seq)/(len(bound)+1):.1f} lines)")

# ---- validation 1: are detected segments more coherent than random cuts? -----
def segments(bset):
    segs=[]; start=0
    for b in sorted(bset)+[len(seq)-1]:
        segs.append(list(range(start,b+1))); start=b+1
    return [s for s in segs if s]
def coherence(segs):
    vals=[]
    for s in segs:
        if len(s)<2: continue
        v=Counter()
        for i in s: v+=seq[i]['feat']
        # mean cosine of each line to its segment centroid
        for i in s: vals.append(cos(seq[i]['feat'], v))
    return np.mean(vals)
obs=coherence(segments(bound))
sizes=[len(s) for s in segments(bound)]
null=[]
for _ in range(500):
    cuts=sorted(rng.choice(range(1,len(seq)-1), size=len(bound), replace=False))
    null.append(coherence(segments(set(cuts))))
null=np.array(null); p=(np.sum(null>=obs)+1)/(len(null)+1)
print(f"within-segment coherence: detected {obs:.3f} vs random {null.mean():.3f}  p={p:.4f}")

# ---- validation 2: do boundaries align with PHYSICAL paragraph breaks? -------
para_lines=set(i for i,s in enumerate(seq) if s['para'])
para_rate=len(para_lines)/len(seq)
hit=sum(1 for b in bound if any((b+off) in para_lines for off in (-1,0,1,2)))
print(f"\nparagraph-start lines: {len(para_lines)} ({100*para_rate:.0f}% of lines)")
print(f"detected boundaries within ±2 lines of a paragraph start: {hit}/{len(bound)} "
      f"({100*hit/max(1,len(bound)):.0f}%)  — chance ~{100*(1-(1-para_rate)**4):.0f}%")
print("  -> high alignment: manifests = paragraphs. low: hidden manifest structure.")

# ---- inventory-completion test ----------------------------------------------
# within a manifest, distinct item-types should accumulate then SATURATE; across a
# boundary, the new-type rate should spike. Compare new-type rate at boundary-lines
# vs interior lines.
seen=set(); newrate=[]
for i,s in enumerate(seq):
    types=set(w for w in s['ws'])
    new=len(types-seen); newrate.append(new/len(types) if types else 0); seen|=types
newrate=np.array(newrate)
b_idx=np.array(sorted(bound))
at_b=newrate[np.clip(b_idx+1,0,len(seq)-1)].mean()
interior=newrate[[i for i in range(len(seq)) if i not in bound and (i-1) not in bound]].mean()
print(f"\ninventory-completion: new-type rate just AFTER a boundary {at_b:.3f} "
      f"vs interior {interior:.3f}")
print("  -> higher at boundaries = each manifest opens a fresh set of items.")

# show a few segment 'signatures' (top morphemes) to see if manifests have identities
print("\nsample manifest signatures (top morphemes per segment):")
for s in segments(bound)[:6]:
    v=Counter()
    for i in s: v+=seq[i]['feat']
    fol=seq[s[0]]['f']
    print(f"  {fol:6s} lines{len(s):2d}: {[m for m,_ in v.most_common(6)]}")
