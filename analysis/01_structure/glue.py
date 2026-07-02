"""
WHAT SURFACE DOES THE GLUE ADHERE TO?
There is no syntax, yet there is glue (generic high-frequency tokens). Glue only
makes sense as connective tissue for SOME structure. Hypothesis: the structure is
not the sentence (grammar) but the LINE/PAGE (layout). Tests:
  1. Is glue EDGE-LOADED? (concentrated at line start/end = it frames the line)
  2. Are line EDGES a specialized distribution vs interior, beyond a within-line
     shuffle? (the line is a real structural surface, not an arbitrary cut)
  3. Do lines have a fixed internal TEMPLATE (position -> word-type), i.e. is it a
     FORM/grid the glue lays out?
  4. Do lines FILL to a regular length (justification = adhering to the page margin)?
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(9)
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'

# prose lines as ordered word lists
lines=[]
for f,L in C['folios'].items():
    for ln in L:
        if ltype(ln['locus'])!='P': continue
        ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
        if len(ws)>=4: lines.append(ws)
allw=[w for ln in lines for w in ln]
freq=Counter(allw)
GLUE=set(w for w,_ in freq.most_common(30))     # the 30 commonest tokens = candidate glue
def H(c):
    n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values()) if n else 0

print(f"prose lines: {len(lines)}, tokens: {len(allw)}, glue set = top 30 words")
print(f"glue covers {100*sum(freq[w] for w in GLUE)/len(allw):.0f}% of all tokens")

# ---- 1. glue edge-loading ----------------------------------------------------
buckets=defaultdict(lambda:[0,0])   # pos -> [glue, total]
for ln in lines:
    n=len(ln)
    for i,w in enumerate(ln):
        pos = 'first' if i==0 else 'last' if i==n-1 else 'second' if i==1 else 'penult' if i==n-2 else 'interior'
        buckets[pos][1]+=1; buckets[pos][0]+= (w in GLUE)
print("\n1. glue fraction by position in line:")
for pos in ['first','second','interior','penult','last']:
    g,t=buckets[pos]; print(f"   {pos:9s}: {100*g/t:4.1f}%  (n={t})")

# ---- 2. edge specialization vs within-line shuffle --------------------------
def edge_dists(ll):
    first=Counter(); last=Counter(); interior=Counter()
    for ln in ll:
        first[ln[0]]+=1; last[ln[-1]]+=1
        for w in ln[1:-1]: interior[w]+=1
    return first,last,interior
def kl(p,q):
    n1=sum(p.values()); n2=sum(q.values()); keys=set(p)|set(q)
    return sum((p.get(k,0)/n1)*math.log2(((p.get(k,0)/n1)+1e-9)/((q.get(k,0)/n2)+1e-9)) for k in keys if p.get(k,0))
f,l,interior=edge_dists(lines)
klf,kll=kl(f,interior),kl(l,interior)
# null: shuffle words within each line, recompute
sh=[list(rng.permutation(ln)) for ln in lines]
fs,ls,ints=edge_dists(sh)
print(f"\n2. line-edge specialization (KL vs interior distribution):")
print(f"   line-FIRST vs interior: real {klf:.2f} bits  | shuffled {kl(fs,ints):.2f}")
print(f"   line-LAST  vs interior: real {kll:.2f} bits  | shuffled {kl(ls,ints):.2f}")
print(f"   most distinctive line-FIRST words: {[w for w,_ in f.most_common(6)]}")
print(f"   most distinctive line-LAST  words: {[w for w,_ in l.most_common(6)]}")

# ---- 3. internal template: word-type by relative position -------------------
# bin each token by relative position (0..1) into 5 slots; how predictable is the
# word-TYPE (glue vs content + which glue) from the slot? high = a fixed form.
slotdist=defaultdict(Counter)
for ln in lines:
    n=len(ln)
    for i,w in enumerate(ln):
        s=min(4,int(i/n*5))
        slotdist[s][w if w in GLUE else '<content>']+=1
print(f"\n3. internal line TEMPLATE — dominant token by relative slot:")
for s in range(5):
    c=slotdist[s]; n=sum(c.values())
    top=[(w,f"{100*v//n}%") for w,v in c.most_common(4)]
    print(f"   slot {s} ({20*s}-{20*s+20}%): {top}")

# ---- 4. line length filling --------------------------------------------------
wl=np.array([len(ln) for ln in lines]); gl=np.array([sum(len(w) for w in ln) for ln in lines])
print(f"\n4. line LENGTH regularity:")
print(f"   words/line : mean {wl.mean():.1f}  CV {wl.std()/wl.mean():.2f}")
print(f"   glyphs/line: mean {gl.mean():.0f}  CV {gl.std()/gl.mean():.2f}  (low CV = lines fill to a target = justified)")
