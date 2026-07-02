"""
COULD IT BE MUSIC?
Music's signature = SEQUENCE structure: repeated phrases (motifs/refrains), forward
melodic predictability, and metrical periodicity. A table/registry has none of these.
Tests:
  1. REPEATED PHRASES — exact word n-grams recurring, real vs within-line shuffle
     (shuffle keeps word frequencies, destroys phrases). Music/formulaic => many real
     repeats above null. Table/bag => real ~ null.
  2. REFRAINS — repeated whole LINES (a chorus reappearing).
  3. MELODIC MEMORY — glyph-stream forward predictability h1->h2->h3 (does knowing more
     past notes sharpen the next? deepening memory is melodic).
  4. METER — periodicity in the word-length sequence (autocorrelation peaks at a fixed
     beat).
Plus the one PRO-music feature we already have: line-final cadence specialization.
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(13)
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
lines=[]
for f,L in C['folios'].items():
    for ln in L:
        if ltype(ln['locus'])!='P': continue
        ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
        if len(ws)>=4: lines.append(ws)

# ---- 1. repeated phrases (n-grams) real vs within-line shuffle ---------------
def ngram_repeat(ll,n):
    c=Counter()
    for ln in ll:
        for i in range(len(ln)-n+1): c[tuple(ln[i:i+n])]+=1
    rep=sum(v for v in c.values() if v>=2)      # tokens that are part of a repeated n-gram
    tot=sum(c.values())
    return rep/tot if tot else 0
print("1. REPEATED PHRASES — fraction of n-grams that recur (real vs shuffled):")
for n in (2,3,4):
    real=ngram_repeat(lines,n)
    sh=np.mean([ngram_repeat([list(rng.permutation(ln)) for ln in lines],n) for _ in range(5)])
    print(f"   {n}-grams: real {100*real:.1f}%  shuffled {100*sh:.1f}%  ratio {real/max(sh,1e-9):.2f}")

# ---- 2. refrains: repeated whole lines ---------------------------------------
linekeys=Counter(tuple(ln) for ln in lines)
rep_lines=[(ln,c) for ln,c in linekeys.items() if c>=2]
# near-identical: same multiset (bag) -> same 'chord set' even if reordered
bags=Counter(frozenset(Counter(ln).items()) for ln in lines)
print(f"\n2. REFRAINS: exact repeated lines: {sum(c for _,c in rep_lines)-len(rep_lines)} repeats "
      f"over {len(rep_lines)} line-types; longest repeated line len "
      f"{max([len(ln) for ln,_ in rep_lines], default=0)}")

# ---- 3. melodic memory: glyph-stream conditional entropy ---------------------
stream=' '.join(w for ln in lines for w in ln)
def cond_entropy(s,k):
    ctx=defaultdict(Counter)
    for i in range(len(s)-k): ctx[s[i:i+k]][s[i+k]]+=1
    n=sum(sum(c.values()) for c in ctx.values())
    return sum(sum(c.values())/n*(-sum(v/sum(c.values())*math.log2(v/sum(c.values())) for v in c.values())) for c in ctx.values())
h=[cond_entropy(stream,k) for k in (0,1,2,3)]
print(f"\n3. MELODIC MEMORY (glyph cond. entropy h0..h3): {[round(x,2) for x in h]}")
print(f"   drop h1->h3 = {h[1]-h[3]:+.2f} bits (deepening memory = more melodic/Markov)")

# ---- 4. meter: periodicity in word-length sequence ---------------------------
wl=np.array([len(w) for ln in lines for w in ln],dtype=float)
wl-=wl.mean()
ac=[np.corrcoef(wl[:-k],wl[k:])[0,1] for k in range(1,13)]
print(f"\n4. METER — word-length autocorrelation at lags 1..12:")
print("   "+"  ".join(f"{a:+.2f}" for a in ac))
print(f"   max |autocorr| beyond lag1: {max(abs(a) for a in ac[1:]):.2f} (a strong periodic peak = meter)")

# ---- pro-music cadence already known ----------------------------------------
finals=Counter(ln[-1] for ln in lines)
print(f"\n(cadence, pro-music) top line-final words: {[w for w,_ in finals.most_common(6)]} "
      f"-> {100*sum(c for _,c in finals.most_common(6))/len(lines):.0f}% of line-ends")
