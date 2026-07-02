"""
UNEXAMINED STRUCTURE.
1) SYNTAX — we did within-word morphology to death but never asked if words form
   ordered sequences. Does word N+1 depend on word N (grammar) or only on position
   (layout) / nothing (bag of words)? Test with bigram info vs a within-line shuffle
   (same words, order destroyed) and a cross-word affix-binding test.
2) CALENDAR — the zodiac medallions: count labels per sign. ~30 => days of a month.
3) LIST — the 'recipes/stars' section: star-bulleted paragraphs. How regular?
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(7)
C=json.load(open('data/corpus/corpus.json')); META=C['meta']
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'

# lines as ordered word sequences (prose only)
lines=[]
for f,L in C['folios'].items():
    for ln in L:
        if ltype(ln['locus'])=='P':
            ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
            if len(ws)>=3: lines.append(ws)
flat=[w for ln in lines for w in ln]
def H(c):
    n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values())

print("="*70)
print("1. SYNTAX — does word ORDER carry information?")
print("="*70)
uni=Counter(flat); Huni=H(uni)
# bigram conditional entropy H(W2|W1), real vs within-line shuffled
def bigram_condH(seqs):
    big=Counter(); ctx=Counter()
    for s in seqs:
        for a,b in zip(s,s[1:]): big[(a,b)]+=1; ctx[a]+=1
    n=sum(big.values())
    # H(W2|W1) = sum p(a) H(W2|a); estimate
    by=defaultdict(Counter)
    for (a,b),c in big.items(): by[a][b]+=c
    return sum(sum(v.values())/n*H(v) for v in by.values())
real_h=bigram_condH(lines)
sh=[list(rng.permutation(s)) for s in lines]            # destroy order, keep line bag
shuf_h=bigram_condH(sh)
print(f"  H(word)            = {Huni:.2f} bits")
print(f"  H(word | prev)     = {real_h:.2f} bits   (real word order)")
print(f"  H(word | prev) shuf= {shuf_h:.2f} bits   (within-line shuffled)")
print(f"  order-specific info= {shuf_h-real_h:+.2f} bits  (how much ORDER adds beyond bag-of-words)")

# cross-word affix binding: does suffix(N) predict prefix(N+1)?
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph']
SUF=['eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y']
def pre(w): return next((p for p in sorted(PRE,key=len,reverse=True) if w.startswith(p)),'∅')
def suf(w): return next((s for s in sorted(SUF,key=len,reverse=True) if w.endswith(s) and len(w)>len(s)),'∅')
pairs=[(suf(a),pre(b)) for s in lines for a,b in zip(s,s[1:])]
sN=Counter(s for s,_ in pairs); pN=Counter(p for _,p in pairs); jt=Counter(pairs); n=len(pairs)
MI=sum(c/n*math.log2((c/n)/((sN[s]/n)*(pN[p]/n))) for (s,p),c in jt.items())
print(f"  I(suffix_N ; prefix_N+1) = {MI:.3f} bits  (cross-word morphological binding)")
print("  => order-info ~0 and binding ~0 would mean: NO syntax, words are a positional bag.")

print("\n" + "="*70)
print("2. CALENDAR — labels per zodiac medallion (~30 = days of a month?)")
print("="*70)
zod=[f for f in C['folios'] if re.match(r'f7[0-3][rv]',f)]
for f in sorted(zod):
    nL=sum(1 for ln in C['folios'][f] if ltype(ln['locus'])=='L' for w in ln['words'] if '?' not in w)
    if nL>=10: print(f"  {f:7s}: {nL} labels")

print("\n" + "="*70)
print("3. LIST — the 'recipes/stars' section (f103+): paragraph regularity")
print("="*70)
rec=[f for f in C['folios'] if re.match(r'f(10[3-9]|11[0-6])[rv]',f)]
plens=[]
for f in rec:
    para=[];
    for ln in C['folios'][f]:
        if ltype(ln['locus'])=='P':
            para+= [w for w in ln['words'] if '?' not in w]
    # paragraphs separated by locus '@P0' starts; approximate by counting P-lines
    nlines=sum(1 for ln in C['folios'][f] if ltype(ln['locus'])=='P')
    if nlines: plens.append(nlines)
print(f"  recipe folios: {len(rec)}, prose-lines/folio: "
      f"mean {np.mean(plens):.0f}, range {min(plens)}-{max(plens)}")
# first-word-of-line vocabulary in recipes: is there a recurring 'entry opener'?
openers=Counter()
for f in rec:
    for ln in C['folios'][f]:
        if ltype(ln['locus'])=='P':
            ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
            if ws: openers[ws[0]]+=1
print(f"  most common line-openers in recipes: {openers.most_common(8)}")
