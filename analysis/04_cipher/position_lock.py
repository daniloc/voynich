"""
Independent replication of the K11 position-lock diagnostic.

Verbose-substitution claim requires: a glyph = a cipher symbol = a plaintext
letter, which therefore appears at MANY positions within words (letters roam).
If Voynich glyphs are instead POSITION-LOCKED (q always onset, n always coda,
e/d medial), they are slot-fillers in a template, not substitution symbols.

Metric: for each glyph, bin its relative within-word position into 5 bins;
compute that glyph's positional entropy (bits); report the frequency-weighted
mean. Low = position-locked (template). Compare Voynich to real languages.
"""
import re, json, math, sys
from collections import Counter, defaultdict
import numpy as np

NBINS = 5
def voy():
    C=json.load(open('data/corpus/corpus.json'))
    return [w for f in C['folios'].values() for ln in f
            for w in ln['words'] if '?' not in w and len(w)>=2]
def txt(path, cap=200_000):
    t=open(path,encoding='utf-8',errors='ignore').read().lower()
    m=re.search(r'\*\*\* start of.*?\*\*\*',t,re.S); t=t[m.end():] if m else t
    m=re.search(r'\*\*\* end of',t,re.S);            t=t[:m.start()] if m else t
    return [w for w in re.findall(r"[a-zÀ-ɏʻ'']+",t)[:cap] if len(w)>=2]

def pos_entropy(words):
    per=defaultdict(lambda: np.zeros(NBINS)); freq=Counter()
    for w in words:
        L=len(w)
        for i,ch in enumerate(w):
            r=i/(L-1) if L>1 else 0.5
            b=min(NBINS-1, int(r*NBINS))
            per[ch][b]+=1; freq[ch]+=1
    tot=sum(freq.values()); H=0.0
    for ch,counts in per.items():
        p=counts/counts.sum(); h=-sum(x*math.log2(x) for x in p if x>0)
        H += (freq[ch]/tot)*h
    return H, len(freq)

if __name__=='__main__':
    print(f"Glyph positional entropy (bits, {NBINS} bins) — LOW = position-locked template\n")
    rows=[('VOYNICH',voy())]+[(a.split(':')[0],txt(a.split(':')[1])) for a in sys.argv[1:]]
    for n,ws in rows:
        H,k=pos_entropy(ws)
        print(f"  {n:11s} H_pos={H:.2f}  (glyphs={k})")
    print("\nIf VOYNICH H_pos sits well below every real language, glyphs are slot-locked")
    print("=> not roaming substitution symbols => verbose cipher disfavored (corroborates K11).")
