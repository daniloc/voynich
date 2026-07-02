"""
WHAT DOES REPETITION SIGNAL IN THE LABELS?
The Voynich's defining anomaly is repetition: running text OVER-repeats (adjacent
'qokedy qokedy', ~3x shuffle). If labels instead UNDER-repeat (collision-resistant),
then text and labels are opposite deviations from the language baseline => different
generative regimes. And in a near-unique naming system, each collision is a SIGNAL:
the system asserting "these two objects are the same".

We measure, against a null, three things repetition lets us detect:
  1. Regime: are labels anti-repetitive while prose is hyper-repetitive?
  2. Effective key-space: how many distinct 'things' can the label process name?
  3. Which collisions are GENERIC (predicted by frequency) vs SURPRISING (=targets).
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(1989)

C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
labels=[]; prose=[]
for f,lines in C['folios'].items():
    for ln in lines:
        t=ltype(ln['locus']); ws=[w for w in ln['words'] if '?' not in w]
        if t=='L': labels+=ws
        elif t=='P': prose+=ws
corpus_freq=Counter(w for f in C['folios'].values() for ln in f for w in ln['words'] if '?' not in w)
Ntot=sum(corpus_freq.values())

def simpson_eff(tokens):
    """1/sum(p^2) = effective number of distinct types (Hill number, order 2).
    Larger => flatter/more name-like; smaller => dominated by a few words."""
    c=Counter(tokens); n=sum(c.values())
    s=sum((v/n)**2 for v in c.values())
    return 1/s, n

print("="*72)
print("1. REGIME — repetition direction vs the language baseline")
print("="*72)
def reprate(tokens):
    c=Counter(tokens); return 1-sum(1 for v in c.values() if v==1)/len(c)  # frac of types that recur
le,ln_=simpson_eff(labels); pe,pn=simpson_eff(prose)
print(f"  recurring-type fraction:   labels {reprate(labels):.2f}   prose {reprate(prose):.2f}")
print(f"  effective vocabulary (1/Simpson):")
print(f"     labels: {le:6.0f} effective types from {ln_} tokens  (ratio {le/ln_:.2f})")
print(f"     prose : {pe:6.0f} effective types from {pn} tokens  (ratio {pe/pn:.2f})")
print("  -> labels flatter/closer to all-unique = a NAMING regime; prose is")
print("     concentrated/repetitive = a different, copy-like regime.")

print("\n" + "="*72)
print("2. EFFECTIVE KEY-SPACE — what the collision rate back-computes")
print("="*72)
# birthday estimator: if n draws give c colliding pairs, effective space V ~ n^2/(2c)
def colliding_pairs(tokens):
    c=Counter(tokens); return sum(v*(v-1)//2 for v in c.values())
for name,tok in [('labels',labels),('prose(size-matched)',None)]:
    if tok is None:
        # match label token count from prose, average over draws
        Vs=[]
        for _ in range(500):
            s=rng.choice(prose,size=len(labels),replace=False)
            cp=colliding_pairs(s); Vs.append(len(s)**2/(2*max(cp,1)))
        print(f"  {name:20s}: effective key-space ~ {np.mean(Vs):6.0f}")
    else:
        cp=colliding_pairs(tok)
        print(f"  {name:20s}: {cp} colliding pairs -> effective key-space ~ {len(tok)**2/(2*max(cp,1)):6.0f}")
print("  -> a LARGER label key-space than prose means the label generator can mint")
print("     more distinct tags before repeating: behaves like a keyed catalog.")

print("\n" + "="*72)
print("3. WHICH COLLISIONS ARE SIGNAL — generic vs surprising")
print("="*72)
# a label collision is 'generic' if the word is already common corpus-wide (birthday
# paradox hits common words first); 'surprising' if the word is rare corpus-wide yet
# still repeats among labels -> that pairing is the system asserting equivalence.
lab_counts=Counter(labels)
repeated=[(w,c) for w,c in lab_counts.items() if c>=2]
def surprisal(w):  # bits: how unexpected is this word corpus-wide
    return -math.log2(corpus_freq[w]/Ntot)
rows=sorted(repeated, key=lambda wc:-surprisal(wc[0]))
print(f"  repeated labels: {len(repeated)} of {len(lab_counts)} distinct")
print("\n  MOST SURPRISING repeated labels (rare corpus-wide but collide in labels)")
print("  -> these are the high-information collisions: prime targets for the test")
for w,c in rows[:12]:
    print(f"     {w:14s} x{c}  corpus-freq={corpus_freq[w]:3d}  surprisal={surprisal(w):.1f} bits")
print("\n  LEAST surprising repeated labels (common words; collisions ~ expected noise)")
for w,c in rows[-6:]:
    print(f"     {w:14s} x{c}  corpus-freq={corpus_freq[w]:3d}  surprisal={surprisal(w):.1f} bits")

# emit the target list for the alignment experiment
targets=[w for w,c in rows if corpus_freq[w]<=8][:20]
json.dump(targets, open('data/grounding/collision_targets.json','w'))
print(f"\n  wrote {len(targets)} surprising-collision target labels -> data/grounding/collision_targets.json")
