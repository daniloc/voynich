"""
IMPLICIT (unsupervised) morphology induction — find the slot structure with NO
hand-fed affix lists, so coverage and residual are discovered, not assumed.
Two parameter-light discovery tools, compared:
  (1) HARRIS branching entropy: a morpheme boundary is where the uncertainty of the
      next glyph SPIKES (many possible continuations). Forward pass -> prefix cuts;
      backward pass -> suffix cuts. Pure information theory, zero linguistic input.
  (2) BPE: greedily merge the most frequent adjacent glyph-pair; the merges that
      survive are the data's preferred recurring units.
Then compare both against the hand-built slot grammar: do they rediscover qo-/-dy etc?
"""
import json, re, math
from collections import Counter, defaultdict
C=json.load(open('data/corpus/corpus.json'))
toks=[w for f in C['folios'].values() for ln in f for w in ln['words'] if '?' not in w and len(w)>=2]
freq=Counter(toks); vocab=list(freq)

# ---------- (1) Harris branching entropy --------------------------------------
def branching_entropy_tables(words, weights):
    """fwd[prefix] = entropy of next char; bwd[suffix] = entropy of prev char."""
    fwd_next=defaultdict(Counter); bwd_prev=defaultdict(Counter)
    for w,wt in zip(words,weights):
        s='^'+w+'$'
        for i in range(1,len(s)):
            fwd_next[s[:i]][s[i]]+=wt
        for i in range(len(s)-1):
            bwd_prev[s[i+1:]][s[i]]+=wt
    def ent(counter):
        n=sum(counter.values()); return -sum(c/n*math.log2(c/n) for c in counter.values()) if n else 0
    fwd={k:ent(v) for k,v in fwd_next.items()}
    bwd={k:ent(v) for k,v in bwd_prev.items()}
    return fwd,bwd
W=[freq[w] for w in vocab]
fwd,bwd=branching_entropy_tables(vocab,W)

def segment(w, thr=0.5):
    """boundary after position i if forward BE rises (local spike) -> prefix end,
    and before position j if backward BE rises -> suffix start. Keep <=2 cuts."""
    s='^'+w+'$'
    fcuts=[]
    for i in range(2,len(s)-1):
        # rise in forward branching entropy at this cut
        if fwd.get(s[:i],0)-fwd.get(s[:i-1],0) > thr:
            fcuts.append(i-1)            # boundary in original-word coords
    bcuts=[]
    for j in range(2,len(s)-1):
        if bwd.get(s[len(s)-j:],0)-bwd.get(s[len(s)-j+1:],0) > thr:
            bcuts.append(len(w)-j+1)
    pre_end = fcuts[0] if fcuts else 0
    suf_start = bcuts[0] if bcuts else len(w)
    if suf_start<=pre_end: suf_start=len(w)
    return w[:pre_end], w[pre_end:suf_start], w[suf_start:]

seg=[segment(w) for w in vocab]
ind_pre=Counter(); ind_suf=Counter()
boundaried=0
for (p,c,s),w in zip(seg,vocab):
    if p: ind_pre[p]+=freq[w]
    if s: ind_suf[s]+=freq[w]
    if p or s: boundaried+=freq[w]
NT=sum(freq.values())
print("="*70)
print("(1) HARRIS branching-entropy induction (no hand input)")
print("="*70)
print(f"  tokens with an induced boundary: {100*boundaried/NT:.1f}%")
print(f"  top induced PREFIXES: {[p for p,_ in ind_pre.most_common(10)]}")
print(f"  top induced SUFFIXES: {[s for s,_ in ind_suf.most_common(10)]}")

# ---------- (2) BPE ------------------------------------------------------------
def bpe(words_freq, merges=40):
    V={ ' '.join(w)+' </w>':f for w,f in words_freq.items() }
    learned=[]
    for _ in range(merges):
        pairs=Counter()
        for word,f in V.items():
            sym=word.split()
            for a,b in zip(sym,sym[1:]): pairs[(a,b)]+=f
        if not pairs: break
        best=max(pairs,key=pairs.get); learned.append(best)
        bigram=' '.join(best); rep=''.join(best)
        V={ w.replace(bigram,rep):f for w,f in V.items() }
    return learned
merges=bpe(freq, 40)
print("\n" + "="*70)
print("(2) BPE merges (data's preferred recurring units, in learning order)")
print("="*70)
print("  ", [a+b for a,b in merges[:24]])

# ---------- compare to hand grammar -------------------------------------------
HAND_PRE={'qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph'}
HAND_SUF={'eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y'}
ip=set(dict(ind_pre.most_common(15))); is_=set(dict(ind_suf.most_common(15)))
print("\n" + "="*70)
print("COMPARISON — did the implicit tools rediscover the hand grammar?")
print("="*70)
print(f"  induced∩hand PREFIXES: {sorted(ip & HAND_PRE)}")
print(f"  induced-only prefixes: {sorted(ip - HAND_PRE)}")
print(f"  induced∩hand SUFFIXES: {sorted(is_ & HAND_SUF)}")
bpe_units=set(a+b for a,b in merges)
print(f"  BPE units that are hand-affixes: {sorted((bpe_units)&(HAND_PRE|HAND_SUF))}")

# ---------- residual at the ceiling -------------------------------------------
resid=[w for (p,c,s),w in zip(seg,vocab) if not p and not s]
print("\n" + "="*70)
print("RESIDUAL at the discovered ceiling (words no tool will segment)")
print("="*70)
print(f"  unsegmentable types: {len(resid)} ({100*sum(freq[w] for w in resid)/NT:.1f}% of tokens)")
print(f"  by onset: {Counter(w[0] for w in resid).most_common(6)}")
print(f"  examples: {[w for w in sorted(resid,key=lambda w:-freq[w])[:14]]}")
