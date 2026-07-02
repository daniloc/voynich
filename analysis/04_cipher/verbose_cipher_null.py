"""
STEP 1 GATE (FINDINGS §6) — does the +1.73 BPE unit-h2 climb (L1) survive a
matched CONTENT-FREE null?

L1's claim: under greedy BPE glyph-pair merging, Voynich's unit-level conditional
entropy climbs far faster than any real language (+1.73 vs +0.58..0.75), so its
glyph redundancy is "packed in recoverable units" = verbose-cipher signature.

The killer confound (§3.3): a ~12% self-citation slot-template generator with NO
content reproduces nearly every Voynich statistic. It is built from the SAME real
prefix/core/suffix morphemes, so it should merge the same way. If the content-free
null ALSO climbs +1.73, the climb is generic to chunked/templated text and is NOT
evidence for a hidden plaintext -> L1 downgraded. Meaning can only live in any
EXCESS structure Voynich has BEYOND the matched null at equal merge count.
"""
import json, re, math, sys
from collections import Counter, defaultdict
import numpy as np
rng = np.random.default_rng(1989)

C = json.load(open('data/corpus/corpus.json'))
ALL = [w for f in C['folios'].values() for ln in f
       for w in ln['words'] if '?' not in w and len(w) >= 1]

# ---- matched content-free generator (residual.py Model B) -------------------
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph']
SUF=['eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s
corefreq=Counter(decomp(w)[1] for w in ALL)
pre_g=defaultdict(Counter); suf_g=defaultdict(Counter)
for w in ALL:
    p,c,s=decomp(w); pre_g[c][p]+=1; suf_g[c][s]+=1
cores=list(corefreq); cw=np.array([corefreq[c] for c in cores],float); cw/=cw.sum()
glyphs=sorted(set(ch for w in ALL for ch in w))
def sample_word():
    c=cores[rng.choice(len(cores),p=cw)]
    p=rng.choice(list(pre_g[c]),p=np.array(list(pre_g[c].values()))/sum(pre_g[c].values()))
    s=rng.choice(list(suf_g[c]),p=np.array(list(suf_g[c].values()))/sum(suf_g[c].values()))
    return p+c+s
def gen_stream(n,p_copy=0.12,window=40,p_mut=0.4):
    out=[]
    for _ in range(n):
        if out and rng.random()<p_copy:
            w=out[-rng.integers(1,min(window,len(out))+1)]
            if rng.random()<p_mut and len(w)>1:
                i=rng.integers(0,len(w)); w=w[:i]+rng.choice(glyphs)+w[i+1:]
            out.append(w)
        else: out.append(sample_word())
    return out

def txt(path, cap=200_000):
    t=open(path,encoding='utf-8',errors='ignore').read().lower()
    m=re.search(r'\*\*\* start of.*?\*\*\*',t,re.S); t=t[m.end():] if m else t
    m=re.search(r'\*\*\* end of',t,re.S);            t=t[:m.start()] if m else t
    return re.findall(r"[a-zÀ-ɏʻ'']+",t)[:cap]

# ---- identical BPE + unit-level conditional entropy -------------------------
def h2_units(words):
    ctx=Counter(); joint=Counter()
    for w in words:
        seq=['^']+w+['$']
        for a,b in zip(seq,seq[1:]): ctx[a]+=1; joint[(a,b)]+=1
    tot=sum(joint.values())
    return -sum((c/tot)*math.log2(c/ctx[a]) for (a,b),c in joint.items())

def bpe_track(tokens, checkpoints):
    words=[list(t) for t in tokens]; out={}
    for m in range(max(checkpoints)+1):
        if m in checkpoints:
            out[m]=round(h2_units(words),2)
        pairs=Counter()
        for w in words:
            for a,b in zip(w,w[1:]): pairs[(a,b)]+=1
        if not pairs: break
        (x,y),_=pairs.most_common(1)[0]
        for w in words:
            i=0
            while i<len(w)-1:
                if w[i]==x and w[i+1]==y: w[i:i+2]=[x+y]
                else: i+=1
    return out

if __name__=='__main__':
    N=len(ALL)
    NULL=gen_stream(N)
    # sanity: content-free null must match the glyph-level (0-merge) anomaly
    corpora=[('VOYNICH',ALL),('NULL(content-free)',NULL),
             ('HAWAIIAN',txt('data/controls/hawaiian.txt')),('LATIN',txt('data/controls/latin.txt'))]
    checks=[0,10,25,50,100]
    print(f"matched-null size: {len(NULL)} tokens vs Voynich {N}\n")
    print(f"{'merges':<8}"+''.join(f"{n:<20}" for n,_ in corpora))
    tr={n:bpe_track(t,checks) for n,t in corpora}
    for m in checks:
        print(f"{m:<8}"+''.join(f"{tr[n][m]:<20}" for n,_ in corpora))
    print("\nclimb (h2@100 - h2@0):")
    for n,_ in corpora:
        print(f"  {n:<20} +{tr[n][100]-tr[n][0]:.2f}")
    print("\nGATE: if NULL climb ≈ VOYNICH climb -> the climb is generic to chunked")
    print("text, not a plaintext signature -> L1 downgraded. If VOYNICH still has")
    print("EXCESS inter-unit structure (lower h2) vs NULL at equal merges -> residual lives.")
