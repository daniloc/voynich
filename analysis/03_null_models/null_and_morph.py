"""
Two things:
(1) NULL-HYPOTHESIS GENERATORS. Build meaningless text that mimics Voynich
    surface statistics. If a "decoding" method later extracts meaning from THIS,
    the method is worthless. This is the test every famous false solution failed.
(2) WORD MORPHOLOGY / SLOT GRAMMAR. Voynich words look strongly positional
    (prefix-core-suffix). Quantify how much of the vocabulary a tiny slot grammar
    explains — the central evidence for the "constructed/abbreviation" view.
"""
import json, math, random
from collections import Counter, defaultdict
random.seed(1989)

C = json.load(open('data/corpus/corpus.json'))
META = C['meta']
words = [w for f in C['folios'].values() for ln in f for w in ln['words'] if '?' not in w]

def adj_repeat(ws):
    return sum(1 for a,b in zip(ws,ws[1:]) if a==b)/(len(ws)-1)
def wl_ratio(ws):
    L=[len(w) for w in ws]; m=sum(L)/len(L); v=sum((x-m)**2 for x in L)/len(L); return m, v/m
def zipf(ws):
    fr=sorted(Counter(ws).values(),reverse=True)
    xs=[math.log(i+1) for i in range(len(fr))]; ys=[math.log(f) for f in fr]
    n=len(xs);sx=sum(xs);sy=sum(ys);sxx=sum(x*x for x in xs);sxy=sum(x*y for x,y in zip(xs,ys))
    return (n*sxy-sx*sy)/(n*sxx-sx*sx)
def profile(name, ws):
    m,r = wl_ratio(ws)
    print(f"  {name:28s} types={len(set(ws)):5d} zipf={zipf(ws):+.2f} "
          f"wlen={m:.2f} var/mean={r:.2f} adj-rep={adj_repeat(ws)*100:.2f}%")

print("="*78)
print("(1) NULL GENERATORS  — can meaningless text fake the Voynich profile?")
print("="*78)
profile("REAL Voynich", words)

# --- Null A: order-2 character Markov chain trained on Voynich words ----------
def train_char_markov(ws, k=2):
    m = defaultdict(Counter)
    for w in ws:
        s = '^'*k + w + '$'
        for i in range(k, len(s)):
            m[s[i-k:i]][s[i]] += 1
    return m
def gen_char_markov(m, k=2):
    s='^'*k
    while True:
        nxt=m[s[-k:]]
        if not nxt: return s.strip('^').replace('$','')
        ch=random.choices(list(nxt), weights=list(nxt.values()))[0]
        if ch=='$': return s.strip('^')
        s+=ch
mk = train_char_markov(words)
null_markov = [gen_char_markov(mk) for _ in range(len(words))]
profile("NULL: char-Markov(2)", null_markov)

# --- Null B: self-citation generator (Timm/Schinner style) -------------------
# Start from a seed word; each new word = copy a recent word, sometimes mutate
# one glyph, sometimes reuse verbatim. Produces Zipf + the repetition anomaly
# with ZERO underlying meaning.
glyphs = list({c for w in words for c in w})
def self_citation(n, window=40, p_verbatim=0.12, p_mutate=0.55):
    out=[random.choice(words[:50])]
    for _ in range(n-1):
        src = random.choice(out[-window:])
        r = random.random()
        if r < p_verbatim:
            out.append(src)                       # exact repeat -> adjacency anomaly
        elif r < p_verbatim+p_mutate and len(src)>1:
            i=random.randrange(len(src))
            out.append(src[:i]+random.choice(glyphs)+src[i+1:])
        else:
            out.append(random.choice(out[-window:]))
    return out
null_self = self_citation(len(words))
profile("NULL: self-citation", null_self)
print("\n  --> If a generator with NO meaning reproduces these stats, then matching")
print("      surface statistics proves nothing. A real claim must do more.")

# ----------------------------------------------------------------------------
print("\n" + "="*78)
print("(2) SLOT GRAMMAR  — are Voynich words built prefix+core+suffix?")
print("="*78)
# Common Voynich morphological elements (well established in the literature)
PREFIXES = ['qo','qok','qot','o','y','ch','sh','d','cph','cth','ckh','cfh']
SUFFIXES = ['dy','edy','eedy','aiin','ain','ain','iin','y','ol','or','ar','al','am','dam','ey','eey','eody']
vocab = Counter(words)
def decompose(w):
    pre = next((p for p in sorted(PREFIXES,key=len,reverse=True) if w.startswith(p)), '')
    rem = w[len(pre):]
    suf = next((s for s in sorted(SUFFIXES,key=len,reverse=True) if rem.endswith(s) and len(rem)>len(s)), '')
    core = rem[:len(rem)-len(suf)] if suf else rem
    return pre, core, suf
explained_types = sum(1 for w in vocab if (lambda d:d[0] and d[2])(decompose(w)))
explained_tok   = sum(f for w,f in vocab.items() if (lambda d:d[0] and d[2])(decompose(w)))
tot_tok = sum(vocab.values())
print(f"  words matching prefix+core+suffix: "
      f"{explained_types/len(vocab)*100:.1f}% of TYPES, "
      f"{explained_tok/tot_tok*100:.1f}% of TOKENS")
core_counts = Counter(decompose(w)[1] for w in words)
print(f"  distinct cores: {len(core_counts):,} (vs {len(vocab):,} whole words)")
print(f"  top cores: {[c for c,_ in core_counts.most_common(15)]}")
print(f"  top prefixes seen: {[p for p,_ in Counter(decompose(w)[0] for w in words if decompose(w)[0]).most_common(8)]}")
print(f"  top suffixes seen: {[s for s,_ in Counter(decompose(w)[2] for w in words if decompose(w)[2]).most_common(8)]}")

# How constrained is glyph ORDER within a word? positional glyph entropy
maxlen=8
pos_alpha=defaultdict(Counter)
for w in words:
    for i,c in enumerate(w[:maxlen]):
        pos_alpha[i][c]+=1
print("\n  positional glyph predictability (entropy in bits by position in word):")
for i in range(maxlen):
    cc=pos_alpha[i]; n=sum(cc.values())
    H=-sum(v/n*math.log2(v/n) for v in cc.values())
    top=cc.most_common(3)
    print(f"    pos {i}: H={H:.2f} bits  dominant={[f'{g}:{100*v//n}%' for g,v in top]}")
