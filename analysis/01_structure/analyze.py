"""
Core statistical battery on the Voynich corpus, with natural-language controls
(Latin = Cicero, English = KJV) run through the *identical* pipeline, plus
null-hypothesis generators. Goal: locate exactly where Voynichese looks like
language and where it violates language.
"""
import json, re, math, random
from collections import Counter, defaultdict

random.seed(1989)  # fixed: scripts can't use Math.random equivalents reproducibly otherwise

# ----------------------------------------------------------------- load Voynich
C = json.load(open('data/corpus/corpus.json'))
META = C['meta']
def folio_words(filt=None):
    out = []
    for f, lines in C['folios'].items():
        lang = META.get(f, {}).get('L', '?')
        if filt and lang != filt: continue
        for ln in lines:
            out += [w for w in ln['words'] if '?' not in w]
    return out
V_ALL = folio_words()
V_A   = folio_words('A')
V_B   = folio_words('B')

# ----------------------------------------------------------------- load controls
def words_from_text(path, keep=200000):
    t = open(path, encoding='utf-8', errors='ignore').read().lower()
    t = re.sub(r'[^a-z\s]', ' ', t)
    ws = t.split()
    return ws[:keep]
EN = words_from_text('data/controls/english.txt')
LA = words_from_text('data/controls/latin.txt')

# ----------------------------------------------------------------- metrics
def char_entropies(words):
    """h0 = log2(alphabet), h1 = unigram char entropy, h2 = conditional entropy
    of next char given previous char. Lower h2 = more predictable/rigid."""
    text = ' '.join(words)
    uni = Counter(text)
    n = sum(uni.values())
    h1 = -sum(c/n * math.log2(c/n) for c in uni.values())
    bi = Counter(zip(text, text[1:]))
    nb = sum(bi.values())
    # H(X2|X1) = H(X1,X2) - H(X1)
    h_joint = -sum(c/nb * math.log2(c/nb) for c in bi.values())
    h_x1 = -sum(c/nb * math.log2(c/nb) for c in Counter(a for a,_ in bi).values())
    h2_cond = h_joint - h_x1
    return len(uni), h1, h2_cond

def zipf_fit(words):
    """Slope of log(freq) vs log(rank); natural language ~ -1."""
    freqs = sorted(Counter(words).values(), reverse=True)
    import math
    xs = [math.log(i+1) for i in range(len(freqs))]
    ys = [math.log(f) for f in freqs]
    n = len(xs); sx=sum(xs); sy=sum(ys); sxx=sum(x*x for x in xs); sxy=sum(x*y for x,y in zip(xs,ys))
    slope = (n*sxy - sx*sy)/(n*sxx - sx*sx)
    return slope

def wordlen_stats(words):
    L = [len(w) for w in words]
    mean = sum(L)/len(L)
    var = sum((x-mean)**2 for x in L)/len(L)
    return mean, var, var/mean  # var/mean: ~1 if Poisson-like; <1 = unnaturally narrow

def adjacent_repeat_rate(words):
    """Fraction of adjacent token pairs that are identical (w_i == w_i+1)."""
    same = sum(1 for a,b in zip(words, words[1:]) if a==b)
    return same/(len(words)-1)

def shuffled_repeat_rate(words):
    w = words[:]; random.shuffle(w)
    return adjacent_repeat_rate(w)

def hapax_fraction(words):
    c = Counter(words)
    return sum(1 for v in c.values() if v==1)/len(c)

# ----------------------------------------------------------------- run
def report(name, words):
    A, h1, h2 = char_entropies(words)
    mean,var,ratio = wordlen_stats(words)
    print(f"\n### {name}  (tokens={len(words):,}, types={len(set(words)):,})")
    print(f"  alphabet size        : {A-1}")           # minus the space char
    print(f"  h1 (char entropy)    : {h1:.3f} bits")
    print(f"  h2 (cond. entropy)   : {h2:.3f} bits   <-- lower = more rigid")
    print(f"  Zipf slope           : {zipf_fit(words):+.3f}   (natural lang ~ -1.0)")
    print(f"  mean word length     : {mean:.2f}")
    print(f"  word-len var/mean    : {ratio:.3f}   (~1 Poisson, <1 = unnaturally uniform)")
    print(f"  hapax fraction       : {hapax_fraction(words):.3f}")
    print(f"  adjacent-repeat rate : {adjacent_repeat_rate(words)*100:.3f}%")
    print(f"    same, shuffled ctrl: {shuffled_repeat_rate(words)*100:.3f}%   <-- repetition is NOT positional artifact if real>>shuffled")

print("="*70)
print("CROSS-CORPUS STATISTICAL BATTERY  (identical pipeline on all)")
print("="*70)
report("VOYNICH (all)", V_ALL)
report("VOYNICH lang-A", V_A)
report("VOYNICH lang-B", V_B)
report("LATIN  (Cicero)", LA)
report("ENGLISH (KJV)", EN)

# ----------------------------------------------------------------- A/B divergence
ca, cb = Counter(V_A), Counter(V_B)
shared = set(ca) & set(cb)
print("\n" + "="*70)
print("CURRIER A vs B  — same script, different statistics?")
print("="*70)
print(f"  vocab A: {len(ca):,}   vocab B: {len(cb):,}   shared types: {len(shared):,}")
print(f"  shared/union: {len(shared)/len(set(ca)|set(cb)):.3f}")
# signature words
def top_distinct(c1, c2, n=12):
    out=[]
    for w,f in c1.most_common(200):
        r1 = f/sum(c1.values()); r2 = (c2.get(w,0)+1)/sum(c2.values())
        out.append((w, r1/r2))
    return sorted(out, key=lambda x:-x[1])[:n]
print(f"  most A-skewed words: {[w for w,_ in top_distinct(ca,cb)]}")
print(f"  most B-skewed words: {[w for w,_ in top_distinct(cb,ca)]}")
