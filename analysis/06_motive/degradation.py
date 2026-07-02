"""
MOTIVE TEST 2 — does the generative DISCIPLINE decay across production order?
  constant discipline  -> a lavished, sustained object (a marvel / devotional feat)
  decay toward the end -> churned work on a deadline (a commodity)
  improvement          -> a maker learning the system as they invent it (organic)

Confound = genre/dialect changes across the book. Control it by running the trend
SEPARATELY WITHIN each Currier dialect (a single hand/register), ordered by the
codicological quire+position codes (Q,P). Caveat: quire order = the REBOUND order,
an approximation of true production order.

Per equal-token chunk along production order, measure four discipline metrics, test
for a monotonic trend (Spearman r vs chunk index), with a permutation null that
reshuffles folio order within the dialect.
"""
import json, math
from collections import Counter, defaultdict
import numpy as np
rng = np.random.default_rng(11)

def load():
    C = json.load(open('data/corpus/corpus.json')); meta = C['meta']
    out = {'A': [], 'B': []}            # list of (Q,P,[tokens]) per folio
    for fol, lines in C['folios'].items():
        m = meta.get(fol, {}); L = m.get('L')
        if L not in ('A', 'B'): continue
        toks = [w for ln in lines for w in ln['words'] if '?' not in w and len(w) >= 1]
        if toks: out[L].append((m.get('Q', 'Z'), m.get('P', 'Z'), toks))
    for L in out: out[L].sort(key=lambda t: (t[0], t[1]))   # codicological order
    return out

def h2_glyph(tokens):
    ctx = Counter(); joint = Counter()
    for w in tokens:
        seq = ['^'] + list(w) + ['$']
        for a, b in zip(seq, seq[1:]): ctx[a] += 1; joint[(a, b)] += 1
    tot = sum(joint.values())
    return -sum((c/tot)*math.log2(c/ctx[a]) for (a, b), c in joint.items())

def adj_rep(tokens):
    return np.mean([tokens[i] == tokens[i+1] for i in range(len(tokens)-1)])

def hapax(tokens):
    c = Counter(tokens); return sum(1 for t in c.values() if t == 1)/len(c)

def wlen(tokens):
    return np.mean([len(w) for w in tokens])

METRICS = [('h2-glyph (template tightness)', h2_glyph),
           ('adj-word repeat (copying)', adj_rep),
           ('hapax frac (still inventing)', hapax),
           ('word length', wlen)]

def chunk_metrics(folio_tokens, K):
    toks = [w for _, _, ts in folio_tokens for w in ts]
    n = len(toks); size = n // K
    chunks = [toks[i*size:(i+1)*size] for i in range(K)]
    return {name: [fn(c) for c in chunks] for name, fn in METRICS}

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = math.sqrt((rx**2).sum()*(ry**2).sum())
    return (rx*ry).sum()/d if d else 0.0

def trend_test(folio_tokens, K=8, nperm=1000):
    obs = chunk_metrics(folio_tokens, K)
    idx = np.arange(K)
    results = {}
    for name, _ in METRICS:
        r = spearman(idx, obs[name])
        null = []
        for _ in range(nperm):
            perm = list(folio_tokens); rng.shuffle(perm)
            null.append(abs(spearman(idx, chunk_metrics(perm, K)[name])))
        p = (np.sum(np.array(null) >= abs(r)) + 1)/(nperm+1)
        results[name] = (r, p, obs[name])
    return results

if __name__ == '__main__':
    data = load()
    for L in ('A', 'B'):
        ft = data[L]; ntok = sum(len(ts) for _, _, ts in ft)
        K = 6 if L == 'A' else 10
        print(f"\n===== Currier {L}  ({len(ft)} folios, {ntok} tokens, {K} chunks, quires "
              f"{ft[0][0]}..{ft[-1][0]}) =====")
        res = trend_test(ft, K=K)
        for name, _ in METRICS:
            r, p, traj = res[name]
            arrow = 'rising ' if r > 0 else 'falling'
            sig = '  *** trend' if p < 0.05 else ''
            tr = ' '.join(f'{v:.2f}' for v in traj)
            print(f"  {name:<32} Spearman {r:+.2f} (p={p:.3f}) {arrow}{sig}")
            print(f"      across production order: [{tr}]")
    print("\nReading: significant FALLING h2 / RISING copying / FALLING hapax toward the end")
    print("= discipline decaying (commodity on a deadline). Flat across all = sustained (marvel).")
