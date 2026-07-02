"""
STEP 1 (HARDENED) — THE VERBOSE-CIPHER DECISION GATE.

unit_collapse.py showed Voynich unit-h2 climbs ~3x faster than real languages under
BPE merging, and read that as a verbose-cipher signature (redundancy packed inside
recoverable glyph-groups). But it never ran the control that killed every other
"signal" in this project: a MATCHED CONTENT-FREE NULL.

The null: a first-order glyph Markov trained on the Voynich TOKEN stream. It
reproduces Voynich's glyph unigram + bigram statistics (hence its low glyph-h2 ~2.1)
EXACTLY, but it has no message and no multi-glyph units -- it's meaningless by
construction. Run identical BPE on it.

Three outcomes:
  (1) VOYNICH climbs, SURROGATE does NOT  -> the climb needs structure the bigrams
      don't carry => recoverable units exist => verbose cipher LIVES. Proceed.
  (2) VOYNICH and SURROGATE climb the SAME -> the climb is just an orthographic
      artifact of templatic spelling, reproducible with zero meaning => h2-climb is
      NOT diagnostic => L1 weakened; stop banking it as cipher-evidence.
  (3) SURROGATE climbs MORE -> anti-diagnostic.

Real-language panel (Hawaiian, Maori, Latin, English) included for the natural band.
All corpora capped to the Voynich token count for fairness.
"""
import re, json, math, sys, os
from collections import Counter, defaultdict
import numpy as np

rng = np.random.default_rng(13)
START, END = '\x02', '\x03'

def voy_tokens():
    C = json.load(open('data/corpus/corpus.json'))
    return [w for f in C['folios'].values() for ln in f
            for w in ln['words'] if '?' not in w and len(w) >= 1]

def txt_tokens(path, cap):
    t = open(path, encoding='utf-8', errors='ignore').read().lower()
    m = re.search(r'\*\*\* start of.*?\*\*\*', t, re.S); t = t[m.end():] if m else t
    m = re.search(r'\*\*\* end of', t, re.S);            t = t[:m.start()] if m else t
    return re.findall(r"[a-zÀ-ɏʻ'']+", t)[:cap]

# ---- content-free surrogate: first-order glyph Markov over the token stream ----
def glyph_markov_surrogate(tokens, n_out):
    """Train START->g->...->END first-order char Markov on the TOKEN stream
    (token-weighted, so glyph uni/bigram stats match), sample n_out meaningless tokens."""
    trans = defaultdict(Counter)
    for w in tokens:
        seq = [START] + list(w) + [END]
        for a, b in zip(seq, seq[1:]):
            trans[a][b] += 1
    # precompute sampling tables
    tbl = {}
    for a, c in trans.items():
        outs = list(c.keys()); ws = np.array([c[o] for o in outs], float); ws /= ws.sum()
        tbl[a] = (outs, ws)
    out = []
    for _ in range(n_out):
        cur = START; w = []
        for _ in range(40):  # safety cap
            outs, ws = tbl[cur]
            nx = outs[rng.choice(len(outs), p=ws)]
            if nx == END: break
            w.append(nx); cur = nx
        if w: out.append(''.join(w))
    return out

# ---- BPE unit-collapse, identical to unit_collapse.py ----
def h2_units(words):
    ctx = Counter(); joint = Counter()
    for w in words:
        seq = ['^'] + w + ['$']
        for a, b in zip(seq, seq[1:]):
            ctx[a] += 1; joint[(a, b)] += 1
    H = 0.0; total = sum(joint.values())
    for (a, b), c in joint.items():
        H -= (c/total) * math.log2(c/ctx[a])
    return H

def bpe_track(tokens, checkpoints):
    words = [list(t) for t in tokens]; out = {}
    for m in range(max(checkpoints) + 1):
        if m in checkpoints:
            alpha = len(set(u for w in words for u in w))
            mpw = float(np.mean([len(w) for w in words]))
            out[m] = (round(mpw, 2), alpha, round(h2_units(words), 2))
        pairs = Counter()
        for w in words:
            for a, b in zip(w, w[1:]): pairs[(a, b)] += 1
        if not pairs: break
        (x, y), _ = pairs.most_common(1)[0]
        for w in words:
            i = 0
            while i < len(w) - 1:
                if w[i] == x and w[i+1] == y: w[i:i+2] = [x+y]
                else: i += 1
    return out

if __name__ == '__main__':
    V = voy_tokens(); N = len(V)
    print(f"Voynich token count = {N}; capping all corpora to N.\n")
    SUR = glyph_markov_surrogate(V, N)
    corpora = [('VOYNICH', V), ('SURROGATE(cf)', SUR)]
    for spec in sys.argv[1:]:
        name, path = spec.split(':')
        corpora.append((name, txt_tokens(path, N)))

    checks = [0, 10, 25, 50, 100]
    tracks = {n: bpe_track(t, checks) for n, t in corpora}
    print("BPE merges applied identically. Cell = units/word | alphabet | unit-h2(bits)\n")
    print(f"{'merges':<8}" + ''.join(f"{n:<24}" for n, _ in corpora))
    for m in checks:
        row = f"{m:<8}"
        for n, _ in corpora:
            mpw, al, h = tracks[n][m]
            row += f"{f'{mpw}/wd a={al} h2={h}':<24}"
        print(row)
    print("\n--- the gate: h2 climb from 0 to 100 merges ---")
    base = {n: tracks[n][0][2] for n, _ in corpora}
    top  = {n: tracks[n][100][2] for n, _ in corpora}
    for n, _ in corpora:
        print(f"  {n:<16} h2 {base[n]:.2f} -> {top[n]:.2f}   climb +{top[n]-base[n]:.2f}")
    vc = top['VOYNICH'] - base['VOYNICH']; sc = top['SURROGATE(cf)'] - base['SURROGATE(cf)']
    print(f"\n  VOYNICH climb +{vc:.2f}  vs  content-free SURROGATE climb +{sc:.2f}")
    if vc > sc + 0.3:
        print("  => Voynich climbs BEYOND its content-free surrogate. Units carry structure")
        print("     the bigrams alone don't. Verbose cipher LIVES -> proceed to Step 2.")
    elif abs(vc - sc) <= 0.3:
        print("  => Voynich and its meaningless surrogate climb the SAME. The climb is an")
        print("     orthographic artifact of templatic spelling, not a cipher signature.")
        print("     h2-climb is NOT diagnostic. Do not bank L1 as cipher-evidence.")
    else:
        print("  => Surrogate climbs MORE. Anti-diagnostic for verbose cipher.")
