"""
STEP 1 — VERBOSE-CIPHER TEST. Voynich's anomaly is that GLYPH h2 (~2.16) is far
below any real language while GLYPH h1 (~3.87) is normal -> redundancy is packed
INSIDE short glyph-groups. Verbose-cipher prediction: there's a segmentation into
units such that the UNIT sequence looks like ordinary plaintext, i.e. unit-level
h2 climbs back into the natural band (~3 bits) over a ~20-40 unit alphabet.

Method: greedy BPE (merge most frequent adjacent glyph pair, within words) applied
IDENTICALLY to Voynich and to real-language controls. Track, as we merge:
  - mean units per word
  - unit-alphabet size
  - unit-level h2 (conditional entropy over the unit stream)
Signature of a verbose cipher: Voynich's unit-h2 RISES toward the controls' as the
spurious within-unit redundancy is absorbed into the units. If instead unit-h2
stays low, the predictability is structural (templatic language / low-info message),
not a recoverable substitution.
"""
import re, json, math, sys
from collections import Counter
import numpy as np

def voy():
    C = json.load(open('data/corpus/corpus.json'))
    return [w for f in C['folios'].values() for ln in f
            for w in ln['words'] if '?' not in w and len(w) >= 1]

def txt(path, cap=200_000):
    t = open(path, encoding='utf-8', errors='ignore').read().lower()
    m = re.search(r'\*\*\* start of.*?\*\*\*', t, re.S);  t = t[m.end():] if m else t
    m = re.search(r'\*\*\* end of', t, re.S);             t = t[:m.start()] if m else t
    return re.findall(r"[a-zÀ-ɏʻ'']+", t)[:cap]

def h2_units(words):
    """conditional entropy H(u_n | u_{n-1}) over the unit stream, boundary-aware."""
    ctx = Counter(); joint = Counter()
    for w in words:
        seq = ['^'] + w + ['$']
        for a, b in zip(seq, seq[1:]):
            ctx[a] += 1; joint[(a, b)] += 1
    H = 0.0; total = sum(joint.values())
    for (a, b), c in joint.items():
        H -= (c/total) * math.log2(c/ctx[a])
    return H

def bpe_track(tokens, merges, checkpoints):
    words = [list(t) for t in tokens]
    out = {}
    for m in range(merges + 1):
        if m in checkpoints:
            alpha = len(set(u for w in words for u in w))
            mpw = np.mean([len(w) for w in words])
            out[m] = (round(mpw, 2), alpha, round(h2_units(words), 2))
        # find most frequent adjacent pair
        pairs = Counter()
        for w in words:
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += 1
        if not pairs: break
        (x, y), _ = pairs.most_common(1)[0]
        for w in words:
            i = 0
            while i < len(w) - 1:
                if w[i] == x and w[i+1] == y:
                    w[i:i+2] = [x + y]
                else:
                    i += 1
    return out

if __name__ == '__main__':
    checks = [0, 10, 25, 50, 100]
    corpora = [('VOYNICH', voy())] + [(a.split(':')[0], txt(a.split(':')[1])) for a in sys.argv[1:]]
    print("Greedy BPE merges applied identically. Cell = (units/word, unit-alphabet, unit-h2 bits)\n")
    print(f"{'merges':<9}" + ''.join(f"{n:<22}" for n, _ in corpora))
    tracks = {n: bpe_track(t, max(checks), checks) for n, t in corpora}
    for m in checks:
        row = f"{m:<9}"
        for n, _ in corpora:
            mpw, al, h = tracks[n][m]
            row += f"{f'{mpw}/wd a={al} h2={h}':<22}"
        print(row)
    print("\nRead the h2 column: if VOYNICH's unit-h2 climbs into the controls' band as")
    print("merges absorb within-unit redundancy, the glyph-groups ARE cipher units.")
