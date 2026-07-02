"""
ENTROPY PANEL — is Voynich's famous "anomalously low glyph entropy" a real
cipher clue, or (like the regen score) another artifact that a simple real
language reproduces?

Compute conditional character entropy h1, h2, h3 (bits) on the token stream of
each corpus, word boundary included as a symbol. The Voynich literature's claim:
h2 ~2.1 bits is far below European languages (~3.0-3.7) -> a cipher signature.
The test: where does HAWAIIAN (13 phonemes, real language) fall? If Hawaiian ~
Voynich, low entropy does NOT distinguish cipher from simple language. Whatever
Voynich does BELOW the simplest real language is the residual worth attacking.
"""
import re, json, math, sys
from collections import Counter
import numpy as np

def voy():
    C = json.load(open('data/corpus/corpus.json'))
    return [w for f in C['folios'].values() for ln in f
            for w in ln['words'] if '?' not in w and len(w) >= 1]

def txt(path, cap=400_000):
    t = open(path, encoding='utf-8', errors='ignore').read().lower()
    m = re.search(r'\*\*\* start of.*?\*\*\*', t, re.S);  t = t[m.end():] if m else t
    m = re.search(r'\*\*\* end of', t, re.S);             t = t[:m.start()] if m else t
    return re.findall(r"[a-zÀ-ɏʻ'']+", t)[:cap]

def cond_entropy(tokens, k):
    """H(c_n | previous k-1 chars), bits, over stream with ' ' between words."""
    s = ' ' + ' '.join(tokens) + ' '
    ctx = Counter(); joint = Counter()
    for i in range(len(s) - k + 1):
        g = s[i:i+k]
        ctx[g[:-1]] += 1; joint[g] += 1
    H = 0.0; total = sum(joint.values())
    for g, c in joint.items():
        p_joint = c / total
        p_cond = c / ctx[g[:-1]]
        H -= p_joint * math.log2(p_cond)
    return H

def run(name, tokens):
    h1 = cond_entropy(tokens, 1)   # marginal char entropy
    h2 = cond_entropy(tokens, 2)   # given previous 1
    h3 = cond_entropy(tokens, 3)   # given previous 2
    alpha = len(set(''.join(tokens)))
    print(f"{name:11s} | alpha={alpha:2d} | h1={h1:4.2f} | h2={h2:4.2f} | h3={h3:4.2f} | "
          f"drop h1->h3={h1-h3:4.2f}")
    return dict(name=name, alpha=alpha, h1=h1, h2=h2, h3=h3)

if __name__ == '__main__':
    print("CONDITIONAL CHARACTER ENTROPY (bits) — lower = more predictable glyphs\n")
    rows = [run('VOYNICH', voy())]
    for a in sys.argv[1:]:
        n, p = a.split(':'); rows.append(run(n, txt(p)))
    print("\n(if HAWAIIAN's h2/h3 ~ VOYNICH, low entropy is a simplicity artifact,")
    print(" not a cipher signature. Any gap Voynich opens BELOW Hawaiian is the residual.)")
    json.dump(rows, open('data/intermediate/_entropy_panel.json','w'), indent=2)
