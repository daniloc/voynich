"""
MOTIVE TEST 1 — was the generator a TRANSMISSIBLE PROCEDURE run by multiple hands,
or one person's idiolect?  (Decides solo-inward vs produced-for-audience.)

Currier A and B are written by different scribes/hands (Davis: >=5 scribes; Currier:
A<->Scribe1, B<->Scribe2). If A and B are the SAME finite-state word-generator with
re-weighted transitions, then one explicit procedure was taught to / shared by
multiple people -> a workshop producing an artifact for someone. If they are
structurally DIFFERENT generators, that points to independent invention / evolution.

Method: build the first-order glyph automaton (START->g->...->END) for each group,
measure the "foreign-dialect penalty": per-char cross-entropy of held-out words from
group Y scored under group X's model, minus under Y's own model. Small penalty = same
system. Calibrate against:
  - SAME-system floor : random split of A into A1/A2 (same generator, two samples)
  - DIFFERENT-system ceiling : English vs Latin word-forms through identical machinery
Place Voynich A-vs-B on that spectrum.
"""
import json, re, math
from collections import Counter, defaultdict
import numpy as np
rng = np.random.default_rng(7)
S, E = '\x02', '\x03'

def voy_by_currier():
    C = json.load(open('data/corpus/corpus.json'))
    meta = C['meta']; A, B = [], []
    for fol, lines in C['folios'].items():
        L = meta.get(fol, {}).get('L')
        toks = [w for ln in lines for w in ln['words'] if '?' not in w and len(w) >= 1]
        if L == 'A': A += toks
        elif L == 'B': B += toks
    return A, B

def text_tokens(path, cap=30000):
    t = open(path, encoding='utf-8', errors='ignore').read().lower()
    m = re.search(r'\*\*\* start of.*?\*\*\*', t, re.S); t = t[m.end():] if m else t
    m = re.search(r'\*\*\* end of', t, re.S);            t = t[:m.start()] if m else t
    return re.findall(r"[a-zà-ÿ]+", t)[:cap]

def split(toks):
    idx = rng.permutation(len(toks)); h = len(toks)//2
    return [toks[i] for i in idx[:h]], [toks[i] for i in idx[h:]]

def model(toks, alpha, k=0.5):
    """first-order char transition log-probs with add-k smoothing over `alpha`."""
    nxt = defaultdict(Counter)
    for w in toks:
        seq = [S] + list(w) + [E]
        for a, b in zip(seq, seq[1:]): nxt[a][b] += 1
    A = list(alpha) + [E]; V = len(A)
    lp = {}
    states = list(alpha) + [S]
    for a in states:
        tot = sum(nxt[a].values())
        lp[a] = {b: math.log((nxt[a][b] + k) / (tot + k*V)) for b in A}
    return lp, set((a, b) for a in nxt for b in nxt[a])   # logp table, allowed-transition support

def xent(toks, lp):
    s = 0.0; n = 0
    for w in toks:
        seq = [S] + list(w) + [E]
        for a, b in zip(seq, seq[1:]):
            tab = lp.get(a)
            if tab is None or b not in tab: s += 20.0      # heavy penalty for impossible
            else: s += -tab[b]
            n += 1
    return s / max(1, n)

def pair(name, X, Y):
    Xtr, Xte = split(X); Ytr, Yte = split(Y)
    alpha = set(c for w in (Xtr+Xte+Ytr+Yte) for c in w)
    Xlp, Xsup = model(Xtr, alpha); Ylp, Ysup = model(Ytr, alpha)
    # foreign penalty in both directions, averaged
    pen_Y = xent(Yte, Xlp) - xent(Yte, Ylp)
    pen_X = xent(Xte, Ylp) - xent(Xte, Xlp)
    jac = len(Xsup & Ysup) / len(Xsup | Ysup)
    return name, jac, (pen_X + pen_Y)/2, xent(Yte, Ylp)

if __name__ == '__main__':
    A, B = voy_by_currier()
    print(f"Currier A tokens={len(A)}  B tokens={len(B)}\n")
    A1, A2 = split(A)
    EN = text_tokens('data/controls/english.txt'); LA = text_tokens('data/controls/latin.txt')

    rows = [
        pair("SAME-system floor  (Voynich A1 vs A2)", A1, A2),
        pair("** Voynich A vs B **", A, B),
        pair("DIFF-system ceiling (English vs Latin)", EN, LA),
    ]
    print(f"{'comparison':<42}{'support-Jaccard':>16}{'foreign-penalty':>18}{'(self xent)':>13}")
    for nm, jac, pen, self_x in rows:
        print(f"{nm:<42}{jac:16.3f}{pen:+18.3f}{self_x:13.2f}")

    floor = rows[0][2]; ab = rows[1][2]; ceil = rows[2][2]
    pos = (ab - floor) / (ceil - floor) if ceil != floor else float('nan')
    print(f"\nForeign-dialect penalty (bits/char): same-system floor {floor:+.3f} | "
          f"A-vs-B {ab:+.3f} | diff-system ceiling {ceil:+.3f}")
    print(f"A-vs-B sits {100*pos:.0f}% of the way from 'same system' to 'different systems'.")
    if pos < 0.33:
        print("=> A and B are ONE generative procedure with re-weighted settings. The system was")
        print("   TRANSMISSIBLE -- multiple hands ran the same machine. Points to a produced,")
        print("   for-an-audience artifact (workshop), NOT a lone private idiolect.")
    elif pos < 0.66:
        print("=> Intermediate: shared skeleton, substantial divergence. A taught-but-drifting")
        print("   procedure, or one evolving over time.")
    else:
        print("=> A and B behave like DIFFERENT systems. Independent invention / strong evolution.")
