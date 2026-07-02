"""
R1 — THE CRYPTANALYTIC ATTACK WITH HELD-OUT GENERALIZATION.
The falsifier of the asemic verdict. This is the standard attack that produces
every fake Voynich "solution" (hill-climb a substitution key to maximize
resemblance to a real language) -- but with the two disciplines those solutions
omit: a POSITIVE CONTROL and a HELD-OUT test.

Method (identical for every corpus):
  1. take the corpus as a stream of characters (alphabet = its own symbols).
  2. split into TRAIN and TEST halves.
  3. hill-climb a bijective char->Latin-letter key on TRAIN ONLY, maximizing the
     decoded text's char-trigram log-likelihood under a Latin language model.
  4. freeze the key; score TEST under it.

Diagnostics:
  - CEILING  = real Latin scored under the Latin LM (what genuine plaintext gets).
  - FLOOR    = random-key decode (what nonsense gets).
  - POSITIVE CONTROL = real Latin enciphered with a known random substitution. The
    attack MUST recover it: TRAIN high AND TEST high (generalizes). Proves the
    method works and that a real cipher would be caught.
  - VOYNICH  : if it's a substitution cipher, TEST ~ TRAIN ~ ceiling. If TEST
    collapses toward FLOOR (overfit, no transferable key), there is no key.
  - SURROGATE (content-free) : the meaningless benchmark Voynich should match if
    it carries no message.

Entropy note (why the ceiling is informative): substitution is a 1:1 relabel, so it
preserves conditional entropy. Voynich h2~2.1 << Latin h2~3.3, so a substitution
decode of Voynich CANNOT reach Latin trigram scores. The gap is a measured fact,
not a search failure.
"""
import re, json, math, sys, os
from collections import Counter, defaultdict
import numpy as np

rng = np.random.default_rng(20260620)

# ---------- corpora as character streams (word breaks kept as space) ----------
def voy_chars():
    C = json.load(open('data/corpus/corpus.json'))
    toks = [w for f in C['folios'].values() for ln in f
            for w in ln['words'] if '?' not in w and len(w) >= 1]
    return ' '.join(toks)

def latin_chars(cap=400_000):
    t = open('data/controls/latin.txt', encoding='utf-8', errors='ignore').read().lower()
    t = re.sub(r'[^a-z]+', ' ', t)            # medieval-ish: collapse to letters+space
    t = t.replace('j', 'i').replace('v', 'u').replace('w', 'u').replace('k', 'c')
    t = re.sub(r'\s+', ' ', t).strip()
    return t[:cap]

def surrogate_chars(voy_toks_text):
    """first-order glyph Markov over Voynich tokens -> meaningless token stream."""
    toks = voy_toks_text.split()
    trans = defaultdict(Counter); START, END = '\x02', '\x03'
    for w in toks:
        seq = [START] + list(w) + [END]
        for a, b in zip(seq, seq[1:]): trans[a][b] += 1
    tbl = {a: (list(c), np.array(list(c.values()), float)) for a, c in trans.items()}
    for a in tbl: tbl[a] = (tbl[a][0], tbl[a][1] / tbl[a][1].sum())
    out = []
    for _ in range(len(toks)):
        cur = START; w = []
        for _ in range(40):
            outs, ws = tbl[cur]; nx = outs[rng.choice(len(outs), p=ws)]
            if nx == END: break
            w.append(nx); cur = nx
        if w: out.append(''.join(w))
    return ' '.join(out)

# ---------- Latin trigram language model ----------
def build_lm(text, alpha):
    aset = set(alpha) | {' '}
    text = ''.join(c if c in aset else ' ' for c in text)
    tri = Counter(); bi = Counter()
    for i in range(len(text) - 2):
        a, b, c = text[i], text[i+1], text[i+2]
        tri[(a, b, c)] += 1; bi[(a, b)] += 1
    V = len(aset)
    def logp(a, b, c):                                   # add-1 smoothed
        return math.log((tri[(a, b, c)] + 1) / (bi[(a, b)] + V))
    return logp, aset

def tri_counts(text):
    """precompute char-trigram counts ONCE; scoring a key is then O(distinct trigrams)."""
    c = Counter()
    for i in range(len(text) - 2):
        c[(text[i], text[i+1], text[i+2])] += 1
    return c, sum(c.values())

def score_counts(counts, total, key, logp):
    m = lambda ch: ' ' if ch == ' ' else key.get(ch, ' ')
    s = 0.0
    for (a, b, c), n in counts.items():
        s += n * logp(m(a), m(b), m(c))
    return s / max(1, total)

def score_plain(text, logp):
    cnt, tot = tri_counts(text)
    return sum(n * logp(a, b, c) for (a, b, c), n in cnt.items()) / max(1, tot)

# ---------- substitution hill-climb (on precomputed train trigram counts) ----------
def hillclimb(train_counts, train_total, src_alpha, dst_alpha, logp, seed_map=None,
              steps=2500, restarts=4):
    src = list(src_alpha); dst = list(dst_alpha)
    n = min(len(src), len(dst)); src = src[:n]
    best_key, best_s = None, -1e9
    for r in range(restarts):
        perm = dst[:n].copy(); rng.shuffle(perm)
        key = dict(zip(src, perm))
        if seed_map:
            for k, v in seed_map.items():
                if k in key and v in key.values():
                    for kk in list(key):
                        if key[kk] == v: key[kk] = key[k]; break
                    key[k] = v
        cur_s = score_counts(train_counts, train_total, key, logp)
        for _ in range(steps):
            a, b = src[rng.integers(n)], src[rng.integers(n)]
            if a == b: continue
            key[a], key[b] = key[b], key[a]
            s = score_counts(train_counts, train_total, key, logp)
            if s >= cur_s: cur_s = s
            else: key[a], key[b] = key[b], key[a]
        if cur_s > best_s: best_s, best_key = cur_s, dict(key)
    return best_key

def attack(name, cipher_text, src_alpha, dst_alpha, logp, seed_map=None):
    half = len(cipher_text) // 2
    train, test = cipher_text[:half], cipher_text[half:]
    tc, tt = tri_counts(train); ec, et = tri_counts(test)
    key = hillclimb(tc, tt, src_alpha, dst_alpha, logp, seed_map)
    return name, score_counts(tc, tt, key, logp), score_counts(ec, et, key, logp)

if __name__ == '__main__':
    L = latin_chars()
    cut = int(len(L) * 0.55)
    Ltrain_lm, Lheld = L[:cut], L[cut:]               # LM on one span, evaluate on a disjoint span
    Lalpha = [c for c, _ in Counter(Ltrain_lm.replace(' ', '')).most_common()]
    logp, _ = build_lm(Ltrain_lm, Lalpha)

    CEIL = score_plain(Lheld, logp)                   # genuine held-out Latin
    randkey = dict(zip(Lalpha, list(rng.permutation(Lalpha))))
    FLOOR = score_plain(''.join(randkey.get(c, ' ') if c != ' ' else ' ' for c in Lheld), logp)

    V = voy_chars()
    Valpha = [c for c, _ in Counter(V.replace(' ', '')).most_common()]
    SUR = surrogate_chars(V)
    Salpha = [c for c, _ in Counter(SUR.replace(' ', '')).most_common()]

    # positive control: encipher the held-out Latin span (disjoint from the LM) with a known key
    encL = ''.join(randkey.get(c, ' ') if c != ' ' else ' ' for c in Lheld)
    print(f"(corpus char-lengths: Latin held={len(Lheld)}, Voynich={len(V)}, surrogate={len(SUR)})")

    print("char-trigram log-likelihood per char under a Latin LM (higher = more Latin-like)\n")
    print(f"  CEILING  real Latin            : {CEIL:7.3f}")
    print(f"  FLOOR    random-key nonsense    : {FLOOR:7.3f}")
    print(f"  (the band any decode lives in: {FLOOR:.2f} .. {CEIL:.2f})\n")

    print(f"{'corpus':<26}{'TRAIN':>9}{'TEST(held-out)':>16}{'generalizes?':>14}")
    rows = []
    rows.append(attack('POSITIVE CTRL (enc Latin)', encL, Lalpha, Lalpha, logp))
    rows.append(attack('VOYNICH', V, Valpha, Lalpha, logp))
    rows.append(attack('VOYNICH +stroke-seed(y=s)', V, Valpha, Lalpha, logp, seed_map={'y': 's'}))
    rows.append(attack('SURROGATE (content-free)', SUR, Salpha, Lalpha, logp))
    for name, tr, te in rows:
        gen = 'YES' if te > FLOOR + 0.6 * (CEIL - FLOOR) else ('partial' if te > FLOOR + 0.3*(CEIL-FLOOR) else 'NO')
        print(f"{name:<26}{tr:9.3f}{te:16.3f}{gen:>14}")

    print("\n--- verdict (compare to the POSITIVE CONTROL, not to absolute scores) ---")
    pc = rows[0][2]; vy = max(rows[1][2], rows[2][2]); su = rows[3][2]
    print(f"  positive control (real cipher) held-out : {pc:.2f}   (ceiling {CEIL:.2f}, floor {FLOOR:.2f})")
    print(f"  VOYNICH best held-out                   : {vy:.2f}")
    print(f"  content-free SURROGATE held-out         : {su:.2f}")
    pc_frac = (pc - FLOOR) / (CEIL - FLOOR)
    vy_frac = (vy - FLOOR) / (CEIL - FLOOR)
    su_frac = (su - FLOOR) / (CEIL - FLOOR)
    print(f"  gap closed -> positive control {100*pc_frac:.0f}% | Voynich {100*vy_frac:.0f}% | surrogate {100*su_frac:.0f}%")
    print(f"  Voynich's lead over the meaningless surrogate: {vy - su:+.3f} bits/char")
    # The discriminator is Voynich-vs-its-matched-content-free-surrogate, NOT the absolute
    # gap fraction: any structured-but-alien text overfits an alphabet permutation to Latin
    # letter frequencies and closes part of the gap with zero meaning (the surrogate proves it).
    if pc_frac < 0.5:
        print("  => POSITIVE CONTROL FAILED to recover -> test underpowered; raise steps/restarts before trusting it.")
    elif vy - su < 0.05:
        print("  => Voynich is cryptanalytically INDISTINGUISHABLE from meaningless noise with the same")
        print(f"     letter statistics (lead {vy-su:+.3f}), and falls {pc-vy:.2f} bits/char short of a real")
        print("     cipher the method DID crack. No substitution key generalizes. R1 NULL.")
        print("     The asemic verdict survives its falsifier.")
    elif vy >= pc - 0.15:
        print("  => Voynich reaches the real-cipher band AND beats its surrogate. OVERTURNED. Escalate.")
    else:
        print("  => Voynich beats its surrogate but trails a real cipher: partial signal; characterize.")
