"""
THE CONTROL THE GENERATOR ARGUMENT NEVER RAN.

Claim under test:
  "A first-order glyph automaton trained on the Voynich lexicon regenerates 61%
   of real word-types; the identical machine on English regenerates only 14%.
   That gap proves Voynich words are automaton output, not language."

Dispute:
  The gap may measure ORTHOGRAPHIC/MORPHOLOGICAL REGULARITY, for which English
  (deep orthography, large effective alphabet, irregular stems) is a maximally
  unfair baseline. A first-order char automaton should ALSO score high on a real,
  meaningful, phonotactically-simple / templatic language. If it does, 61% says
  nothing about meaning.

This script builds ONE automaton and runs it identically on every corpus:
  - train a first-order character Markov over WORD-TYPES (START -> c -> ... -> END)
  - report parameter count, mean word length, alphabet size
  - SELF-REGENERATION: sample many words, % that are attested types (by token & by
    distinct invention) -- the headline number
  - NULL BASELINE: a 0th-order automaton (glyph unigram freqs + length model, NO
    transitions). How much of the score is just "short words from a small alphabet
    densely pack a small space"? The automaton only earns the gap ABOVE this.
"""
import re, json, sys, math
import numpy as np
from collections import Counter

rng = np.random.default_rng(7)
START, END = '\x02', '\x03'

def voynich_types():
    C = json.load(open('data/corpus/corpus.json'))
    toks = [w for f in C['folios'].values() for ln in f
            for w in ln['words'] if '?' not in w and len(w) >= 1]
    return toks

def text_types(path, max_tokens=300_000):
    txt = open(path, encoding='utf-8', errors='ignore').read().lower()
    # strip Gutenberg header/footer if present
    m = re.search(r'\*\*\* start of.*?\*\*\*', txt, re.S)
    if m: txt = txt[m.end():]
    m = re.search(r'\*\*\* end of', txt, re.S)
    if m: txt = txt[:m.start()]
    toks = re.findall(r"[a-zÀ-ɏʻ'']+", txt)  # letters incl. accented + Hawaiian okina
    return toks[:max_tokens]

CAPTYPES = int(__import__('os').environ.get('CAPTYPES', '0'))

class Automaton:
    def __init__(self, types, order=1):
        self.order = order
        self.types = types
        attested = list(set(types))        # train on TYPES (the lexicon shape)
        if CAPTYPES and len(attested) > CAPTYPES:   # match lexicon size across langs
            idx = rng.choice(len(attested), size=CAPTYPES, replace=False)
            attested = [attested[i] for i in idx]
        self.attested = set(attested)
        # first-order transition counts over characters with START/END
        trans = {}
        for w in self.attested:
            seq = (START,) + tuple(w) + (END,)
            for a, b in zip(seq, seq[1:]):
                trans.setdefault(a, Counter())[b] += 1
        self.trans = trans
        self.alpha = sorted(set(ch for w in self.attested for ch in w))
        # 0th-order null: char unigram (over types) + empirical length distribution
        self.uni = Counter(ch for w in self.attested for ch in w)
        self.lendist = Counter(len(w) for w in self.attested)
        # parameter count: nonzero transition entries (the model's real size)
        self.nparams = sum(len(v) for v in trans.values())

    def _sample_markov(self, cap=40):
        out = []
        cur = START
        while True:
            row = self.trans.get(cur)
            if not row: break
            nxt = list(row); w = np.array([row[c] for c in nxt], float); w /= w.sum()
            cur = nxt[rng.choice(len(nxt), p=w)]
            if cur == END or len(out) >= cap: break
            out.append(cur)
        return ''.join(out)

    def _sample_null(self, cap=40):
        lens = list(self.lendist); lw = np.array([self.lendist[l] for l in lens], float); lw /= lw.sum()
        L = lens[rng.choice(len(lens), p=lw)]
        chs = list(self.uni); cw = np.array([self.uni[c] for c in chs], float); cw /= cw.sum()
        return ''.join(chs[i] for i in rng.choice(len(chs), size=min(L, cap), p=cw))

    def regen(self, n=40_000, mode='markov'):
        samp = self._sample_markov if mode == 'markov' else self._sample_null
        gen = [samp() for _ in range(n)]
        gen = [g for g in gen if g]
        by_token = np.mean([g in self.attested for g in gen])
        distinct = set(gen)
        by_type = np.mean([g in self.attested for g in distinct])
        return by_token, by_type, len(distinct)

def run(name, types):
    A = Automaton(types)
    mk_tok, mk_typ, ndist = A.regen(mode='markov')
    nl_tok, _, _ = A.regen(mode='null')
    meanlen = np.mean([len(w) for w in A.attested])
    print(f"{name:14s} | types={len(A.attested):6d} | alpha={len(A.alpha):2d} | "
          f"wlen={meanlen:4.1f} | params={A.nparams:4d} | "
          f"REGEN(tok)={100*mk_tok:5.1f}%  null={100*nl_tok:4.1f}%  lift=x{mk_tok/max(nl_tok,1e-9):.2f}")
    return dict(name=name, types=len(A.attested), alpha=len(A.alpha), wlen=meanlen,
                params=A.nparams, regen_tok=mk_tok, regen_typ=mk_typ, null_tok=nl_tok)

if __name__ == '__main__':
    print("FIRST-ORDER CHARACTER AUTOMATON — self-regeneration across corpora")
    print("(higher REGEN = automaton more easily reproduces the real lexicon)\n")
    rows = [run('VOYNICH', voynich_types())]
    targets = [a.split(':') for a in sys.argv[1:]]  # name:path
    for name, path in targets:
        rows.append(run(name, text_types(path)))
    json.dump(rows, open('data/intermediate/_automaton_control.json', 'w'), indent=2)
