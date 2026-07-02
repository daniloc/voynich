"""
Was the Voynich ENGINEERED to resist the era's cryptanalysis (frequency analysis),
i.e. built to be impenetrable to elite codebreakers? Test the falsifiable part.

The ~1420 attack is frequency analysis (al-Kindi; Italian chancery cryptanalysts).
Its food is REDUNDANCY: skewed symbol frequencies, positional predictability, repeats.
The era's COUNTERMEASURE was the homophonic cipher / nomenclator: give frequent
letters multiple cipher-symbols so frequencies FLATTEN toward uniform, add nulls,
avoid repeats -> the cipher stream moves TOWARD random, away from exploitable structure.

If the Voynich were engineered to resist elite attack it should look like the
COUNTERMEASURE (flat, high-entropy, low-repeat, toward random). If it's a mystique
artifact built to LOOK like language, it should look the opposite: maximally regular,
LOW conditional entropy, high repetition -- the most inviting possible target, which
only "wins" because the attack bottoms out in no content.

Compare on the dimensions a frequency-analyst exploits:
  RANDOM (max resistance) | HOMOPHONIC-Latin (real period countermeasure)
  | PLAINTEXT Latin | VOYNICH
Metric: order-2 REDUNDANCY = (Hmax - h2)/Hmax  (how much exploitable structure a
codebreaker can lean on; high = exposed/attackable, low = resistant).
"""
import json, re, math
from collections import Counter, defaultdict
import numpy as np
rng = np.random.default_rng(3)

def voy_tokens():
    C = json.load(open('data/corpus/corpus.json'))
    return [w for f in C['folios'].values() for ln in f
            for w in ln['words'] if '?' not in w and len(w) >= 1]

def latin_tokens(cap=40000):
    t = open('data/controls/latin.txt', encoding='utf-8', errors='ignore').read().lower()
    t = re.sub(r'[^a-z]+', ' ', t)
    t = t.replace('j','i').replace('v','u').replace('w','u').replace('k','c')
    return t.split()[:cap]

def homophonic(tokens):
    """Real period countermeasure: assign each letter k>=1 homophone symbols
    proportional to its frequency, so cipher symbols are ~equiprobable (flattens
    the frequency distribution frequency-analysis attacks)."""
    freq = Counter(c for w in tokens for c in w)
    tot = sum(freq.values()); N = 40           # target cipher alphabet ~40 symbols
    homo = {}; nxt = 0
    for c, f in freq.items():
        k = max(1, round(N * f / tot)); homo[c] = list(range(nxt, nxt + k)); nxt += k
    out = []
    for w in tokens:
        out.append([homo[c][rng.integers(len(homo[c]))] for c in w])
    return out  # list of symbol-lists

def random_tokens(tokens, alpha=30):
    lens = [len(w) for w in tokens]
    return [[int(rng.integers(alpha)) for _ in range(L)] for L in lens]

def to_symlists(tokens):  # char-tokens -> list of symbol-lists
    return [list(w) for w in tokens]

def h1(symlists):
    c = Counter(s for w in symlists for s in w); tot = sum(c.values())
    return -sum((n/tot)*math.log2(n/tot) for n in c.values()), len(c)

def h2(symlists):
    ctx = Counter(); joint = Counter()
    for w in symlists:
        seq = ['^'] + w + ['$']
        for a, b in zip(seq, seq[1:]): ctx[a]+=1; joint[(a,b)]+=1
    tot = sum(joint.values())
    return -sum((n/tot)*math.log2(n/ctx[a]) for (a,b),n in joint.items())

def adj_rep(symlists):
    flat = [tuple(w) for w in symlists]
    return np.mean([flat[i]==flat[i+1] for i in range(len(flat)-1)])

if __name__ == '__main__':
    voy = to_symlists(voy_tokens())
    lat = to_symlists(latin_tokens())
    homo = homophonic(latin_tokens())
    rnd  = random_tokens(latin_tokens())
    corp = [('RANDOM (max-resist)', rnd), ('HOMOPHONIC-Latin (period countermeasure)', homo),
            ('PLAINTEXT Latin', lat), ('VOYNICH', voy)]
    print(f"{'corpus':<42}{'alpha':>6}{'h1':>7}{'h2':>7}{'order-2 redundancy':>20}{'adj-repeat':>12}")
    rows = []
    for name, s in corp:
        H1, A = h1(s); H2 = h2(s); red = (math.log2(A) - H2)/math.log2(A); rep = adj_rep(s)
        rows.append((name, red, rep, H2))
        print(f"{name:<42}{A:6d}{H1:7.2f}{H2:7.2f}{red:20.2f}{rep:12.3%}")
    print("\nOrder-2 redundancy = exploitable structure a frequency-analyst leans on.")
    print("Engineered-to-RESIST => LOW redundancy (toward random). Built-to-LOOK-like-language => HIGH.")
    red = {r[0]: r[1] for r in rows}
    print(f"\nRedundancy ranking (most resistant -> most exposed):")
    for name, r, rep, H2 in sorted(rows, key=lambda x: x[1]):
        print(f"   {r:.2f}  {name}")
    v = red['VOYNICH']; p = red['PLAINTEXT Latin']; h = red['HOMOPHONIC-Latin (period countermeasure)']
    print(f"\nVoynich redundancy {v:.2f} vs plaintext {p:.2f} vs period countermeasure {h:.2f}.")
    if v >= p:
        print("=> Voynich is MORE exposed to frequency analysis than even plaintext, and FAR more")
        print("   than the period's actual anti-cryptanalysis technology. It was NOT engineered to")
        print("   resist elite codebreakers -- it maximizes the regularities they exploit. It")
        print("   'defeats' them only by having no content behind the regularity: a mirage, not a fortress.")
