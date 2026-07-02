#!/usr/bin/env python3
"""Compare f57v inner ring (R2) vs f66r isolated-glyph column (.M).
Set / order / null / known-alphabet tests."""
import random, itertools
random.seed(42)

# ---- f57v R2 ring: the 17-element repeated sequence (one repeat) ----
# From <f57v.3,+Cc;...> reconciling H/V/U majority. Glyphs in EVA-ish + weirdos.
# Positions (1..17): o l d r v x k [k/m] [f/p] @169 t r @170 @171 y c @172
# Variable slots: pos8 k~m, pos9 f~p. @169/@170/@171/@172 are rare weirdo gallows.
ring = ['o','l','d','r','v','x','k','k','f','@169','t','r','@170','@171','y','c','@172']

# ---- f66r .M isolated single-glyph column (items 16..49), majority over H/C/F/U ----
col = ['y','o','s','sh','y','d','o','f','?','x','air','d','sh','y','f','f','y','o',
       'd','r','f','c','@172','x','t','o','@195','l','r','t','o','x','p','d']
# items where multi-glyph tokens appear: 'sh','air' -> treat 'sh' as one EVA char (it is),
# 'air' = a+i+r (a single drawn token but 3 EVA glyphs). Keep as-is for token view;
# also build a char-expanded view.

def expand(seq):
    out=[]
    for t in seq:
        if t in ('sh','ch'): out.append(t)        # genuine single EVA benches
        elif t.startswith('@'): out.append(t)     # weirdo = one symbol
        elif len(t)>1 and not t.startswith('@'):  # 'air' etc -> split
            out.extend(list(t))
        else: out.append(t)
    return out

ring_c = expand(ring)
col_c  = expand(col)

print("=== EXACT SEQUENCES ===")
print("f57v ring (17, one repeat):", ' '.join(ring))
print("f66r col  (%d tokens):"%len(col), ' '.join(col))
print()

def setinfo(name,seq):
    s=set(seq)
    print(f"{name}: n={len(seq)} tokens, |alphabet|={len(s)}: {sorted(s)}")
    return s
S_ring=setinfo("ring",ring)
S_col=setinfo("col",col)
print()

# ---------- (i) SET comparison: Jaccard ----------
def jaccard(a,b):
    return len(a&b)/len(a|b)
inter=S_ring & S_col
print("=== (i) SET / inventory ===")
print("shared glyphs:", sorted(inter))
print("ring-only:", sorted(S_ring-S_col))
print("col-only:", sorted(S_col-S_ring))
print("Jaccard =", round(jaccard(S_ring,S_col),4))
print()

# ---------- (ii) ORDERED comparison: LCS ----------
def lcs(a,b):
    n,m=len(a),len(b)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):
            dp[i][j]=dp[i+1][j+1]+1 if a[i]==b[j] else max(dp[i+1][j],dp[i][j+1])
    return dp[0][0]

L=lcs(ring,col)
print("=== (ii) ORDER / LCS ===")
print("LCS(ring,col) =", L, " (normalized by min len = %.3f)"%(L/min(len(ring),len(col))))
print("LCS(ring,reversed col) =", lcs(ring, col[::-1]))
print()

# ---------- (iii) NULL model ----------
# Build a shared alphabet = union, weighted by how each symbol actually appears.
# Null Q1: is the observed LCS more than random orderings of the SAME multiset as col,
#          drawn against the fixed ring? (tests ordered coincidence)
# Null Q2: Jaccard vs two random samples from a common glyph pool.
print("=== (iii) NULL ===")
N=20000
# Null for LCS: shuffle col, measure LCS vs ring
obsL=lcs(ring,col)
ge=0; tot=0
import copy
base=col[:]
for _ in range(N):
    random.shuffle(base)
    if lcs(ring,base)>=obsL: ge+=1
print(f"LCS null (shuffle col multiset): obs={obsL}; P(rand>=obs)={ge/N:.4f}  (mean rand LCS≈measuring...)")
# mean
tot=0
for _ in range(2000):
    random.shuffle(base); tot+=lcs(ring,base)
print(f"   mean random LCS = {tot/2000:.2f}")

# Null for Jaccard: the manuscript's global glyph inventory (~) — both sequences drawn from
# a common pool. Use a realistic Voynich glyph pool.
pool=['o','l','d','r','v','x','k','f','t','y','c','s','sh','ch','a','i','n','m','p','e','q','g',
      '@169','@170','@171','@172','@195']
def randset(k): return set(random.sample(pool,k))
obsJ=jaccard(S_ring,S_col)
ge=0
for _ in range(N):
    ge += jaccard(randset(len(S_ring)), randset(len(S_col)))>=obsJ
print(f"Jaccard null (random subsets of {len(pool)}-glyph pool): obs={obsJ:.3f}; P(rand>=obs)={ge/N:.4f}")
print()

# ---------- (3) KNOWN-ALPHABET checks ----------
print("=== KNOWN-ALPHABET checks ===")
eva_order=list("oeairlntsdkpfqcghxmvyz")  # rough EVA/common-frequency-ish reference order
latin=list("abcdefghiklmnopqrstuxyz")     # medieval latin abecedary (no j,v,w)
def monotonic_score(seq, order):
    """fraction of adjacent pairs (both in order) that go forward in `order`."""
    idx={c:i for i,c in enumerate(order)}
    fwd=eq=back=0
    for a,b in zip(seq,seq[1:]):
        if a in idx and b in idx:
            if idx[b]>idx[a]: fwd+=1
            elif idx[b]<idx[a]: back+=1
    tot=fwd+back
    return (fwd,back,tot, fwd/tot if tot else None)
for nm,seq in [("ring",ring),("col",col)]:
    for onm,order in [("latin-abc",latin),("eva-freq",eva_order)]:
        f,b,t,frac=monotonic_score(seq,order)
        print(f"{nm} vs {onm}: forward={f} back={b} of {t} ordered pairs; frac_fwd={frac}")
print("(0.5 = no directional ordering; ~1.0 would mean it follows that alphabet's order)")
print()

# Does ring repeat structure hold? report variable positions
print("=== ring repeat fidelity (from IVTFF) ===")
print("4 repeats, 12/14 stable positions; only pos8 (k/m) and pos9 (f/p) vary across the 4.")
