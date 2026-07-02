#!/usr/bin/env python3
"""
M4 real-herbal control. Supplies the missing botanical control for the
"plants recombine freely" claim. Tags ~25 real medieval-herbal species by
TRUE morphology using the SAME 4-slot scheme as plant_tags.json, coarsens
BOTH sets into ONE shared per-slot vocabulary (so neither set is tagged more
richly than the other), and computes identical metrics:
  - distinct fillers / slot
  - per-slot entropy (bits)
  - inter-slot mutual information  I(infl; root), I(infl; leaf), I(root; leaf)
  - % unique whole-plant combos  vs  free-recombination Monte-Carlo ceiling

Pure-python; no numpy.
"""
import json, math, random
from itertools import combinations
from collections import Counter

random.seed(42)

# ---------------------------------------------------------------------------
# 1. Load Voynich M4 tags
# ---------------------------------------------------------------------------
V = json.load(open("data/grounding/plant_tags.json"))["plants"]

# ---------------------------------------------------------------------------
# 2. COARSENING MAPS  -- the heart of fairness.
#    M4's raw fillers are fine-grained (taproot vs taproot_red vs
#    taproot_red_forked). Real plants would explode into even more uniques if
#    tagged at that grain. So we collapse BOTH sets into a single shared,
#    deliberately COARSE morphological vocabulary per slot. Colour / size
#    qualifiers are dropped for both. This is the only honest comparison.
# ---------------------------------------------------------------------------
ROOT = {  # -> {taproot, fibrous, rhizome, bulb}
    "taproot":"taproot","taproot_branched":"taproot","taproot_red":"taproot",
    "taproot_red_forked":"taproot","taproot_red_heart":"taproot",
    "fibrous_thin":"fibrous","fibrous_dense":"fibrous","fibrous_dense_red":"fibrous",
    "fibrous_fingered":"fibrous",
    "rhizome_horizontal":"rhizome","rhizome_segmented_coiled":"rhizome",
    "bulb":"bulb",
}
STEM = {  # -> {single, branching, multi}  (architecture only)
    "single_erect":"single",
    "branching":"branching",
    "multi_radiating":"multi",
}
LEAF = {  # -> {simple_entire, lobed, divided, linear}
    "lanceolate_drooping":"simple_entire","lanceolate":"simple_entire",
    "ovate_paired":"simple_entire","small_oval":"simple_entire",
    "round_cordate":"simple_entire","hastate_single_large":"simple_entire",
    "basal_brown_mass":"simple_entire","none_small":"simple_entire",
    "palmate_lobed":"lobed",
    "pinnate_fern":"divided",
    "sword_whorl":"linear",
}
INFL = {  # -> {none, umbel, spike, head, cluster, berry}
    "none_leafy":"none","none":"none",
    "umbel":"umbel",
    "spike_red":"spike",
    "composite_blue_head":"head","composite_head":"head",
    "composite_blue_cluster":"cluster","composite_white_cluster":"cluster",
    "berry_pair":"berry","berry_cluster":"berry",
}

def coarsen_voynich(p):
    return (ROOT[p["root"]], STEM[p["stem"]], LEAF[p["leaf"]], INFL[p["infl"]])

vset = [coarsen_voynich(p) for p in V]

# ---------------------------------------------------------------------------
# 3. REAL-HERBAL CONTROL SET
#    ~25 species from the Dioscorides / Tractatus de herbis / Pseudo-Apuleius
#    repertoire. Tagged by TRUE botany, then forced through the SAME coarse
#    vocab above (identical filler categories). Ground truth = real morphology.
#    Vocab per slot:
#      root: taproot | fibrous | rhizome | bulb
#      stem: single | branching | multi
#      leaf: simple_entire | lobed | divided | linear
#      infl: none | umbel | spike | head | cluster | berry
# ---------------------------------------------------------------------------
real = [
 # name,                       root,      stem,       leaf,           infl
 ("mandrake",                 "taproot", "single",   "simple_entire","cluster"),
 ("plantain",                 "fibrous", "single",   "simple_entire","spike"),
 ("fennel (umbellifer)",      "taproot", "branching","divided",      "umbel"),
 ("dill (umbellifer)",        "taproot", "branching","divided",      "umbel"),
 ("hemlock (umbellifer)",     "taproot", "branching","divided",      "umbel"),
 ("wild carrot (umbellifer)", "taproot", "branching","divided",      "umbel"),
 ("iris",                     "rhizome", "single",   "linear",       "head"),
 ("rose",                     "taproot", "branching","divided",      "head"),
 ("violet (viola)",           "rhizome", "single",   "simple_entire","head"),
 ("nettle",                   "rhizome", "single",   "simple_entire","spike"),
 ("grapevine (vine)",         "taproot", "branching","lobed",        "cluster"),
 ("male fern",                "rhizome", "single",   "divided",      "none"),
 ("mallow",                   "taproot", "branching","lobed",        "head"),
 ("wormwood (artemisia)",     "taproot", "branching","divided",      "cluster"),
 ("garlic",                   "bulb",    "single",   "linear",       "head"),
 ("onion",                    "bulb",    "single",   "linear",       "head"),
 ("lily",                     "bulb",    "single",   "linear",       "head"),
 ("sage",                     "taproot", "branching","simple_entire","spike"),
 ("mint",                     "rhizome", "branching","simple_entire","spike"),
 ("poppy",                    "taproot", "single",   "lobed",        "head"),
 ("celandine",                "taproot", "branching","lobed",        "umbel"),
 ("vervain",                  "taproot", "branching","lobed",        "spike"),
 ("comfrey",                  "taproot", "single",   "simple_entire","cluster"),
 ("burdock",                  "taproot", "branching","simple_entire","head"),
 ("centaury",                 "taproot", "branching","simple_entire","cluster"),
 ("saffron crocus",           "bulb",    "single",   "linear",       "head"),
 ("asphodel",                 "rhizome", "single",   "linear",       "spike"),
]
rset = [t[1:] for t in real]

# ---------------------------------------------------------------------------
# 4. METRICS
# ---------------------------------------------------------------------------
SLOTS = ["root","stem","leaf","infl"]

def entropy(vals):
    n=len(vals); c=Counter(vals)
    return -sum((k/n)*math.log2(k/n) for k in c.values())

def mutual_info(xs, ys):
    n=len(xs); cx=Counter(xs); cy=Counter(ys); cxy=Counter(zip(xs,ys))
    mi=0.0
    for (x,y),nxy in cxy.items():
        pxy=nxy/n; px=cx[x]/n; py=cy[y]/n
        mi+=pxy*math.log2(pxy/(px*py))
    return mi

def free_ceiling(dataset, trials=20000):
    """Monte-Carlo: if slots were drawn INDEPENDENTLY from their own marginals
    (free recombination), what % of n draws would be unique whole-plants?
    This is the null M4 invoked. Compare observed %unique to this ceiling."""
    n=len(dataset)
    cols=[[row[i] for row in dataset] for i in range(4)]
    fracs=[]
    for _ in range(trials):
        combos=[tuple(random.choice(cols[i]) for i in range(4)) for _ in range(n)]
        c=Counter(combos)
        uniq=sum(1 for v in c.values() if v==1)
        fracs.append(uniq/n)
    fracs.sort()
    mean=sum(fracs)/len(fracs)
    lo=fracs[int(0.025*len(fracs))]; hi=fracs[int(0.975*len(fracs))]
    return mean, lo, hi

def report(name, ds):
    n=len(ds)
    cols={s:[row[i] for row in ds] for i,s in enumerate(SLOTS)}
    print(f"\n===== {name}  (n={n}) =====")
    print(" distinct fillers / slot:")
    for s in SLOTS:
        print(f"   {s:5s}: {len(set(cols[s])):2d} distinct   H={entropy(cols[s]):.2f} bits")
    print(" inter-slot mutual information (bits):")
    pairs=[("infl","root"),("infl","leaf"),("infl","stem"),("root","leaf")]
    mis={}
    for a,b in pairs:
        mi=mutual_info(cols[a],cols[b])
        # normalise by min entropy -> fraction of one slot's entropy explained
        denom=min(entropy(cols[a]),entropy(cols[b]))
        norm=mi/denom if denom>0 else 0
        mis[(a,b)]=mi
        print(f"   I({a:4s};{b:4s}) = {mi:.3f}   (norm {norm:.2f})")
    avg_key_mi=sum(mis[k] for k in [("infl","root"),("infl","leaf")])/2
    combos=Counter(ds)
    uniq=sum(1 for v in combos.values() if v==1)
    pct=uniq/n
    print(f" whole-plant combos: {len(combos)} distinct of {n};  %unique(seen once)={pct*100:.0f}%")
    mean,lo,hi=free_ceiling(ds)
    print(f" free-recombination ceiling %unique: {mean*100:.0f}%  (95% MC band {lo*100:.0f}-{hi*100:.0f}%)")
    gap=mean-pct
    print(f" gap (ceiling - observed): {gap*100:+.0f} pts   "
          f"[{'AT ceiling' if abs(gap)<=(hi-mean) else 'BELOW ceiling -> syndromes'}]")
    return {"n":n,"pct_unique":pct,"ceiling":mean,"ceil_band":(lo,hi),
            "avg_infl_MI":avg_key_mi,"mis":mis,
            "distinct":{s:len(set(cols[s])) for s in SLOTS},
            "H":{s:entropy(cols[s]) for s in SLOTS}}

print("Shared coarse vocabulary (identical for both sets):")
print("  root: taproot|fibrous|rhizome|bulb")
print("  stem: single|branching|multi")
print("  leaf: simple_entire|lobed|divided|linear")
print("  infl: none|umbel|spike|head|cluster|berry")

rv=report("VOYNICH (M4, coarsened)", vset)
rr=report("REAL HERBAL CONTROL", rset)

print("\n\n############## DISCRIMINATOR SUMMARY ##############")
print(f"{'metric':38s}{'VOYNICH':>12s}{'REAL':>12s}")
print(f"{'avg I(infl;root)+I(infl;leaf) /2':38s}{rv['avg_infl_MI']:>12.3f}{rr['avg_infl_MI']:>12.3f}")
print(f"{'%unique whole-plant':38s}{rv['pct_unique']*100:>11.0f}%{rr['pct_unique']*100:>11.0f}%")
print(f"{'free-recomb ceiling %unique':38s}{rv['ceiling']*100:>11.0f}%{rr['ceiling']*100:>11.0f}%")
print(f"{'gap below ceiling (pts)':38s}{(rv['ceiling']-rv['pct_unique'])*100:>+11.0f} {(rr['ceiling']-rr['pct_unique'])*100:>+11.0f} ")

# sanity: also report on RAW (fine-grained) Voynich, to show coarsening was needed
raw=[(p["root"],p["stem"],p["leaf"],p["infl"]) for p in V]
rawc=Counter(raw)
print(f"\n(For reference: RAW fine-grained Voynich %unique = "
      f"{sum(1 for v in rawc.values() if v==1)/len(raw)*100:.0f}% "
      f"-- this is the 87% M4 reported; coarsening makes the comparison fair.)")

# ---------------------------------------------------------------------------
# 5. PERMUTATION TEST: is each set's inter-slot MI above its OWN shuffle null?
#    Shuffle each slot column independently -> destroys correlation, keeps
#    marginals. Total inter-slot MI = sum over all 6 slot pairs.
# ---------------------------------------------------------------------------
def total_interslot_mi(ds):
    cols=[[row[i] for row in ds] for i in range(4)]
    return sum(mutual_info(cols[a],cols[b]) for a,b in combinations(range(4),2))

def mi_perm_test(name, ds, trials=10000):
    obs=total_interslot_mi(ds)
    cols=[[row[i] for row in ds] for i in range(4)]
    null=[]
    for _ in range(trials):
        sh=[c[:] for c in cols]
        for c in sh: random.shuffle(c)
        ds2=list(zip(*sh))
        null.append(sum(mutual_info([r[a] for r in ds2],[r[b] for r in ds2])
                        for a,b in combinations(range(4),2)))
    null.sort()
    mean=sum(null)/len(null)
    p=(sum(1 for x in null if x>=obs)+1)/(trials+1)
    print(f"  {name:24s} total inter-slot MI = {obs:.3f}  | shuffle-null mean {mean:.3f}  p={p:.3f}")
    return obs,mean,p

print("\n############## MI PERMUTATION TEST (vs own shuffle null) ##############")
print(" If real botany has syndromes, real MI should beat its shuffle null; ")
print(" Voynich (free recombination) should NOT.")
mi_perm_test("VOYNICH", vset)
mi_perm_test("REAL HERBAL", rset)

# ---------------------------------------------------------------------------
# 6. ROBUSTNESS: subsample real set to n=15 (match Voynich) many times,
#    rerun the permutation test each time. Is the syndrome signal robust to
#    small-n, or an artifact of the larger real-n?
# ---------------------------------------------------------------------------
print("\n############## ROBUSTNESS: real set down-sampled to n=15 ##############")
sig=0; ps=[]
for _ in range(200):
    sub=random.sample(rset,15)
    obs=total_interslot_mi(sub)
    cols=[[row[i] for row in sub] for i in range(4)]
    null=[]
    for _ in range(2000):
        sh=[c[:] for c in cols]
        for c in sh: random.shuffle(c)
        ds2=list(zip(*sh))
        null.append(sum(mutual_info([r[a] for r in ds2],[r[b] for r in ds2])
                        for a,b in combinations(range(4),2)))
    p=(sum(1 for x in null if x>=obs)+1)/(len(null)+1)
    ps.append(p)
    if p<0.05: sig+=1
ps.sort()
print(f" real n=15 subsamples significant (p<0.05): {sig}/200 = {sig/2}%   median p={ps[100]:.3f}")
print(f" (Voynich at full n=15 was p=0.22, NOT significant)")

# ---------------------------------------------------------------------------
# 7. TAGGING-SUBJECTIVITY STRESS TEST: randomly corrupt 3 of 27 real tags
#    (simulate my mis-tagging) and see if the syndrome signal survives.
# ---------------------------------------------------------------------------
print("\n############## STRESS: corrupt 3 random real tags, x100 ##############")
vocab={0:["taproot","fibrous","rhizome","bulb"],1:["single","branching","multi"],
       2:["simple_entire","lobed","divided","linear"],
       3:["none","umbel","spike","head","cluster","berry"]}
still_sig=0
for _ in range(100):
    cor=[list(r) for r in rset]
    for idx in random.sample(range(len(cor)),3):
        slot=random.randint(0,3)
        cor[idx][slot]=random.choice(vocab[slot])
    cor=[tuple(r) for r in cor]
    obs=total_interslot_mi(cor)
    cols=[[row[i] for row in cor] for i in range(4)]
    null=[]
    for _ in range(2000):
        sh=[c[:] for c in cols]
        for c in sh: random.shuffle(c)
        ds2=list(zip(*sh))
        null.append(sum(mutual_info([r[a] for r in ds2],[r[b] for r in ds2])
                        for a,b in combinations(range(4),2)))
    p=(sum(1 for x in null if x>=obs)+1)/(len(null)+1)
    if p<0.05: still_sig+=1
print(f" real-with-3-corrupted-tags still significant: {still_sig}/100 = {still_sig}%")
