"""
Z10 — zodiac label->referent grounding test, 10% slice (2026-07-01).
The 2026-07-01 transcript audit found K13 never touched the zodiac (the label-richest
section, explicitly inventoried as "has the power" then bypassed) and that K13's power
controls only ever proved detectability of a PREFIX-carried signal, never a CORE-carried
one. This runs the K13 within-folio shuffle-null machinery on 4 zodiac rings
(f70v2 Pisces, f71r Aries, f72r2 Gemini, f73v Sagittarius; 104 label loci ~ 10% of the
zodiac token corpus), with:
  - binding from 4 independent vision agents using ANCHOR-BASED alignment (labels read
    from pixels and matched to Takahashi, not pure reading-order inference);
  - null = shuffle attr<->label WITHIN folio x tier (kills folio vocab, q-register,
    template grammar, AND tier-layout confounds by construction);
  - power controls carried by prefix AND by core (the audit's gap), each at 100%/75%;
  - computed multiple-comparison control: omnibus max-MI permutation across the whole
    feature x attribute grid (not narrated Bonferroni).
Verdict criteria identical to K13/K12: a referent signal must beat the shuffle null AND
the core signal must exceed the prefix/suffix template floor.
"""
import json, math, random, glob, os, sys
from collections import Counter

random.seed(0)
# optional arg: a directory of fresh bind_*.json files from a new vision pass;
# with no arg, falls back to the repo's committed data/grounding/z10_bindings.json
SLICE_DIR = sys.argv[1] if len(sys.argv) > 1 else "z10_slice"

# ---- morpheme parser (columns.py / l6_grounding.py, unchanged) ----
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph','cfh']
SUF=['eedy','eody','edy','aiin','aiir','ain','iin','dy','ol','or','ar','al','am','dam','ey','eey','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s

def MI(xs,ys):
    n=len(xs); jx=Counter(xs); jy=Counter(ys); jj=Counter(zip(xs,ys))
    return sum(c/n*math.log2((c/n)/((jx[x]/n)*(jy[y]/n))) for (x,y),c in jj.items())
def H(c):
    n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values()) if n else 0.0

FEATS = {
  "prefix":           lambda r: decomp(r["word"])[0],
  "suffix":           lambda r: decomp(r["word"])[2],
  "core_first2":      lambda r: decomp(r["word"])[1][:2],
  "core_len":         lambda r: min(len(decomp(r["word"])[1]),4),
  "core_has_gallows": lambda r: any(g in decomp(r["word"])[1] for g in "ktpf"),
  "wholeword_first2": lambda r: r["word"][:2],
}
CORE_FEATS = {"core_first2","core_len","core_has_gallows"}

def strata_of(rows, within_tier=True):
    by={}
    for i,r in enumerate(rows):
        k=(r["folio"], r["tier"]) if within_tier else (r["folio"],)
        by.setdefault(k,[]).append(i)
    return by

def perm_test(rows, attr_fn, feat_fn, nperm=20000, within_tier=True):
    feats=[feat_fn(r) for r in rows]; attrs=[attr_fn(r) for r in rows]
    obs=MI(feats,attrs); by=strata_of(rows,within_tier); ge=0
    for _ in range(nperm):
        sh=attrs[:]
        for idxs in by.values():
            vals=[attrs[i] for i in idxs]; random.shuffle(vals)
            for j,i in enumerate(idxs): sh[i]=vals[j]
        if MI(feats,sh)>=obs-1e-12: ge+=1
    return obs,(ge+1)/(nperm+1)

def omnibus(rows, attr_fns, nperm=5000, within_tier=True):
    """max-MI over the full feature x attribute grid, one shared permutation stream ->
    a single family-wise p for 'is ANY cell real'."""
    grid=[(an,fn_a,fn,fname) for an,fn_a in attr_fns.items() for fname,fn in FEATS.items()]
    featvals={fname:[fn(r) for r in rows] for fname,fn in FEATS.items()}
    attrvals={an:[fn_a(r) for r in rows] for an,fn_a in attr_fns.items()}
    obs_grid={(an,fname):MI(featvals[fname],attrvals[an]) for an,_,_,fname in grid}
    obs_max=max(obs_grid.values()); by=strata_of(rows,within_tier); ge=0
    for _ in range(nperm):
        mx=0.0
        shuffled={}
        for an in attr_fns:
            sh=attrvals[an][:]
            for idxs in by.values():
                vals=[sh[i] for i in idxs]; random.shuffle(vals)
                for j,i in enumerate(idxs): sh[i]=vals[j]
            shuffled[an]=sh
        for an,_,_,fname in grid:
            m=MI(featvals[fname],shuffled[an])
            if m>mx: mx=m
        if mx>=obs_max-1e-12: ge+=1
    return obs_grid,obs_max,(ge+1)/(nperm+1)

# ---- load bindings (session scratchpad if present, else repo data/grounding/z10_bindings.json) ----
rows=[]
binds=[json.load(open(fp)) for fp in sorted(glob.glob(os.path.join(SLICE_DIR,"bind_*.json")))]
if not binds:
    repo=os.path.join("data","grounding","z10_bindings.json")
    if os.path.exists(repo): binds=list(json.load(open(repo))["folios"].values())
for d in binds:
    for rec in d["records"]:
        w=rec["label"].split(".")[0]           # first word of multi-word locus
        rows.append({"folio":d["folio"],"tier":rec["tier"],"word":w,
                     "conf":rec.get("bind_conf","low"),**{k:rec.get(k,"") for k in
                     ("body","clothing_color","container","star","star_tail","star_hand",
                      "arms","headwear","facing","clock")}})
print(f"loaded {len(rows)} bound label loci from {len(glob.glob(os.path.join(SLICE_DIR,'bind_*.json')))} folios")
for f in sorted(set(r['folio'] for r in rows)):
    sub=[r for r in rows if r['folio']==f]
    print(f"  {f}: n={len(sub)}  conf: {dict(Counter(r['conf'] for r in sub))}")

ATTRS = {
  "body":       lambda r: r["body"] or "unknown",
  "star":       lambda r: r["star"] or "unknown",
  "clothing_color": lambda r: r["clothing_color"] or "none",
  "container":  lambda r: "vessel" if r["container"] in ("barrel","tub") else "none",
  "star_tail":  lambda r: r["star_tail"],
  "arms":       lambda r: r["arms"],
  "headwear":   lambda r: "some" if r["headwear"] not in ("none","") else "none",
  "facing":     lambda r: r["facing"],
  "clock_sector": lambda r: (int(r["clock"])-1)//3 if str(r["clock"]).isdigit() else "?",
}

def battery(rows, title, nperm=20000):
    print("\n"+"="*74); print(title); print("="*74)
    usable={}
    for an,fn in ATTRS.items():
        vals=Counter(fn(r) for r in rows)
        if len(vals)<2 or max(vals.values())>len(rows)-3:
            print(f"  [skip {an}: degenerate {dict(vals)}]"); continue
        usable[an]=fn
        print(f"  attr {an:12s} dist={dict(vals)}  H={H(vals):.2f}b")
    results={}
    for an,fn in usable.items():
        print(f"  --- {an} ---")
        for fname,ffn in FEATS.items():
            obs,p=perm_test(rows,fn,ffn,nperm=nperm)
            results[(an,fname)]=(obs,p)
            flag=" *" if p<0.05 else ""
            print(f"    {fname:18s} MI={obs:.3f}  p={p:.4f}{flag}")
        floor=max(results[(an,'prefix')][0],results[(an,'suffix')][0])
        core=max(results[(an,f)][0] for f in CORE_FEATS)
        print(f"    DISCRIMINATOR: template floor={floor:.3f} core(best)={core:.3f} -> core {'EXCEEDS' if core>floor else 'does NOT exceed'} floor")
    if usable:
        _,omax,op=omnibus(rows,usable)
        ncells=len(usable)*len(FEATS)
        print(f"  OMNIBUS max-MI over {ncells} cells: max={omax:.3f}  FAMILY-WISE p={op:.4f}")
    return results

def power_controls(rows):
    print("\n"+"="*74); print(f"POWER CONTROLS (n={len(rows)}, within folio-x-tier null)"); print("="*74)
    rng=random.Random(7)
    def run_ctrl(name, carrier_fn, consistency):
        import copy
        sim=copy.deepcopy(rows)
        classes={}
        for r in sim:
            v=carrier_fn(r); classes.setdefault(v,len(classes)%2)
        for r in sim:
            true="A" if classes[carrier_fn(r)]==0 else "B"
            r["_a"]=true if rng.random()<consistency else ("B" if true=="A" else "A")
        best=(0,1.0,"")
        for fname,ffn in FEATS.items():
            obs,p=perm_test(sim,lambda r:r["_a"],ffn,nperm=10000)
            if obs>best[0]: best=(obs,p,fname)
        obs,p=perm_test(sim,lambda r:r["_a"],carrier_fn,nperm=10000)
        print(f"  {name:34s} carrier-MI={obs:.3f} p={p:.4f} | best-feat[{best[2]}] MI={best[0]:.3f} p={best[1]:.4f}")
    run_ctrl("POSITIVE prefix-carried (100%)", lambda r: decomp(r["word"])[0], 1.00)
    run_ctrl("PARTIAL  prefix-carried (75%)",  lambda r: decomp(r["word"])[0], 0.75)
    run_ctrl("POSITIVE core-carried (100%)",   lambda r: decomp(r["word"])[1][:2], 1.00)
    run_ctrl("PARTIAL  core-carried (75%)",    lambda r: decomp(r["word"])[1][:2], 0.75)
    import copy
    neg=copy.deepcopy(rows)
    for r in neg: r["_a"]=rng.choice(["x","y"])
    obs,p=perm_test(neg,lambda r:r["_a"],lambda r:decomp(r["word"])[0],nperm=10000)
    print(f"  {'NEGATIVE random attr':34s} prefix-MI={obs:.3f} p={p:.4f}  <- must be ~uniform")

def westfall_young(rows, nperm=10000, label=""):
    """min-p family-wise omnibus (Westfall-Young, shared permutation stream).
    Unlike max-MI, this is fair to low-cardinality features (e.g. core_len)."""
    import bisect
    usable={}
    for an,fn in ATTRS.items():
        vals=Counter(fn(r) for r in rows)
        if len(vals)>=2 and max(vals.values())<=len(rows)-3: usable[an]=fn
    cells=[(an,fname) for an in usable for fname in FEATS]
    fv={f:[FEATS[f](r) for r in rows] for f in FEATS}
    av={a:[usable[a](r) for r in rows] for a in usable}
    obs={c: MI(fv[c[1]],av[c[0]]) for c in cells}
    by=strata_of(rows,within_tier=True)
    nulls={c:[] for c in cells}
    for _ in range(nperm):
        sh={}
        for a in usable:
            s=av[a][:]
            for idxs in by.values():
                vals=[s[i] for i in idxs]; random.shuffle(vals)
                for j,i in enumerate(idxs): s[i]=vals[j]
            sh[a]=s
        for (a,f) in cells: nulls[(a,f)].append(MI(fv[f],sh[a]))
    pcell={c:(sum(1 for v in nulls[c] if v>=obs[c]-1e-12)+1)/(nperm+1) for c in cells}
    sortnull={c: sorted(nulls[c]) for c in cells}
    minp_null=[]
    for k in range(nperm):
        mp=1.0
        for c in cells:
            gt=len(sortnull[c])-bisect.bisect_left(sortnull[c],nulls[c][k]-1e-12)
            p=(gt+1)/(nperm+1)
            if p<mp: mp=p
        minp_null.append(mp)
    obs_minp=min(pcell.values())
    omni=(sum(1 for v in minp_null if v<=obs_minp)+1)/(nperm+1)
    print(f"  [{label}] min raw-p={obs_minp:.4f} over {len(cells)} cells -> WESTFALL-YOUNG family-wise p={omni:.4f}")
    return omni

if rows:
    battery(rows,"TEST A  all bound loci (n=%d): morphology vs referent attributes"%len(rows))
    hm=[r for r in rows if r["conf"] in ("high","med")]
    if len(hm)>=30 and len(hm)<len(rows):
        battery(hm,"TEST B  high+med binding confidence only (n=%d)"%len(hm))
    # tier as attribute needs the weaker within-folio null (tier is the stratum otherwise)
    print("\n"+"="*74); print("TEST C  morphology vs TIER (within-folio null — K12-artifact-prone, caveat)"); print("="*74)
    for fname,ffn in FEATS.items():
        obs,p=perm_test(rows,lambda r:r["tier"],ffn,within_tier=False)
        flag=" *" if p<0.05 else ""
        print(f"    {fname:18s} MI={obs:.3f}  p={p:.4f}{flag}")
    print("\n"+"="*74); print("FAMILY-WISE CORRECTION (Westfall-Young min-p omnibus)"); print("="*74)
    westfall_young(rows, label="all loci")
    hi=[r for r in rows if r["conf"]=="high"]
    if 30<=len(hi)<len(rows): westfall_young(hi, label=f"high-conf only n={len(hi)}")
    power_controls(rows)
else:
    print("no bind_*.json yet")
