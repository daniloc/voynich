"""
L6 within-section referent-variation test (the ledger's "only un-confounded settler").
Q: within ONE section, does a label's MORPHOLOGY track the VISUAL ATTRIBUTE of the
   specific referent it sits on — beyond a within-folio label<->referent shuffle?
Null: permute the attribute<->label binding WITHIN each folio. This preserves section
   vocab, q-register law, label-template grammar, folio identity; destroys ONLY referent
   binding. (Kills confounds #1/#2 + D11 by construction.)
Discriminator (K12-style): the CORE signal must EXCEED the prefix/suffix template floor.
   If core <= prefix/suffix, association is template grammar, not referent semantics.
"""
import math, random
from collections import Counter
random.seed(0)

# ---- morpheme parser (from columns.py) ----
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph','cfh']
SUF=['eedy','eody','edy','aiin','aiir','ain','iin','dy','ol','or','ar','al','am','dam','ey','eey','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s

# ---- grounded pharma records (from vision subagents) ----
# (label, folio, part_class, color, binding_conf)
PHARMA=[
 # f88r @Lc jars
 ("otorchety","f88r","container","red","high"),("otaldy","f88r","container","red","high"),("ofyskydal","f88r","container","red","high"),
 # f88r @Lf
 ("oral","f88r","leaf","green","med"),("orald","f88r","other","brown","med"),("oldar","f88r","leaf","green","med"),
 ("otoky","f88r","root","brown","med"),("otaly","f88r","root","brown","med"),("oram","f88r","leaf","green","med"),
 ("dary","f88r","root","brown","med"),("okol","f88r","root","red","med"),("sorory","f88r","root","brown","med"),
 ("otyda","f88r","root","brown","med"),("otor","f88r","root","brown","low"),("am","f88r","leaf","green","low"),("ofaldo","f88r","leaf","green","med"),
 # f99v @Lc jars
 ("otaramy","f99v","container","red","high"),("okoldody","f99v","container","red","high"),("darolaly","f99v","container","red","high"),("dralas","f99v","container","red","high"),
 # f99v @Lf
 ("otoldy","f99v","root","red","med"),("otorchy","f99v","root","red","med"),("oldy","f99v","root","brown","med"),
 ("darary","f99v","leaf","green","med"),("otaly","f99v","root","red","low"),("olsy","f99v","root","brown","med"),
 ("arol","f99v","root","red","low"),("otoky","f99v","leaf","brown","low"),("oeeesary","f99v","leaf","red","low"),
 ("daiiine","f99v","leaf","mixed","low"),("sory","f99v","leaf","red","low"),("saiino","f99v","root","brown","med"),
 ("otolsar","f99v","other","red","low"),("osary","f99v","other","red","med"),("okechy","f99v","root","brown","med"),
 ("otal","f99v","other","mixed","low"),("chor","f99v","root","red","low"),("olekor","f99v","leaf","mixed","low"),
 ("okeodor","f99v","other","brown","med"),("olky","f99v","other","mixed","low"),("doldam","f99v","leaf","green","med"),("koleearol","f99v","leaf","green","med"),
]

def H(c):
    n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values()) if n else 0.0
def MI(xs,ys):
    n=len(xs); jx=Counter(xs); jy=Counter(ys); jj=Counter(zip(xs,ys))
    return sum(c/n*math.log2((c/n)/((jx[x]/n)*(jy[y]/n))) for (x,y),c in jj.items())

def perm_test(rows, attr_fn, feat_fn, nperm=20000):
    """rows: list of dicts. observed MI(feat;attr) vs within-folio shuffle of attr."""
    feats=[feat_fn(r) for r in rows]
    attrs=[attr_fn(r) for r in rows]
    obs=MI(feats,attrs)
    # group indices by folio for within-folio permutation of attrs
    byf={}
    for i,r in enumerate(rows): byf.setdefault(r["folio"],[]).append(i)
    ge=0
    for _ in range(nperm):
        sh=attrs[:]
        for idxs in byf.values():
            vals=[attrs[i] for i in idxs]; random.shuffle(vals)
            for j,i in enumerate(idxs): sh[i]=vals[j]
        if MI(feats,sh)>=obs-1e-12: ge+=1
    return obs, (ge+1)/(nperm+1)

def run(rows, title, attr_fn, attr_name):
    print("="*70); print(title); print("="*70)
    av=Counter(attr_fn(r) for r in rows)
    print(f"  n={len(rows)}  attribute[{attr_name}] dist: {dict(av)}  H(attr)={H(av):.2f} bits")
    feats={
      "prefix": lambda r: decomp(r["label"])[0],
      "suffix": lambda r: decomp(r["label"])[2],
      "core_first2": lambda r: decomp(r["label"])[1][:2],
      "core_len": lambda r: min(len(decomp(r["label"])[1]),4),
      "core_has_gallows": lambda r: any(g in decomp(r["label"])[1] for g in "ktpf"),
      "wholeword_first2": lambda r: r["label"][:2],
    }
    res={}
    for fn_name,fn in feats.items():
        obs,p=perm_test(rows,attr_fn,fn)
        res[fn_name]=(obs,p)
        card=len(set(fn(r) for r in rows))
        print(f"  {fn_name:18s} MI={obs:.3f} bits  p={p:.4f}  (feat cardinality {card})")
    # discriminator
    pre=res["prefix"][0]; suf=res["suffix"][0]
    floor=max(pre,suf)
    core=max(res["core_first2"][0],res["core_len"][0],res["core_has_gallows"][0])
    print(f"  --- DISCRIMINATOR: template floor max(prefix,suffix)={floor:.3f} | core(best)={core:.3f} -> core {'EXCEEDS' if core>floor else 'does NOT exceed'} floor")
    return res

# Test 1: plant elements only, part_class {root,leaf,other}
plf=[{"label":l,"folio":f,"attr":pc} for (l,f,pc,col,bc) in PHARMA if pc!="container"]
run(plf,"TEST 1  pharma @Lf plant-elements: morphology vs PART-CLASS (root/leaf/other)",
    lambda r:r["attr"],"part_class")

# Test 1b: high+med binding confidence only
plf2=[{"label":l,"folio":f,"attr":pc} for (l,f,pc,col,bc) in PHARMA if pc!="container" and bc in("high","med")]
run(plf2,"TEST 1b  same, restricted to high/med binding confidence",
    lambda r:r["attr"],"part_class")

# Test 2: binary root vs green-foliage via COLOR (most reliably read attribute)
def colbin(r): return "brownred" if r["attr"] in ("brown","red") else ("green" if r["attr"]=="green" else "other")
pcol=[{"label":l,"folio":f,"attr":col} for (l,f,pc,col,bc) in PHARMA if pc!="container"]
run(pcol,"TEST 2  pharma @Lf: morphology vs COLOR-class (brownred/green/mixed)",
    lambda r:colbin(r),"color_class")

# Test 3: include jars -> referent KIND {container, plant}
pall=[{"label":l,"folio":f,"attr":("container" if pc=="container" else "plant")} for (l,f,pc,col,bc) in PHARMA]
run(pall,"TEST 3  all pharma: morphology vs KIND (jar-container vs plant-element)",
    lambda r:r["attr"],"referent_kind")

# ---- POWER CHECK: can this n/test detect a real signal? ----
print("\n"+"="*70); print("POWER / SANITY CONTROLS (n=35 @Lf)"); print("="*70)
import copy
base=[{"label":l,"folio":f,"attr":pc} for (l,f,pc,col,bc) in PHARMA if pc!="container"]
# POSITIVE control: make attribute a deterministic function of the prefix
pos=copy.deepcopy(base)
for r in pos: r["attr"]=("A" if decomp(r["label"])[0] in("ok","ot","o") else "B")
o,p=perm_test(pos,lambda r:r["attr"],lambda r:decomp(r["label"])[0])
print(f"  POSITIVE (attr := f(prefix)):   prefix MI={o:.3f}  p={p:.4f}   <- must be tiny")
# NEGATIVE control: random attribute
neg=copy.deepcopy(base)
rng=random.Random(7)
for r in neg: r["attr"]=rng.choice(["x","y","z"])
o,p=perm_test(neg,lambda r:r["attr"],lambda r:decomp(r["label"])[0])
print(f"  NEGATIVE (attr := random):      prefix MI={o:.3f}  p={p:.4f}   <- must be ~uniform")
# realistic-strength positive: prefix predicts attr 70% of the time
mid=copy.deepcopy(base)
for r in mid:
    true="A" if decomp(r["label"])[0] in("ok","ot","o") else "B"
    r["attr"]=true if rng.random()<0.75 else ("B" if true=="A" else "A")
o,p=perm_test(mid,lambda r:r["attr"],lambda r:decomp(r["label"])[0])
print(f"  PARTIAL  (prefix->attr ~75%):   prefix MI={o:.3f}  p={p:.4f}   <- detectable?")
