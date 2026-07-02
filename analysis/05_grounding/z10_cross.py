"""
Z10b — cross-ring / aggregate zodiac structure tests (2026-07-01).
Answers "could information hide ACROSS the zodiac, invisible to K14's within-ring null?"
Three hypothesis classes K14 was structurally blind to; the two text-only ones tested at
full 12-ring scale (299 Lz loci from LSI_ivtff), the attribute one on z10_bindings.json:
  Z-A  cross-ring positional join: doubled-sign pairs (dark/white Aries f70v1|f71r,
       white/dark Taurus f71v|f72r1, 15 loci each) — does position k in one half-ring
       correspond morphologically to position k in the other (day-alignment)?
       Cyclic-max alignment over all offsets+reversal; null = within-tier order shuffle;
       cross-sign pairs as baseline. RESULT: NULL (same-sign p=0.26/0.40 edit-sim,
       0.93/0.63 morph-match; same-sign never beats the cross-sign baseline).
  Z-B  same-label determinism: among the 104 vision-bound loci (z10_bindings.json), do
       repeated words (otal x5, okal x3, okaly x3, ...) sit on similar-looking nymphs?
       22 same-label pairs, attr-match rate 0.545, within folio-x-tier shuffle p=0.85.
       NULL — the same word recurs on visually unrelated figures. Anti-naming.
  Z-C  collision census (299 loci): 254 types / 299 tokens, 30 repeated types.
       Within-ring duplicate pairs = 12 vs random-assignment null p=0.018 — duplicates
       CLUSTER within rings, the OPPOSITE of naming discipline (a roster avoids giving
       two nymphs in one ring the same name); matches the self-citation/copying
       signature (confound #5). Adjacent-duplicate clustering beyond that: p=0.50.
Untested residue (by design, both K1-class): ring-identity codes (morphology<->which-sign)
are confounded with block vocabulary (confound #2) and need a matched content-free block
generator — the decoded-key version (Occitan month join) already ran null in
zodiac_join.py; steganographic embedding in generator free-choices is unfalsifiable
without a candidate scheme (the adversarial fork's acknowledged open loophole).
"""
import json, math, random, re, os
from collections import Counter
random.seed(0)

PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph','cfh']
SUF=['eedy','eody','edy','aiin','aiir','ain','iin','dy','ol','or','ar','al','am','dam','ey','eey','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s
def clean(t): return re.sub(r'[!?]','',t.split(".")[0])
def lev(a,b):
    m,n=len(a),len(b); d=list(range(n+1))
    for i in range(1,m+1):
        prev=d[0]; d[0]=i
        for j in range(1,n+1):
            cur=d[j]; d[j]=min(d[j]+1,d[j-1]+1,prev+(a[i-1]!=b[j-1])); prev=cur
    return d[n]
def editsim(a,b): return 1-lev(a,b)/max(len(a),len(b),1)
def morphsim(a,b):
    pa,ca,sa=decomp(a); pb,cb,sb=decomp(b)
    return (pa==pb)+(sa==sb)+(ca[:2]==cb[:2])

# ---- extract all 12 rings' Lz sequences from IVTFF ----
FOLIOS=["f70v2","f70v1","f71r","f71v","f72r1","f72r2","f72r3","f72v1","f72v2","f72v3","f73r","f73v"]
data={f:[] for f in FOLIOS}; cur_comment=""
for line in open("data/corpus/LSI_ivtff.txt"):
    m=re.match(r'<(f\d+[rv]\d*)>',line)
    if m: cur_comment=""; continue
    if line.startswith("#"):
        c=line.lstrip("# ").strip()
        if c.startswith("panel") or any(k in c.lower() for k in ("ring","band","circle","center","outside","not in circle")):
            cur_comment=c
        continue
    m=re.match(r'<(f\d+[rv]\d*)\.(\d+[a-z]?),([@&])(\w+);H>\s+(.*)',line)
    if m and m.group(1) in FOLIOS and m.group(4)=="Lz":
        fol,locno,pref,lcode,text=m.groups()
        text=re.sub(r'<![^>]*>','',text).strip().rstrip('.')
        c=cur_comment.lower()
        tier="outside" if ("outside" in c or "not in circle" in c) else ("outer" if "outer" in c else ("inner" if "inner" in c else ("middle" if "middle" in c else "unk")))
        data[fol].append({"tier":tier,"label":text.replace(",",".")})

print("="*74); print("Z-A  CROSS-RING POSITIONAL JOIN (cyclic-max alignment, 15-loci rings)"); print("="*74)
R15={f:{t:[clean(d["label"]) for d in data[f] if d["tier"]==t] for t in ("outer","inner")}
     for f in ["f70v1","f71r","f71v","f72r1"]}
def align_score(A,B,sim):
    best=-1
    for rev in (False,True):
        Bx=B[::-1] if rev else B
        for off in range(len(Bx)):
            s=sum(sim(a,Bx[(i+off)%len(Bx)]) for i,a in enumerate(A))/len(A)
            if s>best: best=s
    return best
def pair_test(f1,f2,sim,nperm=3000):
    obs=sum(align_score(R15[f1][t],R15[f2][t],sim)*len(R15[f1][t]) for t in("outer","inner"))/15
    ge=0
    for _ in range(nperm):
        tot=0
        for t in ("outer","inner"):
            B=R15[f2][t][:]; random.shuffle(B)
            tot+=align_score(R15[f1][t],B,sim)*len(R15[f1][t])
        if tot/15>=obs-1e-12: ge+=1
    return obs,(ge+1)/(nperm+1)
PAIRS=[("f70v1","f71r","SAME-SIGN Aries dark|white"),("f71v","f72r1","SAME-SIGN Taurus white|dark"),
       ("f70v1","f71v","cross-sign baseline"),("f71r","f72r1","cross-sign baseline"),
       ("f70v1","f72r1","cross-sign baseline"),("f71r","f71v","cross-sign baseline")]
for sim,nm in ((editsim,"edit-similarity"),(morphsim,"morph-match(0-3)")):
    print(f"  --- {nm} ---")
    for f1,f2,tag in PAIRS:
        o,p=pair_test(f1,f2,sim)
        print(f"   {tag:28s} {f1}x{f2}: best-aligned={o:.3f}  p={p:.4f}")

print("="*74); print("Z-B  SAME-LABEL DETERMINISM (repeated words on similar nymphs?)"); print("="*74)
binds=json.load(open("data/grounding/z10_bindings.json"))["folios"]
rows=[]
for fol,d in binds.items():
    for r in d["records"]:
        rows.append({"folio":fol,"tier":r["tier"],"word":clean(r["label"]),
                     "attrs":{k:r.get(k,"") for k in("body","container","star_tail","arms","headwear","facing")}})
byword=Counter(r["word"] for r in rows)
print(f"  repeated words among {len(rows)} loci: {({w:c for w,c in byword.items() if c>1})}")
pairs_idx=[(i,j) for i in range(len(rows)) for j in range(i+1,len(rows)) if rows[i]["word"]==rows[j]["word"]]
KEYS=("body","container","star_tail","arms","headwear","facing")
def matchrate(rws,pidx):
    return sum(sum(rws[i]["attrs"][k]==rws[j]["attrs"][k] for k in KEYS) for i,j in pidx)/(len(pidx)*len(KEYS)) if pidx else 0
obs=matchrate(rows,pairs_idx)
by={}
for i,r in enumerate(rows): by.setdefault((r["folio"],r["tier"]),[]).append(i)
ge=0; NP=10000
for _ in range(NP):
    sh=[None]*len(rows)
    for idxs in by.values():
        av=[rows[i]["attrs"] for i in idxs]; random.shuffle(av)
        for j,i in enumerate(idxs): sh[i]=av[j]
    tmp=[{"word":rows[i]["word"],"attrs":sh[i]} for i in range(len(rows))]
    if matchrate(tmp,pairs_idx)>=obs-1e-12: ge+=1
print(f"  {len(pairs_idx)} same-label pairs, attr-match rate={obs:.3f}, shuffle p={(ge+1)/(NP+1):.4f}")

print("="*74); print("Z-C  WHOLE-ZODIAC COLLISION CENSUS (299 loci, 12 rings)"); print("="*74)
words=[clean(d["label"]) for f in FOLIOS for d in data[f]]
c=Counter(words); rep={w:n for w,n in c.items() if n>1}
print(f"  types={len(c)} tokens={len(words)} repeated-types={len(rep)}; top: {sorted(rep.items(),key=lambda kv:-kv[1])[:12]}")
def within_ring_pairs(assign):
    return sum(sum(n*(n-1)//2 for n in Counter(a).values()) for a in assign.values())
obs_w=within_ring_pairs({f:[clean(d["label"]) for d in data[f]] for f in FOLIOS})
sizes={f:len(data[f]) for f in FOLIOS}
ge=0
for _ in range(NP):
    pool=words[:]; random.shuffle(pool); a={}; k=0
    for f in FOLIOS: a[f]=pool[k:k+sizes[f]]; k+=sizes[f]
    if within_ring_pairs(a)>=obs_w: ge+=1
print(f"  within-ring duplicate pairs: observed={obs_w}  random-assignment p={(ge+1)/(NP+1):.4f}"
      f"  (EXCESS duplicates = anti-naming, copying signature)")
adj=0
for f in FOLIOS:
    for t in ("outer","inner","outside"):
        seq=[clean(d["label"]) for d in data[f] if d["tier"]==t]
        adj+=sum(1 for i in range(len(seq)-1) if seq[i]==seq[i+1])
ge=0
for _ in range(NP):
    tot=0
    for f in FOLIOS:
        for t in ("outer","inner","outside"):
            seq=[clean(d["label"]) for d in data[f] if d["tier"]==t]
            random.shuffle(seq)
            tot+=sum(1 for i in range(len(seq)-1) if seq[i]==seq[i+1])
    if tot>=adj: ge+=1
print(f"  adjacent same-label pairs: observed={adj}  within-tier shuffle p={(ge+1)/(NP+1):.4f}")
