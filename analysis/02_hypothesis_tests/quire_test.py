"""
L7 — CODICOLOGY RE-RUN: is confound #2's vocabulary autocorrelation a smooth
READING-ORDER (shelf-position) decay, or a step-function at PRODUCTION units
(quire / scribal hand — leaves written together)?

Why it matters: every control for confound #2 partials out *folio distance in
current binding order*. If the real driver is "same gathering / same scribe," then
(a) signals that "survived" a distance control but actually track quire identity are
weaker than credited, and (b) the correct thing to partial out is quire+hand.

Design: section/quire/reading-distance are mutually confounded (book laid out by
topic in order), so the decisive run is WITHIN ONE SECTION that spans multiple
quires, hands, and both Currier dialects — the herbal. There, topic is held
constant and we can ask what's left driving folio-to-folio vocabulary similarity.

Metric: per-folio L2-normalized CORE-frequency vector (the content field, D3);
pairwise cosine similarity; OLS of similarity on standardized predictors.
Caveat: this tests quire-as-production-unit + scribe, NOT Fagin Davis's specific
bifolio reordering (needs external data). It's the in-repo 80% of L7.
"""
import json, math, re
import numpy as np
from collections import Counter, OrderedDict, defaultdict

PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph','cfh']
SUF=['eedy','eody','edy','aiin','aiir','ain','iin','dy','ol','or','ar','al','am','dam','ey','eey','y']
def core(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return r[:len(r)-len(s)] if s else r

d=json.load(open('data/corpus/body_spatial.json'))
toks=d['tokens']

# folio reading order
order=OrderedDict()
for t in toks: order.setdefault(t['folio'],len(order))

def folio_vectors(section=None):
    vec=defaultdict(Counter); meta={}
    for t in toks:
        if section and t['section']!=section: continue
        if t.get('role')!='paragraph' or t['uncertain'] or '?' in t['word'] or len(t['word'])<2: continue
        f=t['folio']; vec[f][core(t['word'])]+=1
        meta[f]=(t['quire'],t['hand'],t['currier'])
    return vec,meta

def cosine(a,b):
    keys=set(a)|set(b);
    dot=sum(a[k]*b[k] for k in keys)
    na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
    return dot/(na*nb) if na and nb else 0.0

def ols(X,y):
    """return betas, R2 with intercept; X columns already include intercept."""
    beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    yhat=X@beta; ss_res=((y-yhat)**2).sum(); ss_tot=((y-y.mean())**2).sum()
    return beta,1-ss_res/ss_tot

def run(label, section):
    vec,meta=folio_vectors(section)
    fols=[f for f in vec if sum(vec[f].values())>=80]
    fols.sort(key=lambda f:order[f])
    print("="*78); print(f"{label}: {len(fols)} folios "
          f"(quires={len(set(meta[f][0] for f in fols))}, "
          f"hands={sorted(set(meta[f][1] for f in fols))}, "
          f"currier={sorted(set(meta[f][2] for f in fols))})"); print("="*78)
    if len(fols)<12: print("  too few folios\n"); return
    rows=[]
    for i in range(len(fols)):
        for j in range(i+1,len(fols)):
            fi,fj=fols[i],fols[j]
            qi,hi,ci=meta[fi]; qj,hj,cj=meta[fj]
            rd=abs(order[fi]-order[fj])
            rows.append((cosine(vec[fi],vec[fj]), rd,
                         1.0 if qi==qj else 0.0,
                         1.0 if (hi==hj and hi!='?') else 0.0,
                         1.0 if ci==cj else 0.0))
    A=np.array(rows); y=A[:,0]
    rd=A[:,1]; sq=A[:,2]; sh=A[:,3]; sc=A[:,4]
    def z(x): return (x-x.mean())/(x.std()+1e-12)
    # raw correlations
    print(f"  raw corr(similarity, predictor):")
    print(f"    reading-distance {np.corrcoef(y,rd)[0,1]:+.3f} | same-quire {np.corrcoef(y,sq)[0,1]:+.3f} "
          f"| same-hand {np.corrcoef(y,sh)[0,1]:+.3f} | same-currier {np.corrcoef(y,sc)[0,1]:+.3f}")
    # full model, standardized
    feats=[('reading-dist',z(rd)),('same-quire',sq),('same-hand',sh),('same-currier',sc)]
    X=np.column_stack([np.ones(len(y))]+[f for _,f in feats])
    beta,r2=ols(X,y)
    print(f"  OLS similarity ~ all (standardized dist), R2={r2:.3f}:")
    for k,(nm,_) in enumerate(feats):
        print(f"    {nm:14s} beta={beta[k+1]:+.4f}")
    # the key contrast: at MATCHED reading distance, same-quire vs cross-quire
    near=A[(rd>0)&(rd<=8)]  # leaves close in reading order
    if len(near)>20:
        sim_sq=near[near[:,2]==1,0]; sim_xq=near[near[:,2]==0,0]
        print(f"  matched short reading-distance (<=8 leaves): "
              f"same-quire mean sim {sim_sq.mean():.3f} (n={len(sim_sq)}) vs "
              f"cross-quire {sim_xq.mean():.3f} (n={len(sim_xq)})  Δ={sim_sq.mean()-sim_xq.mean():+.3f}")
    # does reading-distance survive controlling for quire+hand+currier?
    Xr=np.column_stack([np.ones(len(y)),sq,sh,sc])
    br,_=ols(Xr,y); resid=y-Xr@br
    pc=np.corrcoef(resid, z(rd))[0,1]
    print(f"  partial corr(reading-distance | quire,hand,currier) = {pc:+.3f}  "
          f"(near 0 => 'smooth shelf decay' was really production-unit identity)")
    print()

run("WITHIN HERBAL (topic held constant; the decisive run)","herbal")
run("WITHIN BIOLOGICAL","biological")
run("WITHIN STARS/RECIPES","stars_recipes")
run("ALL SECTIONS (section uncontrolled — expect section to dominate)",None)

print("READ: if same-quire/same-hand carry the variance and partial-corr(reading-dist)")
print("≈0, then confound #2 is PRODUCTION-UNIT structure, not smooth shelf position —")
print("controls should partial out quire+hand, and distance-only controls were leaky.")
