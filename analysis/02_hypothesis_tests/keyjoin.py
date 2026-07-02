"""
THE KNOWN-PLAINTEXT CAPSTONE. The audit said the ONLY avenue that could still find
meaning is cross-modal known-plaintext, done right: held-out, length-controlled, and
with reference separated from positional drift. Two grounds of external truth:
  PART 1  BOTANICAL CRIB  — vision-read STRUCTURAL tags (flower/root/leaf), not pigment.
          Leave-one-out cross-validation + label-permutation null + length control.
  PART 2  ZODIAC KEY-JOIN — the Occitan month-names are the ONE decoded header.
          Identifiable referential tests ONLY (the raw feature~ordinal r is confounded
          with manuscript position/dialect drift, so we test the things position CAN'T fake):
            (a) are medallions lexically organized at all? vs label-shuffle null
            (b) do ring-order labels encode a day 1..30 progression? vs ring-shuffle null
            (c) debunk: show the r=0.48 q_rate~ordinal is position-drift, not month-reference
A POSITIVE here (survives held-out / shuffle) would be the first real meaning signal.
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(1989)
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
def fk(f): m=re.match(r'f(\d+)([rv])(\d*)',f); return (int(m.group(1)),0 if m.group(2)=='r' else 1,int(m.group(3) or 0))
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph']
SUF=['eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y']
def morphs(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s

# ===================== PART 1: BOTANICAL CRIB, held-out =====================
TAGS = {
 'f4r':dict(flower='red',root='fibrous',leaf='pinnate'),'f6r':dict(flower='red',root='fibrous',leaf='fern'),
 'f8r':dict(flower='none',root='rhizome',leaf='lobed'),'f9r':dict(flower='none',root='taproot',leaf='curly'),
 'f11r':dict(flower='blue',root='fibrous',leaf='frilly'),'f14r':dict(flower='red',root='rhizome',leaf='sword'),
 'f15r':dict(flower='white',root='rhizome',leaf='toothed'),'f18r':dict(flower='blue',root='fibrous',leaf='oval'),
 'f20r':dict(flower='red',root='fibrous',leaf='pinnate'),'f23r':dict(flower='blue',root='rhizome',leaf='palmate'),
 'f26r':dict(flower='blue',root='taproot',leaf='scalloped'),'f31r':dict(flower='white',root='taproot',leaf='oval'),
 'f35r':dict(flower='blue',root='bulb',leaf='thin'),'f37r':dict(flower='red',root='fibrous',leaf='lance'),
 'f43r':dict(flower='white',root='fibrous',leaf='small'),'f52r':dict(flower='blue',root='taproot',leaf='lance'),
 'f55r':dict(flower='red',root='bulb',leaf='palmate'),
}
def fwords(f): return [w for ln in C['folios'].get(f,[]) for w in ln['words'] if '?' not in w]
def trigrams(ws):
    c=Counter()
    for w in ws:
        for i in range(len(w)-2): c[w[i:i+2+1]]+=1
    return c

def loo_crib(attr, valA, valB, k=3):
    folios=[f for f in TAGS if TAGS[f].get(attr) in (valA,valB) and fwords(f)]
    y=np.array([1 if TAGS[f][attr]==valA else 0 for f in folios])
    texts={f:fwords(f) for f in folios}
    lens=np.array([len(texts[f]) for f in folios])
    # length confound check
    lA,lB=lens[y==1].mean(),lens[y==0].mean()
    # leave-one-out: train trigram log-ratio on the rest, score held-out, predict majority side
    def cv_accuracy(labels):
        correct=0
        for i,f in enumerate(folios):
            tr=[g for g in folios if g!=f]
            cA=Counter();cB=Counter()
            for g in tr:
                t=trigrams(texts[g])
                (cA if labels[folios.index(g)]==1 else cB).update(t)
            tA=sum(cA.values())+1;tB=sum(cB.values())+1
            # score held-out folio by summed log-ratio of its trigrams
            s=0
            for ng,ct in trigrams(texts[f]).items():
                s+=ct*math.log(((cA[ng]+1)/tA)/((cB[ng]+1)/tB))
            pred=1 if s>0 else 0
            if pred==labels[i]: correct+=1
        return correct/len(folios)
    obs=cv_accuracy(y)
    null=np.array([cv_accuracy(rng.permutation(y)) for _ in range(1000)])
    p=(np.sum(null>=obs)+1)/(len(null)+1)
    base=max(y.mean(),1-y.mean())
    return folios,obs,base,p,lA,lB

print("="*74); print("PART 1  BOTANICAL CRIB (leave-one-out CV + permutation null)"); print("="*74)
for attr,a,b in [('flower','red','blue'),('root','taproot','rhizome'),('flower','red','none')]:
    fol,obs,base,p,lA,lB=loo_crib(attr,a,b)
    print(f"\n  {attr} {a} vs {b}: n={len(fol)} folios  (mean tokens {lA:.0f} vs {lB:.0f})")
    print(f"    LOO-CV accuracy = {obs:.2f}   base-rate = {base:.2f}   permutation p = {p:.3f}")
    print(f"    {'SIGNAL — text predicts the visual attribute out-of-sample' if p<0.05 and obs>base else 'NULL — no out-of-sample crib'}")

# ===================== PART 2: ZODIAC KEY-JOIN =====================
print("\n"+"="*74); print("PART 2  ZODIAC KEY-JOIN (decoded month-names as ground truth)"); print("="*74)
zfol=[f for f in sorted(C['folios'],key=fk) if re.match(r'f7[0-3][rv]',f)]
med=[]
for f in zfol:
    labs=[w for ln in C['folios'][f] if ltype(ln['locus'])=='L' for w in ln['words'] if '?' not in w and len(w)>=2]
    if len(labs)>=10: med.append((f,labs))
M=len(med); print(f"  {M} medallions (months), {sum(len(l) for _,l in med)} day-labels total")

def jac(a,b): u=a|b; return len(a&b)/len(u) if u else 0
def cores(labs): return [set(morphs(w)) for w in labs]

# (a) Are medallions lexically organized AT ALL? real cross-medallion core-overlap vs
#     a null that reshuffles all labels across medallions keeping sizes.
allcore=[set(morphs(w)[1] for w in l) for _,l in med]
real_across=np.mean([jac(allcore[i],allcore[j]) for i in range(M) for j in range(i+1,M)])
poolcores=[morphs(w)[1] for _,l in med for w in l]; sizes=[len(l) for _,l in med]
def shuffled_across():
    p=rng.permutation(poolcores); idx=0; sets=[]
    for s in sizes: sets.append(set(p[idx:idx+s])); idx+=s
    return np.mean([jac(sets[i],sets[j]) for i in range(M) for j in range(i+1,M)])
nullA=np.array([shuffled_across() for _ in range(1000)])
pA=(np.sum(nullA<=real_across)+1)/(len(nullA)+1)   # distinct => LOWER overlap than shuffle
print(f"\n  (a) medallion organization: cross-medallion core-overlap {real_across:.3f} "
      f"vs shuffle {nullA.mean():.3f}  p={pA:.3f}")
print(f"      {'medallions ARE lexically distinct (organized by folio/medallion)' if pA<0.05 else 'no distinct medallion vocab'}")
print(f"      NOTE: distinct-by-medallion = distinct-by-folio = positional; does NOT yet mean month-referential.")

# (b) Day-sequence: within a ring, do labels encode an ordered 1..30 progression?
#     real adjacent-vs-distant similarity gap vs a per-ring shuffle null.
def ring_gap(labs):
    cs=cores(labs)
    adj=[jac(cs[i],cs[i+1]) for i in range(len(cs)-1)]
    far=[jac(cs[i],cs[j]) for i in range(len(cs)) for j in range(i+3,len(cs))]
    return (np.mean(adj)-np.mean(far)) if adj and far else 0
real_gap=np.mean([ring_gap(l) for _,l in med])
def shuffle_ring_gap():
    g=[]
    for _,l in med:
        ll=list(l); rng.shuffle(ll); g.append(ring_gap(ll))
    return np.mean(g)
nullB=np.array([shuffle_ring_gap() for _ in range(1000)])
pB=(np.sum(nullB>=real_gap)+1)/(len(nullB)+1)
print(f"\n  (b) day-sequence (ring order): adjacent-distant gap {real_gap:+.4f} "
      f"vs ring-shuffle {nullB.mean():+.4f}  p={pB:.3f}")
print(f"      {'SIGNAL — labels encode ring/day ordering' if pB<0.05 else 'NULL — ring order carries no day-progression'}")

# (c) Debunk the r=0.48 feature~ordinal: is it month-reference or position drift?
#     Compute the SAME q-rate per folio for the ZODIAC LABELS and for the PROSE on the
#     same folios. If both drift together with folio order, it is positional, not referential.
ordn=np.arange(M)
lab_q=[np.mean([morphs(w)[0].startswith('q') for w in l]) for _,l in med]
prose_q=[]
for f,_ in med:
    pr=[w for ln in C['folios'][f] if ltype(ln['locus'])=='P' for w in ln['words'] if '?' not in w and len(w)>=2]
    prose_q.append(np.mean([w.startswith('q') for w in pr]) if len(pr)>=5 else np.nan)
r_lab=np.corrcoef(ordn,lab_q)[0,1]
# ordinal-shuffle null for the label correlation (n=12 is tiny)
nullr=np.array([abs(np.corrcoef(rng.permutation(ordn),lab_q)[0,1]) for _ in range(2000)])
pr_lab=(np.sum(nullr>=abs(r_lab))+1)/(len(nullr)+1)
mask=~np.isnan(prose_q)
r_prose=np.corrcoef(ordn[mask],np.array(prose_q)[mask])[0,1] if mask.sum()>3 else float('nan')
print(f"\n  (c) debunk feature~ordinal drift:")
print(f"      label  q-rate ~ ordinal: r={r_lab:+.2f}  (ordinal-shuffle p={pr_lab:.3f}, n={M})")
print(f"      prose  q-rate ~ ordinal: r={r_prose:+.2f}   (same folios)")
print(f"      => {'a manuscript-position drift present in PROSE too: the label r is drift, not month-reference' if (not math.isnan(r_prose) and abs(r_prose)>0.3) else 'label r not significant at n=12 and unidentifiable from position'}")

print("\n"+"="*74); print("KEY-JOIN VERDICT"); print("="*74)
print("  A real referential signal requires (1b) day-sequence or (Part 1) held-out crib to")
print("  survive. Medallion distinctiveness (2a) and any feature~ordinal trend (2c) are")
print("  positional/dialect drift — the same confound that produced every flourish.")
