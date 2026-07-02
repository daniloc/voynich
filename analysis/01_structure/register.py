"""
HOW DO THE TWO PROCESSES RELATE?
Prose and labels differ in their statistics. But do they share a LEXICON?
Decompose every word into prefix+core+suffix (the slot grammar). Then:
  1. SUBSTRATE: do labels and prose draw CORES from the same pool? (shared lexicon)
  2. REGISTER : same cores but different AFFIX weighting? (one system, two registers)
  3. OPERATOR : is there a rule T such that label ~ T(prose word)? (label derives from prose)
  4. LOCALITY : are a folio's labels closer to its OWN prose than to other folios'?
If cores are shared but affixes differ -> two registers of ONE generator, and we can
write the transform that maps one to the other. That transform IS the relationship.
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(1989)
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'

PREFIXES=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph','cfh']
SUFFIXES=['eedy','eody','edy','aiin','aiir','ain','iin','dy','ol','or','ar','al','am','dam','ey','eey','y']
def decomp(w):
    pre=next((p for p in sorted(PREFIXES,key=len,reverse=True) if w.startswith(p)),'')
    rem=w[len(pre):]
    suf=next((s for s in sorted(SUFFIXES,key=len,reverse=True) if rem.endswith(s) and len(rem)>len(s)),'')
    core=rem[:len(rem)-len(suf)] if suf else rem
    return pre,core,suf

labels=[]; prose=[]
lab_by_fol=defaultdict(list); pro_by_fol=defaultdict(list)
for f,lines in C['folios'].items():
    for ln in lines:
        t=ltype(ln['locus']); ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
        if t=='L': labels+=ws; lab_by_fol[f]+=ws
        elif t=='P': prose+=ws; pro_by_fol[f]+=ws

L=[decomp(w) for w in labels]; P=[decomp(w) for w in prose]
Lcore=Counter(c for _,c,_ in L); Pcore=Counter(c for _,c,_ in P)

print("="*70)
print("1. SUBSTRATE — do labels and prose share a CORE inventory?")
print("="*70)
shared=set(Lcore)&set(Pcore)
covl=sum(Lcore[c] for c in shared)/sum(Lcore.values())
print(f"  distinct cores: labels {len(Lcore)}, prose {len(Pcore)}, shared {len(shared)}")
print(f"  Jaccard of core sets         : {len(shared)/len(set(Lcore)|set(Pcore)):.3f}")
print(f"  label core TOKENS also in prose: {covl:.3f}  (high => shared lexicon)")

print("\n" + "="*70)
print("2. REGISTER — same cores, but different AFFIX weighting?")
print("="*70)
def dist(items):
    c=Counter(items); n=sum(c.values()); return {k:v/n for k,v in c.items()}
def kl(p,q):
    keys=set(p)|set(q); return sum(p.get(k,1e-9)*math.log2(p.get(k,1e-9)/q.get(k,1e-9)) for k in keys)
Lpre,Ppre=dist([p for p,_,_ in L]),dist([p for p,_,_ in P])
Lsuf,Psuf=dist([s for _,_,s in L]),dist([s for _,_,s in P])
print(f"  prefix-distribution KL(label||prose): {kl(Lpre,Ppre):.2f} bits")
print(f"  suffix-distribution KL(label||prose): {kl(Lsuf,Psuf):.2f} bits")
def topdiff(Ld,Pd,k=6):
    return sorted(((a, Ld.get(a,0)-Pd.get(a,0)) for a in set(Ld)|set(Pd)), key=lambda x:-x[1])[:k]
print(f"  most LABEL-enriched prefixes: {[(a,round(d,2)) for a,d in topdiff(Lpre,Ppre)]}")
print(f"  most LABEL-enriched suffixes: {[(a,round(d,2)) for a,d in topdiff(Lsuf,Psuf)]}")
print(f"  most PROSE-enriched prefixes: {[(a,round(-d,2)) for a,d in topdiff(Ppre,Lpre)]}")

print("\n" + "="*70)
print("3. OPERATOR — is a label ~ a prose word with affixes swapped?")
print("="*70)
# for each distinct label, is there a prose word sharing its CORE? if so, labels are
# prose stems re-dressed (a derivational relationship, not an independent vocabulary)
labset=set(labels); proset=set(prose)
prose_cores=set(Pcore)
share_core=sum(1 for w in labset if decomp(w)[1] in prose_cores)
exact=len(labset&proset)
print(f"  distinct labels: {len(labset)}")
print(f"  label appears verbatim in prose      : {exact} ({100*exact/len(labset):.0f}%)")
print(f"  label CORE appears in prose (re-dressed): {share_core} ({100*share_core/len(labset):.0f}%)")
print("  => high re-dressed % with low verbatim % = SAME stems, DIFFERENT affixes")
print("     i.e. labels are a derivational register of the prose lexicon.")

print("\n" + "="*70)
print("4. LOCALITY — are a folio's labels tied to its OWN prose?")
print("="*70)
# Jaccard(label-cores of folio f, prose-cores of folio f) vs vs other folios
fols=[f for f in lab_by_fol if len(lab_by_fol[f])>=5 and len(pro_by_fol.get(f,[]))>=20]
def cores(ws): return set(decomp(w)[1] for w in ws)
def jac(a,b): u=a|b; return len(a&b)/len(u) if u else 0
own=[]; other=[]
for f in fols:
    lc=cores(lab_by_fol[f])
    own.append(jac(lc, cores(pro_by_fol[f])))
    others=[g for g in fols if g!=f]
    other.append(np.mean([jac(lc, cores(pro_by_fol[g])) for g in rng.choice(others,size=min(8,len(others)),replace=False)]))
own=np.array(own); other=np.array(other)
# paired permutation
diff=own-other; obs=diff.mean()
perm=np.array([ (diff*rng.choice([-1,1],size=len(diff))).mean() for _ in range(5000)])
p=(np.sum(perm>=obs)+1)/(len(perm)+1)
print(f"  folios tested: {len(fols)}")
print(f"  mean label-core ∩ OWN prose : {own.mean():.3f}")
print(f"  mean label-core ∩ OTHER prose: {other.mean():.3f}")
print(f"  paired diff {obs:+.3f}  permutation p={p:.4f}")
print("  =>", "labels are tied to their OWN page's prose (local derivation)" if p<0.05
      else "labels relate to prose globally, not page-locally")
