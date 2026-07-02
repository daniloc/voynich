"""
THE GENEALOGY HYPOTHESIS — discriminating tests.
A genealogy is a relational DB over time (fits everything we found), but to PRIVILEGE
it over other databases we need its UNIQUE signatures:
  T1 LINEAGE NAMING: a family shares a name-STEM; members differ by an individual
     marker. If folios=families, labels(=members) on a page should share a CORE more
     than chance (controlling for section dialect). The genealogy's strongest tell.
  T2 BEGAT-CHAINING: genealogies chain ("Y begat Z; Z begat ..."). Does an entity in
     line N recur as the SUBJECT (opener) of line N+1 above chance?
  T3 FAMILIES-ON-PAGES: do entities that wear MANY affix-variants (a stem with many
     members) CONCENTRATE on few folios (a family on its page) rather than spread?
If T1-T3 are positive beyond the section confound, genealogy is specifically indicated,
not merely permitted.
"""
import json, re, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(41)
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
def section(f):
    n=int(re.match(r'f(\d+)',f).group(1))
    return ('herbal' if n<=66 else 'astro' if n<=73 else 'bio' if n<=84 else 'cosmo' if n<=86 else 'pharma' if n<=102 else 'recipes')
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph']
SUF=['eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y']
def core(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return r[:len(r)-len(s)] if s else r

# labels (=candidate "members") per folio
labf=defaultdict(list); allwf=defaultdict(list)
for f,L in C['folios'].items():
    for ln in L:
        t=ltype(ln['locus'])
        ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
        if t=='L': labf[f]+=ws
        allwf[f]+=ws

print("="*68); print("T1 LINEAGE NAMING — do labels on one folio share a name-stem?"); print("="*68)
def pair_share(items):
    cs=[core(w) for w in items];
    if len(cs)<2: return None
    same=tot=0
    for i in range(len(cs)):
        for j in range(i+1,len(cs)):
            tot+=1; same+= (cs[i]==cs[j])
    return same/tot
units=[(f,labf[f]) for f in labf if len(labf[f])>=5]
within=np.mean([pair_share(ws) for f,ws in units if pair_share(ws) is not None])
# null: random labels from the SAME section, same group sizes
bysec=defaultdict(list)
for f,ws in units: bysec[section(f)]+=ws
nulls=[]
for _ in range(300):
    vals=[]
    for f,ws in units:
        pool=bysec[section(f)]; samp=list(rng.choice(pool,size=len(ws),replace=False)) if len(pool)>=len(ws) else ws
        v=pair_share(samp);  vals.append(v) if v is not None else None
    nulls.append(np.mean(vals))
nulls=np.array(nulls); p=(np.sum(nulls>=within)+1)/(len(nulls)+1)
print(f"  within-folio label core-sharing: {within:.3f}")
print(f"  null (random same-section labels): {nulls.mean():.3f}  p={p:.4f}")
print("  => within >> null = members on a page share a stem = LINEAGE NAMING")

print("\n"+"="*68); print("T2 BEGAT-CHAINING — entity in line N recurs as opener of line N+1?"); print("="*68)
def fk(f): m=re.match(r'f(\d+)([rv])(\d*)',f); return (int(m.group(1)),0 if m.group(2)=='r' else 1,int(m.group(3) or 0))
chain=tot=0; lines=[]
for f in sorted(C['folios'],key=fk):
    for ln in C['folios'][f]:
        if ltype(ln['locus'])=='P':
            ws=[core(w) for w in ln['words'] if '?' not in w and len(w)>=2]
            if ws: lines.append(set(ws[:-0] or ws)) if False else lines.append((ws))
for a,b in zip(lines,lines[1:]):
    tot+=1; chain+= (b[0] in set(a))     # opener of next ∈ previous line
# shuffle null
op=[l[0] for l in lines]; bodies=[set(l) for l in lines]
sh=op[:]; rng.shuffle(sh)
chain_sh=sum(1 for o,bod in zip(sh[1:],bodies[:-1]) if o in bod)/ (len(lines)-1)
print(f"  next-opener ∈ previous line: real {chain/tot:.3f}  vs shuffled {chain_sh:.3f}")
print("  => real >> shuffled = descent chaining (the 'begat' rhythm)")

print("\n"+"="*68); print("T3 FAMILIES-ON-PAGES — do multi-member stems concentrate on few folios?"); print("="*68)
core_fol=defaultdict(Counter); core_affix=defaultdict(set)
for f,L in C['folios'].items():
    for ln in L:
        for w in ln['words']:
            if '?' in w or len(w)<2: continue
            c=core(w); core_fol[c][f]+=1
            p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
            s=next((x for x in sorted(SUF,key=len,reverse=True) if w.endswith(x) and len(w)>len(x)),'')
            core_affix[c].add((p,s))
def conc(c):  # fraction of a core's tokens on its single top folio
    cc=core_fol[c]; n=sum(cc.values()); return cc.most_common(1)[0][1]/n
multi=[c for c in core_affix if len(core_affix[c])>=8 and sum(core_fol[c].values())>=15]
mono=[c for c in core_affix if len(core_affix[c])<=2 and sum(core_fol[c].values())>=15]
print(f"  high-affix stems (>=8 variants): folio-concentration {np.mean([conc(c) for c in multi]):.3f} (n={len(multi)})")
print(f"  low-affix  stems (<=2 variants): folio-concentration {np.mean([conc(c) for c in mono]):.3f} (n={len(mono)})")
print("  => if high-affix stems are MORE folio-concentrated, families sit on pages.")
