"""
THE CONTACT TEST — does the open key close when you rotate to the column axis?

Hadden move (the film): the data was complete; the wrong DIMENSIONALITY was imposed
on it. Here D2 says within-line word order carries ~0 info ("bag of records") — read
as a pointer, that says the left-to-right axis is wrong and the real unit may run
ACROSS records (down columns), not along them.

D3 found the core morpheme is an OPEN key: ~3000 values, 6.8 bits, refuses to close.
Hypothesis: it closes once you condition on COLUMN position (word-index-within-line).

Spec §5 arbiters built in:
  (1) beat a matched shuffle null (position<->core association destroyed), and
  (2) GENERALIZE out-of-sample (train column-vocab on half the folios, predict the
      other half better than a position-agnostic model).
Confound #1 (Currier dialect) killed by running inside one Currier x section stratum.
Discriminator vs the content-free/template story: the SAME pipeline is run on the
closed enums (prefix, suffix) — known to carry positional template effects — so we can
see whether the OPEN KEY closes BEYOND the template's positional signature.
"""
import json, math, re
import numpy as np
from collections import Counter, defaultdict

rng = np.random.default_rng(0)

# --- slot decomposition (same convention as columns.py / D3) ---
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph','cfh']
SUF=['eedy','eody','edy','aiin','aiir','ain','iin','dy','ol','or','ar','al','am','dam','ey','eey','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s

d=json.load(open('data/corpus/body_spatial.json'))
toks=d['tokens']

def collabel(idx,n):
    if n>1 and idx==n-1: return 'last'
    return 'p%d'%idx if idx<5 else 'mid'

def build(section, currier):
    """rows: list of (folio, [(prefix,core,suffix,collabel), ...]) for paragraph lines."""
    rows=defaultdict(list)
    for t in toks:
        if t['section']!=section or t['currier']!=currier: continue
        if t.get('role')!='paragraph': continue
        if t['uncertain'] or '?' in t['word'] or len(t['word'])<2: continue
        rows[(t['folio'],t['line'])].append(t)
    out=[]
    for (folio,line),g in rows.items():
        g=sorted(g,key=lambda x:x['idx']); n=len(g)
        if n<4: continue
        cells=[(*decomp(x['word']), collabel(i,n)) for i,x in enumerate(g)]
        out.append((folio,cells))
    return out

def H(counter):
    n=sum(counter.values())
    return -sum(v/n*math.log2(v/n) for v in counter.values()) if n else 0.0

def cond_H(pairs):
    """H(field | column) weighted; pairs = [(field, col), ...]"""
    bycol=defaultdict(Counter)
    for f,c in pairs: bycol[c][f]+=1
    N=sum(sum(c.values()) for c in bycol.values())
    return sum(sum(c.values())/N * H(c) for c in bycol.values())

def ig_with_null(pairs, reps=300):
    fields=[f for f,_ in pairs]; cols=[c for _,c in pairs]
    h0=H(Counter(fields)); hc=cond_H(pairs); ig=h0-hc
    nulls=np.empty(reps)
    cols=np.array(cols,dtype=object)
    for r in range(reps):
        perm=rng.permutation(cols)
        nulls[r]=h0-cond_H(list(zip(fields,perm)))
    z=(ig-nulls.mean())/(nulls.std()+1e-12)
    p=(np.sum(nulls>=ig)+1)/(reps+1)
    return h0,hc,ig,nulls.mean(),z,p

def heldout(rows, field_ix):
    """Cross-entropy of test cells under column-conditioned vs position-agnostic
    train model, with shared smoothed vocab. Lower = better prediction.
    Returns (H_global, H_bycol, improvement) on held-out folios."""
    folios=sorted({f for f,_ in rows})
    test_f=set(folios[::2]); # even-indexed folios held out
    train=[(f,cells) for f,cells in rows if f not in test_f]
    test =[(f,cells) for f,cells in rows if f in test_f]
    def cells_of(split): return [cell for _,cl in split for cell in cl]
    tr=cells_of(train); te=cells_of(test)
    V=len({c[field_ix] for c in tr}); a=0.5
    # global model
    gc=Counter(c[field_ix] for c in tr); gn=sum(gc.values())
    def pg(x): return (gc.get(x,0)+a)/(gn+a*(V+1))
    # column model
    cc=defaultdict(Counter)
    for c in tr: cc[c[3]][c[field_ix]]+=1
    cn={k:sum(v.values()) for k,v in cc.items()}
    def pc(x,col):
        c=cc.get(col)
        if not c: return pg(x)
        return (c.get(x,0)+a)/(cn[col]+a*(V+1))
    Hg=-np.mean([math.log2(pg(c[field_ix])) for c in te])
    Hc=-np.mean([math.log2(pc(c[field_ix],c[3])) for c in te])
    return Hg,Hc,Hg-Hc

FIELDS={'prefix':0,'core':1,'suffix':2}

for section,currier,tag in [('herbal','A','herbal/A'),('stars_recipes','B','recipes/B'),
                            ('biological','B','bio/B'),('herbal','B','herbal/B')]:
    rows=build(section,currier)
    ncells=sum(len(c) for _,c in rows)
    print("="*74)
    print(f"STRATUM {tag}  | {len(rows)} paragraph lines, {ncells} words, {len({f for f,_ in rows})} folios")
    print("="*74)
    if ncells<800:
        print("  (too small, skipped)\n"); continue
    print(f"  {'field':7s} {'H(f)':>6} {'H(f|col)':>9} {'IG':>6} {'null_IG':>8} {'z':>6} {'p':>7}   |  held-out CV: {'H_glob':>7} {'H_col':>7} {'Δ(gen)':>7}")
    for name,ix in FIELDS.items():
        pairs=[(c[ix],c[3]) for _,cl in rows for c in cl]
        h0,hc,ig,nmean,z,p=ig_with_null(pairs)
        Hg,Hc,gen=heldout(rows,ix)
        flag = '  <==' if (name=='core' and p<0.01 and gen>0.05) else ''
        print(f"  {name:7s} {h0:6.2f} {hc:9.2f} {ig:6.3f} {nmean:8.3f} {z:6.1f} {p:7.4f}   |              {Hg:7.2f} {Hc:7.2f} {gen:+7.3f}{flag}")
    # closure headline: distinct cores per column
    bycol=defaultdict(set); tot=set()
    for _,cl in rows:
        for c in cl: bycol[c[3]].add(c[1]); tot.add(c[1])
    order=['p0','p1','p2','p3','p4','mid','last']
    print(f"  open-key closure (distinct CORE values): global={len(tot)} | " +
          " ".join(f"{k}:{len(bycol[k])}" for k in order if k in bycol))
    print()

print("READ: a real Contact-style closure = CORE shows p<0.01 vs shuffle AND a")
print("positive held-out Δ(gen) that is LARGE relative to prefix/suffix (the template")
print("floor). If core's Δ(gen) merely tracks the closed enums, the column axis is")
print("just re-expressing known positional template structure — another content-free null.")
