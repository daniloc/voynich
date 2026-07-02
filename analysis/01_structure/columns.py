"""
WHAT ARE THE COLUMNS? and what else does layout-position fix?
Hypothesis: the record is the WORD, with three columns prefix|core|suffix =
attribute|entity|attribute. A database column = a CLOSED small domain; a key = a
large OPEN domain; columns are roughly INDEPENDENT. Tests:
  A. FIELD CARDINALITY & CLOSURE — are prefix/suffix small closed enums and core the
     large open key?
  B. FIELD INDEPENDENCE — mutual information between the three columns (low = orthogonal
     fields, like real table columns).
  C. POSITIONAL INVARIANTS — which field-values are locked by line/word position
     (beyond the daiin row-marker): word-length, gallows, prefix, suffix by position.
  D. CONSERVED WIDTH — do lines fill to a fixed glyph-width (justification invariant):
     does word-count trade off against word-length within a line?
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph','cfh']
SUF=['eedy','eody','edy','aiin','aiir','ain','iin','dy','ol','or','ar','al','am','dam','ey','eey','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s
GALLOWS=set('ktpf')
def has_gallows(w): return any(g in w for g in GALLOWS)

lines=[]
for f,L in C['folios'].items():
    for ln in L:
        if ltype(ln['locus'])!='P': continue
        ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
        if len(ws)>=4: lines.append(ws)
allw=[w for ln in lines for w in ln]
D=[decomp(w) for w in allw]
N=len(allw)

def H(c):
    n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values()) if n else 0
def field_report(name, vals):
    c=Counter(vals); n=sum(c.values())
    top10=sum(v for _,v in c.most_common(10))/n
    print(f"  {name:8s}: {len(c):4d} distinct | top-10 cover {100*top10:3.0f}% | entropy {H(c):.2f} bits | top: {[x for x,_ in c.most_common(6)]}")

print("="*70); print("A. THE THREE COLUMNS — cardinality & closure"); print("="*70)
field_report("prefix", [p for p,_,_ in D])
field_report("core",   [c for _,c,_ in D])
field_report("suffix", [s for _,_,s in D])
print("  => small closed enums (prefix,suffix) + one large open key (core) = a record.")

print("\n"+"="*70); print("B. COLUMN INDEPENDENCE — mutual information between fields"); print("="*70)
def MI(xs,ys):
    n=len(xs); jx=Counter(xs); jy=Counter(ys); jj=Counter(zip(xs,ys))
    return sum(c/n*math.log2((c/n)/((jx[x]/n)*(jy[y]/n))) for (x,y),c in jj.items())
pre=[p for p,_,_ in D]; cor=[c for _,c,_ in D]; suf=[s for _,_,s in D]
print(f"  I(prefix;suffix) = {MI(pre,suf):.3f} bits  (low = orthogonal columns)")
print(f"  I(prefix;core)   = {MI(pre,cor):.3f} bits")
print(f"  I(core;suffix)   = {MI(cor,suf):.3f} bits")

print("\n"+"="*70); print("C. POSITIONAL INVARIANTS — what line-position fixes"); print("="*70)
prof=defaultdict(lambda: defaultdict(list))
for ln in lines:
    n=len(ln)
    for i,w in enumerate(ln):
        pos='1st' if i==0 else 'last' if i==n-1 else f'{i+1}' if i<3 else 'mid'
        p,c,s=decomp(w)
        prof[pos]['len'].append(len(w)); prof[pos]['gal'].append(has_gallows(w))
        prof[pos]['q'].append(p.startswith('q')); prof[pos]['hassuf'].append(bool(s)); prof[pos]['haspre'].append(bool(p))
print(f"  {'pos':4s}{'len':>6}{'gallows%':>9}{'q-pre%':>8}{'has-suf%':>9}{'has-pre%':>9}")
for pos in ['1st','2','3','mid','last']:
    d=prof[pos]
    print(f"  {pos:4s}{np.mean(d['len']):>6.2f}{100*np.mean(d['gal']):>9.0f}{100*np.mean(d['q']):>8.0f}{100*np.mean(d['hassuf']):>9.0f}{100*np.mean(d['haspre']):>9.0f}")
print("  => columns whose values shift with position are LAYOUT-locked invariants.")

print("\n"+"="*70); print("D. CONSERVED WIDTH — justification invariant"); print("="*70)
wc=np.array([len(ln) for ln in lines])
gpl=np.array([sum(len(w) for w in ln) for ln in lines])
mwl=np.array([np.mean([len(w) for w in ln]) for ln in lines])
print(f"  glyphs/line: mean {gpl.mean():.0f} CV {gpl.std()/gpl.mean():.2f}   words/line: mean {wc.mean():.1f} CV {wc.std()/wc.mean():.2f}")
print(f"  corr(word-count, mean-word-length) per line = {np.corrcoef(wc,mwl)[0,1]:+.2f}")
print("    (strong negative => words shrink as count grows => lines fill to a WIDTH)")
# is total glyphs more conserved than words? ratio of CVs
print(f"  CV(glyphs)/CV(words) = {(gpl.std()/gpl.mean())/(wc.std()/wc.mean()):.2f}  (<1 => width better conserved than count)")
