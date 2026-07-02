"""
STRUCTURE ATLAS — operationalizing "reveal as much structure as possible, find edges".
A structural dimension has a LOCATION (which level controls it) and EDGES (where the
rule breaks). We do three things:
  A. COVERAGE — how much of the corpus does the slot grammar explain? The residual
     (unparseable words) is an edge-set: errors, foreign matter, or hidden signal.
  B. WHERE THE HINGE LIVES — the prefix is the register hinge. Condition prefix-choice
     on register / line-position / section; whichever collapses its entropy most is
     where the structure 'lives'.
  C. EDGE MAP — per-folio register score along the manuscript; sharp jumps = edges.
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
C=json.load(open('data/corpus/corpus.json')); META=C['meta']
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph','cfh','s','p','t','k','f','r','l']
SUF=['eedy','eody','edy','aiin','aiir','ain','iin','dy','ol','or','ar','al','am','dam','ey','eey','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s
def H(items):
    c=Counter(items); n=sum(c.values()); return -sum(v/n*math.log2(v/n) for v in c.values())
def condH(pairs):  # H(x|y) = sum_y p(y) H(x|y)
    by=defaultdict(list)
    for x,y in pairs: by[y].append(x)
    n=sum(len(v) for v in by.values())
    return sum(len(v)/n*H(v) for v in by.values())

# gather tokens with context
toks=[]  # (word, register, posbucket, section, folio)
def section(f):
    n=int(re.match(r'f(\d+)',f).group(1))
    return ('herbal' if n<=66 else 'astro' if n<=73 else 'bio' if n<=84 else 'cosmo' if n<=86 else 'pharma' if n<=102 else 'recipes')
for f,lines in C['folios'].items():
    for ln in lines:
        reg=ltype(ln['locus'])
        ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
        for i,w in enumerate(ws):
            pb='init' if i==0 else 'final' if i==len(ws)-1 else 'mid'
            toks.append((w, 'label' if reg=='L' else 'prose' if reg=='P' else 'other', pb, section(f), f))

print("="*70)
print("A. COVERAGE — how much does the slot grammar explain?")
print("="*70)
N=len(toks)
parsed_full=sum(1 for w,*_ in toks if decomp(w)[0] and decomp(w)[2])     # has prefix AND suffix
parsed_any =sum(1 for w,*_ in toks if decomp(w)[0] or decomp(w)[2])      # has at least one affix
resid=[w for w,*_ in toks if not (decomp(w)[0] or decomp(w)[2])]
print(f"  tokens={N}")
print(f"  prefix+core+suffix parse : {100*parsed_full/N:.1f}%")
print(f"  >=1 affix                : {100*parsed_any/N:.1f}%")
print(f"  RESIDUAL (no affix)      : {100*len(resid)/N:.1f}%  ({len(set(resid))} distinct)")
print(f"  residual sample (edges)  : {[w for w,_ in Counter(resid).most_common(12)]}")

print("\n" + "="*70)
print("B. WHERE THE HINGE LIVES — what collapses prefix entropy?")
print("="*70)
prose=[(decomp(w)[0], reg, pb, sec) for w,reg,pb,sec,_ in toks]
H0=H([p for p,_,_,_ in prose])
Hreg=condH([(p,reg) for p,reg,_,_ in prose])
Hpos=condH([(p,pb)  for p,_,pb,_ in prose])
Hsec=condH([(p,sec) for p,_,_,sec in prose])
Hcore=condH([(decomp(w)[0],decomp(w)[1]) for w,*_ in toks])  # prefix given the stem it attaches to
print(f"  H(prefix)              = {H0:.3f} bits")
for name,h in [('| register',Hreg),('| line-position',Hpos),('| section',Hsec),('| core/stem',Hcore)]:
    print(f"  H(prefix {name:15s}) = {h:.3f}   reduces {100*(H0-h)/H0:4.1f}%")
print("  => the dimension with the biggest reduction is where the hinge structure lives.")

print("\n" + "="*70)
print("C. EDGE MAP — register score along the manuscript (find the seams)")
print("="*70)
def folio_key(f):
    m=re.match(r'f(\d+)([rv])(\d*)',f);
    return (int(m.group(1)), 0 if m.group(2)=='r' else 1, int(m.group(3) or 0))
fols=sorted(C['folios'], key=folio_key)
# register score per folio = fraction of PROSE tokens with q-onset (the prose hinge)
score={}
for f in fols:
    ps=[w for ln in C['folios'][f] if ltype(ln['locus'])=='P' for w in ln['words'] if '?' not in w]
    if len(ps)<20: continue
    score[f]=sum(1 for w in ps if w.startswith('q'))/len(ps)
ks=list(score); vals=np.array([score[f] for f in ks])
# sparkline + change points (where successive folios jump > 1.5*std of diffs)
spark='▁▂▃▄▅▆▇█'
lo,hi=vals.min(),vals.max()
line=''.join(spark[min(7,int((score[f]-lo)/(hi-lo+1e-9)*7))] for f in ks)
print(f"  q-onset rate across {len(ks)} folios (each char = one folio, ▁ low … █ high):")
for i in range(0,len(line),60): print("   ",line[i:i+60])
d=np.abs(np.diff(vals)); thr=d.mean()+1.2*d.std()
edges=[(ks[i],ks[i+1],vals[i],vals[i+1]) for i in range(len(d)) if d[i]>thr]
print(f"\n  sharpest register EDGES (jumps in q-rate, |Δ|>{thr:.2f}):")
for a,b,va,vb in edges[:10]:
    print(f"    {a:7s}->{b:7s}  {va:.2f} -> {vb:.2f}  (L={META.get(a,{}).get('L','?')}->{META.get(b,{}).get('L','?')})")
