"""
GENERATIVE FIT. Build the production model implied by our schema and see if it
regenerates ALL the manuscript's statistics. If a schema-sampler + a copying step
reproduces everything (incl. burstiness), then the structure needs no meaning to
exist -> 'generated'. Whatever it CAN'T reproduce is residual where meaning could live.
  Model A: independent record sampling  (core ~ P(core); prefix~P(pre|core); suf~P(suf|core))
  Model B: A + SELF-CITATION (with prob p, re-emit a recent word, maybe 1-glyph mutated)
Compare real vs A vs B on: Zipf, word-length, hapax, char cond-entropy, adjacent-repeat,
and BURSTINESS (the pro-language signal — does copying alone reproduce it?).
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
rng=np.random.default_rng(31)
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph']
SUF=['eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s
def fk(f): m=re.match(r'f(\d+)([rv])(\d*)',f); return (int(m.group(1)),0 if m.group(2)=='r' else 1,int(m.group(3) or 0))
real=[w for f in sorted(C['folios'],key=fk) for ln in C['folios'][f] if ltype(ln['locus'])=='P' for w in ln['words'] if '?' not in w and len(w)>=2]
N=len(real)

# learn the schema
corefreq=Counter(decomp(w)[1] for w in real)
pre_g=defaultdict(Counter); suf_g=defaultdict(Counter)
for w in real:
    p,c,s=decomp(w); pre_g[c][p]+=1; suf_g[c][s]+=1
cores=list(corefreq); cw=np.array([corefreq[c] for c in cores],float); cw/=cw.sum()
glyphs=sorted(set(ch for w in real for ch in w))
def sample_word():
    c=cores[rng.choice(len(cores),p=cw)]
    p=rng.choice(list(pre_g[c]),p=np.array(list(pre_g[c].values()))/sum(pre_g[c].values()))
    s=rng.choice(list(suf_g[c]),p=np.array(list(suf_g[c].values()))/sum(suf_g[c].values()))
    return p+c+s

def gen_A(n): return [sample_word() for _ in range(n)]
def gen_B(n, p_copy=0.18, window=30, p_mut=0.4):
    out=[]
    for _ in range(n):
        if out and rng.random()<p_copy:
            w=out[-rng.integers(1,min(window,len(out))+1)]
            if rng.random()<p_mut and len(w)>1:
                i=rng.integers(0,len(w)); w=w[:i]+rng.choice(glyphs)+w[i+1:]
            out.append(w)
        else: out.append(sample_word())
    return out

# stat battery
def zipf(ws):
    fr=sorted(Counter(ws).values(),reverse=True); x=np.log(np.arange(1,len(fr)+1)); y=np.log(fr)
    return np.polyfit(x,y,1)[0]
def wlen(ws): L=np.array([len(w) for w in ws]); return L.mean(), L.var()/L.mean()
def hapax(ws): c=Counter(ws); return sum(1 for v in c.values() if v==1)/len(c)
def adjrep(ws): return np.mean([a==b for a,b in zip(ws,ws[1:])])
def h2(ws):
    t=' '.join(ws); bi=Counter(zip(t,t[1:])); uni=Counter(t[:-1]); n=sum(bi.values())
    hj=-sum(v/n*math.log2(v/n) for v in bi.values()); hu=-sum(v/n*math.log2(v/n) for v in uni.values())
    return hj-hu
def burst(ws,mink=10):
    pos=defaultdict(list)
    for i,w in enumerate(ws): pos[w].append(i)
    Bs=[(np.diff(p).std()-np.diff(p).mean())/(np.diff(p).std()+np.diff(p).mean()) for p in pos.values() if len(p)>=mink and np.diff(p).std()+np.diff(p).mean()>0]
    return np.mean(Bs)

A=gen_A(N); B=gen_B(N)
def row(name,ws):
    m,vr=wlen(ws)
    print(f"  {name:18s} zipf={zipf(ws):+.2f}  wlen={m:.2f} v/m={vr:.2f}  hapax={hapax(ws):.2f}  "
          f"h2={h2(ws):.2f}  adj-rep={100*adjrep(ws):.2f}%  burst={burst(ws):+.3f}")
print("GENERATIVE FIT — real vs schema-sampler (A) vs schema+self-citation (B):\n")
row("REAL", real); row("A: schema only", A); row("B: schema+copy", B)
print("\nresidual = any column where B differs from REAL = structure beyond 'schema+copy'.")
