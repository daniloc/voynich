"""
THE LAST NULL: a MEANINGLESS topic-stateful generator. Cluster entities into K latent
topics (by co-occurrence — arbitrary statistical topics, no semantics). Generate with
slow topic-drift + the schema + copying. Sweep K: how many topics does it take to match
BOTH the real burstiness (0.138) AND the hapax tail (0.70)?
  small K matches  -> a simple drifting generator suffices; meaningless survives.
  K ~ #folios needed -> the text needs ~one distinct topic per page = genuine
                        page-specific content, which simple generation can't fake.
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
from scipy.cluster.vq import kmeans2, whiten
rng=np.random.default_rng(37)
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
def fk(f): m=re.match(r'f(\d+)([rv])(\d*)',f); return (int(m.group(1)),0 if m.group(2)=='r' else 1,int(m.group(3) or 0))
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph']
SUF=['eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s

folios=[f for f in sorted(C['folios'],key=fk)]
fidx={f:i for i,f in enumerate(folios)}
# per (core, folio) counts
real=[]; core_fol=defaultdict(lambda: np.zeros(len(folios)))
for f in folios:
    for ln in C['folios'][f]:
        if ltype(ln['locus'])!='P': continue
        for w in ln['words']:
            if '?' in w or len(w)<2: continue
            real.append((w,f)); core_fol[decomp(w)[1]][fidx[f]]+=1
stream=[w for w,_ in real]; N=len(stream)
glyphs=sorted(set(ch for w in stream for ch in w))

# schema: affix sampling per core
pre_g=defaultdict(Counter); suf_g=defaultdict(Counter)
for w in stream:
    p,c,s=decomp(w); pre_g[c][p]+=1; suf_g[c][s]+=1
def dress(c):
    p=rng.choice(list(pre_g[c]),p=np.array(list(pre_g[c].values()))/sum(pre_g[c].values()))
    s=rng.choice(list(suf_g[c]),p=np.array(list(suf_g[c].values()))/sum(suf_g[c].values()))
    return p+c+s

def burst(ws,mink=10):
    pos=defaultdict(list)
    for i,w in enumerate(ws): pos[w].append(i)
    Bs=[(np.diff(p).std()-np.diff(p).mean())/(np.diff(p).std()+np.diff(p).mean()) for p in pos.values() if len(p)>=mink and np.diff(p).std()+np.diff(p).mean()>0]
    return np.mean(Bs)
def hapax(ws): c=Counter(ws); return sum(1 for v in c.values() if v==1)/len(c)

cores=[c for c in core_fol if core_fol[c].sum()>=3]
Mat=np.array([core_fol[c]/core_fol[c].sum() for c in cores])   # core distribution over folios
print(f"cores clustered: {len(cores)} over {len(folios)} folios")
print(f"REAL: burstiness={burst(stream):+.3f}  hapax={hapax(stream):.2f}\n")
print(f"{'K topics':>9}{'avg-run':>9}{'burst':>9}{'hapax':>9}")

def generate(labels, K, L):
    topic_cores=defaultdict(list); topic_w=defaultdict(list)
    for c,t in zip(cores,labels):
        topic_cores[t].append(c); topic_w[t].append(core_fol[c].sum())
    tw={t:np.array(w)/sum(w) for t,w in topic_w.items()}
    sizes=np.array([sum(topic_w[t]) for t in range(K)]); sizes=sizes/sizes.sum()
    out=[]; t=rng.choice(K,p=sizes); run=0
    while len(out)<N:
        if run<=0: t=rng.choice(K,p=sizes); run=max(1,int(rng.exponential(L)))
        if rng.random()<0.18 and out:                       # copying step
            out.append(out[-rng.integers(1,min(30,len(out))+1)])
        else:
            cc=topic_cores[t]
            c=cc[rng.choice(len(cc),p=tw[t])]
            out.append(dress(c))
        run-=1
    return out[:N]

for K in [6, 20, 60, 150]:
    cent,labels=kmeans2(whiten(Mat), K, seed=1, minit='++', missing='warn')
    # avoid empty clusters: relabel present
    present=sorted(set(labels)); remap={t:i for i,t in enumerate(present)}; labels=np.array([remap[t] for t in labels]); Keff=len(present)
    for L in [12]:
        g=generate(labels,Keff,L)
        print(f"{Keff:>9}{L:>9}{burst(g):>+9.3f}{hapax(g):>9.2f}")
print("\n(match to REAL burst 0.138 & hapax 0.70 at SMALL K => meaningless generator wins;")
print(" only at K ~ #folios => genuine page-specific content.)")
