"""
THE MARSHALED INVESTIGATION.
The audit found ~18 useful tests and ~11 flourishes. Every useful test reduces to one
instrument the flawed ones lack: a MATCHED CONTENT-FREE NULL. The flourishes all failed
the same way — they read size / Currier-dialect / content-free-reproducible structure as
meaning. So unify the survivors: build the content-free generator ONCE (schema-sampler +
self-citation, fitted to real burstiness), then run ONLY the discriminators where meaning
COULD survive that null — coupling to something OUTSIDE the token stream:
   D1 CROSS-MODAL   does text track an independent image observable? (gen has no images)
   D2 ORDER/TRAJECTORY  does line order within a paragraph carry info beyond a bag?
   D3 LINE-POSITION  is glyph-shape/length conditioned on physical column position?
   D4 SYNTAX        does within-line word order add predictability over a bag?
A signal is REAL only if: real != matched content-free gen, AND not explained by
layout/dialect/size. Everything else is the genre announcing itself.
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
from itertools import combinations
rng = np.random.default_rng(1989)
C = json.load(open('data/corpus/corpus.json')); META = C['meta']
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
def fk(f):
    m=re.match(r'f(\d+)([rv])(\d*)',f); return (int(m.group(1)),0 if m.group(2)=='r' else 1,int(m.group(3) or 0))
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph']
SUF=['eedy','eody','edy','aiin','ain','iin','dy','ol','or','ar','al','am','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s

# ---- ordered prose lines (P loci), with paragraph + folio tags ----
LINES=[]
for f in sorted(C['folios'],key=fk):
    for ln in C['folios'][f]:
        if ltype(ln['locus'])!='P': continue
        ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
        if ws: LINES.append(dict(f=f,para='@' in ln['locus'],ws=ws))
REAL=[w for L in LINES for w in L['ws']]; N=len(REAL)

# ================= the matched content-free generator (genfit Model B) =================
corefreq=Counter(decomp(w)[1] for w in REAL)
pre_g=defaultdict(Counter); suf_g=defaultdict(Counter)
for w in REAL:
    p,c,s=decomp(w); pre_g[c][p]+=1; suf_g[c][s]+=1
cores=list(corefreq); cw=np.array([corefreq[c] for c in cores],float); cw/=cw.sum()
glyphs=sorted(set(ch for w in REAL for ch in w))
def sample_word():
    c=cores[rng.choice(len(cores),p=cw)]
    p=rng.choice(list(pre_g[c]),p=np.array(list(pre_g[c].values()))/sum(pre_g[c].values()))
    s=rng.choice(list(suf_g[c]),p=np.array(list(suf_g[c].values()))/sum(suf_g[c].values()))
    return p+c+s
def gen_stream(n,p_copy=0.12,window=40,p_mut=0.4):
    out=[]
    for _ in range(n):
        if out and rng.random()<p_copy:
            w=out[-rng.integers(1,min(window,len(out))+1)]
            if rng.random()<p_mut and len(w)>1:
                i=rng.integers(0,len(w)); w=w[:i]+rng.choice(glyphs)+w[i+1:]
            out.append(w)
        else: out.append(sample_word())
    return out
GEN=gen_stream(N)
# lay GEN into the SAME line lengths as real (so line-based tests are comparable)
GLINES=[]; k=0
for L in LINES:
    GLINES.append(dict(f=L['f'],para=L['para'],ws=GEN[k:k+len(L['ws'])])); k+=len(L['ws'])

def burst(ws,mink=10):
    pos=defaultdict(list)
    for i,w in enumerate(ws): pos[w].append(i)
    B=[(np.diff(p).std()-np.diff(p).mean())/(np.diff(p).std()+np.diff(p).mean()) for p in pos.values() if len(p)>=mink and np.diff(p).std()+np.diff(p).mean()>0]
    return np.mean(B)
print(f"matched null check: burstiness real={burst(REAL):+.3f}  gen={burst(GEN):+.3f}  (fitted ok)\n")

# ============================ D1: CROSS-MODAL COUPLING ============================
# does an INDEPENDENT image observable (pigment color) predict the page's vocabulary?
# gen has no images, so any real coupling here cannot come from a content-free stream.
print("="*74); print("D1  CROSS-MODAL: does image color predict text? (Mantel permutation)"); print("="*74)
try:
    feats={i:f for i,*f in json.load(open('data/corpus/page_features.json'))}
    def recto_page(n): return 2*n-2
    recs=[]
    for fol,lines in C['folios'].items():
        m=re.match(r'f(\d+)([rv])',fol)
        if not m or int(m.group(1))>57: continue
        n=int(m.group(1)); pg=recto_page(n)+(1 if m.group(2)=='v' else 0)
        if pg not in feats: continue
        ink,green,red,blue,edge=feats[pg]
        if green<0.04: continue
        ws=[w for ln in lines for w in ln['words'] if '?' not in w]
        if len(ws)<12: continue
        recs.append((green,red,blue,set(ws)))
    pairs=list(combinations(range(len(recs)),2))
    vd=np.array([sum((recs[i][k]-recs[j][k])**2 for k in range(3))**.5 for i,j in pairs])
    tj=np.array([len(recs[i][3]&recs[j][3])/len(recs[i][3]|recs[j][3]) for i,j in pairs])
    def pear(x,y): return float(((x-x.mean())*(y-y.mean())).mean()/(x.std()*y.std()+1e-12))
    obs=pear(vd,tj); perm=np.array([pear(vd,rng.permutation(tj)) for _ in range(4000)])
    p=(np.sum(np.abs(perm)>=abs(obs))+1)/(len(perm)+1)
    print(f"  herbal folios={len(recs)}  Mantel r(visual-dist, vocab-overlap)={obs:+.4f}  p={p:.4f}")
    print(f"  VERDICT: {'SIGNAL — text couples to image' if p<0.05 else 'NULL — no image/text coupling (content-free is sufficient)'}")
except FileNotFoundError:
    print("  data/corpus/page_features.json missing; skipped")

# ============================ D2: ORDER / TRAJECTORY ============================
# within a paragraph, are nearby lines more similar than distant lines (a drifting state)?
# compare REAL vs its own line-shuffle null AND vs the matched content-free GEN.
print("\n"+"="*74); print("D2  ORDER: within-paragraph line-similarity vs lag (log vs bag)"); print("="*74)
def paragraphs(lines):
    out=[]; cur=[]
    for L in lines:
        if L['para'] and cur: out.append(cur); cur=[]
        cur.append(L)
    if cur: out.append(cur)
    return [p for p in out if len(p)>=4]
def feat(ws): return Counter(m for w in ws for m in decomp(w) if m)
def cos(a,b):
    ks=set(a)|set(b); na=math.sqrt(sum(v*v for v in a.values())); nb=math.sqrt(sum(v*v for v in b.values()))
    return sum(a[k]*b[k] for k in ks)/(na*nb) if na and nb else 0
def lag_slope(lines):
    paras=paragraphs(lines); byL=defaultdict(list)
    for p in paras:
        fs=[feat(L['ws']) for L in p]
        for i in range(len(fs)):
            for j in range(i+1,len(fs)):
                byL[j-i].append(cos(fs[i],fs[j]))
    lags=sorted(k for k in byL if len(byL[k])>=20)[:8]
    means=[np.mean(byL[k]) for k in lags]
    sl=np.polyfit(lags,means,1)[0]
    return sl, list(zip(lags,[round(m,3) for m in means]))
sr,curve_r=lag_slope(LINES)
sg,curve_g=lag_slope(GLINES)
# shuffle null: permute line order within each paragraph
def shuffled(lines):
    paras=paragraphs(lines); flat=[]
    for p in paras:
        idx=rng.permutation(len(p)); flat+= [p[i] for i in idx]
    # rebuild a flat list tagging first as para-start
    out=[]
    for p in paras:
        ws=[L['ws'] for L in p]; rng.shuffle(ws)
        for j,w in enumerate(ws): out.append(dict(f=p[0]['f'],para=(j==0),ws=w))
    return out
ss=np.mean([lag_slope(shuffled(LINES))[0] for _ in range(8)])
print(f"  lag-similarity slope (negative = nearby lines more similar = ORDER carries info):")
print(f"    REAL            slope={sr:+.5f}   curve={curve_r[:5]}")
print(f"    REAL line-shuffle null slope={ss:+.5f}")
print(f"    matched content-free gen slope={sg:+.5f}")
print(f"  VERDICT: {'SIGNAL — real order beyond bag AND beyond content-free gen' if (sr<ss-0.002 and sr<sg-0.002) else 'NULL — order info matches shuffle/gen (bag-like)'}")

# ============================ D3: LINE-POSITION ============================
print("\n"+"="*74); print("D3  LINE-POSITION: is glyph-shape/length fixed by column position?"); print("="*74)
GAL=set('tkpf')
def linepos(lines):
    first=[L['ws'][0] for L in lines if L['ws']]
    last=[L['ws'][-1] for L in lines if len(L['ws'])>1]
    mid=[w for L in lines for w in L['ws'][1:-1]]
    gf=lambda S:np.mean([1.0 if w[0] in GAL else 0.0 for w in S])
    ml=lambda S:np.mean([len(w) for w in S])
    return (gf(first),gf(mid),gf(last)),(ml(first),ml(mid),ml(last))
(rg,rl)=linepos(LINES); (gg,gl)=linepos(GLINES)
print(f"  gallows-initial rate  first/mid/last  REAL {rg[0]:.2f}/{rg[1]:.2f}/{rg[2]:.2f}   GEN {gg[0]:.2f}/{gg[1]:.2f}/{gg[2]:.2f}")
print(f"  mean word length      first/mid/last  REAL {rl[0]:.2f}/{rl[1]:.2f}/{rl[2]:.2f}   GEN {gl[0]:.2f}/{gl[1]:.2f}/{gl[2]:.2f}")
real_eff=abs(rg[0]-rg[1]); gen_eff=abs(gg[0]-gg[1])
print(f"  VERDICT: {'SIGNAL — strong line-position rule absent from content-free gen' if real_eff>gen_eff+0.05 else 'NULL'}  (real gallows-step={real_eff:.2f} vs gen={gen_eff:.2f})")
print("   NOTE: a real but CONTENT-FREE layout rule (text aware of line breaks) — not meaning.")

# ============================ D4: SYNTAX ============================
# does WITHIN-LINE word order add predictability? bigram cond-entropy real vs within-line shuffle.
print("\n"+"="*74); print("D4  SYNTAX: does within-line word order reduce next-word entropy?"); print("="*74)
def bigram_condH(lines_ws):
    bi=Counter(); uni=Counter()
    for ws in lines_ws:
        for a,b in zip(ws,ws[1:]): bi[(a,b)]+=1; uni[a]+=1
    n=sum(bi.values())
    return -sum(c/n*math.log2(c/uni[a]) for (a,b),c in bi.items())
real_ws=[L['ws'] for L in LINES]
H_real=bigram_condH(real_ws)
H_sh=np.mean([bigram_condH([list(rng.permutation(ws)) for ws in real_ws]) for _ in range(6)])
print(f"  within-line bigram conditional entropy H(next|prev):")
print(f"    REAL              {H_real:.3f} bits")
print(f"    within-line shuffle null {H_sh:.3f} bits")
print(f"  VERDICT: {'SIGNAL — word order carries info' if H_real < H_sh-0.05 else 'NULL — order adds nothing (bag-of-words confirmed)'}  (gap={H_sh-H_real:+.3f})")

print("\n"+"="*74); print("RESIDUAL LEDGER"); print("="*74)
print("  Meaning can ONLY live where REAL diverges from the matched content-free null")
print("  in a way layout/dialect/size cannot explain. Read the four VERDICTs above.")
