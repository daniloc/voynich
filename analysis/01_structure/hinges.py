"""
THE HINGES. Cores are stable across registers; AFFIXES switch. So the affixes are
the functional layer -- the moving parts. Separate the lexical layer (cores) from
the functional layer (affixes) and study the functional layer as the 'grammar':
  H1. The q-operator. Prose onsets qok-/qot-; label onsets ok-/ot-. Is 'q' a
      separable, productive toggle (qW <-> W)? That would be a literal hinge glyph.
  H2. Affix agreement. Do prefixes and suffixes co-vary (a prefix forces a suffix)?
      Mutual information between the prefix slot and suffix slot = grammar strength.
  H3. Positional conditioning. Does the hinge (prefix choice) respond to position
      in the line/paragraph -- i.e. is the register switch driven by page geometry?
"""
import json, re, math, numpy as np
from collections import Counter, defaultdict
C=json.load(open('data/corpus/corpus.json'))
def ltype(loc):
    m=re.search(r'[A-Za-z]',loc); return m.group(0).upper() if m else '?'
allw=[w for f in C['folios'].values() for ln in f for w in ln['words'] if '?' not in w and len(w)>=2]
vocab=set(allw); freq=Counter(allw)

print("="*70)
print("H1. THE q-OPERATOR — is 'q' a separable, productive hinge glyph?")
print("="*70)
qwords=[w for w in vocab if w.startswith('q')]
q_has_twin=sum(1 for w in qwords if w[1:] in vocab)        # qW -> W attested?
print(f"  distinct q-words: {len(qwords)}")
print(f"  'q'-stripped form also attested: {q_has_twin} ({100*q_has_twin/len(qwords):.0f}%)")
# null: strip the FIRST glyph of non-q words; how often is the remainder attested?
nonq=[w for w in vocab if not w.startswith('q')]
base=sum(1 for w in nonq if w[1:] in vocab)/len(nonq)
print(f"  baseline (strip 1st glyph of any word -> attested): {100*base:.0f}%")
# q attaches almost only to 'o': confirm
after_q=Counter(w[1] for w in qwords if len(w)>1)
print(f"  glyph following 'q': {after_q.most_common(4)}  (q is a bound onset, ~always qo-)")
# token-level: is the qW form the PROSE variant and W the LABEL variant?
labels=set(); prose=set()
for f,lines in C['folios'].items():
    for ln in lines:
        t=ltype(ln['locus'])
        for w in ln['words']:
            if '?' in w or len(w)<2: continue
            (labels if t=='L' else prose if t=='P' else set()).add(w)
pairs=[(w, w[1:]) for w in qwords if w[1:] in vocab]
q_in_prose=sum(1 for q,o in pairs if q in prose); o_in_labels=sum(1 for q,o in pairs if o in labels)
print(f"  of qW/W twin pairs: qW occurs in prose {100*q_in_prose/len(pairs):.0f}%, "
      f"W occurs in labels {100*o_in_labels/len(pairs):.0f}%")
print("  => q- behaves like a register/function marker bolted onto a shared stem.")

print("\n" + "="*70)
print("H2. AFFIX AGREEMENT — do prefix and suffix slots co-vary (a grammar)?")
print("="*70)
PRE=['qok','qot','qo','ok','ot','o','y','ch','sh','d','cth','ckh','cph','cfh']
SUF=['eedy','eody','edy','aiin','aiir','ain','iin','dy','ol','or','ar','al','am','dam','ey','eey','y']
def decomp(w):
    p=next((x for x in sorted(PRE,key=len,reverse=True) if w.startswith(x)),'')
    r=w[len(p):]; s=next((x for x in sorted(SUF,key=len,reverse=True) if r.endswith(x) and len(r)>len(x)),'')
    return p,(r[:len(r)-len(s)] if s else r),s
D=[decomp(w) for w in allw]
pre=[p for p,_,_ in D]; suf=[s for _,_,s in D]
# mutual information between prefix and suffix
joint=Counter((p,s) for p,_,s in D); N=len(D)
pp=Counter(pre); ps=Counter(suf)
MI=0
for (p,s),c in joint.items():
    pij=c/N; pi=pp[p]/N; pj=ps[s]/N
    if pij>0: MI+=pij*math.log2(pij/(pi*pj))
Hs=-sum((v/N)*math.log2(v/N) for v in ps.values())
print(f"  mutual information I(prefix;suffix) = {MI:.3f} bits  (of {Hs:.2f} bits suffix entropy)")
print(f"  => {100*MI/Hs:.0f}% of suffix choice is pinned by the prefix: affix AGREEMENT.")
# show the strongest prefix->suffix preferences
for p in ['qok','ok','ot','ch','sh','o']:
    rows=Counter(s for pp_,_,s in D if pp_==p)
    tot=sum(rows.values())
    if tot<50: continue
    top=[(s if s else 'Ø',f"{100*c//tot}%") for s,c in rows.most_common(3)]
    print(f"     {p:4s}- favours suffix {top}")

print("\n" + "="*70)
print("H3. POSITIONAL CONDITIONING — does the hinge respond to page geometry?")
print("="*70)
# for prose lines, prefix distribution by position-in-line
pos_pre=defaultdict(Counter)
for f,lines in C['folios'].items():
    for ln in lines:
        if ltype(ln['locus'])!='P': continue
        ws=[w for w in ln['words'] if '?' not in w and len(w)>=2]
        for i,w in enumerate(ws):
            bucket='line-initial' if i==0 else 'line-final' if i==len(ws)-1 else 'mid'
            pos_pre[bucket][decomp(w)[0]]+=1
print(f"  q-prefix (qo/qok/qot) rate by position in line:")
for b in ['line-initial','mid','line-final']:
    c=pos_pre[b]; n=sum(c.values())
    q=sum(v for k,v in c.items() if k.startswith('q'))
    gall=sum(v for k,v in c.items() if k in ('ch','sh','cth','ckh','cph','cfh'))
    print(f"    {b:13s}: q-onset {100*q/n:4.1f}%   ch/sh-onset {100*gall/n:4.1f}%   (n={n})")
print("  => if q-onset is suppressed line-initially and boosted mid-line, the hinge")
print("     is conditioned by LAYOUT, not just by the word it sits on.")
