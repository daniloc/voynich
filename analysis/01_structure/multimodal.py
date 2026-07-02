"""
LAYOUT-AWARE ANALYSIS — the multimodal bridge without needing OCR of the glyphs.
The IVTFF locus tag for every line encodes its PHYSICAL ROLE on the page:
  P = paragraph (running text)      L = label (a word beside a drawing)
  C = circular line   R = radial line   (the astro roundels)
  t/T = title line                  others = captions etc.
If Voynichese is a real language, single-word LABELS attached to pictures
(a plant name, a star name) should differ statistically from running prose:
shorter, drawn from a constrained naming vocabulary, less repetitive.
We also test the famous "label words mostly start with o-/y-" claim.
"""
import json, re, math
from collections import Counter, defaultdict

C = json.load(open('data/corpus/corpus.json'))
META = C['meta']

# locus type = first letter of the locus code after stripping @/+/- prefixes
def locus_type(loc):
    m = re.search(r'[A-Za-z]', loc)
    return m.group(0).upper() if m else '?'

bytype = defaultdict(list)        # type -> list of words
bytype_lines = defaultdict(list)  # type -> list of word-lists (per line)
for f, lines in C['folios'].items():
    for ln in lines:
        t = locus_type(ln['locus'])
        ws = [w for w in ln['words'] if '?' not in w]
        bytype[t] += ws
        if ws: bytype_lines[t].append(ws)

def wl(ws):
    L=[len(w) for w in ws]; return sum(L)/len(L)
def adj(ws):
    return 100*sum(1 for a,b in zip(ws,ws[1:]) if a==b)/max(1,len(ws)-1)
def startfreq(ws):
    c=Counter(w[0] for w in ws if w); n=sum(c.values())
    return [(g,100*v//n) for g,v in c.most_common(4)]

print("="*74)
print("TEXT STATISTICS BY PHYSICAL ROLE ON THE PAGE (locus type)")
print("="*74)
NAMES={'P':'paragraph prose','L':'label (by drawing)','C':'circular line',
       'R':'radial line','T':'title','X':'extra/caption'}
print(f"{'role':22s}{'tokens':>8}{'types':>7}{'wlen':>7}{'adj-rep%':>9}  top word-initial glyphs")
for t,ws in sorted(bytype.items(), key=lambda kv:-len(kv[1])):
    if len(ws)<60: continue
    print(f"{NAMES.get(t,t):22s}{len(ws):>8}{len(set(ws)):>7}{wl(ws):>7.2f}{adj(ws):>9.2f}  {startfreq(ws)}")

# Labels: how many are single-word lines? and their vocabulary overlap w/ prose
labelwords = bytype['L']
prosewords = set(bytype['P'])
if labelwords:
    only_in_labels = [w for w in set(labelwords) if w not in prosewords]
    print("\n--- LABELS vs PROSE -------------------------------------------------")
    print(f"label tokens: {len(labelwords)}, distinct: {len(set(labelwords))}")
    print(f"label words NEVER seen in prose: {len(only_in_labels)} "
          f"({100*len(only_in_labels)/len(set(labelwords)):.0f}% of label vocab)")
    print(f"most common label words: {[w for w,_ in Counter(labelwords).most_common(12)]}")
    # mean line length per role (labels should be ~1-2 words)
    for t in ['L','P','C','R']:
        if bytype_lines[t]:
            avg=sum(len(x) for x in bytype_lines[t])/len(bytype_lines[t])
            print(f"  mean words/line [{NAMES.get(t,t)}]: {avg:.2f}")

# Positional structure WITHIN the page: does the FIRST word of a paragraph differ?
print("\n--- PARAGRAPH-INITIAL anomaly (the 'gallows on line 1' effect) ----------")
first_words=[]; other_words=[]
for f,lines in C['folios'].items():
    paras=[ln for ln in lines if locus_type(ln['locus'])=='P']
    for ln in paras:
        ws=[w for w in ln['words'] if '?' not in w]
        if not ws: continue
        first_words.append(ws[0]); other_words+=ws[1:]
fc=Counter(w[0] for w in first_words); oc=Counter(w[0] for w in other_words)
nf=sum(fc.values()); no=sum(oc.values())
print("glyph : line-initial%  vs  mid-line%   (lift)")
for g,_ in fc.most_common(8):
    a=100*fc[g]/nf; b=100*oc.get(g,0)/no
    print(f"  {g}   :   {a:5.1f}%        {b:5.1f}%      x{a/max(b,0.1):.1f}")
