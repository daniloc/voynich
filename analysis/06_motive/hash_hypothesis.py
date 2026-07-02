"""
IS THE CAPTION A KEY/HASH RATHER THAN A DESCRIPTION?
A descriptive caption and a hash are opposite objects, and a correlation test
(which we already ran -> null) CANNOT tell a hash from independence: the avalanche
property makes hash output look random w.r.t. its input. So test the two things a
correlation can't hide:
  (A) FUNCTIONAL FORM. A hash is fixed-length; a key/recipe is variable-length and
      should scale with the complexity of what it encodes. Measure caption-length
      variability, and whether it scales with drawing complexity.
  (B) INFORMATION BUDGET. Could the text even hold the image (or vice versa)?
      Compare bits-in-a-caption against the bits the drawing actually carries.
This tells us which DIRECTION any hash could run, and rules out the efficient cases.
"""
import json, math, re, numpy as np
from collections import Counter

C = json.load(open('data/corpus/corpus.json'))
META = C['meta']
feats = {i:f for i,*f in json.load(open('data/corpus/page_features.json'))}  # page->[ink,green,red,blue,edge]

# ---------- corpus-wide word information content -----------------------------
allwords = [w for f in C['folios'].values() for ln in f for w in ln['words'] if '?' not in w]
wc = Counter(allwords); N = len(allwords)
H_word = -sum(c/N*math.log2(c/N) for c in wc.values())        # bits per word (actual, Zipf-aware)
print(f"corpus: {N:,} tokens, {len(wc):,} types")
print(f"effective information per word  : {H_word:.2f} bits  (max if uniform = {math.log2(len(wc)):.1f})")

# ---------- (A) functional form: caption length distribution -----------------
# words per folio = 'caption length' for that page
wpf = {f:sum(len(ln['words']) for ln in lines) for f,lines in C['folios'].items()}
vals = np.array([v for v in wpf.values() if v>0])
print("\n(A) CAPTION-LENGTH (words per folio):")
print(f"    mean={vals.mean():.0f}  median={np.median(vals):.0f}  min={vals.min()}  max={vals.max()}")
print(f"    coefficient of variation = {vals.std()/vals.mean():.2f}   (a fixed-length HASH would be ~0.0)")

# scaling with drawing complexity (herbal, aligned). Drawing complexity proxy =
# PIGMENT area+edges (colored ink = the drawing; black text excluded by color test)
def recto_page(n): return 2*n if n<12 else 2*n-2
rows=[]
for fol,lines in C['folios'].items():
    m=re.match(r'f(\d+)([rv])',fol)
    if not m or int(m.group(1))>57: continue
    n=int(m.group(1)); pg=recto_page(n)+(1 if m.group(2)=='v' else 0)
    if pg not in feats: continue
    ink,green,red,blue,edge=feats[pg]
    pigment=green+red+blue
    if pigment<0.04: continue                    # only pages with a real drawing
    txt=sum(len(ln['words']) for ln in lines)
    if txt<8: continue
    rows.append((txt, pigment))
T=np.array([r[0] for r in rows]); P=np.array([r[1] for r in rows])
def pear(x,y):
    x=(x-x.mean())/x.std(); y=(y-y.mean())/y.std(); return float((x*y).mean())
print(f"\n    drawing-complexity (pigment) vs caption-length, n={len(rows)} herbal folios")
print(f"    correlation r = {pear(T,P):+.3f}")
print("      ~0  => caption length does NOT scale with how complex the plant is")
print("           (consistent with hash/independent generation; against 'recipe whose")
print("            length grows with the thing it builds')")

# ---------- (B) information budget -------------------------------------------
mean_words_caption = vals.mean()
caption_bits = mean_words_caption * H_word
# image 'describable' content from the structural parts-vocabulary we tagged:
ROOTS, LEAVES, FLOWERS, ARRANGE = 6, 8, 4, 4      # generous counts from the sample
image_bits = math.log2(ROOTS*LEAVES*FLOWERS*ARRANGE)
print("\n(B) INFORMATION BUDGET (bits):")
print(f"    average caption        : {mean_words_caption:.0f} words x {H_word:.1f} = {caption_bits:.0f} bits")
print(f"    drawing combinatorics  : log2({ROOTS}*{LEAVES}*{FLOWERS}*{ARRANGE}) = {image_bits:.0f} bits")
print(f"    ratio caption/image     : {caption_bits/image_bits:.0f}x")
print(f"""
    Implications:
      * text = hash(image): a hash needs only ~{image_bits:.0f} bits to be injective over
        the drawings; a {caption_bits:.0f}-bit 'hash' is ~{caption_bits/image_bits:.0f}x too long. So the text is NOT
        an efficient hash of the picture.
      * text = key(image): {caption_bits:.0f} bits is far MORE than enough to specify the
        drawing -> works, but most of the text would be unexplained surplus.
      * image = hash(text): the drawing ({image_bits:.0f} bits) easily fits as a lossy
        fingerprint/icon of the {caption_bits:.0f}-bit text. Information-favored direction.
""")

# ---------- entropy sanity: does the text look like hash OUTPUT at all? -------
# hash output is maximal-entropy & collision-free; measure the opposite signals
adj = sum(1 for a,b in zip(allwords,allwords[1:]) if a==b)/(len(allwords)-1)
top = wc.most_common(1)[0]
print("Does the text even look like hash OUTPUT?")
print(f"    most common word '{top[0]}' = {100*top[1]/N:.1f}% of all tokens (hash output: ~0%)")
print(f"    adjacent-identical-word rate = {100*adj:.2f}% (hash output: ~0%)")
print("    -> Voynichese is LOW-entropy and self-repeating: it does not look like the")
print("       OUTPUT of a hash. If a hash is involved, the text is the INPUT side and")
print("       the picture is the compressed output (image = hash/icon of text).")
