"""
Can a machine recover the manuscript's section structure from IMAGES ALONE,
with no labels and no human telling it where the sections are?
For every page we extract cheap visual features from the native-res image:
  - ink coverage (how much is drawn/written at all)
  - green / red-brown / blue pigment fractions (foliage, roots/flowers, stars)
  - 'text-likeness': density of small dark connected strokes vs large color blobs
Then we let the per-page feature vector speak for itself.
"""
import fitz, io, numpy as np
from PIL import Image

d = fitz.open('images/facsimile/Voynich_Manuscript.pdf')
rows = []
for i in range(d.page_count):
    imgs = d[i].get_images()
    if not imgs:
        rows.append((i, 0,0,0,0,0)); continue
    base = d.extract_image(imgs[0][0])
    im = Image.open(io.BytesIO(base['image'])).convert('RGB').resize((256,355))
    a = np.asarray(im).astype(np.int16)
    R,G,B = a[...,0],a[...,1],a[...,2]
    bright = (R+G+B)/3
    # ink = anything notably darker than the vellum background
    bg = np.median(bright)
    ink = bright < bg-35
    inkfrac = ink.mean()
    # pigment masks (loose, robust to faded medieval inks)
    green = ((G>R+12)&(G>B+4)&(bright<bg-5)).mean()
    redbrown = ((R>G+18)&(R>B+18)).mean()
    blue = ((B>R+12)&(B>G+6)).mean()
    # text-likeness: ink that is "thin" (high local edge density) vs solid fills
    dark = ink.astype(np.uint8)
    edges = np.abs(np.diff(dark,axis=1)).sum()+np.abs(np.diff(dark,axis=0)).sum()
    edge_per_ink = edges/max(1,dark.sum())     # high => stringy strokes (text); low => solid blobs (drawings)
    rows.append((i, inkfrac, green, redbrown, blue, edge_per_ink))

import json
json.dump(rows, open('data/corpus/page_features.json','w'))

# crude auto-segmentation: label each page by its dominant visual character
def label(r):
    i,ink,green,red,blue,edge = r
    if ink < 0.02: return 'blank/cover'
    drawingy = (green+red+blue) > 0.06 and edge < 1.2
    texty    = ink > 0.06 and edge >= 1.2
    if green > 0.05 and red > 0.005: return 'HERBAL (foliage+roots)'
    if green > 0.05: return 'foliage'
    if texty and (green+red+blue) < 0.03: return 'TEXT-DENSE (recipes/stars)'
    if blue > 0.02 and edge < 1.4: return 'ASTRO/COSMO (pigment+radial)'
    if drawingy: return 'drawing-heavy'
    return 'mixed'

from collections import Counter
labels = [label(r) for r in rows]
print("AUTO-DETECTED PAGE CHARACTER (vision only, no text, no metadata):\n")
# print run-length-encoded so the section structure is visible
prev=None; start=0
for i,l in enumerate(labels+[None]):
    if l!=prev:
        if prev is not None:
            print(f"  pages {start:3d}-{i-1:3d}  ({i-start:3d} pp)  {prev}")
        prev=l; start=i
print("\nsummary:", dict(Counter(labels)))
