# Zodiac label→referent binding protocol (one agent per folio)

GOAL: bind every Takahashi Lz label locus on your folio to the SPECIFIC nymph/figure it labels,
and record that referent's visual attributes. This feeds a within-folio shuffle-null test of
whether label morphology tracks referent appearance. HONESTY RULES: never force a binding;
grade every binding's confidence; report count mismatches loudly; your attribute readings must
come from pixels, not from what would be interesting.

## Files (in this directory)
- <folio>_full.jpg     — full-resolution scan
- <folio>_overview.jpg — 1400px overview
- <folio>_{TL,TR,BL,BR}.jpg — native-res quadrants (12% overlap)
- tasklist_<folio>.txt — the Takahashi label sequence, grouped by ring tier, in transcription order

You have python3 with PIL in this directory — make your own tight crops liberally, e.g.:
  python3 -c "from PIL import Image; im=Image.open('f73v_full.jpg'); im.crop((x1,y1,x2,y2)).resize((w*3,h*3)).save('my_crop.jpg')"
Upscale 2-4x when reading labels. Read crops with the Read tool.

## Protocol
1. CENSUS: from overview + quadrants, count figures per tier (outer band / inner band / outside-diagram).
   Compare against the label counts in your tasklist. Report any mismatch (extra/unlabeled figures, shared labels).
2. ENUMERATE: walk each tier CLOCKWISE starting at 12 o'clock. For each figure, record clock position
   (integer 1-12, 12=top) and attributes (schema below), making a tight crop when needed.
3. READ ANCHORS: for as many figures as possible (aim >= half per tier), read the label written beside
   the figure and match it against the candidate words in your tasklist. Labels are short (2-9 glyphs).
   EVA cheat sheet: o=small circle; a=like 'a'; y=figure-9 w/ tail; d=figure-8; e=small 'c'; ee='cc';
   k,t,p,f=GALLOWS (tall, legged, looped — easy to spot); ch=two c's joined by bar ("bench");
   sh=bench w/ plume; iin/in=row of minims w/ final tail; l=looped ascender-less '2'-like; r/s=minim w/ flag;
   q=like '4', word-initial. Discriminate among the CANDIDATE list (closed set) — full open transcription not required.
   Key discriminators: word length, gallows presence/position, final glyph (y vs r vs l vs n-tail).
4. ALIGN: Takahashi transcribes each tier in a consistent rotational order, but the START position and
   DIRECTION are not documented. Use your anchor readings to find the rotation offset + direction that
   best matches the tasklist order for that tier. Report the alignment (start clock position, direction,
   number of anchors matched, number contradicting). If anchors contradict a single consistent rotation,
   SAY SO — do not force it.
5. BIND & RECORD: emit one record per label locus in your tasklist.

## Output — write bind_<folio>.json here AND include it in your final message:
{"folio": "...", "census": {"outer": {"figures": N, "labels": M}, ...},
 "alignment": {"outer": {"start_clock": 12, "direction": "cw", "anchors_matched": 9, "anchors_contradicting": 0}, ...},
 "records": [
   {"locus": 5, "label": "oty", "tier": "outer", "clock": 12,
    "body": "nude|clothed", "clothing_color": ""|"red"|"green"|"blue"|"other",
    "container": "barrel|tub|none", "star": "yes|no", "star_tail": "yes|no",
    "star_hand": "left|right|none", "arms": "one_raised|both_raised|down|other",
    "headwear": "none|veil|crown|hat", "facing": "left|right|front",
    "eva_read": "what you actually read, '' if not attempted", "read_match": "exact|partial|none|not_attempted",
    "bind_conf": "high|med|low", "notes": ""}
 ],
 "anomalies": ["free text — count mismatches, odd figures, illegible zones, layout surprises"]}

bind_conf: high = this label's own glyphs read from pixels match the word; med = both rotational
neighbors are exact-read anchors; low = order-inference only.
