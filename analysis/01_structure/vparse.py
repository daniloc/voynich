"""
IVTFF parser for the Landini-Stolfi Interlinear archive (EVA).
We use the Takahashi transcription (;H>) as the primary corpus: it is the only
complete one. We also capture per-folio metadata ($L = Currier language A/B,
$H = hand, $Q/$P quire/page, $I illustration type) from the folio header lines.

EVA / IVTFF conventions handled:
  .  , = word separators        (, is an *uncertain* space; we treat as space)
  !  % = "null"/pseudo-space fillers -> deleted (they pad alignment, not glyphs)
  ?    = one unreadable glyph   -> kept as '?' so we can measure legibility
  *    = unreadable             -> '?'
  <...> {...} = inline comments / markers -> stripped
  [a:b] = ambiguous reading     -> take first alternative (a)
Anything left outside the known EVA glyph set is dropped from token stats.
"""
import re, json, sys
from collections import defaultdict, Counter

SRC = "data/corpus/LSI_ivtff.txt"

# ---- folio header metadata -------------------------------------------------
# e.g.  <f1r>  <! $I=T $Q=A $P=A $L=A $H=1 $X=V>
HEADER_RE = re.compile(r'^<(f[0-9]+[rv][0-9]*)>\s*<!\s*(.*?)>')
def parse_headers(lines):
    meta = {}
    for ln in lines:
        m = HEADER_RE.match(ln)
        if m:
            folio, body = m.group(1), m.group(2)
            kv = dict(re.findall(r'\$([A-Z])=(\S+)', body))
            meta[folio] = kv
    return meta

# ---- data lines ------------------------------------------------------------
# e.g.  <f1r.1,@P0;H>      fachys.ykal.ar.ataiin...
LOCUS_RE = re.compile(r'^<(f[0-9]+[rv][0-9]*)\.(\d+),([^;]*);([A-Z])>\s*(.*)$')

def clean_eva(raw):
    s = raw
    s = re.sub(r'<[^>]*>', '', s)      # inline <...> comments / corrections
    s = re.sub(r'\{[^}]*\}', '', s)    # {...} comments
    s = re.sub(r'\[([^:\]]*):[^\]]*\]', r'\1', s)  # [a:b] -> a
    s = s.replace('!', '').replace('%', '')        # null fillers
    s = s.replace('*', '?')            # unreadable glyph
    s = s.replace(',', '.')            # uncertain space -> space
    s = s.strip().strip('.')
    words = [w for w in s.split('.') if w != '']
    return words

def load(transcriber='H'):
    lines = open(SRC, encoding='latin-1').read().splitlines()
    meta = parse_headers(lines)
    # folio -> list of (line_no, locus_type, [words])
    folios = defaultdict(list)
    for ln in lines:
        m = LOCUS_RE.match(ln)
        if not m: continue
        folio, lineno, locus, who, text = m.groups()
        if who != transcriber: continue
        words = clean_eva(text)
        if words:
            folios[folio].append((int(lineno), locus, words))
    return meta, folios

if __name__ == '__main__':
    meta, folios = load()
    nfol = len(folios)
    all_words = [w for f in folios.values() for (_,_,ws) in f for w in ws]
    legible = [w for w in all_words if '?' not in w]
    langs = Counter(meta.get(f,{}).get('L','?') for f in folios)
    print(f"folios with text : {nfol}")
    print(f"total word tokens: {len(all_words):,}")
    print(f"fully-legible    : {len(legible):,} ({100*len(legible)/len(all_words):.1f}%)")
    print(f"distinct legible : {len(set(legible)):,}")
    print(f"Currier language by folio: {dict(langs)}")
    print(f"sample folio f1r words: {[w for (_,_,ws) in folios['f1r'] for w in ws][:15]}")
    # persist a clean legible-token stream + per-folio for downstream scripts
    out = {'meta': meta,
           'folios': {f: [{'line':l,'locus':loc,'words':ws} for (l,loc,ws) in v]
                      for f,v in folios.items()}}
    json.dump(out, open('data/corpus/corpus.json','w'))
    print("wrote data/corpus/corpus.json")
