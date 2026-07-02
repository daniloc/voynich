# images/ — directory index

~446 image files. Described by group, not per-file (per-file tables would be
unreadable at this volume and most files are single-use vision-agent crops).
None of these are produced by any script in `analysis/` — `page_vision.py` reads
`images/facsimile/Voynich_Manuscript.pdf` but writes no images; every crop/montage
here was made by a vision subagent with ad hoc PIL crops (the pattern documented
in `data/grounding/z10/PROTOCOL.md`: `Image.open(...).crop(...).resize(...).save(...)`)
or fetched directly from an IIIF endpoint.

---

## images/facsimile/ (32 files)

The Voynich manuscript facsimile itself and derived page renders.

- **`Voynich_Manuscript.pdf`** (54MB) — the full facsimile PDF. This is the one
  re-fetchable/source file in this group (original scan). Read directly by
  `page_vision.py` (PyMuPDF) to build `data/corpus/page_features.json`.
- **`raw_NNN.png`** (18 files, e.g. `raw_000`, `raw_016`, `raw_072`, `raw_204`) —
  native-resolution PNG extractions of specific PDF pages (1108×1536), pulled out
  page-by-page so a vision agent could inspect them via the Read tool (PDF pages
  can't be viewed directly). Page numbers cluster around folios of specific
  interest: `016/026/050` = herbal plant pages behind `plant_tags.json`'s M4
  botany control; `072/081/093/099` = the f57v/f66r/f88r/f99v region (ring/column
  comparison, pharma jars); `108/119/121/123/157` = biological/balneo section;
  `172/174/175/176` = the f86 rosette foldout region; `204` = f116v marginalia
  near the end of the book.
- **`page_NNN.png`** (6 files: `000/072/081/093/099/204`) — smaller (426×590)
  thumbnail renders of the same subset of pages above, likely for quick-look
  context before pulling the full-res `raw_` version.
- **`pdf_pNNN.jpeg`** (7 files, pages `201–206`, `208`) — JPEG page renders
  (1128×1536, 600dpi source) from a different extraction pass, covering pages
  near the very end of the facsimile (post-f116v marginalia / colophon region).
- **Re-fetchable:** yes, all of these are deterministic re-extractions of pages
  from `Voynich_Manuscript.pdf`, which is itself the standard published facsimile
  (ultimately traceable to Yale Beinecke's digitization, `data/iiif/iiif_canvases.json`).

---

## images/crops/ (132 files, incl. `zodiac_z10/`)

Hand-made crops from manuscript images, produced during specific analysis passes.
All derived (not independently re-fetchable except by re-cropping the source
IIIF/facsimile image at the same coordinates) unless noted.

- **`f*` (e.g. `f116v_*`, `f17r_*`, `f1r_rightcol`, `f66r_*`, `f71r_*`)** — folio
  detail crops. The `f116v_*` set (10 files: `L1–L4`, `3lines`, `block`, `top`,
  `wide`, `MICH`, `SIX`, `LAST`) are the marginalia-line crops behind M6's "no new
  tell" anomaly scan of the famous f116v marginal writing. `f66r_*`/`f17r_*` feed
  M7 (ring-vs-column key comparison) and the f17r note anomaly.
- **`z*` (e.g. `z70v_*`, `z72r_*`, `z72v_*`, `z73r_*`, `z73v_*`, `z_75v_row`,
  `z_77r_manifold`, `z_78r_*`, `z_84v_tubes`, `crec_stars`)** — zodiac/cosmological
  detail crops (individual figures, words, and label positions on the zodiac and
  cosmological folios), feeding the K14/zodiac_join.py-era label-reading passes
  before the dedicated `zodiac_z10/` set existed.
- **`bal_*`** (18 files: `75r`–`84v`) — full balneo-section folio crops (the
  nymph/tub/pipe pages), the visual substrate for K13's balneo arm (found to
  carry almost no visual variation — every nymph reads as nude/arms-down).
- **`pf_*`** (6 files: `1r`, `70v`, `72r`, `72v`, `73r`, `73v`) — "page fetch"
  crops, one per zodiac-ring/title folio, used as the base image for early
  label-position work (precursor to the z10 protocol's per-folio quadrant crops).
- **`vis_*`** (4 files: `f57v_full`, `f57v_ring`, `f66r_col`, `f66r_full`) — the
  f57v ring vs f66r isolated-glyph-column comparison images behind M7
  (`ring_column_compare.py`'s hand-transcribed sequences were read off these).
- **`m_*`** (12 files: `l0`, `l3L`, `l3R`, `l4`, `l4b`, `last`, `mich`, `mich2`,
  `michL`, `michR`, `six`, `word1`) — fine-grained f116v marginalia line/word
  crops (more granular than the `f116v_*` set above), for the "Michiton oladabas"
  marginal-writing anomaly check.
- **`c71_*`, `c88_*`, `cf1_*`, `cbal_wrap`, `cherb_split`, `scan_*`, `otoky_zoom`,
  `band_f99r/f99v`, `f99r_band`, `f99v_band`** — pharma/nymph detail crops: jar
  labels and text on f88r (`c88_*`), nymph/root crops on f71 (`c71_*`), the f99v
  band/jar region (`f99r_band`/`f99v_band`/`band_f99*`), and the `otoky` collision
  word close-up (K5's "otoky = root vs bulb" cross-folio collision check).
- **`iiif_*`** (6 files: `67r`, `69r`, `71r`, `78r`, `88r`, `99v`) — direct IIIF
  crops pulled via the Yale canvas URLs in `data/iiif/iiif_canvases.json` /
  `iiif_urls.json`. **Re-fetchable** from Yale Beinecke IIIF at full resolution.
- **`contact_sheet.png`, `glyph_zoom.png`** — general-purpose reference sheets
  (glyph inventory zoom, a contact-sheet overview), not tied to one folio.
- **`_l6_*`, `_tmp_pg*`, `_schurstab_22v.jpg`** — underscore-prefixed = scratch/
  temp crops from specific test runs: `_l6_*` are the K13 (l6_grounding.py)
  pharma-referent crops (f88r/f75v regions); `_tmp_pg*` are disposable page
  crops; `_schurstab_22v.jpg` is the M9 comparison crop of Zürich Ms. C 54
  f22v (hand-bow Sagittarius cousin), sourced from `_schurstab_manifest.json`.
- **`images/crops/zodiac_z10/`** (4 files: `f70v2_full.jpg`, `f71r_full.jpg`,
  `f72r2_full.jpg`, `f73v_full.jpg`) — **the 4 full-resolution zodiac ring scans
  used for the K14 binding pass.** These are the primary source images the 4 K14
  vision agents read from (per `data/grounding/z10/PROTOCOL.md`), producing
  `data/grounding/z10_bindings.json`. Re-fetchable from Yale IIIF at the same
  folios.

---

## images/comparanda/ (276 files across 6 manuscript dirs + 8 top-level files)

Candidate exemplar manuscripts pulled during the M9/M11/M13 zodiac-source hunt
(searching for the Voynich zodiac's iconographic ancestor). Grouped by manuscript.

| Directory | Manuscript | M-item verdict (FINDINGS.md) |
|---|---|---|
| `cand_cgm28/` (6 files) | **BSB Cgm 28** — *Kalender und Praktika, schwäbisch* (Swabian calendar/practica), Munich | Candidate exemplar family member; only calendar pages (`cal12`–`cal16`) + a montage were fetched. **Its Sagittarius folio was never pulled** — open audit item, "one IIIF fetch away." |
| `cand_clm13076/` (56 files) | **BSB Clm 13076**, S-German, dated **1356** | M11: **closest verified figural zodiac relative** — human (not centaur) Sagittarius + reptile-quadruped Scorpio + crayfish Cancer + curl-tail lion match 4–5/6 markers, but has a **longbow** (not crossbow) and is ~50yrs early. "Ancestor-type witness," not the exemplar. Largest comparanda set (multiple crop passes: `btm*`, `foot*`, `hi*`, `p*` page series). |
| `cand_cpg291/` (34 files) | **Heidelberg Cod. Pal. germ. 291** — *Iatromathematisches Hausbuch*, Bavaria, after 1477/before 1496 | M13: one of the two genuine human-**crossbow** Sagittarius witnesses finally found on screen (f16v). Same MS has crayfish-Cancer (f13r) + lobster-Scorpio (f16r) + goat-Capricorn (f17r). Post-window (~50yrs late) and renders Sagittarius as a roundel medallion, not Voynich's ring layout. |
| `cand_ljs449/` (10 files) | **UPenn LJS 449**, c.1446 (in-window) | M11: genre cousin — centaur+longbow Sagittarius (wrong branch) but matching Cancer/Leo iconography. |
| `cand_md2/` (141 files — the largest set) | **Tübinger Hausbuch Md 2**, calendar text dated **1404** (in-window) | M9/M11/M13: family/region/date corroborator, but Sagittarius (f116r, scored in M13) is a **hairy wild-man with a longbow** spanning two arches — mismatch on crossbow + cap. Extensively photographed (full zodiac cycle `z_*r/v`, text pages `t_*`, deep-zoom `sag116_zoom`/`md2_deep*`, plus a `manifest.json`). |
| `csg827/` (21 files) | **Cod. Sang. 827**, St. Gallen, Lake Constance, dated **1425/28** (in-window) | M9 correction: corroborates the genre/region/date, but has **no pictorial zodiac figures at all** — only a Zodiac-Man + dietetic rubrics + Labor-of-the-Months. The "closest named analog" sub-claim built on this MS was withdrawn after pixel inspection. |

Top-level `images/comparanda/` files (not in a manuscript subdirectory):
- **`cgm312_*.jpg`** (5 files: `00171/00172/00173`, `172_full`, `sag`, `sco`) —
  **BSB Cgm 312**, *Losbücher* (fortune-telling books), Swabian, **1450–1473**.
  M13: the *other* genuine human-crossbow Sagittarius witness (f83v, red
  tunic/blue hose) + a reptilian/lizard Scorpio. Also post-window, also a
  roundel medallion format.
- **`SAG_compare.jpg`, `SAG_compare2.jpg`** — side-by-side comparison sheets
  assembling multiple candidate Sagittarius figures (Voynich f73v vs the
  crossbow witnesses above) for the M11/M13 visual-verification passes.

**Re-fetchable:** all comparanda images are re-fetchable from their source IIIF
endpoints — BSB (Cgm 28, Cgm 312, Clm 13076) and e-codices (Cod. Sang. 827); the
`data/iiif/*_manifest.json` files preserve the manifest for the four that have
one. `cand_cpg291` = Heidelberg digital library, `cand_ljs449` = UPenn Kislak
Center (Penn in Hand / OPenn), `cand_md2` = Tübingen digital library — none of
these three have a saved manifest JSON in `data/iiif/`.

---

## images/derived/ (6 files)

Composite/derived images (montages, comparison maps) assembled from other images
— not independently re-fetchable, since they're synthesized, not sourced.

- **`plants_montage_0/1/2.png`** — 3 montage sheets (4 plants each) of herbal
  diagrams, the direct visual source behind `data/grounding/plant_tags.json`'s
  M4 real-botany control tagging.
- **`corner_map.png`** — a page/margin layout map (likely a codicological
  corner/collation reference; exact producing pass not identified with
  certainty — see "could not fully identify" note below).
- **`collision_referents.png`** — comparison image for K5's cross-section word
  collision test (e.g. `otoky`/`okolo` referents side by side).
- **`otoky_compare.png`** — the specific `otoky` collision pixel-comparison
  (drooping roots on f88r vs a striped bulb on f99v) that helped kill K5.

---

## Summary of gaps flagged during this audit

- **`cand_cgm28`'s Sagittarius folio was never fetched** — confirmed open audit
  item; only 5 calendar-page crops + a montage exist for this manuscript.
- **No image in this tree is produced by a checked-in script** — `page_vision.py`
  only reads the facsimile PDF in-memory for pixel statistics and writes JSON,
  never an image file. Every crop/montage/comparison sheet was made ad hoc by a
  vision subagent (consistent with `data/grounding/z10/PROTOCOL.md`'s documented
  crop workflow) — there is no reproducible "rebuild all crops" script.
- **`images/derived/corner_map.png`** — I could not confidently identify which
  analysis pass produced this one (no script references it, and its content
  wasn't distinguishable from the other derived maps at a glance); flagging
  rather than guessing.
- Three comparanda manuscripts (`cand_clm13076`, `cand_cpg291`, `cand_ljs449`,
  `cand_md2`) have no manifest JSON saved in `data/iiif/` — only `cand_cgm28`,
  `cgm312_*`, `csg827`, and the Schürstab crop have a preserved manifest.
