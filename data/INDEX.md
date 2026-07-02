# data/ — file index

Every file under `data/`, grouped by subdirectory. "Produced by" is the script or
process that generated the file (established by grepping `analysis/*/*.py` for
`json.dump`/`open(...,'w')` calls and matching filenames — only 5 scripts in the
repo actually write a data file; see the note at the end). "Consumed by" is every
script found to `open()`/`json.load()` the file by name.

**Path convention:** during the 2026-07-01 reorg every `analysis/*.py` script had its
data-file paths updated to the `data/<subdir>/` layout, verified by `py_compile`
(50/50) and a stale-path grep. Scripts are run **from the repo root**
(`python3 analysis/<dir>/<script>.py`). This index lists scripts by the filename they
reference.

---

## data/corpus/ — the parsed text corpus (the D12 spine's raw material)

| File | What it is | Produced by | Consumed by |
|---|---|---|---|
| `LSI_ivtff.txt` | The Landini–Stolfi Interlinear archive (IVTFF format), all transcribers merged, downloaded from `voynich.nu` (Zattera's 2021-09-21 edit of Stolfi's 1998 release). 38,943 lines; carries per-folio header metadata (`$I` illustration type, `$Q`/`$P` quire/page, `$L` Currier language, `$H` hand, `$X` extra tag) and per-line locus codes. | External download (voynich.nu beta archive) | `vparse.py` (parses it into `corpus.json`, using the Takahashi `;H>` transcriber as primary since it's the only complete one); `z10_cross.py` (re-reads it directly for the full 12-ring, 299-Lz-locus cross-zodiac test, K14b) |
| `eva_takahashi.txt` | **Broken / dead file.** 14 bytes, literal contents `404: Not Found` — a failed HTTP fetch saved as if it succeeded. Not referenced by any script. | Failed external download | none — safe to delete or re-fetch |
| `body.json` | Deterministic per-word parse of the whole manuscript: `{"pages": {...226 folios...}, "tokens": [...37,019 word-records...]}`. Each token carries `folio/line/locus/lprefix/lcode/role/idx/nline/linepos/word/uncertain/section/currier/quire/hand` — i.e. locus role (paragraph/label/circular/radial) and dialect/hand metadata, but **no image-grounding fields**. This is the pre-vision-grounding stage of the D12 spatial spine. | **No script in `analysis/` writes this file** — built inline in a session (not preserved as a checked-in script); likely `vparse.py`'s logic extended with locus-role/line-position parsing that was never saved back into that file. Gap noted for the record. | Superseded by `body_spatial.json` for most later scripts; no script found to load `body.json` by that literal name (scripts that reference "body" data load `body_spatial.json` instead) |
| `body_spatial.json` | **The D12 spine.** Same shape as `body.json` (226 pages / 37,019 tokens) plus two added fields per token: `subject` (the vision-grounded description of what's on that folio) and `topology` (spatial role: `concentric_rings` 1,377 / `scattered_near_figures` 2,955 / `radial_spokes` 164 / `rows` 659 / `mixed` 176 / `other` 769 / unset for plain text pages ~30,919). Role breakdown: paragraph 33,695, circular 1,937, label 1,038, radial 349. Section breakdown: herbal 10,724, stars_recipes 10,670, biological 6,866, pharmaceutical 2,504, cosmological 2,469, text 1,625, zodiac 1,322, astronomical 839. | **No script in `analysis/` writes this file either** — the same gap as `body.json`; produced by merging `body.json` with the 28-folio `grounding.json` vision pass, done inline/by a subagent, never saved as a build script. | `contact.py` (K12 column-registration test), `quire_test.py` (production-unit autocorrelation, confound #2) — and is the substrate cited throughout FINDINGS.md for D11/D12/K12/K13 |
| `corpus.json` | Line-level parse: `{"meta": {folio: {I,Q,P,L,H,X}}, "folios": {folio: [{"line","locus","words"}]}}`. This is the *original* flat corpus almost every early analysis script loads (pre-spatial-spine). Superseded for spatial questions by `body_spatial.json`, but still the working corpus for morphology/entropy/null-model scripts. | `vparse.py` (`SRC = LSI_ivtff.txt`) | The overwhelming majority of `analysis/01_structure`, `02_hypothesis_tests`, `03_null_models`, `04_cipher`, `06_motive` scripts (41 of ~50 scripts reference `corpus.json` directly) |
| `page_features.json` | Per-PDF-page visual feature vectors: `[page_idx, ink_fraction, green_fraction, redbrown_fraction, blue_fraction, edge_per_ink]`, one row per page of the facsimile PDF (209 rows — matches the PDF's page count, not folio count). Cheap pixel statistics (ink coverage, pigment-channel fractions, stroke-vs-blob edge density) extracted with no section labels given. | `page_vision.py` (reads `images/facsimile/Voynich_Manuscript.pdf` via PyMuPDF/PIL, writes this file) | `hash_hypothesis.py` (D6 info-budget test, M-series), `collision.py`, `crossmodal.py`, `residual.py` |

---

## data/controls/ — natural-language control corpora

All are Bible translations *except* `english.txt`/`latin.txt`/`finnish.txt`, which
are Gutenberg literary texts (KJV Bible for English is the one exception — see
row). This matters per confound #4 (English-as-floor): English is a deep-orthography
outlier and several "control" languages are the *same* source text (the Gospel of
Matthew genealogy) in different languages, which is good for controlling genre but
means they aren't independent corpora in the literary sense.

| File | What it is | Produced by | Consumed by |
|---|---|---|---|
| `english.txt` | Project Gutenberg KJV Old Testament (Genesis/Exodus/...). 99,597 lines. | External download (Gutenberg #10) | `analyze.py`, `burst_tables.py`, `ab_procedure.py` |
| `latin.txt` | Project Gutenberg, Cicero's Orations. 1,769 lines — much shorter than the others. | External download (Gutenberg) | `analyze.py`, `burst_tables.py`, `cipher_attack.py`, `decoder_trap.py`, `resistance.py`, `verbose_cipher_null.py`, `ab_procedure.py` — **note (audit item):** every use is classical/Ciceronian Latin; no medieval/vernacular Latin target was ever tried despite the M9 transalpine-exemplar finding |
| `hawaiian.txt` | Hawaiian Bible (Genesis-opening confirmed). 41,899 lines, ~4.4MB. Central to K2 (Hawaiian 30.9% automaton-match beats Voynich). | External download | `verbose_cipher_null.py`; `entropy_panel.py` (comment references it, loads via a generic path) |
| `finnish.txt` | Project Gutenberg, *Kalevala*. 23,559 lines — the one non-Bible, non-English literary control besides Latin. | External download (Gutenberg) | Referenced in `entropy_panel.py`'s D5 panel (loaded generically; see `_entropy_panel.json` for cached FINNISH row) |
| `hungarian.txt` | Bible (Matthew genealogy confirmed), Hungarian. 41,899 lines but only 7,957 non-blank (heavy verse-break whitespace). | External download | Part of the D5 entropy panel / K2-style language band (see `_entropy_panel.json`); no direct filename hit in checked-in scripts — likely loaded by an uncommitted control-loader |
| `turkish.txt` | Bible, Turkish. 41,899 lines, 28,784 non-blank. | External download | Same as above — entropy-panel / automaton band language |
| `swahili.txt` | Bible, Swahili (Matthew genealogy). 41,899 lines, 7,851 non-blank. Cited in FINDINGS K2 ("Voynich sits below Swahili ×97 on the lift metric"). | External download | Same as above |
| `maori.txt` | **Māori Bible, but in TEI/CES XML markup despite the `.txt` extension** (misnamed — contains `<cesHeader>`/`<div>` tags, not plain text). 62,484 lines. | External download | Not found in any checked-in script by literal name — the K2 Māori analysis was run from an ad hoc/session-only loader, not preserved here (ledger gap) |
| `maori_bible.xml` | Māori Bible, CES-XML encoded, explicit `.xml` extension (looks like a cleaner re-fetch of the same source as `maori.txt`). 96,273 lines. | External download | Same gap as `maori.txt` — no script reference found |
| `_mri_tatoeba` / `_mri_tatoeba.bz2` | Māori sentences from the Tatoeba project (conversational, **not Bible-derived** — everyday sentences like "I'm tired!", "Is the plane on time?"). The `.bz2` is the compressed original; the plain file is the decompressed text (25,837 bytes). | External download (Tatoeba) | Not consumed by any script yet — this is the fix for the audit's "unclosed okina caveat": a non-Bible Māori corpus to control for genre effects in K2's Māori/Hawaiian comparison, fetched but never run through the harness |

---

## data/grounding/ — vision-grounding artifacts (image↔text binding)

None of the files in this directory are written by a checked-in `analysis/` script
— they were authored directly by vision subagents (Claude reading manuscript page
images and recording structured JSON) during grounding passes, or hand-assembled
from those passes. `_method`/`built` fields inside several files document this.

| File | What it is | Produced by | Consumed by |
|---|---|---|---|
| `grounding.json` | The 28-folio vision-grounding payload: `{folio: {folio, identified_subject, layout_topology, elements: [...]}}` — free-text descriptions + structured element lists (nymphs, tubs, stars, rings, jars) per illustrated folio. This is the source that got merged into `body_spatial.json`'s `subject`/`topology` fields. | 28 vision subagents (one per folio), 2026-06-20 pass (D12) | Merged manually into `body_spatial.json`; no script re-reads it directly |
| `_ground_payload.json` | The **input worklist** for the grounding pass above: a 28-entry list of `{folio, section, image, extra_images, labelsets}`, where `labelsets` are the raw IVTFF label loci (locus code + word list) each agent had to bind. This is the task spec, not the result. | Assembled from `corpus.json`/`body.json` label extraction (script not preserved) | Fed to the 28 grounding subagents as their brief |
| `plant_tags.json` | Vision tags for 27 herbal plant diagrams: `{"_method": "...claude-opus-4-8, 2026-06-20...", "plants": [{"id","src","root","stem","leaf","infl"}]}` — per-plant part-type categories (taproot/fibrous, single_erect, lanceolate_drooping, etc.) read off 3 montage images + 3 full-res renders. This is the real-botany control set for M4. | Vision subagent (claude-opus-4-8) reading `images/derived/plants_montage_*.png` + `images/facsimile/raw_016/026/050.png` | `herbal_control.py` (M4: tests whether Voynich plant-part combinations show real botanical cross-part constraints — they don't, vs a 27-species control that does) |
| `cribs.json` | Two label groups used as "constrained subject" anchors (L2, picture-cribs): `medallions` (12 zodiac-medallion folios → their label lists) and `pharma_labels` (15 pharma-jar folios → labels). This is the raw anchor set L2 describes as "untested at scale." | Extracted from `corpus.json`/label loci (script not preserved) | Not consumed by any checked-in script — L2's crib-anchored held-out decode was never built as specified (audit item); this file is the un-exploited input for that test |
| `verify_pairs.json` | 17 pre-registered word-pair verification targets, e.g. `["p", ["f49v","f66r"]]` — cross-folio recurrence pairs flagged for manual pixel/label verification. | Hand-picked during a session (K5-adjacent) | Audit notes only 2 of 17 pairs were ever actually inspected — most of this list is unexploited |
| `collision_targets.json` | 20 cross-section word-collision targets (e.g. `"otolor"`) — words that recur in visually unrelated contexts, used for K5's "keyed catalog" kill (pixel test found no consistent referent). | `collision_signal.py` (writes this file) | Feeds the manual IIIF pixel-inspection step of K5 (done outside any script) |
| `z10_bindings.json` | The K14 zodiac-label→referent bindings: `{"built","method","folios": {folio: {census, alignment, labels:[...]}}}` for the 4-ring, 104-locus 10% zodiac slice (f70v2 Pisces, f71r Aries, f72r2 Gemini, f73v Sagittarius). Anchor-based: labels read from pixels and matched against the Takahashi candidate set, not inferred from reading order. | 4 vision subagents, 2026-07-01, using `data/grounding/z10/` protocol materials | `z10_grounding.py` (reads it as fallback when no live `bind_*.json` scratch files exist; runs the K14 within-folio shuffle-null attribute test), `z10_cross.py` (K14b cross-ring tests) |
| `_labels/*.json` (28 files) | Per-folio raw label-locus extracts, one file per grounded folio (`f1r.json` … `f99v.json`), e.g. `[{"locus":"@Lx","role":"label","words":["d","g"]}, ...]`. This is the per-folio breakdown of what `_ground_payload.json` bundles together. | Extracted from `corpus.json` label loci (script not preserved) | Consumed by the vision grounding subagents (one file per agent's folio) |

### data/grounding/z10/ — K14 protocol materials (the 10%-slice zodiac binding pass)

| File | What it is | Produced by | Consumed by |
|---|---|---|---|
| `PROTOCOL.md` | The written protocol given to each of the 4 K14 vision agents: goal, file manifest (full-res scan + quadrant crops + tasklist), step-by-step binding procedure (census → clockwise enumerate → read anchors → grade confidence), and an EVA glyph cheat-sheet for reading labels off pixels. | Authored by the orchestrating session | Read by each of the 4 z10 vision subagents before starting |
| `worklist.json` | Per-ring label sequences for 4 folios (f70v2/f71r/f72r2/f73v — only), each entry `{locus, tier, ring_comment, label}` in transcription order. The task input for the 4-agent pass (subset of the full 12-ring zodiac). | Extracted from `LSI_ivtff.txt` Lz loci (script not preserved) | Fed to the 4 K14 vision agents alongside the per-folio `tasklist_*.txt` files |
| `tasklist_f70v2.txt`, `tasklist_f71r.txt`, `tasklist_f72r2.txt`, `tasklist_f73v.txt` | Human-readable versions of `worklist.json`, one per folio, grouped by ring tier — "TAKAHASHI Lz LABEL SEQUENCE... in transcription order," multi-word labels dot-joined. | Same extraction as `worklist.json` | Given directly to each folio's vision agent as its reading reference |
| `labels_raw.json` | Raw label dump for 4 folios (same set as `worklist.json`) — likely an earlier/intermediate form before tier-tagging. | Extraction pass (script not preserved) | Superseded by `worklist.json`/tasklists for the actual binding pass |
| `all12_labels.json` | Label sequences for **all 12** zodiac ring-folios (f70v1/f70v2/f71r/f71v/f72r1/f72r2/f72r3/f72v1/f72v2/f72v3/f73r/f73v) — the full-scale version of `worklist.json`'s 4-folio subset. This is the un-touched 90% the K14b/K14 overturn-condition scale-up would need. | Extraction pass (script not preserved) | Not yet consumed by any binding pass — the obvious next step flagged in FINDINGS §6 ("remaining 90% of the zodiac... obvious scale-up") |
| `z10_results.txt` | Saved console output of the K14 attribute test: binding counts per folio (104 loci, 4 folios), attribute distributions (body/star/clothing_color/container/star_tail/arms/headwear/facing/clock_sector) and per-attribute morphology MI/p-value tables. | `z10_grounding.py` (stdout, manually saved) | Reference copy of the K14 result; not re-loaded programmatically |

---

## data/iiif/ — IIIF manifests (Voynich + comparanda manuscripts)

All fetched directly from institutional IIIF endpoints (not produced by any Python
script — these are raw manifest downloads, re-fetchable from source at any time).

| File | What it is | Source (re-fetchable) | Consumed by |
|---|---|---|---|
| `iiif_urls.json` | 3 direct Yale IIIF image-service URLs keyed by folio number (`69r`, `88r`, `99v`) — the folios needing high-res pixel inspection for K5's collision test. | Yale Beinecke IIIF (`collections.library.yale.edu`) | Manual pixel-comparison step for K5 |
| `iiif_canvases.json` | Full IIIF canvas list for the Voynich manuscript itself — 213 entries `[label, width, height, image-service-url]` covering front cover through back matter. | Yale Beinecke IIIF manifest | General-purpose lookup for any folio's high-res image URL (used to fetch the `images/crops/iiif_*.jpg` and `pf_*.jpg` crops) |
| `cgm28_manifest.json` | IIIF manifest for **BSB Cgm 28**, "Kalender und Praktika, schwäbisch" (Swabian calendar/practica). M9/audit candidate exemplar. | Bayerische Staatsbibliothek IIIF | Used to fetch `images/comparanda/cand_cgm28/*` (calendar pages only — **its Sagittarius folio was never pulled**, an open audit item) |
| `cgm312_manifest.json` | IIIF manifest for **BSB Cgm 312**, "Losbücher" (fortune-telling books). M13's Sagittarius/Scorpio crossbow-witness manuscript. | Bayerische Staatsbibliothek IIIF | Used to fetch `images/comparanda/cgm312_*` |
| `csg0827_manifest.json` | IIIF manifest for **Cod. Sang. 827**, St. Gallen Stiftsbibliothek. M9's corrected/withdrawn "closest analog" (Lake Constance, 1425/28, in-window but no pictorial zodiac). | e-codices (St. Gallen) | Used to fetch `images/comparanda/csg827/*` |
| `_schurstab_manifest.json` | IIIF manifest for **Zürich Zentralbibliothek Ms. C 54** ("Schürstab" codex). M9's hand-bow Sagittarius cousin (~1472). | e-codices (Zürich) | Used to fetch `images/crops/_schurstab_22v.jpg` |

*(`cand_clm13076`, `cand_cpg291`, `cand_ljs449`, `cand_md2` comparanda images were fetched without saving a manifest JSON — no corresponding file in this directory.)*

---

## data/intermediate/ — cached script outputs and scratch arrays

| File | What it is | Produced by | Consumed by |
|---|---|---|---|
| `_spectacle_rows.json` | Per-side (226 entries) production/effort table: `{folio, sec, I, Q, P, H, prod, tok, lines, gelem}` — token count, line count, gallows-element count, and a `prod` (production-order) index per page-side. The substrate for M8's "no spectacle front-loading" finding. | Subagent (M8 test), script not preserved | M8 analysis (effort-calibrated set-piece score, done inline — not a checked-in script) |
| `_automaton_control.json` | Cached results of the K2 automaton test: 5 rows (`VOYNICH`, `HAWAIIAN`, `MAORI`, `LATIN`, `ENGLISH`?), each `{name, types, alpha, wlen, params, regen_tok, regen_typ, null_tok}` — regeneration rates at a matched 4,000-type lexicon cap. | `automaton_control.py` (writes this file) | Cached/reference copy of K2's headline numbers |
| `_worklist.json` | The 28-folio grounding worklist: `{folio, section, nlabels, codes, images}` — which folios needed vision grounding, how many label loci each has, which locus codes (Lx/Lz/Cc/Ri/Ro/L0/Ls) and which crop images to use. This is the plan behind the D12 28-agent grounding pass. | Extracted from `corpus.json`/`body.json` locus counts (script not preserved) | Fed to the 28 D12 grounding subagents (parallels `_ground_payload.json` in `data/grounding/`) |
| `_audit_results.json` | Self-catalog of 29 `analysis/` scripts: `{script, category, purpose}` for each — e.g. `pivot.py` / "investigatory" / (entity-declension test description). A metadata table describing what each analysis script does. | An audit/cataloging pass (script not preserved) | Reference for understanding script purposes; likely a source document for FINDINGS.md/the 2026-07-01 audit |
| `_entropy_panel.json` | Cached D5 entropy-panel results: 7 rows (`VOYNICH`, `HAWAIIAN`, `ENGLISH`, `LATIN`, `FINNISH`, `TURKISH`, +1 more), each `{name, alpha, h1, h2, h3}` — character-entropy figures underlying "h1=3.87 normal, h2/h3 anomalously low." | `entropy_panel.py` (writes this file) | Cached/reference copy of D5's numbers |
| `_sides.json` | `{"sides": [...202 entries...], "secof": {...202 folio→section...}}` — per-physical-side (recto/verso) table: `{side, I, sec, Q, P, prod, tok, panels, gel}`. A compact companion/precursor to `_spectacle_rows.json` (202 vs 226 sides — likely differs in how facing-folio and label-only sides are counted). | Same M8-adjacent pass as `_spectacle_rows.json` (script not preserved) | Not directly re-loaded by any checked-in script |
| `_sides_full.json` | Expanded per-side table, 202 entries: `{folio, section, nlabels, codes, images}` — same shape as `_worklist.json` but for all 202 sides rather than just the 28 illustrated ones. | Same extraction pass as `_worklist.json` (script not preserved) | Not directly re-loaded by any checked-in script |
| `_order.npy` | 1-D float array, 209 entries (`[0,1,2,...]`) — a page-order index. Length matches `page_features.json`'s 209 rows. | Script not preserved (companion to a page-level analysis, likely M8-related) | Not re-loaded by any checked-in script found |
| `_spec.npy` | 1-D float array, 209 entries, values in [0,1] (e.g. `0.996, 0.288, 0.9, ...`) — almost certainly a per-page "spectacle"/effort score (matches `_spectacle_rows.json`'s domain). | Script not preserved | Not re-loaded by any checked-in script found |
| `_labs.npy` | 1-D string array (`<U11`), 209 entries, values like `mixed`, `herbal` — per-page section/label array, same length as `_order.npy`/`_spec.npy`. | Script not preserved | Not re-loaded by any checked-in script found |

---

## Summary of gaps flagged during this audit

- **`eva_takahashi.txt` is a dead file** — contents are a literal `404: Not Found` string, not transcription data. Safe to delete or re-fetch from the actual Takahashi source.
- **No checked-in script produces `body.json` or `body_spatial.json`** (the D12 spatial spine) — only `vparse.py` (→`corpus.json`), `entropy_panel.py`, `automaton_control.py`, `collision_signal.py`, and `page_vision.py` write any data file. The spatial-spine build (IVTFF locus/role parse + 28-agent grounding merge) was run inline/ad hoc and never saved back as a reproducible script. Same gap applies to every file in `data/grounding/` (except `collision_targets.json`) and most of `data/intermediate/` — they're all vision-subagent or one-off-session outputs with no preserved builder.
- **`maori.txt`/`maori_bible.xml`/`_mri_tatoeba`** have no script reference at all — the K2 Māori analysis that produced the cited 30.9%/33.5% figures was not run from a script preserved in this repo (consistent with the 2026-07-01 audit's note that K2's Māori figures are "unsourced in audited transcripts").
- **`_mri_tatoeba`** (non-Bible Māori control) is fetched but unused — it's the fix for K2's "unclosed okina caveat" and is sitting ready to run.
- **`cribs.json`/`verify_pairs.json`** are largely unexploited: the L2 crib-anchored held-out decode was never built against `cribs.json`, and only 2 of 17 `verify_pairs.json` pairs were ever inspected.
- **`data/grounding/z10/all12_labels.json`** is the ready-made input for scaling K14 from its 10% slice (4 rings) to the full 12-ring zodiac — flagged in FINDINGS §6 as the obvious next step, not yet run.
