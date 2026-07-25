# Run Ledger — every analysis run, traced to its evidence

*Companion to `FINDINGS.md` (the authoritative claims ledger). Where `FINDINGS.md` states the
verdict, this file traces each run: what script produced it, what the transcript shows, and
what soft spots the 2026-07-01 audit (`docs/AUDIT-2026-07-01.md`) flagged. Sources: `FINDINGS.md`,
the `.py` script docstrings (all live at repo root), `docs/AUDIT-2026-07-01.md`, and
`transcripts/readable/*.txt`. Session-ID note: the audit found `FINDINGS.md`'s header session
attributions partly wrong — corrected mapping used here: K12 ran in `4e11ffbd`
(`transcripts/readable/short_session_4e11.txt`), K13 in `dcc71ad3`
(`transcripts/readable/k13_grounding_session.txt`, which runs `l6_grounding.py`), and the
M-series in `ac7bc224` (`transcripts/readable/cipher_and_motive_session.txt`, which runs
`ab_procedure.py`/`degradation.py`/`resistance.py`). The readable extracts were renamed to
match their content during the 2026-07-01 reorg. Where a run could not be pinned
to a script, docstring, or transcript passage, this file says so explicitly rather than guessing.*

---

## Campaign 1 — 2026-06-20 main session (structure discovery: K3–K10, D1–D8)

Transcript: `transcripts/readable/main_session.txt`. Session `308ba40b`.

### vparse.py
- **Question:** Can the IVTFF/EVA transcription be parsed into a clean token stream with metadata (Currier language, hand, quire/page, illustration type) preserved?
- **Method & control:** Deterministic parser over the Takahashi transcription (the only complete one); documented handling of uncertain spaces, nulls, unreadable glyphs, inline comments.
- **Result:** Produces the corpus substrate (`corpus.json`) every later script builds on.
- **Ledger:** infrastructure, not a claim row.
- **Caveats:** none noted.

### analyze.py
- **Question:** Where does Voynichese look like language, and where does it violate language, across a core statistical battery?
- **Method & control:** Same pipeline run on Voynich, Latin (Cicero), English (KJV), plus null-hypothesis generators.
- **Result:** Establishes the baseline stats (Zipf, entropy, word length) later scripts build on.
- **Ledger:** D1 (Zipf slope −0.89), D5 (entropy panel groundwork).
- **Caveats:** none noted.

### null_and_morph.py
- **Question:** (1) Do meaningless generators mimic Voynich surface stats? (2) How much of the vocabulary does a hand-built prefix-core-suffix slot grammar explain?
- **Method & control:** Compare real text to constructed nulls; measure slot-grammar coverage.
- **Result:** Early version of the null-generator method later formalized in `residual.py`/`genfit.py`; early slot-grammar coverage measure.
- **Ledger:** feeds D3 (slot-template morphology) and confound #3 (content-free reproducibility).
- **Caveats:** none noted.

### implicit_morph.py
- **Question:** Does unsupervised morphology induction (no hand-fed affix lists) rediscover the same prefix/core/suffix structure as the hand-built slot grammar?
- **Method & control:** Harris branching-entropy (forward/backward passes) and BPE merges, compared against the hand slot grammar for `qo-`/`-dy` rediscovery.
- **Result:** Three independent methods (hand, Harris, BPE) converge on the same morpheme inventory with zero linguistic input.
- **Ledger:** D1 — "slot grammar rediscovered by 3 independent unsupervised methods."
- **Caveats:** none noted.

### multimodal.py
- **Question:** Do labels beside drawings differ statistically from running prose (as a real naming vocabulary would), and does glyph choice depend on physical line-position?
- **Method & control:** Compare token stats (word count/line, vocabulary overlap, word-initial glyph distribution) across IVTFF locus roles (paragraph, label, circular line, radial line); compare paragraph-initial glyph frequency vs mid-paragraph.
- **Result:** Labels average 1.18 words/line vs 8.17 for prose; 54% of label vocabulary never appears in prose; labels start with `o-` 52% of the time vs ~20% in prose. **Paragraph-initial gallows law:** first word of a paragraph is 17.5× more likely to start with `p` (gallows) than mid-paragraph, `t` at 5.3×, `y` at 4.4×, while `c`/`o` are suppressed (full lift table: d ×1.8, s ×1.3, y ×4.4, o ×0.6, q ×0.8, t ×5.3, p ×17.5, c ×0.2).
- **Ledger:** UNLEDGERED — the "line-initial-gallows law" itself has no FINDINGS.md row (it feeds D3's "positional invariants" and D11 narratively but isn't stated with its own numbers there); the label-vocabulary-disjointness result is broader than D11 and also has no dedicated row. Found in `transcripts/readable/main_session.txt` (search "PARAGRAPH-INITIAL anomaly"). Audit doc calls this "the strongest non-null signal in the whole corpus."
- **Caveats:** audit doc flags the ~54% label/prose disjointness figure as under-logged relative to D11.

### decoder_trap.py
- **Question:** Does a naive frequency-rank "decoder" (Voynich glyphs → Latin letters, scored by Latin trigram likelihood) produce meaningfully different output for real Voynich vs. meaningless null text?
- **Method & control:** Run the same decoder on real Voynich and on constructed null text; compare trigram scores.
- **Result:** Both score about the same — demonstrates that "it comes out looking like Latin" proves nothing without a null.
- **Ledger:** methodological precursor to R1/`cipher_attack.py`'s positive-control discipline; not its own ledger row.
- **Caveats:** none noted.

### crossmodal.py
- **Question:** Does the picture predict the text on the same herbal folio (do visually-similar plants carry more-similar text than random folio pairs)?
- **Method & control:** Extract pigment features per herbal page; compare text similarity for visually-similar vs random folio pairs.
- **Result:** demonstrates the method; feeds into the later `crib.py`/`keyjoin.py` cross-modal line of attack.
- **Ledger:** precursor to K4 (superseded by `crib.py`→`keyjoin.py` held-out CV).
- **Caveats:** honest framing in the docstring: folio↔PDF alignment is approximate, color is a crude visual proxy.

### hash_hypothesis.py
- **Question:** Is a caption a key/hash rather than a description — and could either direction even hold the required information?
- **Method & control:** (A) functional-form test: does caption length scale with drawing complexity? (B) information-budget: bits in caption vs bits the drawing carries.
- **Result:** Average caption ~1,734 bits vs plant's categorical content ~10 bits — 181× too big to be a hash of the image.
- **Ledger:** D6 — "image ≈ lossy fingerprint of text, not text describing image."
- **Caveats:** audit doc: "D6 (181× info-budget rests on constants the adversarial fork's own script audit called fabricated)" — soft spot on the constants used.

### collision.py
- **Question:** Are labels collision-resistant (name-like) or reused (descriptive), and are repeated labels visually consistent?
- **Method & control:** Compare label collision-resistance to size-matched running prose; test whether repeated labels land in visually-consistent contexts.
- **Result:** feeds into `collision_signal.py`'s more developed test.
- **Ledger:** precursor to K5.
- **Caveats:** none noted beyond K5's own (below).

### collision_signal.py
- **Question:** What does repetition signal in labels — is the naming regime anti-repetitive (opposite of prose's hyper-repetition), and which collisions are generic vs surprising?
- **Method & control:** Compare label vs prose repetition regime against a null; classify collisions as frequency-predicted vs surprising.
- **Result:** Rare cross-section collisions (e.g. `otoky`) flagged as candidate "same object" signals — then killed by the pixel test.
- **Ledger:** K5 — "Keyed catalog" killed: rare collisions (`otoky` = drooping roots f88r vs striped bulb f99v; `okolo` = bathing nymph vs plant root) tag categorically different objects; no match on correct high-res IIIF pages.
- **Caveats:** audit doc: "K5 (2 of 17 pre-registered collision pairs inspected; reused K3's discredited folio-range classifier)" — the collision test's coverage and classifier are both flagged as soft.

### columns.py
- **Question:** Are the word's three parts (prefix/core/suffix) database-like columns — closed enums vs an open key, roughly independent?
- **Method & control:** Field cardinality/closure test; mutual-information between the three fields; positional-invariant tests; line-width conservation test.
- **Result:** prefix: 15 values/3.3 bits (closed); core: 3,042 values/6.8 bits (open key); suffix: 18 values/3.3 bits (closed). I(pre;suf)=0.20, I(pre;core)=1.33, I(core;suf)=1.61.
- **Ledger:** D3 — slot-template morphology.
- **Caveats:** none noted.

### structure_gaps.py
- **Question:** Does word order carry information (syntax), and are there calendar/list regularities (zodiac label counts, star-bulleted paragraphs)?
- **Method & control:** Bigram-information test vs within-line shuffle null.
- **Result:** within-line shuffle ≈ identical bigram entropy — no syntax signal.
- **Ledger:** historical D2 result, superseded by Campaign 8's lower-cardinality held-out morph-class test.
- **Caveats:** exact-word entropy was too sparse and in-sample to support the general "bag of records" claim.

### glue.py
- **Question:** Is there a "surface" the manuscript's high-frequency glue words adhere to — line-edges, or a fixed line template?
- **Method & control:** Glue fraction by line position; KL-divergence of line-edge word distributions vs interior, vs a shuffle null; per-slot dominant-token template; line-length regularity (CV).
- **Result:** glue fraction by position: first 15.1%, second 31.2%, interior 27.9%, penult 25.4%, last 17.7%. Line-edge specialization: line-first vs interior 6.82 bits real vs 3.70 shuffled; line-last vs interior 6.51 bits real vs 3.80 shuffled. Line length: mean 8.4 words/line (CV 0.32), mean 44 glyphs/line (CV 0.31). Most distinctive line-openers: `daiin, saiin, dain, sor, sol, sain`; line-closers: `daiin, dy, dam, am, dal, ol`. `daiin` is 2.2% of all tokens and dominates both opener and closer slots.
- **Ledger:** UNLEDGERED — the "daiin-as-line-punctuation" reading (`daiin` best read as a line-boundary marker, not a lexeme) has no FINDINGS.md row. Found in `transcripts/readable/main_session.txt` (search "daiin"). Feeds D3's positional-invariants line but the specific claim and numbers aren't logged.
- **Caveats:** audit doc lists this among under-logged positives.

### manifest.py
- **Question:** Does the prose (established as a word-order-free bag) segment into discrete "manifests" (inventories) with internally-coherent vocabulary and detectable boundaries?
- **Method & control:** TextTiling (vocabulary-dissimilarity sliding window) over all 4,116 prose lines; validated against random segmentation and against physical paragraph marks; plus an inventory-completion (new-type accumulation) test.
- **Result:** 308 candidate manifest boundaries (~13 lines apart). Segments more internally coherent than random cuts: 0.686 vs 0.668, p=0.002. Boundaries align with physical paragraph breaks 42% of the time vs 23% chance (58% of vocabulary shifts fall between paragraphs). New-type rate spikes just after a boundary: 0.294 vs 0.249 interior.
- **Ledger:** UNLEDGERED — the TextTiling manifest-boundary result (p=0.002) has no FINDINGS.md row. Found in `transcripts/readable/main_session.txt` (search "TextTiling"). Interpreted as "overlapping inventories drawn from one shared item-pool with soft boundaries," not crisply-separated manifests — consistent with, but not identical to, D2's "bag of records."
- **Caveats:** audit doc lists this as under-logged; effect size is explicitly called "small" in the transcript itself.

### log_test.py
- **Question:** Is the text a manifest (exchangeable bag) or a transition-log (trajectory with directional drift), within paragraphs?
- **Method & control:** T1 lag-decay, T2 directional drift, T3 proportion-shift, each vs within-paragraph shuffle nulls.
- **Result:** T2 directional-drift null: real +0.031 vs shuffled +0.027 — negligible; no A→B process.
- **Ledger:** K6 — "Transition log / process trajectory" killed.
- **Caveats:** none noted.

### roster_test.py
- **Question:** Does the text behave like a roster/org-chart (persistent named entities recurring across snapshots, with changing attributes)?
- **Method & control:** Persistent-membership test; core-population size/tail test; entity-attribute-change test; prefix-nesting/hierarchy test.
- **Result:** feeds into D7 (per-entity record stability) and D3 (core as open key).
- **Ledger:** feeds D7; no independent kill/survive row of its own.
- **Caveats:** none noted.

### genealogy.py
- **Question:** Is the text a genealogy (relational DB over time) — do folios show lineage-naming (T1), begat-chaining (T2), or families concentrated on pages (T3)?
- **Method & control:** Three discriminating tests vs the section-dialect confound.
- **Result:** discriminating predictions (lineage-naming, begat-chaining) failed; library imprint reads as medical genres, not charters/registers.
- **Ledger:** K7 — "Genealogy / lineage record" killed.
- **Caveats:** none noted.

### music_test.py
- **Question:** Could Voynichese be music (repeated phrases/motifs, melodic memory, metrical periodicity)?
- **Method & control:** Exact word n-gram repeats vs within-line shuffle; repeated whole-line search; glyph-stream h1→h2→h3; word-length autocorrelation.
- **Result:** no motifs beyond 3-grams (4-grams = 0%); zero repeated lines anywhere; no meter (word-length autocorrelation 0.06).
- **Ledger:** K8 — "It's music" killed.
- **Caveats:** none noted.

### index_test.py
- **Question:** Is the folio-opener word a real index entry that predicts the folio's section/picture-type?
- **Method & control:** Opener-predicts-section vs a random-word-from-the-same-folio control.
- **Result:** opener predicts section *below* the majority baseline (36% vs 56% always-guess-herbal); no ordinal trend.
- **Ledger:** K9 — "folio-opener is an index" killed.
- **Caveats:** none noted. (Note: `pivot.py`/main-session narrative separately found 153 distinct, 74%-unique, gallows-headed folio-openers — a real "formatting spine" — but confirms it "points at nothing," consistent with K9.)

### pivot.py
- **Question:** Pivoted along the entity (core) axis instead of the word-row axis: is the entity-attribute space low-dimensional, do entities cluster into classes, and do those classes map to sections/label-vs-prose?
- **Method & control:** Dimensionality reduction on entity-attribute profiles; clustering; cross-tabulation vs section and locus role; folio-opener sequence as a candidate index column.
- **Result:** folio-openers: 153 distinct across 206 folios (74% unique), all gallows-headed — read as a genuine structural invariant that nonetheless "points at nothing" (formatting spine, not table of contents; opener's prefix is *less* section-informative than a random word's, 0.12 vs 0.27).
- **Ledger:** corroborates K9; not a separate ledger row.
- **Caveats:** none noted.

### library_shadow.py
- **Question:** Does the corpus preserve the "shadow" of a real library of source-works — a catalog with sizes, a cross-reference network, a shared "glue" stratum, hub/isolated sections?
- **Method & control:** Recover works/sizes from illustration-label folio ranges; cross-reference edges between sections; shared-vocabulary stratum.
- **Result:** "works" were hand-drawn folio-number ranges that don't match the real IVTFF illustration labels (pharma bucket is half herbal pages; astro fuses 3 things; cosmo is 2 folios); cross-reference edges track Currier dialect + section size, not "function."
- **Ledger:** K3 — "Six recovered works / library catalog" killed (killed specifically by the adversarial-fork re-run of this script, `1bb4a48a`).
- **Caveats:** none noted beyond the kill itself.

### hinges.py
- **Question:** Is `q-` a separable, productive toggle between prose and label registers (a literal hinge glyph)? Do prefix/suffix co-vary? Is the hinge position-conditioned?
- **Method & control:** H1 q-operator toggle test; H2 prefix/suffix mutual information; H3 positional conditioning of prefix choice.
- **Result:** establishes q- as a register-switching hinge; feeds into D11's later, fuller confirmation on the spatial body.
- **Ledger:** precursor to D11.
- **Caveats:** none noted.

### structure_atlas.py
- **Question:** How much of the corpus does the slot grammar explain (coverage), where does the register "hinge" live (lexicon vs position), and where are the sharp structural edges across folios?
- **Method & control:** Coverage measurement; prefix-entropy collapse under different conditioning variables; per-folio register score with jump detection.
- **Result:** locates the hinge in the lexicon, modulated by page position.
- **Ledger:** feeds D3/D11 narratively; no independent ledger row.
- **Caveats:** none noted.

### stroke.py
- **Question:** Does decomposing EVA glyphs into pen-stroke primitives reveal a real scribal tradition underneath the invented alphabet?
- **Method & control:** Stroke-primitive inventory count; gallows+bench compositionality; minim-run length distribution; figure-9/figure-8 positional analysis; compared against Cappelli's Latin abbreviation paleography via a subagent.
- **Result:** 17 stroke primitives = standard Latin minuscule motor set; figure-9 (EVA `y`) is 84% word-final, matching Cappelli's round-9 `-us/-um/con-` siglum's positional grammar; gallows+bench ligature in 11% of gallows tokens; minim runs cluster 1–2, never 4+.
- **Ledger:** D8 — ⚪ Latin-trained abbreviating scribe (external corroboration via Zandbergen/Petersen literature, per subagent).
- **Caveats:** none flagged in audit; subagent flagged an internal tension (cadel/looped gallows read as slightly more Burgundian/French than Italian) reflected honestly in D8/D9.

### ring_column_compare.py
- **Question:** Are f57v's inner ring (R2) and f66r's isolated-glyph column the same reused alphabet/key?
- **Method & control:** Set-overlap (Jaccard), ordered-LCS, known-alphabet-order tests, vs null.
- **Result:** set Jaccard 0.48 (null p=0.32), ordered LCS at chance, no Latin/EVA alphabet order fit.
- **Ledger:** M7 — "No authored symbol-key" — independent display flourishes, not a reused alphabet/key.
- **Caveats:** none noted (M7 entry notes a self-correction: f66r = a word-column + a separate 34-glyph isolated column, not one object).

### burst_tables.py
- **Question:** Do words clump topically (burstiness) more than a Poisson/random spread, and is the corpus a single table or a collection of section-specific schemas?
- **Method & control:** Burstiness coefficient B on inter-occurrence gaps, vs English/Latin and vs global/within-section shuffles; schema-invariance and domain-divergence tests.
- **Result:** feeds into the burstiness numbers later re-scored by `residual.py`/`topic_gen.py` (IoD 1.31 vs 0.99 i.i.d. floor).
- **Ledger:** feeds K10's burstiness numbers; see audit note below on the metric swap.
- **Caveats:** audit doc: `topic_gen.py`'s burstiness metric (B=0.138) and `residual.py`'s metric (IoD) were never reconciled on the same statistic — flagged as the "K1/K10 metric swap."

### genfit.py
- **Question:** Does a schema-sampler + self-citation copying step reproduce all of Voynich's headline statistics (Zipf, word-length, hapax, entropy, adjacent-repeat, burstiness)?
- **Method & control:** Model A (independent record sampling) vs Model B (A + self-citation) vs real data.
- **Result:** copying-augmented model reproduces most statistics; burstiness is the key discriminator later refined by `topic_gen.py`/`residual.py`.
- **Ledger:** feeds K1/K10 (content-free reproducibility), confound #3.
- **Caveats:** none noted beyond the metric-swap issue above.

### topic_gen.py
- **Question:** How many meaningless latent "topics" does a drifting generator need to match Voynich's real burstiness (0.138) and hapax tail (0.70)?
- **Method & control:** Sweep topic-count K; compare generated burstiness/hapax to real values.
- **Result:** at the time, called "the strongest result of the entire investigation" for a pro-content reading (an unreachable burstiness/hapax combination) — later superseded by `residual.py`'s broader content-free sweep on a different metric (IoD).
- **Ledger:** superseded input to K10; the reconciliation itself is UNLEDGERED per the audit (K1/K10 metric swap, never re-run on the same statistic).
- **Caveats:** audit doc: explicitly flagged as never reconciled with `residual.py`'s IoD metric.

### register.py
- **Question:** Do prose and labels share a lexicon (substrate), differ only in affix weighting (register), and is there a deterministic operator mapping one to the other?
- **Method & control:** Core-morpheme overlap (token-weighted); KL-divergence of prefix/suffix distributions between registers; verbatim-vs-re-dressed label matching; own-folio vs other-folio locality test with permutation null.
- **Result:** 78.6% of label core morphemes also occur as prose cores (token-weighted coverage; raw Jaccard 0.07 is misleadingly low due to prose's larger core inventory). Only 46% of labels appear verbatim in prose, but 73% share a core with a prose word ("re-dressed"). KL divergence: prefix 0.62 bits, suffix 0.23 bits. Operator: `label ≈ {o/ot/ok} + (shared core) + {y/al/ar}`. Locality: mean label-core ∩ own prose 0.088 vs other prose 0.078, paired diff +0.010, permutation p=0.0288 (flagged in-transcript as likely section-dialect confound, not real page-local derivation).
- **Ledger:** UNLEDGERED — the affix-swap transform itself has no FINDINGS.md row (called "the strongest positive finding of the whole investigation" in the transcript, and cited in FINDINGS.md's own confound/lead prose but never given a numbered row). The invertibility follow-up (does each stem have one canonical label form? a clean invertible rule = grammar) proposed in the transcript was never run. Found in `transcripts/readable/main_session.txt` (search "affix-swap").
- **Caveats:** audit doc flags this explicitly as under-logged, plus the never-run invertibility follow-up.

### crib.py (first pass)
- **Question:** Is there a glyph n-gram over-represented in text grouped by a visual plant attribute (e.g. flower colour), beyond what random regrouping of the same folios would give?
- **Method & control:** Vision-tagged structural morphology grouped by attribute value; n-gram over-representation vs random-regrouping null.
- **Result:** in-sample near-miss (root p=0.056) later shown to be dialect leakage.
- **Ledger:** feeds K4 (superseded by `keyjoin.py` held-out CV, see Campaign 2).
- **Caveats:** see K4 below.

### zodiac_join.py (first run)
- **Question:** Does the one decoded key in the manuscript (the Occitan zodiac month-names) join referentially to the ~30 day-labels around each medallion?
- **Method & control:** (A) within- vs across-medallion label-core overlap. (B) correlation of label features (mean length, q-rate, suffix-y rate, ok-prefix rate) with month ordinal. (C) within-ring adjacent-vs-distant label similarity (day-progression test).
- **Result:** (A) day-labels share only ~10% core vocabulary across medallions (collision-resistant naming). (B) `q_rate` vs month-ordinal looked real at r=+0.48, but diagnosed as the manuscript-wide q-gradient (folio-position confound, since the zodiac folios span exactly where global q-rate rises) rather than a month signal; other features flat (mean_len r=+0.15, suf_y r=+0.38, pre_ok r=-0.27 — none robust). (C) adjacent-label similarity 0.153 vs distant 0.130, gap +0.023 — negligible day-count signal.
- **Ledger:** UNLEDGERED — this null result (the decoded month-key does not join referentially to its own day-records) has no FINDINGS.md row, despite being called "the cleanest possible confirmation of the whole arc" in the transcript. Found in `transcripts/readable/main_session.txt` (search "zodiac_join"). Re-run later in the adversarial fork (see Campaign 2) with the same null outcome, and the L5/L2 ledger entries reference the ring-order result (+0.0235, p=0.043) without crediting this script by name.
- **Caveats:** audit doc lists `zodiac_join.py`'s referential null explicitly as under-logged; also notes the script's own narrative claims "|r|~0, month-blind" while the actual q_rate correlation is r=+0.48 — a hand-waved near-miss, though correctly diagnosed as confounded folio-position drift, not a month signal.

### page_vision.py
- **Question:** Can section structure (herbal/astro/balneo/pharma/recipe) be recovered from page images alone, with no labels?
- **Method & control:** Cheap per-page visual features (ink coverage, pigment fractions, text-likeness) let speak for themselves (unsupervised).
- **Result:** feeds the later, more developed color/pigment agent work (see M-series iconography).
- **Ledger:** exploratory; no dedicated ledger row.
- **Caveats:** none noted.

---

## Campaign 2 — Adversarial fork (content-free generator, K1/K10, body_spatial build, D11/D12)

Transcript: `transcripts/readable/adversarial_fork.txt`. Session `1bb4a48a`.

### library_shadow.py (adversarial re-run)
- **Question:** Does the "six recovered works" catalog claim survive an adversarial audit of its own classifier?
- **Method & control:** Re-examine the folio-range classifier against the real IVTFF illustration labels.
- **Result:** classifier mismatches (pharma bucket half herbal; astro fuses 3 things; cosmo is 2 folios); cross-reference edges track Currier dialect + section size.
- **Ledger:** K3 — "Six recovered works / library catalog" killed.
- **Caveats:** none beyond K3's own.

### crib.py / keyjoin.py (held-out capstone)
- **Question:** Does a vision-scored botanical crib (flower colour, structural tags) predict text beyond a held-out, label-permutation-controlled, length-controlled test?
- **Question (Part 2):** Does the decoded zodiac month-key join referentially under the same discipline?
- **Method & control:** Leave-one-out cross-validation + label-permutation null + length control (Part 1); identifiable referential tests immune to position-drift confounding (Part 2).
- **Result:** Dead out-of-sample: root near-miss p=0.056 in-sample collapses to p=0.53 held-out; in-sample signal was Currier A/B dialect leakage.
- **Ledger:** K4 — "Text describes the plants" killed.
- **Caveats:** none beyond K4's own.

### automaton_control.py
- **Question:** Does a first-order glyph automaton's 61%-vs-14% Voynich/English regeneration gap prove Voynich words are non-linguistic automaton output, or does it just measure orthographic regularity for which English is an unfair baseline?
- **Method & control:** Same automaton run on Voynich, English, Latin, plus templatic/simple-phonotactic real languages (Hawaiian, Māori); a 0th-order null baseline to isolate the automaton's earned gap.
- **Result:** At a matched 4,000-type lexicon, Hawaiian matches 30.9% and Māori 33.5%, beating Voynich (24–48% depending on cap). On the structure-isolating lift metric, Voynich (×57) sits below Swahili (×97).
- **Ledger:** K2 — "The automaton proves it isn't a language" killed.
- **Caveats:** audit doc: "K2 (Māori 33.5%/lift figures unsourced in audited transcripts; unclosed okina caveat on the Hawaiian control)."

### residual.py
- **Question:** Unifying re-run: build ONE content-free generator (schema-sampler + self-citation fitted to real burstiness), then test only where meaning could survive it — cross-modal coupling, order/trajectory, line-position, within-line syntax.
- **Method & control:** Matched content-free null vs Voynich on: cross-modal (D1), order/trajectory (D2), line-position (D3), syntax (D4).
- **Result:** Real burstiness IoD 1.31 is only ~30% above the i.i.d. noise floor (0.99); naive copying overshoots (4.57); real value sits inside the content-free range. D3 (line-position, gallows-initial/word-length by column) is the one surviving REAL signal (real 0.22/0.06/0.05 vs generator flat 0.09/0.08/0.09) but reclassified as a content-free layout rule (the line-initial-gallows law), not meaning.
- **Ledger:** K1 ("sealed reference database" reframe) and K10 ("burstiness proves topical content") both killed.
- **Caveats:** the D3 line-position survival is exactly the "line-initial-gallows law" — itself unledgered by name (see `multimodal.py` entry above).

### body_spatial.json build (28-folio vision grounding)
- **Question:** Can the manuscript's spatial spine (line-position, locus role, section, Currier, hand) be reconstructed deterministically from IVTFF, and can labels be grounded to drawn elements via vision?
- **Method & control:** Deterministic IVTFF parse (Takahashi primary) for `body.json`; 28 vision-grounding agents (one per illustrated folio) binding labels to drawn elements, layout topology, and per-object labels.
- **Result:** 37,019 tokens carrying IVTFF locus role, line-position, `$I` section, Currier, hand; 28 illustrated folios image-grounded (zodiac rings; sun-rosette 17-caption↔17-star 1:1 match; pharma jar columns; nymph scenes); 6,100 tokens on grounded folios in the first assembly pass, 367 zodiac day-labels keyed to a sign.
- **Ledger:** D12 — "Spatial body reconstructed (L3/R2 DONE)."
- **Caveats:** none noted beyond general vision-grounding confidence caveats carried into K13/K14.

### D11 (q-register law), derived from body_spatial.json
- **Question:** Is `q-`/`qo-` running-text scaffolding, near-absent from naming (labels)?
- **Method & control:** Label-vs-prose split on the rebuilt spatial body, checked in every section.
- **Result:** label q-initial 0–3% vs prose 10–25%, holds in every section (bio 3/25, pharma 0/13, herbal 0/10, zodiac 0/–).
- **Ledger:** D11.
- **Caveats:** none noted.

---

## Campaign 3 — Dispute fork (K2 corroboration, K11 corroboration)

Transcript: `transcripts/readable/dispute_fork.txt`. Session `780e52c1`.

### automaton_control.py (dispute-fork re-run)
- **Question:** Does K2's kill hold up under a fresh adversarial dispute?
- **Method & control:** same as Campaign 2's `automaton_control.py`.
- **Result:** reconfirms K2's numbers.
- **Ledger:** K2.
- **Caveats:** same as above.

### entropy_panel.py
- **Question:** Is Voynich's low glyph conditional entropy (h2~2.1) a real cipher clue, or does a simple templatic real language (Hawaiian) reproduce it?
- **Method & control:** Conditional entropy h1/h2/h3 on Voynich, English, Latin, Finnish, Hawaiian.
- **Result:** h1=3.87 (normal, in the pack with English/Latin); h2=2.12, h3=1.90; h1→h3 drop of 1.97 bits ≈ 2× any real language in the panel.
- **Ledger:** D5 — "Low entropy = sequential constraint over a normal alphabet, not a small alphabet."
- **Caveats:** none noted.

### position_lock.py
- **Question:** Independent replication of the position-lock diagnostic against verbose-cipher: are Voynich glyphs position-locked (template slot-fillers) rather than roaming (cipher symbols)?
- **Method & control:** Relative within-word position binned into 5 bins; frequency-weighted mean positional entropy; compared to real languages.
- **Result:** Voynich 0.67 bits positional entropy vs 1.10–1.28 for every real language tested; directional replication on a 5-bin relative-position metric gives Voynich 1.70 bits (lowest) vs Hawaiian 1.95 / Latin 2.15 / Finnish 2.16.
- **Ledger:** K11 (corroborating probe b3).
- **Caveats:** none noted; ledger explicitly flags this gap as "more moderate than the absolute-position metric."

### unit_collapse.py
- **Question:** Does Voynich's unit-level h2 climb back into the natural ~3-bit band under BPE glyph-merging (the verbose-cipher signature — redundancy packed inside recoverable glyph-groups)?
- **Method & control:** Greedy BPE applied identically to Voynich and real-language controls; track unit-alphabet size and unit-h2 across merges.
- **Result:** Voynich's unit-h2 climbs ~3× faster than real languages under BPE merging (originally read as pro-cipher evidence, L1).
- **Ledger:** input to K11 (later killed by the matched null in `cipher_gate.py`/`verbose_cipher_null.py`).
- **Caveats:** see K11 below.

### verbose_cipher_null.py
- **Question:** Does the unit-h2 BPE climb (L1/`unit_collapse.py`) survive a matched content-free null?
- **Method & control:** A first-order glyph-Markov surrogate trained on the Voynich token stream (reproduces glyph unigram/bigram stats exactly, no message, no multi-glyph units), run through identical BPE.
- **Result:** Voynich climbs +1.73; the meaningless surrogate climbs +1.95 — *more* than Voynich. The climb is an orthographic artifact of templatic spelling, not recoverable cipher units.
- **Ledger:** K11 — "Verbose-cipher signature" killed (part 1 of 3 probes).
- **Caveats:** none noted; corroborated independently by a second content-free null (a morpheme-slot template) in the same K11 row, which also climbs +1.72 — at/above Voynich's +1.73 at every merge depth.

---

## Campaign 4 — Cipher campaign (K11, R1, K12, L7)

Transcripts: `transcripts/readable/cipher_and_motive_session.txt` (session `ac7bc224`, R1) and `transcripts/readable/short_session_4e11.txt` (session `4e11ffbd`, K12/L7).

### cipher_gate.py
- **Question:** Formal decision gate on L1 (verbose cipher): does the BPE unit-h2 climb need structure the surrogate's bigrams don't carry?
- **Method & control:** Voynich vs the glyph-Markov surrogate vs a real-language panel (Hawaiian, Māori, Latin, English), all capped to Voynich's token count.
- **Result:** Voynich and surrogate climb the same — the gate FAILED (outcome 2 of 3 possible: "climb is a generic artifact, not diagnostic").
- **Ledger:** K11 (the decisive kill of L1, the project's last live verbose-cipher hope). FINDINGS.md §6: "Step 1 RESULT (2026-06-20, ran `cipher_gate.py`): the gate FAILED."
- **Caveats:** none noted.

### cipher_attack.py
- **Question:** R1 — does a standard substitution hill-climb (with a positive control and a held-out test, the two disciplines every fake Voynich "solution" omits) recover a substitution key that makes Voynich read as Latin, generalizing to held-out text?
- **Method & control:** Bijective char→Latin-letter key hill-climbed on TRAIN half only (char-trigram LL vs a Latin LM), frozen and scored on TEST half. Diagnostics: ceiling (real Latin under the LM), floor (random-key decode), positive control (real Latin enciphered with a known key — must generalize), Voynich, and a content-free surrogate.
- **Result:** Held-out char-trigram LL/char (band floor −3.65 … ceiling −2.22): positive control −2.49 (81% of gap, method works); Voynich −2.70; surrogate −2.70; Voynich − surrogate = +0.002 bits/char (zero). Voynich falls 0.21 bits/char short of a cipher the method did crack, despite 6× more training data than the positive control. Stroke-seed (y=s) changed nothing.
- **Ledger:** R1 — "NULL." Major cipher families excluded (combined with the entropy bar and K11's position-locking).
- **Caveats:** audit doc: "R1 used a Latin-only target LM (classical Cicero). No medieval Latin, German, or Italian vernacular target was ever tried, despite M9's transalpine exemplar finding" — the "major cipher families excluded" claim is broader than what was actually tested; flagged as a cheap re-run worth doing.

### contact.py
- **Question:** The "Contact"/Hadden move — does the open-key core morpheme close (become predictable) when conditioned on column position (word-index-in-line) instead of the (already-dead) left-to-right row axis?
- **Method & control:** Condition core entropy on column position within one Currier×section stratum (controls dialect confound); position-shuffle null in-sample; held-out generalization test (train column-vocab on half the folios, predict the other half); discriminator vs the closed-enum (prefix/suffix) template floor.
- **Result:** In-sample: cuts core entropy 0.43–0.75 bits, beats the position-shuffle null at z=17–40, p≈0.003 in all 4 Currier×section strata. Held-out: column→core gain collapses to +0.03–0.07 bits (~1% of the key's ~6.5-bit entropy); 359–511 of 1146 distinct cores remain undetermined even fixing the column (recipes/B). Discriminator: core's generalizable column signal does NOT exceed the prefix/suffix template floor (herbal/A: core +0.031 < prefix +0.092; B strata core ≈ prefix).
- **Ledger:** K12 — "Column registration" killed. Same mirage class as K1/K10/K11: the content-free positional template, not a hidden semantic field.
- **Caveats:** none noted.

### quire_test.py
- **Question:** L7 — is the vocabulary autocorrelation behind confound #2 a smooth reading-order/shelf-distance decay, or a step-function at production units (quire/scribal hand)?
- **Method & control:** Within the herbal section only (spans multiple quires, hands, both Currier dialects — topic held constant); per-folio L2-normalized core-frequency vectors; pairwise cosine similarity; OLS on standardized predictors (same-Currier, same-quire, same-hand, reading-distance).
- **Result:** similarity driven by same-Currier (raw r +0.52) far more than same-quire/same-hand; smooth reading-distance ~0 once blocks controlled (partial r = −0.012); real quire-boundary step (+0.059 at matched distance) invisible to a distance-only control.
- **Ledger:** confound #2 (refined) and L7 ("DONE" — sharpened the control, no new meaning).
- **Caveats:** docstring's own caveat: tests quire-as-production-unit + scribe, NOT Fagin Davis's specific bifolio-reordering hypothesis (needs external codicological data) — "the in-repo 80% of L7." Audit doc separately flags M2 (which uses the same quire-order assumption) as needing a re-run on Davis's true codicological collation (quire order ≈ rebound order, not necessarily production order).

---

## Campaign 5 — Grounding campaign (K13)

Transcript: `transcripts/readable/k13_grounding_session.txt` (session `dcc71ad3`).

### l6_grounding.py
- **Question:** L6's only promotable, un-confounded form: within ONE section, does a label's morphology track the visual attribute of the specific referent it sits on, beyond a within-folio label↔referent shuffle?
- **Method & control:** Within-folio shuffle null (preserves section vocab, q-register law, template grammar, folio identity; destroys only referent binding). Discriminator: the core signal must exceed the prefix/suffix template floor (K12-style). Positive control (attr := f(prefix), must fire) and a noisy partial control (75%-consistent prefix→attr, ~25% binding error, must still be caught).
- **Result:** 8 vision agents bound labels to referents on pharma + balneo folios. Balneo arm dead on arrival (nymphs carry almost no visual variation — nearly every figure nude/arms-down/holds-nothing). Pharma arm (n=35 plant-elements, real root/foliage/flower and brown/red/green variation): no morpheme field beats the shuffle null (best part-class p=0.12; color core p=0.06, marginal and cardinality-driven, dies under multiple-comparison correction). The one sub-0.05 hit (jar-vs-plant core-length, p=0.02) is the K12 template artifact (ornate-jar labels are simply longer), not referent semantics. Positive control fires p<0.0001; the 75%-consistent noisy partial control still caught at p=0.0001 — confirms the null is a powered test, not underpowered.
- **Ledger:** K13 — "Within-section referent semantics" killed, and L6 marked "DEAD → K13."
- **Caveats:** audit doc: "K13 power gap (remaining half): K13's controls only proved detectability of a prefix-carried signal at pharma n=35, never a core-carried one; the balneo arm was never statistically run (eyeballed), though the raw 8-agent JSON has real variation." This gap was the direct motivation for K14 (`z10_grounding.py`), which added core-carried power controls at zodiac scale.

---

## Campaign 6 — Motive campaign (M1–M14)

Transcript: `transcripts/readable/cipher_and_motive_session.txt` (session `ac7bc224`).

### ab_procedure.py
- **Question:** M1 — Is Currier A/B one transmissible generative procedure run by multiple hands (workshop, produced-for-audience) or independent idiolects?
- **Method & control:** First-order glyph automaton per group; "foreign-dialect penalty" (cross-entropy of held-out words from group Y scored under group X's model, minus under Y's own model), calibrated against a same-system floor (random A1/A2 split) and a different-system ceiling (English vs Latin through identical machinery).
- **Result:** foreign-dialect penalty +0.229 bits/char = only 25% of the way from the same-system floor (A1/A2, +0.00) to the different-system ceiling (Eng/Lat, +0.90); support-Jaccard 0.77 ≈ floor.
- **Ledger:** M1 — combined with Davis's ≥5 hands, implies multiple people ran one shared machine → produced for an audience (kills motive classes A/D/E).
- **Caveats:** none noted.

### degradation.py
- **Question:** M2 — does generative discipline decay across production order (rushed commodity) or stay constant (sustained/lavished feat)?
- **Method & control:** Per equal-token chunk along quire+page codicological order, within each Currier dialect separately (controls genre/dialect confound); four discipline metrics; Spearman trend test with a permutation null reshuffling folio order.
- **Result:** within-dialect, quire-ordered: h2 flat, copying flat, hapax flat ~0.72–0.76 to the final quire (never recycles); only trend is word-length rising (B +0.76, p=0.011) = elaboration, not fatigue.
- **Ledger:** M2 — no deadline-rush churn → lavished/sustained feat, not a rushed commodity.
- **Caveats:** docstring's own caveat: quire order = the rebound order, an approximation of true production order. Audit doc lists "M2's re-run on Davis's true codicological collation" as an open next step (quire ≈ rebound, not production order).

### resistance.py
- **Question:** M10 — was Voynichese engineered to resist period frequency-analysis cryptanalysis (built to be impenetrable), or does it maximize exactly the regularities a codebreaker exploits?
- **Method & control:** Order-2 redundancy metric (Hmax−h2)/Hmax and adjacent-repeat rate, compared across random, homophonic-cipher (the era's real countermeasure), plaintext Latin, and Voynich.
- **Result:** order-2 redundancy: random 0.04 < homophonic cipher 0.17 < plaintext Latin 0.24 < Voynich 0.51; adjacent-repeat: ciphers ≈0%, Voynich 0.81%.
- **Ledger:** M10 — NOT security-engineered; it maximizes the regularities cryptanalysts feed on. Mandate = debunk-proof, not decode-proof.
- **Caveats:** none noted.

### herbal_control.py
- **Question:** M4 — does the "plants recombine freely, no cross-part constraints" claim (paralleling the text's free affixation) survive a real 27-species botanical control tagged with the same coarse scheme?
- **Method & control:** 27 real medieval-herbal species tagged by true morphology, same 4-slot scheme as `plant_tags.json`, coarsened into one shared per-slot vocabulary; matched permutation null on part-correlations, robustness checks (down-sampling, tag corruption).
- **Result:** raw mutual information is a mirage (real 3.07 ≈ Voynich 3.10), but the matched permutation null splits them cleanly: real botany's part-correlations beat their shuffle null p<0.0001 (species syndromes); Voynich's are indistinguishable from its own null, p=0.22.
- **Ledger:** M4 — corrected claim: "no syndromes vs null," not "MI≈0." Voynich plants recombine without the cross-part constraints real botany imposes.
- **Caveats:** none noted; the docstring itself frames this as supplying "the missing botanical control" for an earlier, weaker version of the claim.

### Subagent-only runs (no .py script): M3, M5, M6, M8, M9, M11, M12, M13, M14
- **Question (M3):** Does the imagery borrow real drawing-grammar while inventing composition (costume without content)?
- **Question (M5):** What is the production order (image-first vs text-first)?
- **Question (M6):** Do the manuscript's rarest glyphs/longest tokens hide an automaton-rule violation (a "tell")?
- **Question (M8):** Is effort/spectacle front-loaded (cheap-fraud signature) or distributed?
- **Question (M9/M11/M13):** What is the closest real-world exemplar tradition for the zodiac imagery (region/date/figure-type)?
- **Question (M12/M14):** What kind of patron/customer profile and what archival test could corroborate or falsify it?
- **Method & control:** Vision + web/IIIF subagent investigation in each case; no null-hypothesis statistical test (these are iconographic/codicological/archival, not corpus-statistical).
- **Result (headline numbers):** M3 ~85% borrowed-real drawing-grammar / ~75% invented composition; M8 effort-calibrated set-piece score ρ=+0.08 (n.s.), center-of-mass 0.59 (back half); M9 5 convergent South-German/Lake-Constance markers, crossbow type dates ~1425; M11 Clm 13076 matches 4–5/6 markers but ~50 yrs early; M13 two genuine human-crossbow Sagittarius witnesses (Heidelberg Cod.Pal.germ.291, BSB Cgm 312) finally pixel-verified, both post-window; M12 ranked patron types (signorial court with a standing astrology habit ranked best fit); M14 no centralized Visconti treasury pre-1468, 1447 records collapse, but Pellegrin's 1426 Pavia library inventory (988 entries) is published and unscanned for a Voynich fingerprint.
- **Ledger:** M3, M5, M6, M8, M9, M11, M12, M13, M14 (see `FINDINGS.md` §7 for full text).
- **Caveats:** audit doc: BSB Cgm 28 was fetched but its Sagittarius folio never pulled (`images/comparanda/cand_cgm28/`); Egerton 2724 and ÖNB 1842 deprioritized from literature descriptions alone (after the project twice proved that literature unreliable); the swallowtail-merlon check on the rosettes castle is "one IIIF crop away"; the herbal (~98-plant alchemical-herbal canon) and balneo (De Balneis Puteolanis tradition) never got the full exemplar hunt the zodiac got.

---

## Campaign 7 — 2026-07-01 (audit, K14, K14b)

No readable transcript for this campaign (it postdates the seven `transcripts/readable/*.txt` files; its own session `9966cd54` is the audit session itself, described narratively in `docs/AUDIT-2026-07-01.md`). Documented here from `FINDINGS.md`, the script docstrings, and `data/grounding/z10/PROTOCOL.md`.

### Transcript audit (docs/AUDIT-2026-07-01.md)
- **Question:** Five Sonnet analysts re-read all seven prior session transcripts end-to-end (plus ~113 embedded subagent reports) hunting for untested ideas, dropped threads, overclaimed kills, and unexploited/under-logged artifacts.
- **Method & control:** Cross-checked every claim against `FINDINGS.md`.
- **Result:** Verdict: asemic conclusion safe, no stone looked likely to flip it, but real gaps found in three classes — untested substrates (zodiac reference, cross-zodiac structure, R1's Latin-only target LM, L2's crib-anchored decode never built at scale), ledger integrity (under-logged positives: `register.py` affix-swap, `zodiac_join.py` null, the ~54% label/prose disjointness, the line-initial-gallows law, `daiin`-as-punctuation, TextTiling boundaries, the rosette causeway graph, the balneo pool-topology dataset), and iconography/motive asymmetries (herbal/balneo never got the zodiac's exemplar-hunt treatment).
- **Ledger:** meta — motivated K14/K14b (below).
- **Caveats:** the audit's own session-attribution corrections are the basis for this file's transcript↔campaign mapping (see header note).

### z10_grounding.py
- **Question:** K14 — the zodiac (highest-power label set, 1,222 labels) was inventoried by K13 as "has the power" then bypassed for pharma n=35. Does zodiac label morphology encode a nymph's visual attributes, tested at 3× K13's scale with a core-carried power control (the audit's identified gap)?
- **Method & control:** 10% slice: 4 rings (f70v2 Pisces, f71r Aries, f72r2 Gemini, f73v Sagittarius), 104 label loci, bound by 4 vision agents using anchor-based alignment (labels read from pixels and matched to the Takahashi candidate set, not pure reading-order inference — protocol in `data/grounding/z10/PROTOCOL.md`). 9 referent attributes (body, clothing-color, container, star-presence, star-tail, arms, headwear, facing, clock-sector) × 6 morpheme fields, within folio×tier shuffle null; power controls carried by both prefix and core, each at 100%/75% consistency; omnibus max-MI permutation for multiple-comparison control (not narrated Bonferroni).
- **Result:** ~60% exact reads, 0 rotational contradictions across all 10 tiers. NULL: Westfall–Young family-wise p=0.20 (all 104 loci), p=0.44 (high-confidence only, n=76). The 7 raw sub-0.05 cells behave like noise (star-presence cells vanish, clothing-color weakens, when low-quality bindings removed). Core-carried controls fire at p=0.0001 at both 100% and 75% consistency (prefix-carried 75%: p=0.004) — confirms the test had power to catch even a sloppy core-carried signal.
- **Ledger:** K14 — "Zodiac labels encode their nymph's visual attributes" killed. Closes the audit's identified K13 power gap.
- **Caveats:** vision by-catch noted in the ledger: f72r2 has a "clothed zone" (4 clothed figures on pedestals) plus one genuinely unlabeled figure; f73v locus-1 nymph has no star (3 stars serve 4 outside nymphs); label-placement conventions differ per folio (f71r labels sit at the star-tip between figures, so naive nearest-word binding would misbind the whole ring by one) — the anchor-based method was built specifically to avoid this failure mode.

### z10_cross.py
- **Question:** K14b — three hypothesis classes K14's within-ring shuffle null is structurally blind to: cross-ring positional correspondence, whole-zodiac same-label determinism, and ring-identity codes. Could information hide across the zodiac rather than within one ring?
- **Method & control:** Z-A cross-ring positional join: doubled-sign pairs (dark/white Aries f70v1/f71r; white/dark Taurus f71v/f72r1; 15 loci each) tested via cyclic-max alignment over all offsets+reversal, null = within-tier order shuffle, cross-sign pairs as baseline. Z-B same-label determinism: among the 104 K14-bound loci, do repeated words sit on similar-looking nymphs, vs a within folio×tier shuffle null. Z-C collision census: within-ring duplicate rate vs a random-assignment null, at full 12-ring scale (299 Lz loci).
- **Result:** Z-A: NULL — same-sign pairs never beat the cross-sign baseline (p=0.26–0.93 across edit-similarity/morph-match measures). Z-B: NULL, and anti-naming — 22 same-label pairs (`otal`×5, `okal`×3, `okaly`×3, …) match on referent attributes at only 0.545 vs shuffle p=0.85; the same word recurs on visually unrelated figures. Z-C: 254 types/299 tokens; within-ring duplicate pairs = 12, an EXCESS over the random-assignment null (p=0.018) — duplicates cluster within rings, the opposite of naming discipline; read as confound #5's copying signature.
- **Ledger:** K14b — killed. Untested residue flagged as K1-class by construction: ring-identity codes are confounded with block vocabulary (the decoded-key version already ran null in `zodiac_join.py`), and steganographic embedding in generator free-choices remains unfalsifiable without a candidate scheme.
- **Caveats:** none beyond the explicitly-acknowledged untested residue above (an open loophole the adversarial fork already flagged as unfalsifiable without a specific candidate scheme).

---

## Campaign 8 — 2026-07-22 (held-out follow-ups and conclusion audit)

### sequence_parallel.py / morph_voynich_transfer.py
- **Question:** Does word-class order generalize beyond exact-word sparsity and known line-position/production-block effects?
- **Method & control:** EVA prefix/suffix/affix/shape representations; train-only counts and vocabulary; complete-quire holdout; target-position-conditioned unigram baseline; uncertain and one-character tokens retained as adjacency breaks; quire-level bootstrap/sign-flip inference.
- **Result:** EVA prefix adds `+0.02025` held-out bits/transition, 13/16 quires positive, Bonferroni `p=0.00146`. Affix-pair forward-vs-deep-interior-reversed direction is `+0.1440`, positive in 16/16 quires, but the high-cardinality affix predictor has negative calibrated gain (`-0.458`). Cross-hand transfer is weak; Currier transfer is asymmetric.
- **Ledger:** D2 revised from "no word-order syntax / positional bag" to a narrow morph-class ordering result, then further qualified by Campaign 10.
- **Caveats:** the baseline does not contain exact previous-word identity. Campaign 10 shows the morph-class gain is nonpositive after a stronger surface hierarchy. The qualitative boundary dependency also predates this repository: Smith and Ponzi, [*Cryptologia* 43(6), 2019](https://doi.org/10.1080/01611194.2019.1596998), and Parisel, [arXiv:2604.19762](https://arxiv.org/abs/2604.19762).

### morph_control_benchmark.py
- **Question:** Is Voynich directional morphology distinguishable from known-meaningful text and verbose ciphertext?
- **Method & control:** Exact Voynich line/quire template imposed on official Naibbe ciphertext and Pliny controls; leave-one-block-out scoring under generic two-character boundaries and a separately marked Voynich-selected EVA sensitivity; official Naibbe revision and asset hashes recorded.
- **Result:** Generic affix-pair direction is Voynich `+0.10249`, meaningful Naibbe `+0.03245`, ordinary Latin `+0.13083`; Voynich is indistinguishable from Latin (`p=0.20367`). Under the selected EVA map, Voynich prefix gain is `+0.02025` while Naibbe is `-0.01695`, but this map was selected on Voynich and is not substitution-invariant.
- **Ledger:** D2 defended as structure, not semantics; L1 reopened but constrained.
- **Caveats:** control line/quire boundaries are artificial; generic high-cardinality predictors remain too sparse to beat their position-only baseline.

### naibbe_benchmark.py
- **Question:** Can the K11 diagnostics reject a known-meaningful verbose homophonic ciphertext?
- **Method & control:** Official author repository at pinned revision `f2675ec5...`; token-count/line-template matched comparisons; BPE, position-lock, EVA slots, lexical tail, order, and the repository R1 attack.
- **Result:** Naibbe is more position-locked than Voynich (`1.574` vs `1.682`), has a similar BPE climb (`+1.613` vs `+1.742`), and a close EVA slot/MI profile. R1 fails on it despite passing a simple-substitution control. Naibbe still mismatches Voynich hapax rate, burstiness, exact-token order, and selected-prefix transfer.
- **Ledger:** K11 split: BPE is still non-evidence, but the position-lock exclusion of verbose ciphertext is retracted.
- **Caveats:** Naibbe is a counterexample to the diagnostic, not evidence that the manuscript uses Naibbe.

### morph_procedural_nulls.py
- **Question:** Do the repository's content-free procedures reproduce the held-out directional signal?
- **Method & control:** 60 independent replicas each of strict layout exchange, `residual.py` schema/copy, and aggressive local self-citation; complete-quire scoring; preserved adjacency breaks; transition-profile residuals.
- **Result:** layout exchange and schema/copy direction intervals remain near zero (schema/copy affix `[-0.0150,+0.0158]`) versus real affix `+0.1440`. Aggressive self-citation can bracket generic prefix/suffix predictive gain but not directionality, and is degenerate (`none->none` prefix transition about 74% vs real 5%).
- **Ledger:** D2 and confound #3 revised; the existing generator is not a matched null for directional order.
- **Caveats:** generator schema is globally estimated, copy parameters are inherited hard-coded values rather than demonstrated fits, and 60 replicas give minimum raw empirical `p=0.0164`.

---

## Campaign 9 — 2026-07-23 (blind hidden-channel search)

### stego_channel_search.py / stego_bit_search.py / stego_slot_cipher.py
- **Question:** Is a low-complexity message carried by a fixed word/glyph subsequence, a binary glyph/word feature, or prefix/suffix free choices?
- **Method & control:** 332 pre-specified extraction channels; 2,560 Bacon-5/ASCII-7/ASCII-8 variants with shuffled bit-stream controls; homophonic substitution on prefix, suffix, pair, and interleaved streams with keys fitted on the first half and frozen on the second. Synthetic English and Latin homophonic streams are the positive controls.
- **Result:** No readable held-out output. The best fixed-channel outputs remain over 1 nat/character below language ceilings. Binary leaders have low printable fractions or letter soup and only small gains over shuffle. Voynich slot streams are at least 1.23 nats/character below their ceilings, while the synthetic controls recover fluent held-out text.
- **Ledger:** K15, limited to the enumerated fixed and monoalphabetic channels.
- **Caveats:** keyed routes, higher-radix arithmetic, variable per-page keys, and unconstrained generative encodings are outside this battery.

### naibbe_style_attack.py
- **Question:** Can an unknown Naibbe-like mixture of one-letter atomic words and two-letter compounds be segmented and solved without its plaintext key?
- **Method & control:** Naibbe's published table supplies only atomic/compound labels and boundaries for calibration; plaintext letter values are withheld. A structural ranker/classifier is fitted on the first half of official Naibbe and frozen on its second half and on Voynich. Homophonic keys are then fitted independently on train halves. Direct true-key application is used only as a diagnostic upper bound after blind scoring.
- **Result:** Held-out Naibbe token classification is 97.1%; complete emission boundaries are right for 93.4% of tokens. The true key over those noisy boundaries recovers clear Pliny, proving the segmentation is usable. Blind paired-swap key search scores Naibbe `-3.119` and Voynich `-3.206` against a Latin ceiling of `-2.040`, a separation of only `0.087`; neither blind output is plaintext.
- **Ledger:** L1 remains live but low-prior; this attack is inconclusive, not a Voynich lead.
- **Caveats:** the homophonic optimizer still leaves a large gap to the known-key control, so failure cannot reject the variable one/two-letter family. Naibbe's component shapes were designed from EVA and are a mechanism control, not independent evidence for that scheme.

### stego_transposition_search.py
- **Question:** Does a fixed nonstandard reading order expose substituted English or Latin?
- **Method & control:** 208 deterministic folio/line/word/glyph orders: reversals, odd-even unweaving, boustrophedon rows, and ragged vertical reads. Substitution keys are train-half fitted and held-out tested; initial leaders and ordinary order receive heavier 12-restart refits.
- **Result:** No readable output. Selection was optimizer-sensitive: ordinary forward order becomes the best English result on refinement (`-2.777`, ceiling `-2.093`). The refined Latin leader is `-2.778` versus ceiling `-2.142` and remains repetitive pseudo-Latin; ordinary order is `-2.820`.
- **Ledger:** K15 for these fixed route/transposition choices.
- **Caveats:** arbitrary keyed column widths, page-specific routes, diagram-specific spirals, and transposition combined with a variable homophonic code remain outside scope.

---

## Campaign 10 — 2026-07-23 (surface residualization and solver hardening)

### morph_markov_residual.py
- **Question:** Does the suffix/prefix arrow add held-out predictive information beyond observed word-level and local-copy structure?
- **Method & control:** Leave one complete quire out; train-only vocabulary/counts; uncertain and one-character tokens remain hard breaks. Nested next-prefix models add exact previous word, rare-core pooling, causal copy-lag state, and target position before testing source prefix or suffix. A separately fitted deep-interior analysis and 24 word-Markov surrogates use the same scorer.
- **Result:** Raw suffix(N)→prefix(N+1) is strong (`+0.07374` bits/transition, 14/16 quires, block-flip `p=0.00089`). Its primary surface residual is negative (`-0.01629`, 95% CI `[-0.02059,-0.01219]`, 1/16 positive); prefix→prefix is `-0.02411` with 0/16 positive. Real residuals are less negative than all 24 parametric surrogates, but morphology never improves absolute held-out likelihood over the declared surface model.
- **Ledger:** D2 retained as a descriptive dependency; K16 kills it as an independent solver clue.
- **Caveats:** prefix and suffix are deterministic functions of the exact source word, so this can only test sparse-context backoff. The surrogate generator is fitted globally and the 24-replicate empirical minimum is `p=0.04`.

### solver_residual_controls.py
- **Question:** Can source-prefix pooling after an exact-word hierarchy distinguish meaningful text/cipher from procedural output?
- **Method & control:** Matched 16-block leave-one-out tests on Voynich, Naibbe, Latin, substitution-invariant Latin, English, layout exchange, schema/copy, and self-citation. Proper held-out log score is primary; accuracy and novel-word score are secondary.
- **Result:** The surface exact-word stage detects English (`+0.19615` bits/event, 16/16 blocks, `p=0.00002`), showing pipeline sensitivity. Prefix pooling is negative for Voynich (`-0.06520`), Naibbe, Latin, and English. It is strongly positive only for aggressive self-citation (`+0.19417`, 16/16), making it a local-production diagnostic rather than a meaning detector.
- **Ledger:** K16.
- **Caveats:** non-Voynich controls use artificial Voynich line/block templates; procedural arms are single deterministic calibration points.

### residual_cipher_solver.py
- **Question:** Can a Naibbe-style variable one/two-character code produce held-out Latin likelihood beyond surface role and word-position statistics without being rewarded for generic directionality?
- **Method & control:** Four outer folds over 16 complete blocks, with 8 fit / 4 inner-validation / 4 outer-test blocks. Uncertain, one-character, unresolved, and out-of-fit-inventory emissions are hard breaks; no n-gram crosses a break, line, block, or fold boundary. Codebook size and sequential-mixture weight are selected on inner validation with an `M log2(n)/(2n)` mapped-symbol penalty. `lambda=0` is the exact inactive baseline.
- **Result:** The official Naibbe oracle activates in all folds at mean `+0.5353` held-out bits/emission. Blind Naibbe and blind Voynich both activate in 0/4 folds because every active candidate has negative inner-validation residual before the complexity penalty. Voynich therefore receives exactly zero residual evidence.
- **Ledger:** L1 remains live but underpowered.
- **Caveats:** At this stage blind recovery also failed the meaningful Naibbe control, so this run alone excluded no variable-key family. Campaign 11 supersedes that optimizer limitation for the declared Naibbe family.

---

## Campaign 11 — 2026-07-23 (powered known-family recovery)

### recurrence_decoder_prototype.py / recurrence_key_synth.py
- **Question:** Does first-occurrence recurrence encoding transfer information across randomized homophonic keys without sharing glyph identities?
- **Method & control:** Randomized Naibbe-style keys over disjoint Latin control text; recurrence/frequency/context features exclude glyph strings and official values. Models freeze before official ciphertext scoring. The second implementation separately reports oracle structural boundaries and a synthetic-trained blind segmenter.
- **Result:** The minimal statistical probe is weak but above its destroyed-recurrence control. The larger synthetic-key learner reaches **92.32%** weighted recovery on unseen synthetic keys and **59.05%** on official Naibbe with oracle structure; blind structure reduces official character recovery to **24.25%**.
- **Ledger:** L1 calibration only.
- **Caveats:** These are applicability probes, not generic Voynich decoders. Structural segmentation, not recurrence evidence, is the dominant bottleneck.

### naibbe_permutation_decoder.py
- **Question:** Can the official Naibbe key be recovered without reading its published letter values, and does the recovery generalize to untouched ciphertext?
- **Method & control:** Treat the cipher specification as 18 independently permuted 23-symbol role/table blocks. Tune rank/context initialization and deterministic tetragram refinement only on three randomized synthetic keys. Canonically sort glyph surfaces so CSV row order cannot encode the alphabet; freeze full and both half-stream keys before opening official plaintext. The Naibbe checkout was `f2675ec5dd275268bc64dd48ea64fc0e0e9827a2`; exact asset hashes and the recovered key are serialized.
- **Result:** Identity-row accuracy is **4.26%**. Unseen synthetic recovery is **99.21–99.69%**; official full-stream recovery is **99.61%**. Fit-first/test-second recovers **99.46%** and fit-second/test-first **99.64%**. The frozen output begins `ipomiferaearboresquaequemitioribus...`.
- **Ledger:** powered positive control for K17; the old blind-optimizer caveat is resolved for this known family.
- **Caveats:** This assumes the published Naibbe role/table inventory and unrespaced word boundaries. It is not a generic cipher solution.

### voynich_latent_channel_gate.py
- **Question:** Do the published inventory itself, or morphology-nearest/clustered latent channel assignments, transfer the powered Naibbe solver to held-out Voynich quires?
- **Method & control:** Four outer folds over 16 complete blocks: eight fit, four inner validation, four test. Segmentation statistics, channel maps, and permutation keys are fit-only; lambda is validation-selected; physical lines and hard breaks are preserved; mapped-symbol complexity receives an MDL penalty. Each channel hypothesis must first activate in all four corresponding Naibbe folds.
- **Result:** Three controls are powered: exact inventory and K23 nearest morphology average about **+0.484 bits/emission**, K12 nearest **+0.346**, all 4/4 active. Their Voynich counterparts are 0/4 active with mean raw residuals **−0.0384**, **−0.0649**, and **−0.0217** respectively. Morphology clustering fails its own Naibbe control and is excluded.
- **Ledger:** K17. No tested Naibbe-lineage channel hypothesis passes.
- **Caveats:** This closes only exact published membership and the declared nearest-morphology assignments under a Latin tetragram target. A materially different latent grouping remains outside scope.

### permutation_heldout_controls.py
- **Question:** Does the table-permutation optimizer manufacture held-out Latin gains from key breaks, block exchange, or matched table-Markov streams?
- **Method & control:** Two directionally held-out Naibbe attacks and 20 plaintext-free adversarial attacks across three null families. The key is frozen before the test half; the threshold is the strict maximum null residual and LM score.
- **Result:** First→second and second→first Naibbe residuals are **+1.1427** and **+1.0099 bits/tetragram**, with **99.46% / 99.64%** plaintext accuracy. Both clear the strict worst-null residual (**+0.7536**) and LM (**−4.5558**) cutoffs; 0/20 null attacks clear both.
- **Ledger:** K17 control.
- **Caveats:** This predeclared threshold calibrates one attack. It cannot be imported unchanged after searching additional latent groupings.

---

## Campaign 12 — 2026-07-23 (edit carriers and the remaining row axis)

### edit_operation_channel.py
- **Question:** Does the ordered edit that turns an adjacent or nearest-local-source word into its target carry a stationary homophonic plaintext symbol?
- **Method & control:** Forty-two predeclared source/distance/representation channels cover edit operation, position, changed glyph, source lag, and full source→target delta. Keys fit on complete fit quires, channel/language selection uses disjoint validation quires, and keys freeze before complete test quires. Position/lag-matched shuffles repeat shortlist selection. Embedded English/Latin uses the same physical layout.
- **Result:** Selected Voynich keys gain **+0.1118 nats/tetragram** over a frequency baseline, below the matched-null maximum **+0.1213** (`p=0.40`); output is unreadable and averages **−2.6095 nats/character** from its language ceiling. Embedded controls average **−0.1875** from ceiling and recover continuous plaintext.
- **Ledger:** K18.
- **Caveats:** This tests stationary many-to-one edit-symbol mappings, not arbitrary stateful or keyed edit ciphers. Four null replicates give coarse empirical resolution; the absolute language gap is the stronger discriminator.

### edit_choice_bitstream.py
- **Question:** Could the writer encode bits by choosing an odd/even-ranked alternative under the same source or edit context?
- **Method & control:** Rank dictionaries fit only on complete fit quires. Fifty direct/conditional channels are decoded under 80 fixed Bacon-5/ASCII phase, polarity, and bit-order variants; channel and codec are validation-selected and frozen for test. Event order is shuffled within quire for the null, preserving every context, alternative count, and bit imbalance. Bacon controls are embedded at observed per-quire capacities.
- **Result:** Embedded English/Latin Bacon payloads recover exactly (mean ceiling gap **−0.0299**). Voynich averages **−5.4894** below ceiling, is unreadable, and is worse than the null panel (`p=0.8889`).
- **Ledger:** K18.
- **Caveats:** Conditional rank is only one enumerative coding convention; keyed arithmetic coding or a changing rank dictionary is outside scope.

### row_axis_gate.py
- **Question:** Do successive prose rows behave as aligned table records after ordinary previous-row copying is controlled?
- **Method & control:** For 27,730 target cells, complete-quire held-out mixtures add (1) numeric-position vocabulary, (2) copying from anywhere in the preceding row, then (3) a same-numeric-column point mass. EM weights fit on train quires and freeze for test. Page-preserving row shuffles repeat the six-representation family-wise search. An 8% same-column copy injection is the positive control.
- **Result:** Raw same-column core matching is **7.43%** versus **7.08%** in neighboring source columns, but the row-bag component absorbs it. Every equal-quire diagonal residual is nonpositive (best **−0.00000013 bits/cell**; core **−0.0000403**), family-wise `p=1.0`. The injected table signal reaches **+0.1271 bits/cell** and improves 16/16 word blocks.
- **Ledger:** K19; L8 closed.
- **Caveats:** This excludes literal numeric-column continuity. More elaborate ragged alignment rules would be new hypotheses and need their own selection-adjusted null.

### Crib-input audit
- **Question:** Does `data/grounding/cribs.json` provide the ~200 known plaintext anchors assumed by L2?
- **Method & control:** Direct schema/content audit against the grounding and binding files.
- **Result:** No. It contains only Voynich label strings grouped by medallion/pharma context; there are no translations, plant names, month names, or other candidate plaintext values. K13/K14 already tested the available broad visual-class information.
- **Ledger:** L2 corrected from an unrun decode lead to a missing-input requirement.
- **Caveats:** A future independently established lexeme-level anchor would reopen a genuine crib attack.

---

## Campaign 13 — 2026-07-23 (parallel anchors and bounded line-state synthesis)

### parallel_passage_gate.py
- **Question:** Does the corpus contain a cross-quire near-parallel prose line strong enough to support differential cryptanalysis without a plaintext hint?
- **Method & control:** Lines from the same section and Currier dialect but different quires are proposed through uncommon shared cores. Order-preserving matching blocks receive inverse-document-frequency weights. Thirty-two strict nulls preserve quire, section, Currier, and physical word-position buckets and repeat the complete maximum-over-pairs search.
- **Result:** The observed maximum normalized score is **0.2169**, below the strict-null maximum **0.3089** and below the mean null maximum **0.2425** (family-wise **p=0.8182**). The leading pair's longest contiguous core match is only two words. No credible differential anchor is present.
- **Ledger:** K20.
- **Caveats:** This searches line-scale ordered core recurrence. A longer parallel with heavy synonymy or a transformation not preserving core order would not be proposed.

### stateful_line_program_search.py
- **Question:** Does one character per Voynich word emerge under a compact line-reset recurrence over deterministic word morphology?
- **Method & control:** Eight declared coordinates (length, edges, glyph sums, and polynomial whole/core/affix values) are crossed with fifteen direct, lag, line-seed, position, and running-register programs and Latin/English targets: **240 candidates**. Global 23-letter output keys fit on complete fit quires; feature/program/language selection uses disjoint validation quires; keys freeze before test quires. Eight strict nulls preserve quire/section/Currier/position and repeat the full 240-way selection. Three unknown-program/unknown-key Latin controls use the same line geometry.
- **Result:** `lag_minus`, `seed_minus`, and `running_plus` controls recover **100% plaintext accuracy**. Voynich averages **−12.4525** unconditional log likelihood per tetragram, **−1.1947** below the selected language ceiling. It loses to the matched-null maximum on final likelihood (**−12.2213**, empirical **p=0.8889**) and residual gain (**+1.4173** null maximum versus Voynich **+1.2513**, **p=0.3333**). Exact candidate consistency is only **0.50** and all outputs are repetitive pseudo-language.
- **Ledger:** K20.
- **Caveats:** The gate covers one global permutation key, one emitted character per retained word, modulus 23, line resets, and at most one running register. Multi-register, variable-length, page-keyed, or non-arithmetic transducers remain outside scope.

---

## Campaign 14 — 2026-07-23 (visual graphs as decoder state)

### multimodal_visual_key_gate.py
- **Question:** Do the invented-looking plants or zodiac figures carry a key that becomes meaningful only when combined with the text, rather than naming the pictured object directly?
- **Method & control:** Two annotation-matched gates. The plant arm aligns 15 tagged herbal drawings to 142 prose runs (1,010 words), reduces root/stem/leaf/inflorescence tags to eight fixed graph invariants, and adds each invariant with coefficient ±1 to ten representative K20 state programs modulo 23. A global substitution key fits on complete fit quires; candidate/language selection uses disjoint validation quires; test quires remain untouched. The full **2,720-candidate** search is repeated after rematching complete plant records within quire and within Currier language. Two hidden Latin controls use actual plant keys, then repeat under mismatched drawings. The zodiac arm uses 95 inner/outer nodes on four anchor-bound rings, crosses eight text coordinates, three current/lag operations, four moduli, and sixteen current/transition visual features (**1,536 candidates**), and nests train/validation/test by whole folio. Its null nontrivially rotates the complete visual sequence within each folio×tier, preserving circular order and autocorrelation.
- **Result:** Plant authentic pairing gains **+0.1137 nats/tetragram** over text-only. All eight rematches do better: within-quire mean **+0.4333** and within-Currier mean **+0.3540**, both empirical **p=1.0**; selected formulas differ in all four folds. The simple paired control recovers **99.88%** plaintext and the planted feature/program in 4/4 folds, versus **34.64%** accuracy after image rematching. The harder running/whole-graph control reaches **78.19%** versus **2.58%** rematched but misses its predeclared 80% pass threshold. Zodiac held-out gain is **−0.0901 bits/node**, only 1/4 folds positive, cyclic-rotation **p=0.3951**. Its 10%-noise control gains **+0.5567 bits/node** and selects the planted transition in 4/4 folds.
- **Ledger:** K21.
- **Caveats:** This excludes fixed additive plant page shifts and the declared local zodiac state/transition vocabulary, not arbitrary image steganography. The plant complex-state arm has only partial power. A plant↔zodiac transfer is not identifiable because the committed plant annotations are page-level organ slots while zodiac bindings are ordered figure nodes. Scaling the K14 protocol to all 12 rings and defining a common graph schema are data prerequisites for a stronger multimodal attack.

---

## Campaign 15 — 2026-07-23 (all-zodiac visual binding and cross-domain transfer)

### build_zodiac_visual_nodes.py / build_multimodal_visual_graphs.py
- **Question:** Can the missing visual data prerequisite be closed without inventing semantic attributes for the eight unannotated zodiac rings?
- **Method & control:** Yale scans are assigned explicit panel crops, centers, elliptical ring radii, and clockwise tier geometry. Four folios retain their pixel-read K14 phase; the other eight use the anchored modal start and carry ±1-node uncertainty. Figure-core descriptors exclude raw ink/edge/entropy and remove glyph-sized connected components; larger context crops remain only for leakage testing. QC overlays expose every node placement. The second builder applies the identical pixel schema and an explicit graph vocabulary to 15 herbal and 12 zodiac diagrams.
- **Result:** **286** inner/outer zodiac nodes, 12 QC overlays, and **27** common-schema page records are generated reproducibly. All label counts reconcile; 13 outside-diagram labels remain separately recorded.
- **Ledger:** K22 data.
- **Caveats:** Eight ring phases remain geometry-inferred, not label-read. The common graph fields have domain-specific construction provenance and are not assumed semantically equivalent.

### multimodal_graph_transfer_gate.py
- **Question:** Does the all-ring visual sequence predict label morphology, or does one pigment/graph relation transfer between herbal and zodiac pages?
- **Method & control:** The node arm searches **2,112** modular text-coordinate × guarded pixel-state/transition candidates with whole-folio train/validation/test splits. Sixty nulls rotate complete visual sequences within folio×tier and repeat selection. Fixed ±1-node phase scenarios, a planted channel, four-ring pixel→semantic prediction, and core/context label-length leakage probes audit the image bindings. The cross-domain arm tests 15 herbal against 12 zodiac page records. Its primary gate uses pigment only, requires one shared feature pair and sign in both domains, rank-residualizes folio order and text sample size, and repeats pair selection in 1,000 complete text-record permutations.
- **Result:** Zodiac gain is **−0.0456 bits/node**, 4/12 folds positive, cyclic **p=0.131**; all phase scenarios remain negative. The planted channel passes at **+0.3057** and is selected in 8/12 folds. Pixel→semantic calibration is **−0.1252**, 0/4 positive, **p=0.410**. Raw context pixels leak label length (**p=0.0365**), but the guarded core does not (**p=0.309**). Exploratory green pigment tracks `o`-initial rate in both domains (**rho +0.571/+0.591, p=0.032**). The controlled common-pair gate instead selects blue pigment vs type-token ratio (**rho −0.521/−0.538**) and fails selection-adjusted significance (**p=0.152**); its zodiac correlation drops to **−0.171** after shared-canvas aggregation. A post-hoc green/gallows node-localization check is negative (**−0.0066 bits/node, p=0.927**).
- **Ledger:** K22. No visual key or plaintext recovered; freeze the raw green relation only as an independent replication target.
- **Caveats:** The raw transfer is compatible with production order, canvas exposure, register, or other page-level effects. The semantic calibration failure limits what a null from automatically extracted silhouettes can exclude.

---

## Campaign 16 — 2026-07-24 (external abbreviation controls and illustration routes)

### build_historical_abbreviation_controls.py
- **Question:** Can a variable-length decoder be restricted by attested medieval abbreviation practice instead of an invented expansion inventory?
- **Method & control:** The builder selects Latin Cappelli records whose stated period intersects the fourteenth or fifteenth century, preserving Cappelli's notation without treating every bracket as omitted text. It separately extracts line-level diplomatic `<expan>/<ex>` events from Nuremberg Letterbooks 2–5 (1408–1423), grouped by book and writer. Source URLs, licenses, input hashes, selection rules, and exclusion counts are embedded in the artifact.
- **Result:** **8,869** Cappelli source rows become **8,420** deduplicated records. The Nuremberg arm contains **118,681** usable expansion events and **18,179** distinct book/writer-grouped pairs. Every one of the 12 declared decoder chunks is attested in the Cappelli solution field; this licenses the inventory but does not identify an EVA sign.
- **Ledger:** L4 calibration.
- **Caveats:** Cappelli is a lexicon rather than continuous prose. The Nuremberg manuscript is German running text with Latin scribal inheritance, so its event coverage calibrates abbreviation frequency and diversity, not a Voynich target language.

### abbreviation_transducer_gate.py
- **Question:** Does a bounded, historically licensed variable-length expansion map make held-out Voynich decode more economically as Latin than the same fitted one-character key?
- **Method & control:** Four complete-quire folds fit a 22-way monographic base using fit-block frequencies only, select its restart on a disjoint validation fold, and freeze it. The variable arm exhausts all **432** null-or-expansion maps over five position-conditioned `y/q/ch/sh` slots on fit blocks. The primary score pools full Latin-trigram log probability over identical source observations; every added output character pays its code length and the variable arm pays **8.7549 map-description bits per fold**. Nineteen schema/copy replicates provide a rank screen. A planted synthetic-Latin transducer tests recovery.
- **Result:** Voynich is negative in **4/4 folds**, pooled **−0.002264 bits/source observation**, and ranks **19/20** against the procedural panel (18/19 nulls score higher). Its variable output expands 188,372 source observations to 191,136 decoded characters while worsening joint code length. Synthetic Latin is positive in 4/4 folds (**+0.02810**), but active-slot recovery is only **80.97% globally / 62.42% minimum fold**, below the locked 90% power threshold. The bounded screen fails.
- **Ledger:** L4 remains live at low prior; this is no evidence for Latin or the declared EVA-to-siglum assignments.
- **Caveats:** The result is a pilot rank, not a calibrated significance test. EVA is not a diplomatic allograph transcription, the language model is Ciceronian rather than matched fifteenth-century technical Latin, and the synthetic recovery failure prevents a family-wide exclusion.

### illustration_route_gate.py
- **Question:** Do the illustrations specify the order in which existing text units should be read, rather than changing their values?
- **Method & control:** All 24 inner/outer zodiac rings are scored clockwise and counterclockwise as closed cycles, removing arbitrary start phase. Each of 1,000 null replicates draws one random circular order per ring and scores that route together with its exact reverse before taking the same two-candidate maximum. The planted counterclockwise control uses donor prose excluded, together with the target quire, from its reference transition model.
- **Result:** Clockwise scores **−6.50845** and counterclockwise **−6.54769 bits/transition**; the paired route/reverse null gives **p=0.35265**. The donor-held-out planted counterclockwise route is selected at **p=0.000999**. Canonical zodiac direction is unsupported.
- **Ledger:** K23.
- **Caveats:** Existing herbal data contain aggregate graph counts but no observed node-edge topology. Constructed DFS/BFS outputs are software diagnostics only, carry no inferential p-values, and do not test real plant routes.

---

## Campaign 17 — 2026-07-24 (causal production-algorithm inversion)

### production_algorithm_gate.py
- **Question:** What smallest normalized causal source model assigns the best probability to untouched Voynich words: a character grammar, register conditioning, previous-word context, recent exact copying, or a general mutation channel?
- **Method & control:** Four complete-quire outer folds use two fit folds, one disjoint validation fold, and one untouched test fold over 33,133 gap-safe prose words. Every candidate is a normalized word code. Character models reset at word boundaries; register state uses Currier, section, and line position but excludes hand. The folio model updates transition counts only after scoring each word. Exact-copy and canonical prefix-decodable edit channels use only the preceding eight words, with mixture weights fitted on training. A planted 14%-copy/18%-edit source, a base-only source, and Latin reflowed into the same layout calibrate the channel.
- **Result:** Character trigram scores **12.3423 bits/word**; register conditioning improves it by **0.2518**. Causal folio state adds **0.5215** at fixed concentration 32. After that state, exact-copy+edit adds only **0.00388**, with fitted copy weight **0.25–0.42%** and edit weight about **0.004%**. The planted operation control is recovered at **92.23%** with **+1.5078 bits/word**; base-only false gain is **+0.00056**. Latin copy/edit gain is **+0.1012**.
- **Ledger:** D13; K24 kills recent copy/mutation as the main production engine.
- **Caveats:** This is a probabilistic source description, not plaintext. Canonical edit scripts do not sum over alternative alignments. Folio adaptation can represent meaningful topical/register variation or a content-free page-specific procedure.

### adaptive_horizon_gate.py
- **Question:** Is the state a short copying cache, a line process, or a persistent folio profile, and what prior concentration does it require?
- **Method & control:** The same nested folds select among static, line-cumulative, rolling 8/16/32/64/128/256-word, and full-folio caches. A second validation scan selects concentrations 4–256. An independent generator measures generic online-estimation gain; a planted 32-word cache tests horizon localization; Latin measures ordinary-language behavior.
- **Result:** Full-folio state wins **4/4 folds** before concentration tuning; rolling performance improves monotonically with horizon (8: **11.9445**, 32: **11.7707**, 128: **11.6077**, 256: **11.5752**, folio: **11.5690 bits/word**). Nested concentration selection chooses **16 twice and 32 twice**, for **+0.5283 bits/word** over static register and about **11.5622 bits/word** total. The independent generator gains **+0.1520** and selects weaker concentrations 32–256. The planted 32-word cache is localized to rolling-32 in 3/4 folds. Latin gains **+0.6827** and selects concentrations 8–32.
- **Ledger:** D13.
- **Caveats:** The Latin result is decisive for interpretation: persistent folio state is compatible with meaningful language and is not proof of an asemic generator. Residual lag-1 difficulty remains, but Latin's residual is comparable; it is not a Voynich-specific second channel.

### state_boundary_gate.py
- **Question:** Is the inferred state truly page-specific rather than generic online adaptation or quire/register drift, and does the source require one reset boundary or a hierarchy?
- **Method & control:** The boundary tournament crosses line, paragraph, page side, physical leaf, repository quire, Currier×section register, Currier, and global reset units with concentrations 4–256. A second declared family layers a page-side cache over leaf/quire/register/Currier/global outer state. Selection uses validation quires and scoring uses untouched test quires. The state-swap gate requires at least 80 words per side, selects 16/32/64 prefix words and concentration on validation, freezes the prefix state, and always scores words 65–128. Donor states are restricted to the same repository quire, Currier, and section; each donor's conditional counts are normalized context-by-context to the target's evidence totals before 500 within-stratum permutations. The corpus does not safely identify true bifolia, so the available `r/v` pair is called a physical leaf.
- **Result:** Every fold selects the hierarchical family: three `side over Currier`, one `side over register`. Selected held-out score is **11.3709 bits/word**. The descriptive best hierarchy (`side over register`, outer/inner concentration 64/64) scores **11.3667**, a **+0.09684** gain over the best single register boundary (**11.4635**). The frozen own-page prefix beats quire×Currier×section-matched permutations by **+0.28082 bits/word** over 5,954 suffix words on 120 sides, with **p=0.001996 in all four folds**; own state ranks first on **56/120** sides. Across 99 usable `r/v` pairs it beats the opposite side by **+0.1869 bits/word** on average. The signal spans herbal, biological, cosmological, pharmaceutical, recipe, and text-only sections. The independent source is null (**+0.00394**, no fold significant); the planted page-state source gives **+1.2602**, 119/120 rank-1, with all folds significant. Latin is stronger at **+0.60555** and 50/120 rank-1.
- **Ledger:** D13 is refined from a single folio cache to hierarchical Currier/register plus page-side state.
- **Caveats:** This establishes a reusable page-side distribution beyond quire, dialect, and section. Latin proves that such a state can arise from ordinary topical language; the result does not decide whether the Voynich state is semantic, cryptographic, or procedural. The next discriminating use is to predict this state from independent image features under whole-quire holdout and within-stratum image rematching.

---

## Campaign 18 — 2026-07-24 (illustration-to-page-state prediction)

### build_page_illustration_features.py
- **Question:** Can full-facsimile image features be built without allowing the target writing itself to leak into the visual predictor?
- **Method & control:** The 209-page pinned PDF is aligned to the IIIF canvas order after removing four edge views. Detection runs at 384×512 after local paper-background subtraction. Connected foreground survives only if it spans at least 18 detection pixels vertically; large components entering the guarded scan-edge zone are rejected. Primary features are global component statistics, 8×8 silhouette density plus row/column projections, and four 8×8 pigment grids. Small components are retained only as an excluded leakage diagnostic. A reproducible eight-page overlay marks admitted illustration pixels green and excluded foreground red.
- **Result:** 204 labeled manuscript-side records are produced; 171 labels exactly overlap corpus folios. Composite foldout labels are deliberately not guessed. Pixel inspection confirms that ordinary text is excluded while substantial plant, zodiac, figure, root, and diagram structure survives.
- **Ledger:** K25 feature substrate.
- **Caveats:** Disconnected short drawing strokes are sacrificed with the text, some page blemishes remain, and composite foldouts require a manually verified many-to-many map. These are coarse whole-page descriptors, not object recognition.

### image_state_gate.py
- **Question:** Does independent illustration structure identify a better donor for the frozen page-specific text state established by D13?
- **Method & control:** Four complete-quire folds reuse the register-conditioned word-reset trigram source and a frozen 64-word page prefix. Fit pages alone scale image features; validation pages choose among four guarded families, 1/3/5 nearest donors, and concentrations 8/16/32/64. On untouched test pages every donor is another page in the exact same repository quire×Currier×section stratum. The primary null performs 500 complete image-vector relabelings inside those same strata. The excluded small-component grid is run separately. A target-prefix state oracle chooses the nearest other textual state and tests whether the donor pool contains recoverable structure.
- **Result:** The 100 test pages contribute 4,869 suffix words in 15 matched strata. Guarded-image selection loses **0.05752 bits/word** to the mean matched donor and is worse than the pooled relabeling null (**p=0.99601**); no fold is individually significant. The excluded text-layout diagnostic gains only **0.00178**, **p=0.43713**. The prefix-state oracle succeeds at **+0.14368**, and the target's own prefix gains **+0.26589**, so the negative is not caused by an empty donor signal. A second full run is byte-identical.
- **Ledger:** K25 closes coarse whole-page silhouette/pigment nearest-donor prediction.
- **Caveats:** The result does not exclude object-level graph correspondence, corrected composite-foldout alignment, trained image embeddings, local label-to-object links, or a nonlinear joint text/image model. It establishes no plaintext or visual semantics.

---

## Campaign 19 — 2026-07-24 (learned visual embeddings to text-state axes)

### build_guarded_dinov2_embeddings.py
- **Question:** Can an externally pretrained visual backbone retain useful object similarity after the anti-text guard that defeated coarse image features?
- **Method & control:** The frozen 22,056,576-parameter DINOv2-S/14 backbone is loaded from Meta's official repository at revision `7764ea0f...`; the 84.1MB weights hash to `b938bf1b...60cd9`. The same reproducible PDF/IIIF mapping produces 204 sides. Four 224×224 guarded views are embedded: full-layout and tight union-of-components crops, each in original color and black silhouette. CLS and mean-patch pooling yield eight 384-dimensional families. No Voynich label or text statistic fits the backbone. A dedicated montage records the exact color inputs.
- **Result:** All 204 records receive all eight embeddings. The input audit confirms no ordinary prose is presented, but also exposes the price of the strict guard: disconnected short strokes are omitted, making some ring and figure inputs sparse.
- **Ledger:** K26 feature substrate.
- **Caveats:** DINOv2 was trained on modern natural images, not manuscript drawings. The frozen embeddings are global/tight page summaries, not detected medieval objects. Model code/weights are Apache-2.0 and are fetched rather than vendored.

### visual_state_axis_gate.py
- **Question:** Can learned image features predict a compact coordinate system for the D13 page state even when raw image-neighbor matching fails?
- **Method & control:** In each complete-quire fold, fit pages alone define standardized PCA axes of log first-64-word character-transition counts and fit kernel-ridge maps from the frozen image embeddings. Validation chooses eight visual representations, four ridges, 2/4/8/16 axes, 1/3/5 donors, and concentrations 16/32/64. Untouched suffixes always begin at word 65. Donors and 500 complete image relabelings are restricted to exact repository quire×Currier×section. Target prefix coordinates are unavailable to the primary predictor and appear only in selected-axis and full-state oracles. The excluded small-component text-layout grid runs through the same supervised pipeline.
- **Result:** Over 100 test pages, 4,869 suffix words, and 15 strata, image-selected states gain **+0.02842 bits/word** over the mean donor. That is not a pairing result: relabeled images gain **+0.02470**, leaving only **+0.00372** for authentic pairing (**p=0.30739**). Direct image→axis reconstruction is **R²=−0.7701**; selected visual families differ in all four folds and fold p-values are `0.222/0.701/0.703/0.287`. The text-layout diagnostic is null (**p=0.71657**). The selected-axis oracle gains **+0.02873** and the full-state oracle **+0.14368**, so the state and donor pool remain recoverable. Leading axes repeatedly contrast `ed→y / dy→end` with `ch/sh→o / ho→l,r`, but the images do not predict those coordinates.
- **Ledger:** K26 closes the guarded global/tight DINOv2 plus linear-kernel state-axis family.
- **Caveats:** The positive raw gain is regression toward a few central donor states, not image information. This does not test complete object-region segmentation, traced illustration graphs, corrected composite foldouts, manually aligned local features, nonlinear fine-tuning, or joint training on an independent manuscript-image control.

---

## Campaign 20 — 2026-07-25 (public complete-object visual attack)

### setup_public_object_attack.sh
- **Question:** Can the complete-object attack be mounted reproducibly from public resources rather than ad hoc local models or repeated permission-gated commands?
- **Method & control:** One idempotent shell entry point installs pinned inference runtimes, downloads public artifacts into the gitignored cache, verifies published or locally pinned digests, and smoke-tests every checkpoint load. It pins Surya layout revision `0aee81d...`, the HORAE fine-tuned YOLO12s release, official SAM 2.1 tiny, and the same official DINOv2-S/14 revision/weights used in K26. The generated SHA-256 manifest covers every inference artifact.
- **Result:** All four model families load from the verified cache. The bootstrap is the sole network/install step required by the subsequent builder and gate.
- **Ledger:** K27 public-resource substrate.
- **Caveats:** Surya and DINOv2 are not Voynich-trained. HORAE transfers weakly by itself; it is retained as a secondary proposal source, not treated as a semantic classifier.

### build_public_object_embeddings.py
- **Question:** Do complete illustration objects and their topology carry the page-state signal lost by K25/K26's coarse masks?
- **Method & control:** A manually audited normalized crop map resolves every composite foldout scan into all **49** previously missing corpus targets, yielding exactly **225 unique folios**. Two transcript units share the same f90v photographed field and are explicitly excluded from gate fitting/scoring. Surya visual boxes, HORAE decoration boxes, and directional green/blue/red pigment components propose regions. SAM supplies spatial guards; retained masks are substantial foreground-connected components with page-frame rejection, contour filling, and narrow pigment support. DINOv2 embeds full/tight RGB and silhouette views; a separate descriptor records object, ink, pigment, skeleton, component, endpoint/junction, grid, and projection topology. A seven-page montage audits a plant, circle, foldout chart, biological page, mixed object pages, and a text-only negative.
- **Result:** The artifact contains all eight DINO feature families plus topology for all 225 folios. There are **196 proposal-bearing** and **29 blank** sides, **49 foldout-mapped** sides, and **2 shared-canvas exclusions**. Completed objects cover median **11.4%** of a page (maximum **27.3%**), avoiding the page-scale leakage seen during development. The final QC retains complete plants, rings, charts, and figures while f103r's ordinary prose remains blank.
- **Ledger:** K27 feature substrate.
- **Caveats:** The masks are publicly pretrained proposals plus deterministic color/geometry guards, not human gold segmentation. Disconnected, uncolored micro-strokes can still be lost, and object semantics are not labeled.

### public_object_state_axis_gate.py
- **Question:** Does the stronger foldout-complete object representation predict D13's text-state axes out of quire?
- **Method & control:** The K26 four-fold state-axis implementation, split logic, validation grid, exact quire×Currier×section donor restriction, 500 within-stratum image relabelings per fold, and textual oracles are reused. Only the visual artifact changes. Shared photographed canvases are unavailable to both fit and scoring.
- **Result:** The primary gate fails over **120 untouched pages / 5,954 suffix words / 18 strata**. Image-selected donors score **11.71727 bits/word** versus **11.70006** for the matched mean, a **−0.01721 bits/word** loss. The observed pairing is ordinary under pooled relabeling (**p=0.26148**) and direct held-out axis prediction is **R²=−11.4447**. Fold gains are `−0.0273/−0.0338/−0.0308/+0.0193`; the lone positive fourth fold is nominally significant but does not replicate. The selected-axis oracle remains positive (**+0.00804**) and the full-state oracle gains **+0.11810**, so the failure is visual prediction rather than an empty donor pool.
- **Ledger:** K27 closes the public detector/SAM complete-object DINO/topology representation under the frozen linear-kernel state-axis gate.
- **Caveats:** This does not exclude manually traced local icon/text correspondences, object-to-label alignment, manuscript-specific segmentation training, or nonlinear joint multimodal learning. It recovers no plaintext or visual semantics.

---

## Campaign 21 — 2026-07-25 (historical table/counter reconstruction)

### historical_counter_mechanism_gate.py
- **Question:** Could a fifteenth-century scribe have generated the text by advancing a short tally or wheel and consulting a compact glyph-choice table?
- **Method & control:** Five manually executable counters advance by word, glyph, or page line and reset only at lines, paragraphs, or pages. Periods 2/3/4/5/7/12 and two shrinkage levels give 60 candidates. Each table contains only phase×glyph multipliers over D13's train-only register trigram and causal page state. Complete-quire folds fit on two folds, select on a third after paying a lower-bound one description bit per table entry, and score the fourth. The baseline with no table is explicit. Thirty-two nulls independently rotate the phase origin in every reset unit and repeat full selection. Base-only, Latin reflow, and a planted word-line mod-4 source are controls. A per-line offset audit tests the manual-error prediction.
- **Result:** The first screen passes statistically but not algorithmically. All four Voynich folds improve, pooled **+0.0012465 bits/glyph** (**+0.007784 bits/word**), phase-origin **p=0.030303**. Three folds choose page-line counters, but with incompatible periods **3, 4, and 5**; the fourth chooses word-line period **12**. Base-only and Latin choose no table in 4/4 folds. The planted source recovers word-line mod 4 in 4/4, gains **+0.031277 bits/glyph**, and reaches **p=0.030303**. Offset-0 and half-line persistence are only weakly above chance. The heterogeneous periods and phase tables, dominated by rare position-sensitive glyphs, trigger the absolute-layout discriminator rather than an algorithm claim.
- **Ledger:** K28 provisional screen; superseded in interpretation by the residual gate below.
- **Caveats:** A positive family-level phase-origin test does not identify one reproducible table. The 32-null p-value is resolution-limited, and most individual test folds do not repay their table description cost.

### historical_counter_residual_gate.py
- **Question:** Does the apparent repeating cycle survive after ordinary absolute layout conventions are learned?
- **Method & control:** Train quires fit two nonperiodic nuisance profiles by coordinate descent: exact page-line indices 0–10 with one 11+ class, and exact word-in-line indices 0–10 with one 11+ class. The page-line counter tournament is then rerun on residual glyph probabilities with the same validation description charge, explicit no-counter baseline, untouched test quires, and 32 page-origin rotations. A fresh planted page-line mod-4 source tests whether a real repeating cycle remains recoverable after the nuisance profiles.
- **Result:** The absolute page-line/word-slot profiles themselves add **+0.007452 bits/glyph in all four held-out folds**. Fitted independently, exact page-line contributes **+0.004763** and exact word slot **+0.003069**. After that structure is present, Voynich selects no residual table in **4/4 folds**: pooled gain **0**, no positive fold, **p=1.0**. The independent base-only source also selects no table. The planted line cycle selects a table in 4/4 folds, recovers period 4 in 3/4 (period 2 in one fold), gains **+0.008048 bits/glyph**, and reaches **p=0.030303**. Thus the first screen was stable absolute layout structure, not a repeating counter. The period-12 word arm is observationally confounded with exact word position on normal line lengths and is closed rather than decoded.
- **Ledger:** K28 closes the declared small repeating table/tally/wheel mechanism.
- **Caveats:** This does not exclude nonperiodic per-page recipes, semantic writing with layout conventions, a keyed table whose state is not tied to visible boundaries, or tables operating on untranscribed allographic distinctions.

---

## Vision-agent runs

- **28-folio grounding pass → D12** (Campaign 2, `1bb4a48a`/`transcripts/readable/adversarial_fork.txt`): one vision agent per illustrated folio, building `body_spatial.json` (37,019 tokens with locus role/line-position/section/Currier/hand) from deterministic IVTFF parsing plus vision-grounded label→drawing binding. Produced D11 (q-register law) and D12 itself, and seeded the later crib/zodiac datasets (`data/grounding/cribs.json`, `grounding.json`).
- **8-agent K13 pass** (Campaign 5, `dcc71ad3`/`transcripts/readable/k13_grounding_session.txt`): 8 vision agents bound pharma + balneo labels to referents and visual attributes for `l6_grounding.py`'s within-folio shuffle-null test. NULL (see K13 above); the balneo arm's raw JSON was never statistically re-run (flagged by the audit as the K13 power gap).
- **4-agent K14 pass** (Campaign 7, 2026-07-01): 4 vision agents, one per zodiac folio (f70v2, f71r, f72r2, f73v), following the anchor-based binding protocol in `data/grounding/z10/PROTOCOL.md` (census → enumerate clockwise → read anchors → align rotation/direction → bind & record, with explicit confidence grading: high = own glyphs read from pixels match the word, med = both rotational neighbors are exact-read anchors, low = order-inference only). Results feed `data/grounding/z10_bindings.json` and both `z10_grounding.py` (K14) and `z10_cross.py` (K14b).
- **M9/M11/M13 comparanda hunts** (Campaign 6, `ac7bc224`/`transcripts/readable/cipher_and_motive_session.txt`): vision + web/IIIF subagents comparing the Voynich zodiac (esp. f73v Sagittarius) against real manuscript exemplars. Candidate images and crops archived under `images/comparanda/`: `cand_clm13076/` (BSB Clm 13076, 1356 — ancestor-type witness), `cand_md2/` (Tübinger Hausbuch, 1404 — genre/date match, longbow mismatch), `csg827/` (Cod. Sang. 827, Lake Constance 1425/28 — no pictorial zodiac, corroborates genre/region/date only), `cand_ljs449/` (UPenn LJS 449, c.1446 — wrong branch, centaur+longbow), `cand_cpg291/` (Heidelberg Cod.Pal.germ.291 — verified human-crossbow Sagittarius, post-window), `cand_cgm28/` (BSB Cgm 28 — fetched but Sagittarius folio never pulled, per the audit), plus `cgm312_*.jpg` (BSB Cgm 312 — verified human-crossbow Sagittarius + reptilian Scorpio, post-window) and `SAG_compare*.jpg` (side-by-side comparison crops).

---

## Runs the audit flagged but this file could not fully verify with a dedicated result

Per the audit doc's own list, these remain open/under-specified as of 2026-07-01 and are not claimed as closed above:
- The rosette causeway graph and the balneo pool-topology dataset — mentioned as under-logged artifacts in `docs/AUDIT-2026-07-01.md` §2, but no corresponding script or transcript passage with a discrete result was located in this pass; **result not recovered from sources**.
- R1's cipher-family exclusion re-run against medieval Latin / German / Italian vernacular target LMs — not run (audit-flagged gap, still open).
- L2's crib-anchored held-out decode at the ~200-entry scale specified — not built (audit-flagged gap, still open).
- Named-rival confrontation: public Scott Schechter and Naibbe materials have now been inspected/run; Montemurro & Zanette's clustering metric and a tonal-language control remain open.
