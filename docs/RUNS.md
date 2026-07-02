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
- **Ledger:** D2 — no word-order syntax ("bag of records").
- **Caveats:** none noted.

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
- Named-rival confrontation (scott-schechter Latin-Occitan glossary, the Naibbe cipher, Montemurro & Zanette's clustering metric, a tonal-language control) — none run through the held-out harness (audit-flagged gap, still open).
