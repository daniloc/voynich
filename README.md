# What is the Voynich manuscript, anyway?

I am not a cryptographer, linguist, statistician, or data scientist. I have almost no usable skills for pursuing the hidden depths of the [Voynich manuscript](https://en.wikipedia.org/wiki/Voynich_manuscript).

But I do have a Claude subscription, three years of LLM tool intuition, and a love of mystery. So for three days, between other tasks, I asked Opus 4.8 to explore various strategies to unearth hidden meaning within this strange text.

We found no information. Just structure.

But perhaps you have a different approach. More domain knowledge, more structured LLM strategy, or some other secret sauce. Here's a repo full of strategies and conclusions to either build from or falsify. Have fun! **Pull requests welcome.** If you find more than I did, or simply tighten the ratchet further, drop me some commits.

## In the robot's words

This repo is a from-scratch attempt to figure out what the Voynich manuscript
*is*, using statistics, null models, and multimodal vision — run as a series
of Claude sessions with human direction, followed by a Codex adversarial
follow-up, June–July 2026. The premise was
deliberately amateur: what would a curious person with an LLM and a couple of
evenings actually be able to establish, working from the public facsimile and
nothing else? The answer turned out to be more than expected, and it points
the opposite direction from most Voynich writing: not "here is the hidden
message," but a chain of controlled experiments, each with a working positive
control, that closes off one decipherment avenue after another. The interest
here isn't a decoded manuscript — it's what's left standing after you insist
that every finding beat a matched null before you believe it.

## The conclusion, in five sentences

No plaintext or visual-referent code has been recovered, and the powered
label-to-picture tests remain null. Cross-word suffix/prefix dependence is
real, but it was already published and its held-out incremental value becomes
negative after exact-word/core, causal-copy, and position controls. A
leakage-clean known-family solver now recovers meaningful Naibbe at **99.61%**,
including **99.46% / 99.64%** on mutually held-out halves and clear separation
from 20 adversarial null attacks, while the same exact and morphology-nearest
families activate on Naibbe in every fold but on held-out Voynich in none.
New powered attacks on local edit symbols, conditional edit-rank bits, aligned
row records, 240 compact line-reset recurrence candidates, plant drawings as
page keys, all-12 clockwise zodiac visual transitions, and nuisance-controlled
herbal↔zodiac pigment transfer recover their paired controls but remain null
on Voynich; a separate 24-ring direction test is also null, while a
historically constrained abbreviation pilot is negative in all four folds but
underpowered for a family exclusion. A small historical counter screen is
initially positive but collapses completely after exact absolute page-line and
word-slot profiles, while its planted cycle remains recoverable. The best
held-out production model is now
explicit: a word-reset character-trigram grammar, conditioned on register and
line position, with persistent Currier/register state and a second page-side
transition state layered over it; recent exact copying adds only **0.0039
bits/word** and a general edit channel collapses to zero. Latin shows the same
hierarchical page-state behavior more strongly, so this recovers a source
algorithm but neither plaintext nor a discriminator between meaningful
topical variation and a content-free page-specific procedure; a direct
follow-on over 100 held-out pages finds that guarded whole-page silhouettes
and pigment grids do not predict that state. A stronger frozen DINOv2 model
mapped to train-only text-state axes also fails exact-stratum image
relabeling (`p=0.307`, held-out axis `R²=-0.77`), despite successful textual
state oracles. A stronger public-resource attack maps all 225 folios,
including 49 foldout targets, through Surya/HORAE proposals, SAM-completed
objects, DINOv2, and explicit topology; it also fails (`−0.0172 bits/word`,
`p=0.261`, `R²=-11.44`) while the textual oracles remain positive.

## Repo map

```
FINDINGS.md              the ledger — single source of truth for every claim, kill, and live lead
analysis/                Python scripts, ten stages from parsing to causal source inversion
  INDEX.md               what each script does and which FINDINGS.md row it produced
data/                    corpus, controls, vision-grounding JSON, IIIF manifests, cached intermediates
  INDEX.md               every data file: what it is, what wrote it, what reads it, and known gaps
images/                  ~446 files: the facsimile, manuscript crops, and comparanda manuscripts
  INDEX.md               grouped index (not per-file — most are single-use vision-agent crops)
transcripts/
  raw/                   full session logs, one JSON object per line (PII-scrubbed)
  readable/              flattened, role-tagged text extractions of the raw logs, for skimming
  subagent_reports/      just each dispatched subagent's task + final report, pulled out separately
docs/
  RUNS.md                every analysis run traced to its script, method, result, and caveats
  SESSIONS.md            a reader's guide to the eight sessions — what each did, in what order
  AUDIT-2026-07-01.md    the project's self-audit: what it found, what it closed, what's still soft
moves.tsv                the file-relocation manifest from the 2026-07-01 reorg (src → dst)
```

## Where to start reading

1. **This README** — orientation.
2. **`FINDINGS.md`** — the whole argument in one document: durable findings,
   killed claims, the recurring confounds, live leads, the spec any real
   solution must satisfy, and the motive-forensics series. Read this before
   anything else; everything else in the repo is either evidence for a row in
   here or a trace back to how that row was produced.
3. **`docs/RUNS.md`** — once you want to know *how* a specific claim was
   established: method, control, exact numbers, and honestly-logged caveats,
   organized by campaign.
4. **`docs/SESSIONS.md`** — the narrative version: which of the eight Claude
   sessions ran which tests, in what order, and the false starts along the
   way (including a "sealed reference database" reframe the project talked
   itself into, then reversed).
5. **`transcripts/`** — the primary source, if you want to see the actual
   reasoning (and dead ends) behind a ledger row, not just its conclusion.

## How to reproduce

Scripts are plain `python3`, run **from the repo root**:

```
python3 analysis/04_cipher/cipher_attack.py
```

Dependencies: `numpy`, `scipy`, `Pillow` (`PIL`), `pymupdf` (`fitz`). A
`.venv` was used locally during development but is gitignored — create your
own (`python3 -m venv .venv && source .venv/bin/activate && pip install numpy
scipy Pillow pymupdf`) before running anything.

Rebuilding the Cappelli historical-abbreviation derivative additionally
requires `openpyxl`; the checked-in transducer and route gates otherwise use
the standard library.

The selected `analysis/08_breakthrough` permutation, recurrence-statistical,
and multimodal visual-key solvers use only the standard library plus `numpy`.
The optional causal recurrence experiment additionally requires `torch`.
Rebuilding the K26 guarded DINOv2 embeddings also requires `torch` and fetches
the pinned official Meta repository and weights; the checked-in embeddings
record the exact revision, URL, hash, and Apache-2.0 attribution.

K27's public complete-object attack is bootstrapped and checksum-verified in
one command:

```
analysis/10_generator_inversion/setup_public_object_attack.sh
python3 analysis/10_generator_inversion/build_public_object_embeddings.py
python3 analysis/10_generator_inversion/public_object_state_axis_gate.py
```

The setup script installs pinned inference runtimes and caches the public
Surya, HORAE, SAM 2.1, and DINOv2 weights under the gitignored
`.cache/public_object_attack/` directory.

K28's historically constrained table/counter screen and its required
absolute-layout discriminator run without network access:

```
python3 analysis/10_generator_inversion/historical_counter_mechanism_gate.py
python3 analysis/10_generator_inversion/historical_counter_residual_gate.py
```

Some scripts fetch manuscript imagery live from the Yale Beinecke IIIF
endpoint — expect network calls for anything touching high-resolution folio
crops. All script data-paths were updated for the `data/<subdir>/` layout
during the 2026-07-01 reorg; the current 94-script tree passes `compileall`.
If you hit a stale path anyway, `moves.tsv` has the full old→new mapping.

Two key data files have **no builder script checked in** — they were
assembled by vision subagents reading manuscript pages, not generated by a
Python pipeline: `data/corpus/body_spatial.json` (the spatial spine — every
token's locus role, line position, section, and image-grounded subject/
topology) and `data/grounding/grounding.json` (the 28-folio vision-grounding
payload merged into it). See `data/INDEX.md` for the full list of files with
no preserved builder, and what reads them instead.

## How to join in

The rules of the game, in full in `FINDINGS.md` §3 and §5, boiled down:

- **Every claim needs a ledger row** before it's allowed to influence the
  next move — 🟢 survived / 🔴 killed / 🟡 live / ⚪ external.
- **Beat a matched content-free null**, not just "looks like English." A
  ~15% self-citation generator with zero topics or meaning reproduces most of
  the manuscript's surface statistics; a finding only counts if it clears
  that bar.
- **Cross-validate held-out.** Fit on part of the corpus, predict the rest.
  Every dead "solution" in the literature fit a handful of words and
  couldn't decode the next page.
- **Check the six recurring confounds** (§3): Currier A/B dialect leakage,
  production-unit/block autocorrelation (not smooth shelf-distance),
  content-free reproducibility, English-as-an-unfair-baseline, and local
  copying that mimics order.
- **Any real solution must satisfy §5's spec:** reproduce the measured surface
  ordering without claiming it as payload evidence; improve held-out likelihood
  beyond the exact-word/core/copy/position baseline; beat content-free
  competitors; and make recurring readings hold everywhere the entity recurs.

### Open problems

**Desk-runnable now, from data already in this repo:**
- A permutation test on the existing balneo K13 agent data — the arm was
  eyeballed, never statistically run.
- Re-run R1 (`cipher_attack.py`) against medieval Latin / German / Italian
  vernacular target language models — the original ran Ciceronian Latin only.
- Obtain a real lexeme-level crib before attempting crib-anchored decoding.
  `data/grounding/cribs.json` is only grouped Voynich labels, not known
  plaintext or translations; its broad visual classes were already tested by
  K13/K14.
- Reconcile K1/K10's metric swap: run `residual.py`'s content-free generator
  through `topic_gen.py`'s burstiness (B) and hapax metrics on the same
  footing.
- Extend the now-powered cipher attack only with a predeclared mechanism
  materially different from the tested exact/nearest Naibbe, local edit-choice,
  and one-register line-reset channels; repeat the same max-over-candidates
  search inside matched nulls.
- Fit candidate generative/cipher models to the joint residual panel: lexical
  tail, burstiness, exact-token order, and held-out likelihood beyond the
  word/core/copy/position baseline. Boundary directionality is a surface
  calibration requirement, not an independent target.
- Score the Montemurro–Zanette long-range clustering metric against the
  content-free null.
- Add a tonal language to the control panel (Bowern & Lindemann's
  counter-proposal was never tested).
- Independently replicate K22's frozen raw page-level lead: green pigment
  versus `o`-initial/gallows rate. The all-12 in-repo result does not survive
  the nuisance-controlled common-pair gate or source-canvas aggregation, and
  it does not localize to individual figures. New pages or independently
  measured crops are required; refitting another feature on these 27 diagrams
  would not be a replication.
- Replace the eight geometry-only zodiac phases with label-read anchors, then
  rerun the frozen K22 node gate. The current ±1-node sensitivity is uniformly
  negative, but the independent pixel→semantic calibration also fails, so the
  automatic silhouette binding is not strong enough to exclude richer visual
  attributes.
- Trace actual herbal node coordinates, edges, and sibling geometry from the
  scans before testing plant-derived DFS/BFS or other routes. The current
  aggregate graph counts cannot identify a route, so their constructed-tree
  diagnostics are deliberately excluded from inference.

**Needs external data or resources:**
- Build a diplomatic glyph/allograph transcription and a matched
  fifteenth-century technical Latin language model before revisiting L4. The
  current EVA-based 432-map pilot gives no positive signal and its synthetic
  active-slot recovery misses the locked power threshold.
- A re-run of M2 (production-discipline test) on Lisa Fagin Davis's true
  codicological collation, rather than quire/rebound order.
- A hand-search of Pellegrin's 1426 Pavia ducal-library inventory (988
  published entries) for a no-gold, anonymous *liber secretorum* candidate
  (M14).
- Digitization of Nürnberg Staatsarchiv Hs.426 — the one in-window,
  catalog-attested human-crossbow-Sagittarius witness, currently unviewable.
- A full exemplar hunt for the herbal (alchemical-herbal canon) and balneo
  (De Balneis Puteolanis tradition) imagery, which never got the treatment
  the zodiac did.
- Fetching BSB Cgm 28's actual Sagittarius folio — the manuscript was
  fetched, but that one folio never pulled.

**Standing discrimination target** — a serious content-free or
meaning-bearing model must predict held-out text and jointly match the lexical
tail, burstiness, exact-token order, and observed transition profile. It only
earns sequential evidence by beating the nested surface baseline. A cipher
must additionally recover constrained held-out crib content; a referential
account must beat the within-folio shuffle at scale.

## Honesty

The project audited itself (`docs/AUDIT-2026-07-01.md`): five parallel
analysts re-read all prior transcripts hunting for untested ideas,
overclaimed kills, and unexploited artifacts. The dominant failure mode found
was under-logging of real positive results, not overclaiming — but there are
genuine soft spots, left visible rather than quietly patched:

- **D6** (the image-as-fingerprint information-budget argument) rests on
  constants a script audit called fabricated and flagged for re-derivation.
- **K5** (label-collision kill) inspected only 2 of 17 pre-registered
  verification pairs.
- **K2** (the automaton/language-floor kill) cites Māori figures that are
  unsourced in the audited transcripts.
- **The former terminal asemic conclusion was overclaimed.** The 2026-07-22
  Naibbe control invalidates K11's position-lock exclusion, and the 2026-07-23
  residual analysis removes the morphotactic arrow as affirmative evidence.
  A later leakage-clean solver now recovers Naibbe at 99.61% and rejects the
  corresponding exact/nearest channel transfers on held-out Voynich. Broader
  content-free and meaning-bearing models remain unresolved, but solver
  underpower no longer protects that Naibbe-shaped branch. Powered edit-symbol,
  edit-rank-bit, and row-axis controls now reject those additional branches as
  well.
- **`data/corpus/eva_takahashi.txt`** is a dead file — 14 bytes of a failed
  HTTP fetch (`404: Not Found`) saved as if it were transcription data. Safe
  to delete.

## Provenance

Investigation run with Claude (Anthropic) models. Main analysis sessions:
2026-06-20 through 2026-06-21. Self-audit plus K14/K14b (the zodiac
grounding tests) and this repo reorganization/documentation pass: 2026-07-01.
Held-out register, semantic, Naibbe, morphotactic, and residual-solver re-tests
were run with OpenAI Codex on 2026-07-22 through 2026-07-25.
All session transcripts are preserved in `transcripts/`. The Claude archives
were PII-scrubbed on 2026-07-02; the Codex follow-up is a direct message-record
export from its on-disk session file.
Manuscript imagery is drawn from Yale University's Beinecke Rare Book &
Manuscript Library, MS 408, via its public IIIF endpoint.

Third-party materials — the LSI transcription, the language-control corpora,
and the comparanda-manuscript images — are credited in **`ATTRIBUTIONS.md`**,
with their original licenses. Those terms govern reuse of those assets; this
repository's `LICENSE` covers only its own code, docs, and derived data.
