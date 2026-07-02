# The Voynich Manuscript: A Falsification Campaign

This repo is a from-scratch attempt to figure out what the Voynich manuscript
*is*, using statistics, null models, and multimodal vision — run as a series
of Claude sessions with human direction, June–July 2026. The premise was
deliberately amateur: what would a curious person with an LLM and a couple of
evenings actually be able to establish, working from the public facsimile and
nothing else? The answer turned out to be more than expected, and it points
the opposite direction from most Voynich writing: not "here is the hidden
message," but a chain of controlled experiments, each with a working positive
control, that closes off one decipherment avenue after another. The interest
here isn't a decoded manuscript — it's what's left standing after you insist
that every finding beat a matched null before you believe it.

## The conclusion, in five sentences

The manuscript is a generated, language-shaped artifact with no recoverable
propositional content: rule-dense (a real slot-grammar, a compact finite-state
word-engine, stable per-entity records) but referent-free. Every major
decipherment avenue was tested against a matched content-free null and a
working positive control, and every one came back null — no word-order
syntax, no substitution or verbose cipher, no label-to-picture reference
within a section (pharma/balneo), on zodiac labels, or across zodiac rings.
The likeliest account of *why* it exists is a sustained, multi-hand workshop
production — a "manufactured wonder" of secret learnedness for an audience,
not a private idiolect or a quick fraud. Two things this evidence cannot
settle: whether the makers sincerely believed the work held real knowledge or
knowingly built a magnificent enigma, and whether some low-bit-rate
steganographic scheme is embedded in the generator's free choices — a claim
that is unfalsifiable by construction, since absence of evidence for a
specific scheme is not evidence against all possible schemes. Short of new
external evidence (an undigitized manuscript, an archival record), the
meaning-recovery question is closed; the open question is the maker's intent,
not the text's content.

## Repo map

```
FINDINGS.md              the ledger — single source of truth for every claim, kill, and live lead
analysis/                ~50 Python scripts, six stages from parsing to motive forensics
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

Some scripts fetch manuscript imagery live from the Yale Beinecke IIIF
endpoint — expect network calls for anything touching high-resolution folio
crops. All script data-paths were updated for the `data/<subdir>/` layout
during the 2026-07-01 reorg and verified to compile (50/50); if you hit a
stale path anyway, `moves.tsv` has the full old→new mapping.

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
- **Check the five recurring confounds** (§3): Currier A/B dialect leakage,
  production-unit/block autocorrelation (not smooth shelf-distance),
  content-free reproducibility, English-as-an-unfair-baseline, and local
  copying that mimics order.
- **Any real solution must satisfy §5's spec:** meaning isn't in word order;
  information (if any) lives in slot-choices; it must generalize to
  held-out pages; it must beat a content-free null; and any reading of a
  recurring entity must hold everywhere that entity recurs.

### Open problems

**Desk-runnable now, from data already in this repo:**
- A permutation test on the existing balneo K13 agent data — the arm was
  eyeballed, never statistically run.
- Re-run R1 (`cipher_attack.py`) against medieval Latin / German / Italian
  vernacular target language models — the original ran Ciceronian Latin only.
- L2's crib-anchored held-out decode, using `data/grounding/cribs.json`
  (~200 anchors), never actually built at scale.
- Reconcile K1/K10's metric swap: run `residual.py`'s content-free generator
  through `topic_gen.py`'s burstiness (B) and hapax metrics on the same
  footing.
- Run the named rival solutions (the scott-schechter Latin-Occitan glossary,
  the Naibbe cipher) through the held-out harness already built for R1.
- Score the Montemurro–Zanette long-range clustering metric against the
  content-free null.
- Add a tonal language to the control panel (Bowern & Lindemann's
  counter-proposal was never tested).
- Scale K14 from its 10%-slice zodiac test to all 12 rings —
  `data/grounding/z10/all12_labels.json` is the ready-made input, and
  `data/grounding/z10/PROTOCOL.md` is the binding protocol to reuse.

**Needs external data or resources:**
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

**Standing falsification conditions** — what would overturn the asemic
verdict: a substitution/cipher key that beats the content-free surrogate on
held-out crib text; a referent signal that beats the within-folio shuffle
null at scale (the zodiac's remaining 90%, or a higher-resolution grounding
pass); or an outside adversarial replication that finds a real flaw in one of
the kills above rather than just re-asserting the killed claim.

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
- **`data/corpus/eva_takahashi.txt`** is a dead file — 14 bytes of a failed
  HTTP fetch (`404: Not Found`) saved as if it were transcription data. Safe
  to delete.

## Provenance

Investigation run with Claude (Anthropic) models. Main analysis sessions:
2026-06-20 through 2026-06-21. Self-audit plus K14/K14b (the zodiac
grounding tests) and this repo reorganization/documentation pass: 2026-07-01.
All session transcripts are preserved in `transcripts/` (PII-scrubbed
2026-07-02), including the transcript of this cleanup session itself.
Manuscript imagery is drawn from Yale University's Beinecke Rare Book &
Manuscript Library, MS 408, via its public IIIF endpoint.

Third-party materials — the LSI transcription, the language-control corpora,
and the comparanda-manuscript images — are credited in **`ATTRIBUTIONS.md`**,
with their original licenses. Those terms govern reuse of those assets; this
repository's `LICENSE` covers only its own code, docs, and derived data.
