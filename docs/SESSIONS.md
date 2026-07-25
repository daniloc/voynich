# Session transcripts — a reader's guide

The original guide below covers the eight Claude Code sessions. A subsequent
Codex investigation ran from 2026-07-22 through 2026-07-25 and is documented
at the end of this file.

This project's actual history lives in eight Claude Code sessions, run over two sittings
(2026-06-20/21, then a 2026-07-01 audit). One session did the founding investigation;
five more forked off it the same day/night to attack, dispute, meta-analyze, and extend
its claims; a sixth continued the thread the next morning; a seventh, ten days later,
audited all of it. `FINDINGS.md` is the durable output — a ledger of claims tagged
🟢 survived / 🔴 killed / 🟡 live / ⚪ external. This document is the map back from
ledger rows to the actual conversations that produced them, in case you want to see the
reasoning (and the false starts) behind a given `D`/`K`/`L`/`M`/`R` number.

Read chronologically, the arc is: a huge single-session sprint (`main_session`) builds
the whole statistical/morphological/image-grounding toolkit and a first cut at nearly
every ledger category in one sitting — including proposing and then reversing its own
"sealed reference database" reframe. Three forks split off from it that same evening to
stress-test it from different angles: one *adversarially* attacks its biggest claims
(kills K1, K3; builds the spatial corpus that yields D11/D12), one is handed a rival
*dispute* theory to test on its own terms (kills K2; corroborates K11), and one runs a
*meta-analysis* asking "are we making progress?" — which turns into the session that
runs the two decisive cipher falsifiers (K11, R1) and then the whole motive/production
forensics series (M1–M14). The morning after, two more short sessions pick up single
threads: a *Contact*-inspired "hidden dimension" reread of the manuscript's geometry
(K12), and a check of whether the project's own "largely exhausted" self-assessment was
premature (it wasn't quite — K13). A ninth-day-later audit session (this one) rereads
everything end to end, finds real gaps, and closes two of them same-day (K14, K14b).

One session (`tiny_session_aca2`) is a dead end: a duplicate seed prompt that was never
continued.

**Cross-session convergence worth noticing:** K11 (the verbose-cipher kill) is the one
claim independently re-derived in *three* different sessions — `cipher_gate.py` in
`cipher_and_motive_session`, `verbose_cipher_null.py` in `main_session`, and `position_lock.py`
in `dispute_fork` — each catching the same result by a different route, at one point
literally colliding mid-edit when a fork noticed `FINDINGS.md` had already been updated
by a parallel session running the same kill.

---

### main_session (`308ba40b-cc89-42d8-850e-0d6a083ed87c`, 2026-06-20 05:10 – 2026-06-21 01:59)

**What it was:** The origin session. The user poses a hypothetical — a "savvy amateur"
claims to have cracked the Voynich manuscript with LLM tools; how would they have
approached it? — then says "Proceed" and hands over the actual manuscript. What follows
is a single ~20-hour sprint that builds essentially the whole toolkit from scratch:
Zipf/entropy/morphology statistics against Latin/English controls, null-hypothesis
generators, slot-grammar decomposition, IIIF image-grounding of herbal/zodiac/balneo/
rosette pages, cross-modal crib search, a "hash vs. key" reframe, label-collision tests,
a prosopographical maker-sieve, paleographic stroke analysis, and a production-cost
model. Midway, the session talks itself into a "sealed reference database" reframe that
turns every prior null into an apparent "confirmation" — before the user's own
falsifier challenge cracks it back open. It ends with the verbose-cipher gate and a
1:1 decoder-trap both coming back null/vacuous, and the "asemic artifact" verdict
forming in real time.

**Key outcomes:** D1, D2, D3, D5, D6, D7 (the split-half entity-stability falsifier,
run directly here), D8, D9 — plus early/seed versions of D4 and D12 (fully built later).
K4, K5, K6 and the seeds of K7/K8/K9. K10 and K11 are both run here — K10 asserted then
self-reversed, K11's `verbose_cipher_null.py` independently corroborating the parallel
fork's `cipher_gate.py` kill. Runs an early, simpler R1 precursor (`decoder_trap.py`);
the fuller held-out `cipher_attack.py` that FINDINGS.md credits for R1 belongs to
`cipher_and_motive_session`, not here.

**Notable moments:**
- The epistemic frame set before any data is touched: *"the failure mode to watch for is flexibility masquerading as a solution... was the key frozen before the test folios were decoded, and does it survive the gibberish control?"*
- A hypothesis dying via self-caught confound: the "root morpheme" crib near-signal (p=0.056) turns out to be Currier A/B dialect leakage, not anything visual.
- The session's biggest detour: a long stretch re-narrating the entire investigation as confirmation of a "sealed reference database," reversed by the session's own end — *"K10 kills my burstiness argument... K1 kills the 'sealed reference database'... So my 'needle moved back toward real content' was unearned."*
- The only genuinely readable plaintext in the whole manuscript, transcribed by a marginalia subagent: f116v's "michiton oladabas" and the Occitan/Catalan month-names (mars, abril, may, yony, yoller, augst...) — *"the only readable plaintext in the entire object... is in plain script, not Voynichese. There is no Rosetta surface."*
- The closing verdict: *"A generated, language-shaped artifact with no recoverable propositional content — Codex Seraphinianus class, four centuries early."*

**Files:** `transcripts/raw/308ba40b-cc89-42d8-850e-0d6a083ed87c.jsonl` · `transcripts/readable/main_session.txt` · `transcripts/subagent_reports/main_session_subagent_reports.txt` (33 subagents)

---

### tiny_session_aca2 (`aca27691-3a41-43ef-b148-7a5dcbd3037e`, 2026-06-20 05:27 – 05:27)

**What it was:** A false start, not a session in any real sense. It contains the exact
same two-part seed prompt ("interesting math" / the full "credible source claims to have
cracked the Voynich manuscript..." prompt) that opens `main_session`, sent about 17
minutes after `main_session` began — almost certainly a duplicate/parallel window the
user opened and then abandoned once `main_session` took off. No assistant turns, no
investigation, nothing to characterize beyond "the seed prompt, twice, unanswered."

**Key outcomes:** None — no ledger rows trace here.

**Notable moments:** N/A — the entire transcript is the seed prompt.

**Files:** `transcripts/raw/aca27691-3a41-43ef-b148-7a5dcbd3037e.jsonl` · `transcripts/readable/tiny_session_aca2.txt` (no subagents, no subagent report)

---

### adversarial_fork (`1bb4a48a-f47a-4a8d-824b-2145420c4578`, 2026-06-20 20:32 – 2026-06-21 02:00)

**What it was:** Forked from `main_session` that evening and handed its "six works"
library-catalog claim to attack. It reproduces the source script's numbers exactly, then
demolishes the *interpretation*: the "works" are hand-drawn folio-number ranges that
contradict the corpus's own IVTFF illustration labels, and the headline cross-reference
edge sits *below* a size-permutation null. The user then hands over a second, bigger
target — the "sealed reference database" reframe that had converted a decade of
negative results into apparent positive evidence — which the fork kills by building a
matched content-free generator (~15% local self-citation) that reproduces every
signature the reframe cited. It escalates via a user-invited mass fan-out: 31
script-audit subagents (one per `.py` file) followed by 28 vision-grounding subagents
(one per illustrated folio), the latter building the spatially-grounded corpus that
becomes D12. It ends by catching its own newest finding (label morphology "predicting"
referent class) as a confound-driven near-false-positive, and hands a decision gate back
to the user rather than declaring victory.

**Key outcomes:** K1 (sealed reference database), K3 (six recovered works), D11 (the
q-register law), D12 (spatial body reconstruction — this fork's main constructive
output). Independently re-derives K11's kill via its own BPE unit-h2 test. Produces L6
(domain-cores `ted`/`kor`) as a new live-but-confounded lead.

**Notable moments:**
- The kill-shot against the reframe: *"it is observationally identical to 'structured gibberish with no content,' which is the one hypothesis it has to beat and the one it never mentions."*
- The one-line demolition after the content-free generator ran: *"copy a recent word ~15% of the time... reproduces the burstiness, the source-clustering, the A/B seam, and the label/body split, jointly, with one parameter."*
- Self-correction as discipline: after a result that "looks like a win," the agent catches itself — *"referent class ≡ section... It's a confounded near-false-positive, not R1 evidence. Let me log it honestly."*
- On motive: *"Labor tracks the value of the object to whoever resourced it — and an object's value can be entirely in its appearance."*
- 59 subagents total (31 script-audits + 28 vision-grounding); the script audits mostly landed on "mixed" (sound machinery undermined by missing nulls or hardcoded parameters) rather than clean pass/fail — only one script (`vparse.py`) was graded fully sound.

**Files:** `transcripts/raw/1bb4a48a-f47a-4a8d-824b-2145420c4578.jsonl` · `transcripts/readable/adversarial_fork.txt` · `transcripts/subagent_reports/adversarial_fork_subagent_reports.txt` (59 subagents)

---

### dispute_fork (`780e52c1-20db-4332-8958-8bfd0d08a8e2`, 2026-06-20 21:27 – 2026-06-21 01:54)

**What it was:** Another same-evening fork, handed a fully-articulated rival theory
in-house style ("dispute? Found it — at the level the data can actually support"): a
"~472-parameter finite-state glyph automaton" claim that the 61%-vs-14%
self-regeneration gap against English "proves" Voynichese is meaningless. The session
accepts the description but attacks the inference, arguing English is an unfairly deep
baseline and the real control should be a templatic, shallow-orthography language. It
builds `automaton_control.py`, sources real Bible-parallel corpora (Hawaiian, Turkish,
Swahili, Finnish, Māori...), and gets a decisive result: Hawaiian beats Voynich at a
matched lexicon size. It then pivots to "assume the manuscript IS crackable, what next?"
— building an entropy/BPE pipeline that briefly looks like a positive verbose-cipher
signal, until a `FINDINGS.md` re-read triggers a pre-registered content-free null that
kills its own hopeful finding, discovering mid-edit that a parallel fork had already
logged the same kill by a different route.

**Key outcomes:** K2 (confirmed: Hawaiian 30.2–30.9% / Māori beat or match Voynich
19.5–28.7% at matched lexicon; agglutination alone doesn't drive the gap, phonotactic
simplicity does). K11 corroboration: a second, independent content-free null
(morpheme-slot template, not glyph-Markov) reproduces the h2 climb, and
`position_lock.py` independently replicates position-locking on a 5-bin relative metric.

**Notable moments:**
- Opening pitch, in the rival theory's own confident voice: *"The maker had solved the problem of manufacturing convincing text at scale without composing meaning... The words were the cheap part."*
- The rebuttal's core move, stated before testing: *"The control you need is not English... My strong prediction: the gap collapses."*
- The turn: *"Hawaiian (30.2%) regenerates more than Voynich (21.0%) — a real, fully meaningful human language beats Voynich on the exact metric claimed to prove Voynich isn't language."*
- The cross-session collision: *"The file moved while I was editing it — a parallel fork already ran a cipher_gate.py and logged K11... Two different nulls converge on the same kill."*
- This reads as an in-house stress test (devil's advocate), not an external claim relayed from elsewhere — there's no citation or outside source, and the session folds its result into the same shared `FINDINGS.md` ledger as if it were simply the next entry in one ongoing investigation.

**Files:** `transcripts/raw/780e52c1-20db-4332-8958-8bfd0d08a8e2.jsonl` · `transcripts/readable/dispute_fork.txt` (no subagents dispatched, no subagent report)

---

### cipher_and_motive_session (`ac7bc224-d570-492b-91a2-4be14a3267c4`, 2026-06-20 23:01 – 2026-06-21 05:19)

**What it was:** The cipher-gate/R1 falsifiers plus nearly the entire
motive/production-forensics series (M1–M14). (Originally extracted under a misleading
name; renamed to match its content in the 2026-07-01 reorg.) It opens as pure meta-analysis: the user asks it to find
prior JSONL logs and assess "are we making progress?" It digests the earlier sessions,
concludes strong methodological progress and zero decipherment progress, notices three
forks had all independently reset to the same prompt, and writes `FINDINGS.md` plus a
project-memory file specifically to stop that circling. The user then asks "are you
prepared to attempt the cipher?" — pivoting into `cipher_gate.py` (K11) and
`cipher_attack.py` (R1), both of which fail, converting the "asemic artifact" reading
from hypothesis to standing conclusion. From there it runs a long motive-forensics tail
(partly directly, partly via subagents): production-discipline tests, iconographic
exemplar-hunting (the Hausbuch/crossbow-Sagittarius chase), and archival patron research
(Visconti-Sforza). It ends not at a terminal verdict but at a genuine evidentiary wall —
two concrete non-AI-tractable next steps (image an undigitized Nuremberg manuscript;
hand-search the published Pellegrin Pavia inventory) — offered to the user as a resting
point rather than declared as closure.

**Key outcomes:** K11 and R1 (both run here — R1's positive control on real enciphered
Latin succeeds; Voynich scores identically to its content-free surrogate, +0.002
bits/char, effectively zero). The full M1–M14 motive series: M1 (A/B one transmissible
procedure), M2 (no production decay), M3 (imagery = costume), M4 (herbal-control
correction), M5 (image-first production order), M6 (no new tell), M7 (no authored
symbol-key), M8 (no spectacle front-loading), M9 (zodiac exemplar family, partially
withdrawn), M10 (not security-engineered), M11 (Clm 13076, closest verified relative),
M12 (patron profile), M13 (two crossbow-Sagittarius witnesses pixel-verified), M14
(Pellegrin archival test made executable).

**Notable moments:**
- The meta-analysis's blunt self-assessment: *"You're making strong progress on what the object is and how to study it rigorously, and no progress on what it says... the effort has reached the point of diminishing returns... and is starting to spin."*
- R1's kill-shot: *"Voynich is cryptanalytically indistinguishable from meaningless noise with the same letter statistics, and falls [0.21] bits/char short of a real cipher the method DID crack. No substitution key generalizes."*
- The exemplar hunt's honesty pattern — propose, pixel-verify, withdraw: Cod. Sang. 827 was cited from literature as the "closest named analog," then vision-checked and found to have "NO pictorial zodiac figures — only a Zodiac-Man" — *"the 'closest named analog' sub-claim is withdrawn (was a literature cite, failed pixel inspection)."*
- The closing framing: *"The productive question is why it was made, not what it says"* — and, literally at the end, *"the remaining advances require primary sources that aren't digitized or haven't been searched."*
- Unusually self-auditing throughout: it repeatedly catches its own errors (a positive-control bug, an MI mirage caught and rescued with a proper permutation null) rather than letting them stand.

**Files:** `transcripts/raw/ac7bc224-d570-492b-91a2-4be14a3267c4.jsonl` · `transcripts/readable/cipher_and_motive_session.txt` · `transcripts/subagent_reports/cipher_and_motive_session_subagent_reports.txt` (13 subagents)

---

### short_session_4e11 (`4e11ffbd-ea23-4be9-b3ec-9df6d7b1f8c0`, 2026-06-21 01:58 – 02:15)

**What it was:** A short, focused session picking up right where the previous night's
forks left off. The user invokes *Contact*'s Hadden reveal (the raster images have a
hidden third dimension) and asks for an analogous move. The assistant takes the analogy
literally: D2's "word order carries ~0 information" finding gets reread not as a dead
end but as a possible registration mark pointing at the wrong reading axis — maybe
records run down columns, not along lines. It builds and runs the column-registration
test, which looks spectacular in-sample and collapses on held-out data, then follows up
with a quire-order refinement before recommending R1 as the last real swing.

**Key outcomes:** K12 (column registration — beats a shuffle null at z=17–40 in-sample
in all four Currier×section strata, but held-out generalization gain collapses to
+0.03–0.07 bits, ~1% of the core's entropy). L7 (confound refinement: folio-position
autocorrelation is discrete block structure — same-Currier dominates — not smooth
shelf-distance decay).

**Notable moments:**
- The pitch: *"The primer was complete the whole time. What was wrong was the dimensionality the humans imposed on it... Rotate the substrate, and constraints that looked unsatisfiable suddenly close."*
- The counterweight stated before running anything: *"Hadden had a guarantee. We have a filter."*
- The kill: *"In-sample, the column axis looks electric... Held-out, it evaporates... The open key does not close... behind it was the template, not a message."*
- Closing verdict: *"We made the move, twice, and there is no hidden third dimension in the geometry of the substrate."*

**Files:** `transcripts/raw/4e11ffbd-ea23-4be9-b3ec-9df6d7b1f8c0.jsonl` · `transcripts/readable/short_session_4e11.txt` (no subagents, no subagent report)

---

### k13_grounding_session (`dcc71ad3-9b7b-4441-895f-9e47d63b1b97`, 2026-06-21 05:22 – 06:21)

**What it was:** The session that ran K13, the last un-confounded semantic test.
(Originally extracted under a misleading name; renamed to match its content in the
2026-07-01 reorg.) It opens with the user quoting
"the AI-tractable analysis is now largely exhausted" and asking whether that's true. The
assistant partly disagrees, flagging that `FINDINGS.md` itself names L6 (within-section
referent grounding — does a label's morphology track the visual attribute of the
specific referent it sits on) as the one live, un-confounded test never actually run. The
user says "proceed," and the session builds a vision-grounding pipeline: personal
grounding of two pharma folios, then a fan-out of 8 parallel vision subagents binding
labels to nymphs (balneo) and plant-elements (pharma) across ten folios, followed by a
within-folio shuffle null with power controls. Result: null. The session closes with two
reflective exchanges on what "exhausted" and "consensus" would even mean for a project
like this.

**Key outcomes:** K13 — balneo arm dead on arrival (nymphs carry almost no visual
variation to encode); pharma arm (n=35, real root/foliage/flower and color variation)
comes back null (best p=0.12), with the lone sub-0.05 hit identified as the same K12
template artifact. Power validated: positive control p=0.0000, a noisy 75%-consistent
partial control still caught at p=0.0001 — proving the null was informative, not
underpowered.

**Notable moments:**
- The opening question, quoted verbatim from the user: *"the AI-tractable analysis is now largely exhausted... do you agree... or is there another avenue remaining?"* — answered *"Largely yes for meaning-recovery — but no, the ledger isn't exhausted."*
- The balneo death: *"Balneo nymphs are visually near-homogeneous — almost every nymph came back nude, arms down, holds none... no attribute variation to carry semantics."*
- The closing summary line: *"Reference is excluded not just between classes... but within a class where no confound can hide."*
- The session ends on a genuinely reflective note rather than a scoreboard update — a four-part "how would we know we'd exhausted all avenues" framework, and a proposal to quarantine the speculative M-series from the rigorous null for any eventual publication.

**Files:** `transcripts/raw/dcc71ad3-9b7b-4441-895f-9e47d63b1b97.jsonl` · `transcripts/readable/k13_grounding_session.txt` · `transcripts/subagent_reports/k13_grounding_session_subagent_reports.txt` (8 subagents)

---

### audit session (`9966cd54-3dfe-4b04-891c-0e4793a49e04`, 2026-07-01 19:40 – 20:59 UTC, *snapshot*)

**What it was:** Ten days after the June sessions, this session reread the whole project
end to end. Five parallel Sonnet analyst agents were dispatched to read all seven prior
transcripts plus their ~113 embedded subagent reports, hunting for untested substrates,
dropped threads, overclaimed kills, and under-logged positives — checked against
`FINDINGS.md`. The verdict: the asemic conclusion held, but the audit surfaced a real,
specific gap — the zodiac's 1,222 labels had been inventoried by K13 as "has the power"
and then never tested. That gap was closed the same day: **K14** ran a 10% zodiac slice
(104 loci, anchor-based vision binding, 4 rings) and came back null at family-wise
p=0.20, with power controls firing correctly; **K14b** then checked three hypothesis
classes K14's within-ring null was structurally blind to (cross-ring positional
correspondence, same-label determinism, ring-identity codes) and found the same null,
plus an actively anti-referential signature (within-ring label duplicates are an
*excess* over random, the signature of copying, not naming discipline). The session also
did a repo-cleanup pass (see `moves.tsv`, the file-relocation manifest into
`data/`/`images/`/`docs/`) and produced `docs/AUDIT-2026-07-01.md`, this document's
sibling. **This transcript is a snapshot**: you are reading the write-up from inside the
same session that produced it, and its raw `.jsonl` only runs through 20:59:08 UTC —
before this very documentation task began.

**Key outcomes:** K14 and K14b (see `FINDINGS.md` §2 for full detail). The audit itself
is not a ledger row but is the reason K14/K14b exist; it also flagged several
under-logged-but-real prior results (the `register.py` affix-swap transform, the
`zodiac_join.py` referential null, the line-initial-gallows law, `daiin`-as-punctuation,
TextTiling boundaries) that remain open bookkeeping tasks, not test failures.

**Notable moments:**
- The audit's framing of its own mandate: five analysts "hunting for unturned stones... checked against FINDINGS.md," explicitly distinguishing "the dominant failure mode was under-logging of positives, not overclaimed kills."
- The session-attribution correction that confirms this document's own mapping: *"K12 ran in session 4e11ffbd, K13 in dcc71ad3, and the M-series in ac7bc224 — FINDINGS.md's header attributions are partly wrong."*
- K14's binding methodology fix, explicitly named as repairing K13's weakness: *"anchor-based alignment... NOT reading-order inference"* — catching, among other things, that f71r's labels sit at the star-tip *between* figures, where naive nearest-word binding would have misbound the whole ring by one.
- K14b's closing observation reframes a design choice as evidence: within-ring duplicate labels are an *excess* over chance (p=0.018) — *"duplicates cluster within rings, the opposite of naming discipline (a roster avoids giving two nymphs in one ring the same name)."*

**Files:** `transcripts/raw/9966cd54-3dfe-4b04-891c-0e4793a49e04.jsonl` (no separately extracted `readable/` or `subagent_reports/` file — the full analyst reports and the K14/K14b vision-agent reports live in the raw jsonl itself) · `docs/AUDIT-2026-07-01.md` (this session's own written summary) · `FINDINGS.md` (K14, K14b rows)

---

### Codex follow-up (`019f90e8-2ebb-76c2-8e91-73ad1f7b70dd`, 2026-07-22 – 2026-07-25)

**What it was:** An adversarial continuation that began by reviewing whether
the repository's terminal asemic conclusion followed from its evidence. It
first weakened that conclusion: meaningful Naibbe invalidated the old
position-lock exclusion, and the published suffix/prefix dependency lost its
incremental value after exact-word/core/copy/position controls. The session
then built powered known-family solvers and attacked local edit carriers,
aligned rows, compact line-reset programs, illustration-derived keys and
routes, historically licensed Latin abbreviations, causal production models,
and image-to-page-state prediction. It ended by reframing the problem around a
fifteenth-century workshop and testing a physically executable small
table/tally/wheel mechanism.

**Key outcomes:** K15–K28 and D13. The strongest constructive result is a
normalized source model: word-reset character trigram grammar, register state,
and persistent page-side adaptation. Recent copy/edit is not the main engine.
The small-counter screen initially passes, but incompatible selected periods
reduce completely to absolute page-line and word-slot profiles under a powered
residual gate. Across the broader campaign, paired meaningful, planted, and
oracle controls activate while no plaintext, visual key, or referential
mapping is recovered.

**Files:** `transcripts/raw/codex-019f90e8-2ebb-76c2-8e91-73ad1f7b70dd-messages.jsonl`
contains the actual timestamped user/assistant message records exported from
the on-disk Codex session. Executable details and caveats are in Campaigns
11–21 of `docs/RUNS.md`; scripts live in `analysis/07_followups` through
`analysis/10_generator_inversion`; derived outputs are indexed in
`data/INDEX.md`.

---

## How to read a raw `.jsonl`

Each raw file in `transcripts/raw/` is one JSON object per line — a full Claude Code
session log. The fields worth knowing:
- `type`: `"user"` or `"assistant"` (also occasional bookkeeping types like `mode`,
  `file-history-snapshot`, `attachment` that aren't conversational turns).
- `message.content`: either a plain string (simple user text) or a list of typed
  blocks — `"text"` (visible prose), `"thinking"` (extended reasoning, assistant-only),
  `"tool_use"` (a tool call with its input), and `"tool_result"` (that tool's output,
  attached to the next `user`-typed line).
- `timestamp`: ISO-8601 UTC.
- `sessionId`, `parentUuid`, `uuid`: thread the conversation together; sidechains
  (subagent turns run inline) are marked `isSidechain: true`.

The `transcripts/readable/*.txt` files are a flattened, role-tagged extraction of just
the `text`/`tool_use`/`tool_result` blocks from each raw jsonl, for skimming without a
JSON parser. The `transcripts/subagent_reports/*.txt` files pull out just each dispatched
subagent's task prompt and final report (not its full internal transcript).

**Full per-subagent transcripts** were removed from the public tree on 2026-07-02
(they embedded verbatim copies of third-party web pages and library-licensed images);
each subagent's task prompt and final report survives in `transcripts/subagent_reports/`.
