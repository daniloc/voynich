# Analysis scripts index

All scripts are run **from the repo root**: `python3 analysis/<dir>/<script>.py`.
Ledger column = the FINDINGS.md item the script produced or feeds ("unledgered" = result
exists only in a transcript; see docs/RUNS.md for the recovered numbers and
docs/AUDIT-2026-07-01.md §2 for why that matters). Detailed per-run summaries with
methods, controls, and numbers: **docs/RUNS.md**.

## 01_structure — corpus parsing, morphology, entropy, layout

| script | purpose | ledger |
|---|---|---|
| vparse.py | IVTFF parser; builds the token stream every other script consumes | infra |
| analyze.py | first-pass corpus statistics (Zipf, word lengths, hapax) | D1 |
| implicit_morph.py | unsupervised morphology (Harris branching entropy + BPE) | D1 |
| columns.py | slot-template morphology: prefix/core/suffix decomposition, MI structure | D3 |
| entropy_panel.py | conditional-entropy panel vs real-language controls | D5 |
| structure_gaps.py | word-order shuffle tests | D2 |
| structure_atlas.py | slot-grammar coverage + where the register hinge lives | unledgered |
| glue.py | high-frequency "glue" tokens; line-edge loading; daiin-as-punctuation | unledgered |
| hinges.py | q-operator strippability; prefix↔suffix agreement | unledgered |
| pivot.py | entity-axis pivot: core = entity key, attribute profiles (§4 never printed — see audit) | unledgered |
| register.py | label vs prose registers share a lexicon via affix-swap transform | unledgered |
| multimodal.py | label-vs-prose statistics from IVTFF locus roles; line-initial-gallows law | D11-adjacent, unledgered |
| stroke.py | stroke-primitive inventory; figure-9 = Cappelli -us/-um siglum | D8 |
| burst_tables.py | burstiness + single-table-vs-collection schema tests | K10-adjacent |

## 02_hypothesis_tests — specific referential hypotheses (mostly kills)

| script | purpose | ledger |
|---|---|---|
| collision.py / collision_signal.py | label collision/determinism: hash vs descriptive vs filler | K5 |
| crib.py / keyjoin.py | picture-crib decode attempts with held-out CV | K4 |
| index_test.py | folio-opener as index/TOC | K9 |
| log_test.py | transition-log / process-trajectory | K6 |
| music_test.py | is-it-music (motifs, meter, repeated lines) | K8 |
| genealogy.py | lineage/"begat" chaining | K7 |
| roster_test.py | roster/org-chart persistent-membership | unledgered (null) |
| library_shadow.py | "six recovered works" library catalog | K3 (killed by fork audit) |
| zodiac_join.py | the decoded Occitan month key joined to ring labels | unledgered (null) |
| crossmodal.py | pigment features ↔ folio text similarity | unledgered (weak, wrong-signed) |
| quire_test.py | production-unit autocorrelation: block identity vs shelf distance | L7 → confound #2 |
| manifest.py | TextTiling hidden topic boundaries | unledgered (p=0.002) |

## 03_null_models — content-free generators and controls

| script | purpose | ledger |
|---|---|---|
| null_and_morph.py | first null generators + slot-grammar quantification | D3-adjacent |
| genfit.py | schema-sampler + self-citation generator vs full statistic panel | K1 groundwork |
| residual.py | the content-free generator sweep (the K1/K10 killer) | K1, K10 |
| topic_gen.py | topic-stateful meaningless generator (its pro-content result was superseded on a different metric — audit soft spot) | K1-adjacent |
| automaton_control.py | regeneration test vs matched-lexicon real languages | D4, K2 |
| herbal_control.py | real-botany control for plant-part correlations | M4 |
| decoder_trap.py | frequency-rank "decoder" scores nulls as well as Voynich | R1 precursor |

## 04_cipher — cipher hypotheses (all excluded)

| script | purpose | ledger |
|---|---|---|
| unit_collapse.py | BPE unit-merge: does unit-h2 climb back to a language band? | L1 → K11 |
| cipher_gate.py | the matched surrogate gate that killed the verbose-cipher lead | K11 |
| verbose_cipher_null.py | second content-free null reproducing the h2 climb | K11 |
| position_lock.py | glyph positional-freedom vs real languages | K11 |
| cipher_attack.py | substitution hill-climb vs Latin LM, positive control + held-out | R1 |
| contact.py | column-registration ("rotate the substrate") test | K12 |

## 05_grounding — image↔text binding and referent tests

| script | purpose | ledger |
|---|---|---|
| page_vision.py | PDF page rendering/cropping for vision passes | infra |
| l6_grounding.py | within-section referent-variation settler (pharma n=35 + balneo) | K13 |
| z10_grounding.py | zodiac 10% slice: 104 anchor-bound loci, Westfall-Young omnibus, core-carried power controls | K14 |
| z10_cross.py | cross-ring positional join, same-label determinism, collision census | K14b |

## 06_motive — production forensics (the "why")

| script | purpose | ledger |
|---|---|---|
| ab_procedure.py | Currier A/B = one transmissible procedure, two settings | M1 |
| degradation.py | discipline across production (no decay; caveat: rebound order) | M2 |
| hash_hypothesis.py | image↔text information budget (audit: constants flagged, needs re-derivation) | D6 ⚠ |
| ring_column_compare.py | f57v ring vs f66r column: no shared authored key | M7 |
| resistance.py | is it cryptanalysis-resistant? (no — opposite) | M10 |
