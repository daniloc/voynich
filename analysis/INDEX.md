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
| build_zodiac_visual_nodes.py | all-12 ring geometry, guarded figure-core pixel descriptors, graph coordinates, and QC overlays | K22 data |
| build_multimodal_visual_graphs.py | shared pixel/graph schema for 15 herbal and 12 zodiac diagrams | K22 data |

## 06_motive — production forensics (the "why")

| script | purpose | ledger |
|---|---|---|
| ab_procedure.py | Currier A/B = one transmissible procedure, two settings | M1 |
| degradation.py | discipline across production (no decay; caveat: rebound order) | M2 |
| hash_hypothesis.py | image↔text information budget (audit: constants flagged, needs re-derivation) | D6 ⚠ |
| ring_column_compare.py | f57v ring vs f66r column: no shared authored key | M7 |
| resistance.py | is it cryptanalysis-resistant? (no — opposite) | M10 |

## 07_followups — 2026-07-22–23 adversarial re-tests

| script | purpose | ledger |
|---|---|---|
| sequence_parallel.py | morph-class adjacency, held-out directionality, exploratory parallel-line search | D2 |
| morph_voynich_transfer.py | calibrated leave-one-quire-out prefix/affix prediction and cross-hand/Currier transfer | D2 |
| morph_control_benchmark.py | matched meaningful Naibbe/Latin controls under generic and EVA-selected representations | D2, K11, L1 |
| morph_procedural_nulls.py | layout exchange, schema/copy, and local self-citation challenges to directional morphotactics | D2, confound #3 |
| morph_markov_residual.py | nested held-out test of morphotactic value after exact-word/core, causal-copy, and position controls | D2, K16 |
| solver_residual_controls.py | meaningful and procedural calibration of prefix pooling after an exact-word surface model | K16 |
| naibbe_benchmark.py | known-meaningful verbose-cipher positive control for BPE, position-lock, slots, lexical tail, and R1 | K11, L1 |
| register_cv.py | held-out label/prose locality and canonical-form transform tests | D11-adjacent |
| semantic_cv.py | leave-one-quire-out illustration-class coupling with metadata/layout controls | K13/K14-adjacent |
| stego_channel_search.py | fixed word/glyph-stride and edge-glyph channels with held-out substitution attack | K15 |
| stego_bit_search.py | binary glyph/word-feature streams under Bacon and ASCII codecs | K15 |
| stego_slot_cipher.py | homophonic payload attack on prefix/suffix slot choices | K15 |
| naibbe_style_attack.py | structurally calibrated atomic/compound homophonic attack, benchmarked on known Naibbe | L1 |
| residual_cipher_solver.py | complete-block, gap-safe, complexity-penalized residual Naibbe-style attack | L1 |
| stego_transposition_search.py | 208 fixed hierarchical, boustrophedon, and ragged-column reading orders | K15 |

## 08_breakthrough — 2026-07-23 powered cipher recovery

| script | purpose | ledger |
|---|---|---|
| recurrence_decoder_prototype.py | first-occurrence recurrence applicability probe with a NumPy statistical decoder | L1 calibration |
| recurrence_key_synth.py | randomized-key recurrence learner with oracle and blind Naibbe structure | L1 calibration |
| naibbe_permutation_decoder.py | leakage-clean 18-block permutation solver, synthetic controls, and full/held-out official Naibbe recovery | K17 positive control |
| permutation_heldout_controls.py | split-key, block-exchange, and table-Markov false-positive calibration | K17 control |
| voynich_latent_channel_gate.py | four-fold exact/nearest/clustered latent-channel transfer from powered Naibbe controls to Voynich | K17 |
| recurrence_decoder.py | optional PyTorch causal recurrence model; end-to-end smoke tested, not used for ledger evidence | experimental |
| edit_operation_channel.py | complete-quire homophonic attack on local edit operation, position, glyph-delta, and source-lag symbols with embedded controls and matched nulls | K18 |
| edit_choice_bitstream.py | conditional edit-choice rank extraction under frozen Bacon/ASCII codecs with exact embedded payload controls | K18 |
| row_axis_gate.py | held-out previous-row bag-copy versus added same-column mixture, row-shuffle nulls, and injected table control | K19, L8 |
| parallel_passage_gate.py | selection-adjusted cross-quire rare-core alignment search under strict full-search shuffles | K20 |
| stateful_line_program_search.py | 240-candidate line-reset modular recurrence tournament with exact hidden-program controls and complete-search nulls | K20 |
| multimodal_visual_key_gate.py | plant organ graphs as stateful page keys plus clockwise zodiac visual-transition prediction, with paired controls and matched remapping nulls | K21 |
| multimodal_graph_transfer_gate.py | all-12 zodiac node gate, semantic/pixel calibration, phase and leakage controls, and controlled herbal↔zodiac pigment transfer | K22 |

## 09_external_horizon — 2026-07-24 external controls and route tests

| script | purpose | ledger |
|---|---|---|
| build_historical_abbreviation_controls.py | reproducible XIV/XV Latin Cappelli inventory plus diplomatic Nuremberg Letterbooks expansion controls | L4 calibration |
| abbreviation_transducer_gate.py | frozen-base, exhaustive historically licensed variable-length transducer with complete-quire folds and procedural nulls | L4 |
| illustration_route_gate.py | illustration-as-permutation test: paired clockwise/reverse zodiac routes and a non-inferential plant topology diagnostic | K23, plant route open |

## 10_generator_inversion — 2026-07-24–25 causal source-model recovery

| script | purpose | ledger |
|---|---|---|
| production_algorithm_gate.py | normalized held-out source-model ladder: word-reset character grammar, register and folio state, exact-copy mixture, and canonical edit channel | D13, K24 |
| adaptive_horizon_gate.py | nested selection of line, rolling, and folio-cumulative transition memory with concentration scan and planted-horizon/Latin controls | D13 |
| state_boundary_gate.py | nested reset-boundary and hierarchical-state tournament plus quire/Currier/section-matched page-prefix state swaps | D13 |
| build_page_illustration_features.py | reproducible PDF/IIIF mapping and text-guarded whole-page silhouette/pigment features with visual QC overlay | K25 substrate |
| image_state_gate.py | nested whole-quire image-to-frozen-page-state donor test with exact-stratum relabeling nulls and textual-state oracle | K25 |
| build_guarded_dinov2_embeddings.py | pinned frozen DINOv2 embeddings of full/tight, color/silhouette anti-text views with exact input QC | K26 substrate |
| visual_state_axis_gate.py | train-only text-state PCA and visual kernel-ridge prediction with nested selection, exact-stratum relabeling, and state oracles | K26 |
| setup_public_object_attack.sh | one-command pinned download, checksum verification, and load smoke test for public Surya, HORAE, SAM 2.1, and DINOv2 artifacts | K27 substrate |
| build_public_object_embeddings.py | all-225-side foldout mapping, public figure proposals, SAM-guarded complete objects, DINO views, and explicit topology with seven-page QC | K27 substrate |
| public_object_state_axis_gate.py | K26's frozen state-axis discriminator rerun on complete-object DINO/topology representations with shared-canvas exclusion | K27 |
| historical_counter_mechanism_gate.py | compact visible-boundary table/tally/wheel tournament with description cost, phase-origin nulls, planted controls, and counter-slip diagnostics | K28 screen |
| historical_counter_residual_gate.py | page-line counter retest after exact nonperiodic page-line and word-slot profiles, with a residual planted-cycle control | K28 |
