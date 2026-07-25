#!/usr/bin/env python3
"""
Bounded multimodal gates for plant folios and zodiac rings.

The plant gate treats the drawing as a page key for the stateful decoder:

    symbol_i = text_program(word_i, state_i) + c * visual_feature(page) mod 23

The visual grammar is fixed from coarse graph invariants in plant_tags.json.
Candidate selection is nested by quire (fit/validation/test), and the output
substitution is fitted only on fit quires.  Nulls rematch complete plant records
to text folios either within quire (strict) or within Currier language.

The zodiac gate uses the stronger node-level bindings available for four rings.
It asks whether a visual state or visual transition predicts a modular text
coordinate on a held-out whole ring.  The null cyclically rotates the complete
visual sequence against the labels within each folio and tier, preserving both
the visual inventory and its circular autocorrelation.

These are deliberately separate tests.  The plant annotations are page-level;
the zodiac annotations are ordered node-level.  Pooling them would manufacture
a statistical unit that the source data do not contain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Optional, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import stateful_line_program_search as stateful  # noqa: E402


PLANT_TAGS = ROOT / "data" / "grounding" / "plant_tags.json"
ZODIAC_BINDINGS = ROOT / "data" / "grounding" / "z10_bindings.json"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "followups_multimodal_visual_key_gate.json"
)
SEED = 20260723

# The red labels printed into the three committed montages establish these
# alignments.  The three raw renders have their folio in both id and source.
PLANT_FOLIOS = {
    "m0_a": "f4r",
    "m0_b": "f6r",
    "m0_c": "f9r",
    "m0_d": "f11r",
    "m1_a": "f15r",
    "m1_b": "f18r",
    "m1_c": "f20r",
    "m1_d": "f23r",
    "m2_a": "f31r",
    "m2_b": "f35r",
    "m2_c": "f43r",
    "m2_d": "f52r",
    "r016_f8": "f8r",
    "r026_f16": "f16r",
    "r050_f26": "f26r",
}

# This representative basis retains every state mechanism in K20 while avoiding
# a redundant cross-product of all similar three-term affine variants.
PLANT_PROGRAM_NAMES = (
    "direct",
    "position_plus",
    "position_minus",
    "lag_minus",
    "lag_plus",
    "seed_minus",
    "seed_plus",
    "lag_minus_seed_plus",
    "running_plus",
    "running_minus",
)

ZODIAC_ATTRIBUTES = (
    "body",
    "container",
    "star_tail",
    "star_hand",
    "arms",
    "headwear",
    "facing",
)
ZODIAC_MODULI = (2, 3, 5, 7)
ZODIAC_TEXT_OPERATIONS = ("current", "delta", "lag_plus")


@dataclass(frozen=True)
class VisualCandidate:
    text_feature: str
    program: str
    visual_feature: str
    coefficient: int

    @property
    def name(self) -> str:
        return (
            f"{self.text_feature}|{self.program}|"
            f"{self.visual_feature}|{self.coefficient:+d}"
        )


@dataclass
class VisualFold:
    fold: int
    fit_blocks: list[str]
    validation_blocks: list[str]
    test_blocks: list[str]
    candidate: str
    text_feature: str
    program: str
    visual_feature: str
    coefficient: int
    language: str
    validation_initial: float
    validation_final: float
    validation_penalized: float
    test_initial: float
    test_final: float
    test_residual: float
    gap_to_language_ceiling: float
    swaps: int
    plaintext_accuracy: Optional[float]
    known_plaintext: int
    sample: str


@dataclass(frozen=True)
class ZodiacNode:
    folio: str
    tier: str
    word: str
    attrs: tuple[tuple[str, str], ...]
    clock: Optional[int]

    def attr(self, name: str) -> str:
        return dict(self.attrs).get(name, "unknown")


@dataclass(frozen=True)
class ZodiacCandidate:
    text_feature: str
    operation: str
    modulus: int
    visual_feature: str

    @property
    def name(self) -> str:
        return (
            f"{self.text_feature}|{self.operation}|"
            f"mod{self.modulus}|{self.visual_feature}"
        )


def clean_label(value: str) -> str:
    first = value.split(".")[0]
    return "".join(
        glyph
        for glyph in first
        if glyph in stateful.GLYPH_VALUE
    )


def load_plants() -> tuple[list[stateful.WordLine], dict[str, dict], list[str]]:
    tagged = json.loads(PLANT_TAGS.read_text(encoding="utf-8"))["plants"]
    plants = {
        PLANT_FOLIOS[row["id"]]: row
        for row in tagged
        if row["id"] in PLANT_FOLIOS
    }
    all_lines, _ = stateful.load_word_lines()
    lines = [line for line in all_lines if line.folio in plants]
    blocks = sorted({line.block for line in lines})
    missing = sorted(set(plants) - {line.folio for line in lines})
    if missing:
        raise RuntimeError(f"plant folios have no usable prose: {missing}")
    return lines, plants, blocks


def root_axes(row: dict) -> int:
    root = row["root"]
    if root.startswith("fibrous"):
        return 3
    return 1


def root_branch(row: dict) -> int:
    root = row["root"]
    return int(
        "branched" in root
        or "forked" in root
        or root.startswith("fibrous")
    )


def root_horizontal(row: dict) -> int:
    return int(row["root"].startswith("rhizome"))


def stem_axes(row: dict) -> int:
    return {
        "single_erect": 1,
        "branching": 2,
        "multi_radiating": 3,
    }[row["stem"]]


def leaf_order(row: dict) -> int:
    leaf = row["leaf"]
    if leaf in {"none_small", "basal_brown_mass"}:
        return 0
    if leaf == "palmate_lobed":
        return 2
    if leaf == "pinnate_fern":
        return 3
    return 1


def inflorescence_arity(row: dict) -> int:
    infl = row["infl"]
    if infl.startswith("none"):
        return 0
    if "pair" in infl:
        return 2
    if "cluster" in infl or infl == "umbel":
        return 3
    return 1


def color_count(row: dict) -> int:
    values = (row["root"], row["leaf"], row["infl"])
    return sum(
        int(any(color in value for color in ("red", "blue", "white", "brown")))
        for value in values
    )


PLANT_VISUAL_FUNCTIONS: dict[str, Callable[[dict], int]] = {
    "root_axes": root_axes,
    "root_branch": root_branch,
    "root_horizontal": root_horizontal,
    "stem_axes": stem_axes,
    "leaf_order": leaf_order,
    "infl_arity": inflorescence_arity,
    "color_count": color_count,
    "whole_graph_walk": lambda row: (
        root_axes(row)
        + 3 * root_branch(row)
        + 5 * root_horizontal(row)
        + 7 * stem_axes(row)
        + 11 * leaf_order(row)
        + 13 * inflorescence_arity(row)
        + 17 * color_count(row)
    ),
}


def plant_visual_values(
    plants: dict[str, dict],
    assignment: Optional[dict[str, str]] = None,
) -> dict[str, dict[str, int]]:
    assignment = assignment or {folio: folio for folio in plants}
    return {
        name: {
            folio: function(plants[assignment[folio]]) % stateful.MODULUS
            for folio in plants
        }
        for name, function in PLANT_VISUAL_FUNCTIONS.items()
    }


def visual_candidates(
    programs: Sequence[stateful.Program],
) -> list[VisualCandidate]:
    result = [
        VisualCandidate(feature, program.name, "none", 0)
        for feature in stateful.FEATURES
        for program in programs
    ]
    result.extend(
        VisualCandidate(feature, program.name, visual, coefficient)
        for feature in stateful.FEATURES
        for program in programs
        for visual in PLANT_VISUAL_FUNCTIONS
        for coefficient in (-1, 1)
    )
    return result


def base_symbol_lines(
    word_lines: Sequence[stateful.WordLine],
    programs: Sequence[stateful.Program],
    numeric_by_feature: dict[str, Sequence[stateful.NumericLine]],
) -> dict[tuple[str, str], list[stateful.SymbolLine]]:
    return {
        (feature, program.name): [
            stateful.apply_program(line, program) for line in numeric
        ]
        for feature, numeric in numeric_by_feature.items()
        for program in programs
    }


def add_visual_shift(
    base: Sequence[stateful.SymbolLine],
    word_lines: Sequence[stateful.WordLine],
    values: dict[str, dict[str, int]],
    candidate: VisualCandidate,
) -> list[stateful.SymbolLine]:
    if candidate.visual_feature == "none":
        return list(base)
    per_folio = values[candidate.visual_feature]
    result = []
    for symbols, source in zip(base, word_lines):
        shift = candidate.coefficient * per_folio[source.folio]
        result.append(
            stateful.SymbolLine(
                symbols.block,
                tuple(
                    (symbol + shift) % stateful.MODULUS
                    for symbol in symbols.symbols
                ),
                symbols.truth,
            )
        )
    return result


def candidate_symbol_lines(
    base: dict[tuple[str, str], list[stateful.SymbolLine]],
    word_lines: Sequence[stateful.WordLine],
    values: dict[str, dict[str, int]],
    candidate: VisualCandidate,
) -> list[stateful.SymbolLine]:
    return add_visual_shift(
        base[(candidate.text_feature, candidate.program)],
        word_lines,
        values,
        candidate,
    )


def search_plant_fold(
    word_lines: Sequence[stateful.WordLine],
    base: dict[tuple[str, str], list[stateful.SymbolLine]],
    visual_values: dict[str, dict[str, int]],
    candidates: Sequence[VisualCandidate],
    program_lookup: dict[str, stateful.Program],
    languages: dict[str, stateful.LanguageSpec],
    blocks: Sequence[str],
    fold: int,
    shortlist: int,
    maximum_swaps: int,
    key_restarts: int,
) -> VisualFold:
    fit_blocks, validation_blocks, test_blocks = stateful.block_roles(
        blocks, fold
    )
    initial_rows = []
    for candidate in candidates:
        lines = candidate_symbol_lines(
            base, word_lines, visual_values, candidate
        )
        for language_name, language in languages.items():
            key = stateful.initialize_key(lines, fit_blocks, language)
            quads, weights = stateful.quad_counter(lines, validation_blocks)
            score = stateful.score_key(quads, weights, key, language)
            initial_rows.append(
                (score, candidate.name, candidate, language_name, key)
            )
    initial_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)

    refined = []
    for initial_score, _, candidate, language_name, key in initial_rows[
        :shortlist
    ]:
        language = languages[language_name]
        lines = candidate_symbol_lines(
            base, word_lines, visual_values, candidate
        )
        seed_material = (
            f"{candidate.name}|{language_name}|{fold}".encode("ascii")
        )
        restart_rng = random.Random(
            int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8], "big"
            )
        )
        trials = []
        for restart in range(key_restarts):
            trial_key = list(key)
            if restart:
                # Perturb the fit-only initializer, then select the restart by
                # fit score alone.  Validation remains untouched until the key
                # is frozen, so candidate selection does not leak.
                for _ in range(2 + 2 * restart):
                    first, second = restart_rng.sample(
                        range(stateful.MODULUS), 2
                    )
                    trial_key[first], trial_key[second] = (
                        trial_key[second],
                        trial_key[first],
                    )
            trials.append(
                stateful.refine_key(
                    lines,
                    fit_blocks,
                    language,
                    trial_key,
                    maximum_swaps,
                )
            )
        key, fit_initial, fit_final, swaps = max(
            trials, key=lambda item: item[2]
        )
        quads, weights = stateful.quad_counter(lines, validation_blocks)
        validation = stateful.score_key(quads, weights, key, language)
        sample_size = max(1.0, float(weights.sum()))
        visual_complexity = int(candidate.visual_feature != "none")
        complexity = (
            program_lookup[candidate.program].complexity + visual_complexity
        )
        penalty = complexity * math.log(max(sample_size, 2.0)) / (
            2.0 * sample_size
        )
        refined.append(
            (
                validation - penalty,
                candidate.name,
                candidate,
                language_name,
                key,
                initial_score,
                validation,
                fit_initial,
                fit_final,
                swaps,
            )
        )
    winner = max(refined, key=lambda item: (item[0], item[1]))
    (
        validation_penalized,
        _,
        candidate,
        language_name,
        key,
        validation_initial,
        validation_final,
        _fit_initial,
        _fit_final,
        swaps,
    ) = winner
    language = languages[language_name]
    lines = candidate_symbol_lines(base, word_lines, visual_values, candidate)
    quads, weights = stateful.quad_counter(lines, test_blocks)
    initial_key = stateful.initialize_key(lines, fit_blocks, language)
    test_initial = stateful.score_key(quads, weights, initial_key, language)
    test_final = stateful.score_key(quads, weights, key, language)
    accuracy, known = stateful.plaintext_metrics(lines, test_blocks, key)
    return VisualFold(
        fold=fold,
        fit_blocks=sorted(fit_blocks),
        validation_blocks=sorted(validation_blocks),
        test_blocks=sorted(test_blocks),
        candidate=candidate.name,
        text_feature=candidate.text_feature,
        program=candidate.program,
        visual_feature=candidate.visual_feature,
        coefficient=candidate.coefficient,
        language=language_name,
        validation_initial=validation_initial,
        validation_final=validation_final,
        validation_penalized=validation_penalized,
        test_initial=test_initial,
        test_final=test_final,
        test_residual=test_final - test_initial,
        gap_to_language_ceiling=test_final - language.ceiling,
        swaps=swaps,
        plaintext_accuracy=accuracy,
        known_plaintext=known,
        sample=stateful.render_sample(lines, test_blocks, key),
    )


def run_plant_search(
    word_lines: Sequence[stateful.WordLine],
    numeric_by_feature: dict[str, Sequence[stateful.NumericLine]],
    visual_values: dict[str, dict[str, int]],
    candidates: Sequence[VisualCandidate],
    programs: Sequence[stateful.Program],
    languages: dict[str, stateful.LanguageSpec],
    blocks: Sequence[str],
    shortlist: int,
    maximum_swaps: int,
    key_restarts: int,
) -> list[VisualFold]:
    base = base_symbol_lines(word_lines, programs, numeric_by_feature)
    lookup = {program.name: program for program in programs}
    return [
        search_plant_fold(
            word_lines,
            base,
            visual_values,
            candidates,
            lookup,
            languages,
            blocks,
            fold,
            shortlist,
            maximum_swaps,
            key_restarts,
        )
        for fold in range(4)
    ]


def summarize_plant_folds(folds: Sequence[VisualFold]) -> dict:
    selections = Counter(fold.candidate + "|" + fold.language for fold in folds)
    return {
        "mean_test_final": mean(fold.test_final for fold in folds),
        "mean_test_residual": mean(fold.test_residual for fold in folds),
        "mean_gap_to_language_ceiling": mean(
            fold.gap_to_language_ceiling for fold in folds
        ),
        "mean_plaintext_accuracy": (
            mean(
                fold.plaintext_accuracy
                for fold in folds
                if fold.plaintext_accuracy is not None
            )
            if any(fold.plaintext_accuracy is not None for fold in folds)
            else None
        ),
        "visual_selection_rate": mean(
            fold.visual_feature != "none" for fold in folds
        ),
        "selection_consistency": max(selections.values()) / len(folds),
        "selection_counts": dict(sorted(selections.items())),
    }


def rematch_plants(
    word_lines: Sequence[stateful.WordLine],
    plants: dict[str, dict],
    rng: random.Random,
    mode: str,
) -> tuple[dict[str, str], int]:
    meta = {
        line.folio: (line.block, line.currier)
        for line in word_lines
    }
    groups: dict[str, list[str]] = defaultdict(list)
    for folio in plants:
        block, currier = meta[folio]
        key = block if mode == "within_quire" else currier
        groups[key].append(folio)
    assignment = {}
    moved = 0
    for folios in groups.values():
        source = folios[:]
        if len(source) > 1:
            for _ in range(20):
                rng.shuffle(source)
                if any(a != b for a, b in zip(folios, source)):
                    break
        for target, origin in zip(folios, source):
            assignment[target] = origin
            moved += target != origin
    return assignment, moved


def synthetic_visual_numeric(
    word_lines: Sequence[stateful.WordLine],
    plaintext: str,
    program: stateful.Program,
    visual_values: dict[str, dict[str, int]],
    visual_feature: str,
    coefficient: int,
    rng: random.Random,
) -> list[stateful.NumericLine]:
    key = list(range(stateful.MODULUS))
    rng.shuffle(key)
    inverse_key = [0] * stateful.MODULUS
    for cipher_symbol, plain_letter in enumerate(key):
        inverse_key[plain_letter] = cipher_symbol
    plain_ids = [stateful.ALPHABET.index(character) for character in plaintext]

    cursor = 0
    result = []
    for line in word_lines:
        truth: list[Optional[int]] = [None] * len(line.words)
        raw_cipher: list[Optional[int]] = [None] * len(line.words)
        start = 1 if program.skip_first else 0
        shift = coefficient * visual_values[visual_feature][line.folio]
        for index in range(start, len(line.words)):
            letter = plain_ids[cursor % len(plain_ids)]
            cursor += 1
            truth[index] = letter
            raw_cipher[index] = (inverse_key[letter] - shift) % stateful.MODULUS
        values = stateful.invert_program(program, raw_cipher, rng)
        result.append(
            stateful.NumericLine(line.block, values, tuple(truth))
        )
    return result


def empirical_upper_p(observed: float, null: Sequence[float]) -> float:
    return (
        1 + sum(value >= observed - 1e-12 for value in null)
    ) / (len(null) + 1)


def run_plant_gate(args: argparse.Namespace) -> dict:
    word_lines, plants, blocks = load_plants()
    languages = stateful.load_languages(args.languages)
    programs_by_name = {
        program.name: program for program in stateful.PROGRAMS
    }
    programs = [programs_by_name[name] for name in PLANT_PROGRAM_NAMES]
    candidates = visual_candidates(programs)
    baseline_candidates = [
        candidate
        for candidate in candidates
        if candidate.visual_feature == "none"
    ]
    values = plant_visual_values(plants)
    numeric = {
        feature: stateful.feature_lines(word_lines, feature)
        for feature in stateful.FEATURES
    }

    baseline_folds = run_plant_search(
        word_lines,
        numeric,
        values,
        baseline_candidates,
        programs,
        languages,
        blocks,
        args.plant_shortlist,
        args.maximum_swaps,
        args.key_restarts,
    )
    baseline_summary = summarize_plant_folds(baseline_folds)
    observed_folds = run_plant_search(
        word_lines,
        numeric,
        values,
        candidates,
        programs,
        languages,
        blocks,
        args.plant_shortlist,
        args.maximum_swaps,
        args.key_restarts,
    )
    observed_summary = summarize_plant_folds(observed_folds)
    observed_gain = (
        observed_summary["mean_test_final"]
        - baseline_summary["mean_test_final"]
    )
    if args.progress:
        print(
            "plant observed: "
            f"final={observed_summary['mean_test_final']:+.4f} "
            f"gain={observed_gain:+.4f} "
            f"visual-rate={observed_summary['visual_selection_rate']:.2f}",
            flush=True,
        )

    latin_clean = stateful.clean_latin(
        stateful.CONTROL_PATHS["latin"].read_text(
            encoding="utf-8", errors="ignore"
        )
    )
    control_plaintext = latin_clean[-stateful.LM_CEILING_CHARS :]
    synthetic_specs = (
        ("stem_axes", 1, "lag_minus"),
        ("whole_graph_walk", -1, "running_plus"),
    )
    synthetic = []
    for index, (visual, coefficient, program_name) in enumerate(
        synthetic_specs
    ):
        synthetic_numeric = synthetic_visual_numeric(
            word_lines,
            control_plaintext,
            programs_by_name[program_name],
            values,
            visual,
            coefficient,
            random.Random(SEED + 100 + index),
        )
        folds = run_plant_search(
            word_lines,
            {"synthetic": synthetic_numeric},
            values,
            [
                VisualCandidate(
                    "synthetic",
                    program.name,
                    visual_name,
                    candidate_coefficient,
                )
                for program in programs
                for visual_name in PLANT_VISUAL_FUNCTIONS
                for candidate_coefficient in (-1, 1)
            ],
            programs,
            {"latin": languages["latin"]},
            blocks,
            args.plant_shortlist,
            args.maximum_swaps,
            args.key_restarts,
        )
        summary = summarize_plant_folds(folds)
        summary["true_visual_feature"] = visual
        summary["true_coefficient"] = coefficient
        summary["true_program"] = program_name
        summary["passed"] = (
            summary["mean_plaintext_accuracy"] is not None
            and summary["mean_plaintext_accuracy"] >= 0.80
            and summary["mean_gap_to_language_ceiling"] >= -0.75
        )
        mismatch_assignment, moved = rematch_plants(
            word_lines,
            plants,
            random.Random(SEED + 300 + index),
            "within_currier",
        )
        mismatched_values = plant_visual_values(
            plants, mismatch_assignment
        )
        mismatch_folds = run_plant_search(
            word_lines,
            {"synthetic": synthetic_numeric},
            mismatched_values,
            [
                VisualCandidate(
                    "synthetic",
                    program.name,
                    visual_name,
                    candidate_coefficient,
                )
                for program in programs
                for visual_name in PLANT_VISUAL_FUNCTIONS
                for candidate_coefficient in (-1, 1)
            ],
            programs,
            {"latin": languages["latin"]},
            blocks,
            args.plant_shortlist,
            args.maximum_swaps,
            args.key_restarts,
        )
        mismatch_summary = summarize_plant_folds(mismatch_folds)
        summary["paired_accuracy_advantage"] = (
            summary["mean_plaintext_accuracy"]
            - mismatch_summary["mean_plaintext_accuracy"]
        )
        synthetic.append(
            {
                "summary": summary,
                "folds": [asdict(fold) for fold in folds],
                "mismatched_control": {
                    "moved_folios": moved,
                    "assignment": mismatch_assignment,
                    "summary": mismatch_summary,
                    "folds": [
                        asdict(fold) for fold in mismatch_folds
                    ],
                },
            }
        )
        if args.progress:
            print(
                f"plant synthetic {visual}/{program_name}: "
                f"accuracy={summary['mean_plaintext_accuracy']:.3f} "
                f"mismatch={mismatch_summary['mean_plaintext_accuracy']:.3f} "
                f"gap={summary['mean_gap_to_language_ceiling']:+.3f} "
                f"pass={summary['passed']}",
                flush=True,
            )

    null_reports = []
    rng = random.Random(SEED + 500)
    for mode in ("within_quire", "within_currier"):
        for replicate in range(args.plant_nulls):
            assignment, moved = rematch_plants(
                word_lines, plants, rng, mode
            )
            null_values = plant_visual_values(plants, assignment)
            folds = run_plant_search(
                word_lines,
                numeric,
                null_values,
                candidates,
                programs,
                languages,
                blocks,
                args.plant_shortlist,
                args.maximum_swaps,
                args.key_restarts,
            )
            summary = summarize_plant_folds(folds)
            summary["gain_over_observed_baseline"] = (
                summary["mean_test_final"]
                - baseline_summary["mean_test_final"]
            )
            null_reports.append(
                {
                    "mode": mode,
                    "replicate": replicate + 1,
                    "moved_folios": moved,
                    "assignment": assignment,
                    "summary": summary,
                    "folds": [asdict(fold) for fold in folds],
                }
            )
            if args.progress:
                print(
                    f"plant null {mode} {replicate + 1}/"
                    f"{args.plant_nulls}: "
                    f"gain={summary['gain_over_observed_baseline']:+.4f} "
                    f"moved={moved}",
                    flush=True,
                )

    pvalues = {}
    for mode in ("within_quire", "within_currier"):
        values_for_mode = [
            row["summary"]["gain_over_observed_baseline"]
            for row in null_reports
            if row["mode"] == mode
        ]
        pvalues[mode] = empirical_upper_p(observed_gain, values_for_mode)

    return {
        "design": {
            "folios": sorted(plants),
            "n_folios": len(plants),
            "n_lines": len(word_lines),
            "n_words": sum(len(line.words) for line in word_lines),
            "blocks": blocks,
            "text_features": list(stateful.FEATURES),
            "programs": [program.name for program in programs],
            "visual_features": list(PLANT_VISUAL_FUNCTIONS),
            "coefficients": [-1, 1],
            "languages": list(args.languages),
            "candidate_count": len(candidates) * len(args.languages),
            "null_unit": "complete plant-record rematching",
        },
        "baseline": {
            "summary": baseline_summary,
            "folds": [asdict(fold) for fold in baseline_folds],
        },
        "observed": {
            "summary": observed_summary,
            "gain_over_baseline": observed_gain,
            "folds": [asdict(fold) for fold in observed_folds],
        },
        "synthetic_controls": synthetic,
        "nulls": null_reports,
        "p_gain": pvalues,
    }


def load_zodiac_sequences() -> dict[tuple[str, str], list[ZodiacNode]]:
    source = json.loads(ZODIAC_BINDINGS.read_text(encoding="utf-8"))[
        "folios"
    ]
    sequences = {}
    for folio, page in source.items():
        by_tier: dict[str, list[ZodiacNode]] = defaultdict(list)
        for record in page["records"]:
            tier = record["tier"]
            if tier not in {"inner", "outer"}:
                continue
            word = clean_label(record["label"])
            if len(word) < 2:
                continue
            attrs = tuple(
                (name, str(record.get(name) or "none"))
                for name in ZODIAC_ATTRIBUTES
            )
            clock_raw = record.get("clock")
            try:
                clock = int(clock_raw)
            except (TypeError, ValueError):
                clock = None
            by_tier[tier].append(
                ZodiacNode(folio, tier, word, attrs, clock)
            )
        for tier, nodes in by_tier.items():
            if len(nodes) >= 5:
                sequences[(folio, tier)] = nodes
    return sequences


def zodiac_visual_value(
    current: ZodiacNode,
    previous: ZodiacNode,
    name: str,
) -> str:
    if name.startswith("current_"):
        return current.attr(name.removeprefix("current_"))
    if name.startswith("same_"):
        attr = name.removeprefix("same_")
        return "same" if current.attr(attr) == previous.attr(attr) else "change"
    if name == "change_count":
        changed = sum(
            current.attr(attr) != previous.attr(attr)
            for attr in ZODIAC_ATTRIBUTES
        )
        return str(min(changed, 3))
    if name == "clock_gap":
        if current.clock is None or previous.clock is None:
            return "unknown"
        gap = (current.clock - previous.clock) % 12
        return str(min(gap, 3))
    raise ValueError(name)


ZODIAC_VISUAL_FEATURES = (
    tuple(f"current_{name}" for name in ZODIAC_ATTRIBUTES)
    + tuple(f"same_{name}" for name in ZODIAC_ATTRIBUTES)
    + ("change_count", "clock_gap")
)


def zodiac_text_value(
    current: ZodiacNode,
    previous: ZodiacNode,
    feature: str,
    operation: str,
    modulus: int,
) -> int:
    now = stateful.feature_value(current.word, feature)
    lag = stateful.feature_value(previous.word, feature)
    if operation == "current":
        value = now
    elif operation == "delta":
        value = now - lag
    elif operation == "lag_plus":
        value = now + lag
    else:
        raise ValueError(operation)
    return value % modulus


def zodiac_candidates() -> list[ZodiacCandidate]:
    return [
        ZodiacCandidate(feature, operation, modulus, visual)
        for feature in stateful.FEATURES
        for operation in ZODIAC_TEXT_OPERATIONS
        for modulus in ZODIAC_MODULI
        for visual in ZODIAC_VISUAL_FEATURES
    ]


def categorical_log_gain(
    sequences: dict[tuple[str, str], list[ZodiacNode]],
    candidate: ZodiacCandidate,
    train_folios: set[str],
    score_folios: set[str],
    alpha: float = 4.0,
) -> tuple[float, int]:
    base_counts: dict[str, Counter[int]] = defaultdict(Counter)
    visual_counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    for (folio, tier), nodes in sequences.items():
        if folio not in train_folios:
            continue
        for index, current in enumerate(nodes):
            previous = nodes[index - 1]
            outcome = zodiac_text_value(
                current,
                previous,
                candidate.text_feature,
                candidate.operation,
                candidate.modulus,
            )
            visual = zodiac_visual_value(
                current, previous, candidate.visual_feature
            )
            base_counts[tier][outcome] += 1
            visual_counts[(tier, visual)][outcome] += 1

    gain = 0.0
    count = 0
    for (folio, tier), nodes in sequences.items():
        if folio not in score_folios:
            continue
        base_total = sum(base_counts[tier].values())
        if base_total == 0:
            continue
        for index, current in enumerate(nodes):
            previous = nodes[index - 1]
            outcome = zodiac_text_value(
                current,
                previous,
                candidate.text_feature,
                candidate.operation,
                candidate.modulus,
            )
            visual = zodiac_visual_value(
                current, previous, candidate.visual_feature
            )
            base_probability = (
                base_counts[tier][outcome] + 1.0
            ) / (base_total + candidate.modulus)
            conditioned = visual_counts[(tier, visual)]
            visual_probability = (
                conditioned[outcome] + alpha * base_probability
            ) / (sum(conditioned.values()) + alpha)
            gain += math.log2(visual_probability / base_probability)
            count += 1
    return (gain / count if count else float("-inf")), count


def run_zodiac_search(
    sequences: dict[tuple[str, str], list[ZodiacNode]],
    candidates: Sequence[ZodiacCandidate],
) -> list[dict]:
    folios = sorted({folio for folio, _ in sequences})
    folds = []
    for fold, test in enumerate(folios):
        validation = folios[(fold + 1) % len(folios)]
        train = set(folios) - {test, validation}
        ranked = []
        for candidate in candidates:
            score, n_validation = categorical_log_gain(
                sequences, candidate, train, {validation}
            )
            ranked.append((score, candidate.name, candidate, n_validation))
        _, _, winner, n_validation = max(
            ranked, key=lambda item: (item[0], item[1])
        )
        test_gain, n_test = categorical_log_gain(
            sequences, winner, train, {test}
        )
        folds.append(
            {
                "fold": fold,
                "train_folios": sorted(train),
                "validation_folio": validation,
                "test_folio": test,
                "candidate": winner.name,
                "text_feature": winner.text_feature,
                "operation": winner.operation,
                "modulus": winner.modulus,
                "visual_feature": winner.visual_feature,
                "validation_gain_bits_per_node": max(ranked)[0],
                "test_gain_bits_per_node": test_gain,
                "n_validation": n_validation,
                "n_test": n_test,
            }
        )
    return folds


def rotate_zodiac_visuals(
    sequences: dict[tuple[str, str], list[ZodiacNode]],
    rng: random.Random,
) -> tuple[dict[tuple[str, str], list[ZodiacNode]], dict[str, int]]:
    result = {}
    offsets = {}
    for key, nodes in sequences.items():
        offset = rng.randrange(1, len(nodes))
        attrs = [node.attrs for node in nodes]
        clocks = [node.clock for node in nodes]
        attrs = attrs[offset:] + attrs[:offset]
        clocks = clocks[offset:] + clocks[:offset]
        result[key] = [
            ZodiacNode(
                node.folio,
                node.tier,
                node.word,
                attr,
                clock,
            )
            for node, attr, clock in zip(nodes, attrs, clocks)
        ]
        offsets["|".join(key)] = offset
    return result, offsets


def zodiac_summary(folds: Sequence[dict]) -> dict:
    selections = Counter(fold["candidate"] for fold in folds)
    return {
        "mean_test_gain_bits_per_node": mean(
            fold["test_gain_bits_per_node"] for fold in folds
        ),
        "positive_test_folds": sum(
            fold["test_gain_bits_per_node"] > 0 for fold in folds
        ),
        "selection_consistency": max(selections.values()) / len(folds),
        "selection_counts": dict(sorted(selections.items())),
    }


def synthetic_zodiac_control(
    sequences: dict[tuple[str, str], list[ZodiacNode]],
) -> dict:
    # A direct held-out power check for the categorical prediction machinery.
    # Outcome is a noisy binary function of a real visual transition; no text is
    # used, and whole folios remain the unit of generalization.
    folios = sorted({folio for folio, _ in sequences})
    rng = random.Random(SEED + 900)
    rows = {}
    for key, nodes in sequences.items():
        values = []
        for index, current in enumerate(nodes):
            previous = nodes[index - 1]
            value = int(current.attr("headwear") != previous.attr("headwear"))
            if rng.random() < 0.10:
                value = 1 - value
            values.append(value)
        rows[key] = values

    features = list(ZODIAC_VISUAL_FEATURES)

    def gain(
        feature: str, train: set[str], score: set[str]
    ) -> tuple[float, int]:
        base = Counter()
        conditional: dict[str, Counter[int]] = defaultdict(Counter)
        for key, nodes in sequences.items():
            if key[0] not in train:
                continue
            for index, current in enumerate(nodes):
                previous = nodes[index - 1]
                visual = zodiac_visual_value(current, previous, feature)
                outcome = rows[key][index]
                base[outcome] += 1
                conditional[visual][outcome] += 1
        total = sum(base.values())
        score_value = 0.0
        count = 0
        for key, nodes in sequences.items():
            if key[0] not in score:
                continue
            for index, current in enumerate(nodes):
                previous = nodes[index - 1]
                visual = zodiac_visual_value(current, previous, feature)
                outcome = rows[key][index]
                p0 = (base[outcome] + 1) / (total + 2)
                bucket = conditional[visual]
                p1 = (bucket[outcome] + 4 * p0) / (
                    sum(bucket.values()) + 4
                )
                score_value += math.log2(p1 / p0)
                count += 1
        return score_value / count, count

    folds = []
    for fold, test in enumerate(folios):
        validation = folios[(fold + 1) % len(folios)]
        train = set(folios) - {test, validation}
        ranked = [(gain(feature, train, {validation})[0], feature)
                  for feature in features]
        validation_gain, winner = max(ranked)
        test_gain, n_test = gain(winner, train, {test})
        folds.append(
            {
                "fold": fold,
                "test_folio": test,
                "validation_folio": validation,
                "selected_visual_feature": winner,
                "validation_gain_bits_per_node": validation_gain,
                "test_gain_bits_per_node": test_gain,
                "n_test": n_test,
            }
        )
    summary = {
        "true_visual_feature": "same_headwear",
        "noise_rate": 0.10,
        "mean_test_gain_bits_per_node": mean(
            fold["test_gain_bits_per_node"] for fold in folds
        ),
        "correct_selection_rate": mean(
            fold["selected_visual_feature"] == "same_headwear"
            for fold in folds
        ),
    }
    summary["passed"] = (
        summary["mean_test_gain_bits_per_node"] > 0.10
        and summary["correct_selection_rate"] >= 0.50
    )
    return {"summary": summary, "folds": folds}


def run_zodiac_gate(args: argparse.Namespace) -> dict:
    sequences = load_zodiac_sequences()
    candidates = zodiac_candidates()
    observed_folds = run_zodiac_search(sequences, candidates)
    observed_summary = zodiac_summary(observed_folds)
    if args.progress:
        print(
            "zodiac observed: "
            f"gain={observed_summary['mean_test_gain_bits_per_node']:+.4f} "
            f"positive-folds={observed_summary['positive_test_folds']}/4",
            flush=True,
        )
    control = synthetic_zodiac_control(sequences)
    if args.progress:
        summary = control["summary"]
        print(
            "zodiac synthetic: "
            f"gain={summary['mean_test_gain_bits_per_node']:+.4f} "
            f"selection={summary['correct_selection_rate']:.2f} "
            f"pass={summary['passed']}",
            flush=True,
        )

    rng = random.Random(SEED + 1200)
    nulls = []
    for replicate in range(args.zodiac_nulls):
        rotated, offsets = rotate_zodiac_visuals(sequences, rng)
        folds = run_zodiac_search(rotated, candidates)
        summary = zodiac_summary(folds)
        nulls.append(
            {
                "replicate": replicate + 1,
                "offsets": offsets,
                "summary": summary,
                "folds": folds,
            }
        )
        if args.progress and (
            replicate < 4 or (replicate + 1) % 10 == 0
        ):
            print(
                f"zodiac null {replicate + 1}/{args.zodiac_nulls}: "
                f"gain={summary['mean_test_gain_bits_per_node']:+.4f}",
                flush=True,
            )
    null_gain = [
        row["summary"]["mean_test_gain_bits_per_node"] for row in nulls
    ]
    return {
        "design": {
            "folios": sorted({folio for folio, _ in sequences}),
            "sequences": [
                {
                    "folio": folio,
                    "tier": tier,
                    "nodes": len(nodes),
                }
                for (folio, tier), nodes in sorted(sequences.items())
            ],
            "n_nodes": sum(len(nodes) for nodes in sequences.values()),
            "text_features": list(stateful.FEATURES),
            "operations": list(ZODIAC_TEXT_OPERATIONS),
            "moduli": list(ZODIAC_MODULI),
            "visual_features": list(ZODIAC_VISUAL_FEATURES),
            "candidate_count": len(candidates),
            "null_unit": "nonzero cyclic visual rotation within folio and tier",
        },
        "observed": {
            "summary": observed_summary,
            "folds": observed_folds,
        },
        "synthetic_control": control,
        "nulls": nulls,
        "p_gain": empirical_upper_p(
            observed_summary["mean_test_gain_bits_per_node"], null_gain
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant-nulls", type=int, default=6)
    parser.add_argument("--zodiac-nulls", type=int, default=40)
    parser.add_argument("--plant-shortlist", type=int, default=18)
    parser.add_argument("--maximum-swaps", type=int, default=48)
    parser.add_argument("--key-restarts", type=int, default=3)
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=tuple(stateful.CONTROL_PATHS),
        default=("latin", "english"),
    )
    parser.add_argument("--skip-plants", action="store_true")
    parser.add_argument("--skip-zodiac", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plant_nulls < 1 or args.zodiac_nulls < 1:
        raise ValueError("null counts must be positive")
    if (
        args.plant_shortlist < 1
        or args.maximum_swaps < 0
        or args.key_restarts < 1
    ):
        raise ValueError("invalid search limits")
    if "latin" not in args.languages:
        raise ValueError("Latin is required for the synthetic plant controls")

    report = {
        "experiment": "multimodal_visual_key_gate",
        "seed": SEED,
        "plant_gate": None,
        "zodiac_gate": None,
        "cross_domain_transfer": {
            "status": "not_identifiable",
            "reason": (
                "plant annotations are page-level organ graphs while zodiac "
                "annotations are ordered figure nodes; no shared visual "
                "coordinate is present in the committed data"
            ),
        },
    }
    if not args.skip_plants:
        report["plant_gate"] = run_plant_gate(args)
    if not args.skip_zodiac:
        report["zodiac_gate"] = run_zodiac_gate(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.progress:
        print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
