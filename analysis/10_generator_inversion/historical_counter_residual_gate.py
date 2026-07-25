#!/usr/bin/env python3
"""
Discriminate a repeating page-line counter from ordinary layout position.

historical_counter_mechanism_gate.py detects a small but held-out-stable
alignment to page-relative line number.  That can arise without a counter if
opening lines, early page lines, or absolute word slots have distinct glyph
profiles.  This follow-up fits those nonperiodic profiles from training quires:

* exact page-line index 0..10, then a shared 11+ class;
* exact word-in-line index 0..10, then a shared 11+ class.

It then repeats the counter search on residual glyph choices, restricted to a
page-reset line counter.  Validation pays a one-bit lower-bound description
charge per phase/glyph entry and may select no counter.  The full phase-origin
null repeats selection after independently rotating every page.  A planted
line-mod-4 source verifies that a genuine repeating cycle remains recoverable
after the absolute-position profiles are removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import historical_counter_mechanism_gate as counter
import production_algorithm_gate as source


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_historical_counter_residual_gate.json"
)
SEED = 20260725
POSITION_CLIP = 11
NUISANCE_ALPHA = 256.0
NUISANCE_ITERATIONS = 4
MODULI = counter.MODULI
ALPHAS = counter.ALPHAS
DEFAULT_NULLS = 32
CONTROL_NULLS = 8
PLANTED_MODULUS = 4


@dataclass
class StructuralFold:
    raw: counter.PreparedFold
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    nuisance: dict[str, object]


@dataclass
class ResidualTable:
    observed: np.ndarray
    expected: np.ndarray
    multipliers: np.ndarray


def classes(
    split: counter.EncodedSplit,
    coordinate: str,
) -> np.ndarray:
    return np.minimum(
        split.coordinates[coordinate], POSITION_CLIP
    ).astype(np.int32)


def normalized_matrix(
    prepared: counter.PreparedFold,
    split: counter.EncodedSplit,
    line_multiplier: np.ndarray,
    word_multiplier: np.ndarray,
    include_line: bool = True,
    include_word: bool = True,
) -> np.ndarray:
    matrix = prepared.base_vectors[split.group].copy()
    if include_line:
        matrix *= line_multiplier[
            classes(split, "line_in_page")
        ]
    if include_word:
        matrix *= word_multiplier[
            classes(split, "word_in_line")
        ]
    matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1e-300)
    return matrix


def fit_multiplier(
    matrix: np.ndarray,
    split: counter.EncodedSplit,
    labels: np.ndarray,
    states: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphabet_size = matrix.shape[1]
    flat = labels * alphabet_size + split.symbol
    observed = np.bincount(
        flat, minlength=states * alphabet_size
    ).reshape(states, alphabet_size).astype(np.float64)
    expected = np.zeros_like(observed)
    for state in range(states):
        indices = labels == state
        if np.any(indices):
            expected[state] = matrix[indices].sum(axis=0)
    totals = expected.sum(axis=1, keepdims=True)
    shares = np.divide(
        expected,
        totals,
        out=np.full_like(expected, 1.0 / alphabet_size),
        where=totals > 0,
    )
    multiplier = (
        observed + alpha * shares
    ) / np.maximum(expected + alpha * shares, 1e-12)
    return multiplier, observed, expected


def adaptive_matrix_log2(
    split: counter.EncodedSplit,
    matrix: np.ndarray,
) -> float:
    observed = matrix[np.arange(split.symbols), split.symbol]
    probability = (
        split.page_count + counter.PAGE_BACKOFF * observed
    ) / (split.page_total + counter.PAGE_BACKOFF)
    return float(np.log2(np.maximum(probability, 1e-300)).sum())


def fit_structural_fold(
    prepared: counter.PreparedFold,
) -> StructuralFold:
    states = POSITION_CLIP + 1
    alphabet_size = len(prepared.alphabet)
    line_multiplier = np.ones((states, alphabet_size))
    word_multiplier = np.ones((states, alphabet_size))
    train_line = classes(prepared.train, "line_in_page")
    train_word = classes(prepared.train, "word_in_line")
    train_base = prepared.base_vectors[prepared.train.group]
    independent_line, _line_obs, _line_exp = fit_multiplier(
        train_base,
        prepared.train,
        train_line,
        states,
        NUISANCE_ALPHA,
    )
    independent_word, _word_obs, _word_exp = fit_multiplier(
        train_base,
        prepared.train,
        train_word,
        states,
        NUISANCE_ALPHA,
    )
    line_observed = line_expected = word_observed = word_expected = None

    for _iteration in range(NUISANCE_ITERATIONS):
        without_line = normalized_matrix(
            prepared,
            prepared.train,
            line_multiplier,
            word_multiplier,
            include_line=False,
        )
        (
            line_multiplier,
            line_observed,
            line_expected,
        ) = fit_multiplier(
            without_line,
            prepared.train,
            train_line,
            states,
            NUISANCE_ALPHA,
        )
        without_word = normalized_matrix(
            prepared,
            prepared.train,
            line_multiplier,
            word_multiplier,
            include_word=False,
        )
        (
            word_multiplier,
            word_observed,
            word_expected,
        ) = fit_multiplier(
            without_word,
            prepared.train,
            train_word,
            states,
            NUISANCE_ALPHA,
        )

    assert line_observed is not None
    assert line_expected is not None
    assert word_observed is not None
    assert word_expected is not None
    original_test = counter.baseline_log2(prepared.test)
    line_only_test = adaptive_matrix_log2(
        prepared.test,
        normalized_matrix(
            prepared,
            prepared.test,
            independent_line,
            np.ones_like(independent_word),
        ),
    )
    word_only_test = adaptive_matrix_log2(
        prepared.test,
        normalized_matrix(
            prepared,
            prepared.test,
            np.ones_like(independent_line),
            independent_word,
        ),
    )
    return StructuralFold(
        raw=prepared,
        train=normalized_matrix(
            prepared,
            prepared.train,
            line_multiplier,
            word_multiplier,
        ),
        validation=normalized_matrix(
            prepared,
            prepared.validation,
            line_multiplier,
            word_multiplier,
        ),
        test=normalized_matrix(
            prepared,
            prepared.test,
            line_multiplier,
            word_multiplier,
        ),
        nuisance={
            "states_per_profile": states,
            "line_multiplier_range": [
                float(line_multiplier.min()),
                float(line_multiplier.max()),
            ],
            "word_multiplier_range": [
                float(word_multiplier.min()),
                float(word_multiplier.max()),
            ],
            "line_observations": int(line_observed.sum()),
            "word_observations": int(word_observed.sum()),
            "independent_line_profile_test_log2_gain": (
                line_only_test - original_test
            ),
            "independent_line_profile_test_gain_bits_per_symbol": (
                (line_only_test - original_test)
                / prepared.test.symbols
            ),
            "independent_word_profile_test_log2_gain": (
                word_only_test - original_test
            ),
            "independent_word_profile_test_gain_bits_per_symbol": (
                (word_only_test - original_test)
                / prepared.test.symbols
            ),
        },
    )


def observed_probabilities(
    fold: StructuralFold,
    split_name: str,
) -> np.ndarray:
    split = getattr(fold.raw, split_name)
    matrix = getattr(fold, split_name)
    return matrix[np.arange(split.symbols), split.symbol]


def baseline_log2(
    fold: StructuralFold,
    split_name: str,
) -> float:
    split = getattr(fold.raw, split_name)
    observed = observed_probabilities(fold, split_name)
    probability = (
        split.page_count + counter.PAGE_BACKOFF * observed
    ) / (split.page_total + counter.PAGE_BACKOFF)
    return float(np.log2(np.maximum(probability, 1e-300)).sum())


def fit_residual_table(
    fold: StructuralFold,
    candidate: counter.Candidate,
    phases: np.ndarray,
) -> ResidualTable:
    split = fold.raw.train
    alphabet_size = len(fold.raw.alphabet)
    flat = phases * alphabet_size + split.symbol
    observed = np.bincount(
        flat, minlength=candidate.modulus * alphabet_size
    ).reshape(candidate.modulus, alphabet_size).astype(np.float64)
    expected = np.zeros_like(observed)
    for phase in range(candidate.modulus):
        indices = phases == phase
        if np.any(indices):
            expected[phase] = fold.train[indices].sum(axis=0)
    totals = expected.sum(axis=1, keepdims=True)
    shares = np.divide(
        expected,
        totals,
        out=np.full_like(expected, 1.0 / alphabet_size),
        where=totals > 0,
    )
    multipliers = (
        observed + candidate.alpha * shares
    ) / np.maximum(expected + candidate.alpha * shares, 1e-12)
    return ResidualTable(observed, expected, multipliers)


def residual_log2(
    fold: StructuralFold,
    split_name: str,
    candidate: counter.Candidate,
    phases: np.ndarray,
    table: ResidualTable,
) -> float:
    split = getattr(fold.raw, split_name)
    matrix = getattr(fold, split_name)
    denominator = np.empty(split.symbols, dtype=np.float64)
    for phase in range(candidate.modulus):
        indices = phases == phase
        if np.any(indices):
            denominator[indices] = np.sum(
                matrix[indices] * table.multipliers[phase],
                axis=1,
            )
    observed = (
        matrix[np.arange(split.symbols), split.symbol]
        * table.multipliers[phases, split.symbol]
        / np.maximum(denominator, 1e-300)
    )
    probability = (
        split.page_count + counter.PAGE_BACKOFF * observed
    ) / (split.page_total + counter.PAGE_BACKOFF)
    return float(np.log2(np.maximum(probability, 1e-300)).sum())


def phases(
    fold: StructuralFold,
    split_name: str,
    modulus: int,
    random_origin_seed: int | None,
) -> np.ndarray:
    split = getattr(fold.raw, split_name)
    rng = None
    if random_origin_seed is not None:
        rng = np.random.default_rng(counter.phase_seed(
            random_origin_seed,
            fold.raw.fold,
            "line_in_page",
            modulus,
            split_name,
        ))
    return counter.phase_values(
        split, "line_in_page", modulus, rng
    )


def score_fold(
    fold: StructuralFold,
    random_origin_seed: int | None = None,
) -> dict[str, object]:
    original_validation = counter.baseline_log2(
        fold.raw.validation
    )
    original_test = counter.baseline_log2(fold.raw.test)
    baseline_validation = baseline_log2(fold, "validation")
    baseline_test = baseline_log2(fold, "test")
    scored = []
    for modulus in MODULI:
        train_phase = phases(
            fold, "train", modulus, random_origin_seed
        )
        validation_phase = phases(
            fold, "validation", modulus, random_origin_seed
        )
        test_phase = phases(
            fold, "test", modulus, random_origin_seed
        )
        for alpha in ALPHAS:
            candidate = counter.Candidate(
                "line_in_page", modulus, alpha
            )
            table = fit_residual_table(
                fold, candidate, train_phase
            )
            validation_log2 = residual_log2(
                fold,
                "validation",
                candidate,
                validation_phase,
                table,
            )
            test_log2 = residual_log2(
                fold,
                "test",
                candidate,
                test_phase,
                table,
            )
            description = (
                modulus
                * len(fold.raw.alphabet)
                * counter.TABLE_ENTRY_DESCRIPTION_BITS
            )
            scored.append((
                validation_log2 - description,
                validation_log2,
                test_log2,
                candidate,
                description,
            ))

    best = max(
        scored,
        key=lambda row: (
            row[0],
            -MODULI.index(row[3].modulus),
            -ALPHAS.index(row[3].alpha),
        ),
    )
    selected = best[0] > baseline_validation
    adjusted, validation_log2, test_log2, candidate, description = best
    if not selected:
        validation_log2 = baseline_validation
        test_log2 = baseline_test
        description = 0.0
    ranking = sorted(scored, key=lambda row: row[0], reverse=True)[:5]
    gain = test_log2 - baseline_test
    return {
        "fold": fold.raw.fold,
        "selected": asdict(candidate) if selected else None,
        "selected_name": candidate.name if selected else "baseline",
        "nuisance": fold.nuisance,
        "validation": {
            "symbols": fold.raw.validation.symbols,
            "original_source_baseline_log2_probability": (
                original_validation
            ),
            "baseline_log2_probability": baseline_validation,
            "absolute_layout_profile_gain_bits_per_symbol": (
                (baseline_validation - original_validation)
                / fold.raw.validation.symbols
            ),
            "candidate_log2_probability": validation_log2,
            "gain_bits_per_symbol": (
                (validation_log2 - baseline_validation)
                / fold.raw.validation.symbols
            ),
            "net_gain_after_table_description_bits": (
                adjusted - baseline_validation if selected else 0.0
            ),
        },
        "test": {
            "symbols": fold.raw.test.symbols,
            "words": fold.raw.test.words,
            "original_source_baseline_log2_probability": original_test,
            "baseline_log2_probability": baseline_test,
            "absolute_layout_profile_gain_bits_per_symbol": (
                (baseline_test - original_test)
                / fold.raw.test.symbols
            ),
            "candidate_log2_probability": test_log2,
            "gain_bits_per_symbol": gain / fold.raw.test.symbols,
            "gain_bits_per_word": gain / fold.raw.test.words,
            "net_gain_after_table_description_bits": (
                gain - description
            ),
        },
        "validation_top_five": [
            {
                "candidate": row[3].name,
                "raw_gain_bits_per_symbol": (
                    (row[1] - baseline_validation)
                    / fold.raw.validation.symbols
                ),
                "net_gain_after_table_description_bits": (
                    row[0] - baseline_validation
                ),
            }
            for row in ranking
        ],
    }


def summarize(
    name: str,
    events: Sequence[counter.PositionedEvent],
    nulls: int,
    progress: bool,
) -> dict[str, object]:
    prepared, assignment, loads = counter.prepare_folds(events)
    folds = [fit_structural_fold(fold) for fold in prepared]
    observed = []
    for fold in folds:
        result = score_fold(fold)
        observed.append(result)
        if progress:
            print(
                f"{name} fold {fold.raw.fold}: "
                f"{result['selected_name']} "
                f"{result['test']['gain_bits_per_symbol']:+.5f} bits/symbol",
                flush=True,
            )
    symbols = sum(int(row["test"]["symbols"]) for row in observed)
    words = sum(int(row["test"]["words"]) for row in observed)
    log_gain = sum(
        float(row["test"]["candidate_log2_probability"])
        - float(row["test"]["baseline_log2_probability"])
        for row in observed
    )
    layout_log_gain = sum(
        float(row["test"]["baseline_log2_probability"])
        - float(
            row["test"][
                "original_source_baseline_log2_probability"
            ]
        )
        for row in observed
    )
    line_profile_log_gain = sum(
        float(
            row["nuisance"][
                "independent_line_profile_test_log2_gain"
            ]
        )
        for row in observed
    )
    word_profile_log_gain = sum(
        float(
            row["nuisance"][
                "independent_word_profile_test_log2_gain"
            ]
        )
        for row in observed
    )

    null_gains = []
    for replicate in range(nulls):
        seed = SEED + 500_000 + replicate
        replicate_gain = 0.0
        for fold in folds:
            result = score_fold(fold, seed)
            replicate_gain += (
                float(result["test"]["candidate_log2_probability"])
                - float(result["test"]["baseline_log2_probability"])
            )
        null_gains.append(replicate_gain / symbols)
        if progress and (
            replicate == 0
            or (replicate + 1) % 8 == 0
            or replicate + 1 == nulls
        ):
            print(
                f"{name} null {replicate + 1}/{nulls}",
                flush=True,
            )

    gain = log_gain / symbols
    null_array = np.asarray(null_gains)
    p_value = (
        (
            1 + int(np.count_nonzero(null_array >= gain))
        ) / (nulls + 1)
        if nulls else None
    )
    selected = Counter(row["selected_name"] for row in observed)
    moduli = Counter(
        int(row["selected"]["modulus"])
        for row in observed
        if row["selected"] is not None
    )
    summary = {
        "absolute_layout_profile_gain_bits_per_symbol": (
            layout_log_gain / symbols
        ),
        "absolute_layout_positive_test_folds": sum(
            float(
                row["test"][
                    "absolute_layout_profile_gain_bits_per_symbol"
                ]
            ) > 0
            for row in observed
        ),
        "independent_line_profile_gain_bits_per_symbol": (
            line_profile_log_gain / symbols
        ),
        "independent_word_profile_gain_bits_per_symbol": (
            word_profile_log_gain / symbols
        ),
        "gain_bits_per_symbol": gain,
        "gain_bits_per_word": log_gain / words,
        "positive_test_folds": sum(
            float(row["test"]["gain_bits_per_symbol"]) > 0
            for row in observed
        ),
        "table_selected_folds": sum(
            row["selected"] is not None for row in observed
        ),
        "selected_candidate_counts": dict(selected),
        "selected_modulus_counts": {
            str(key): value for key, value in moduli.items()
        },
        "phase_origin_null_p_one_sided": p_value,
        "null_mean_gain_bits_per_symbol": (
            float(null_array.mean()) if nulls else None
        ),
        "null_95th_percentile_gain_bits_per_symbol": (
            float(np.quantile(null_array, 0.95)) if nulls else None
        ),
        "null_replicates": nulls,
    }
    return {
        "name": name,
        "fold_assignment": assignment,
        "fold_event_loads": loads,
        "folds": observed,
        "null_gain_bits_per_symbol": null_gains,
        "summary": summary,
    }


def planted_line_events(
    template: Sequence[counter.PositionedEvent],
    model: source.CharacterModel,
    seed: int,
) -> list[counter.PositionedEvent]:
    rng = random.Random(seed)
    words = [
        counter.sample_planted_word(
            model,
            row.event,
            row.line_index % PLANTED_MODULUS,
            rng,
        )
        for row in template
    ]
    rebuilt = source.rebuild_histories(
        [row.event for row in template], words
    )
    return counter.replace_events(template, rebuilt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=counter.CORPUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--nulls", type=int, default=DEFAULT_NULLS)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positioned, audit = counter.load_positioned(args.corpus)
    real = summarize("VOYNICH", positioned, args.nulls, args.progress)
    panels = [real]
    controls: dict[str, object] = {}

    if not args.skip_controls:
        plain = [row.event for row in positioned]
        full_model = source.CharacterModel(
            plain, order=2, cell_mode="register"
        )
        channel = source.EditChannel(plain)
        base_only, _truth = source.generated_events(
            plain,
            full_model,
            channel,
            {"base": 1.0, "copy": 0.0, "edit": 0.0},
            SEED + 20_000,
        )
        base_panel = summarize(
            "SYNTHETIC_BASE_ONLY",
            counter.replace_events(positioned, base_only),
            min(args.nulls, CONTROL_NULLS),
            args.progress,
        )
        planted_panel = summarize(
            "SYNTHETIC_PLANTED_LINE_PAGE_MOD4",
            planted_line_events(
                positioned, full_model, SEED + 40_000
            ),
            args.nulls,
            args.progress,
        )
        panels.extend((base_panel, planted_panel))
        planted = planted_panel["summary"]
        controls = {
            "planted": {
                "scope": "line_in_page",
                "modulus": PLANTED_MODULUS,
                "preference_multiplier": counter.PLANTED_STRENGTH,
            },
            "planted_selection_pass": (
                int(
                    planted["selected_modulus_counts"].get("4", 0)
                ) >= 3
            ),
            "planted_gain_pass": (
                float(planted["gain_bits_per_symbol"]) > 0.005
            ),
            "planted_null_pass": (
                planted["phase_origin_null_p_one_sided"] is not None
                and float(
                    planted["phase_origin_null_p_one_sided"]
                ) <= 0.05
            ),
            "base_only_rejection_pass": (
                int(
                    base_panel["summary"]["table_selected_folds"]
                ) == 0
            ),
        }

    real_summary = real["summary"]
    controls_pass = (
        all(
            bool(value)
            for key, value in controls.items()
            if key.endswith("_pass")
        )
        if controls else None
    )
    primary_pass = (
        float(real_summary["gain_bits_per_symbol"]) > 0
        and int(real_summary["positive_test_folds"]) >= 3
        and int(real_summary["table_selected_folds"]) >= 3
        and real_summary["phase_origin_null_p_one_sided"] is not None
        and float(real_summary["phase_origin_null_p_one_sided"]) <= 0.05
    )
    result = {
        "experiment": "historical_counter_layout_residual_gate",
        "seed": SEED,
        "claim_boundary": (
            "A pass supports a repeating page-line cycle beyond absolute "
            "layout position. It still does not identify plaintext or meaning."
        ),
        "parameters": {
            "position_profiles": {
                "page_line": (
                    "exact indices 0..10; shared 11+ class"
                ),
                "word_in_line": (
                    "exact indices 0..10; shared 11+ class"
                ),
                "alpha": NUISANCE_ALPHA,
                "coordinate_descent_iterations": NUISANCE_ITERATIONS,
            },
            "residual_scope": "line_in_page",
            "moduli": list(MODULI),
            "alphas": list(ALPHAS),
            "minimum_description_bits_per_phase_glyph_entry": (
                counter.TABLE_ENTRY_DESCRIPTION_BITS
            ),
            "null_replicates": args.nulls,
            "null": (
                "full validation selection after independent page-level "
                "rotations of the line-counter origin"
            ),
        },
        "assets": {
            source.asset_name(args.corpus): counter.sha256(args.corpus),
        },
        "corpus_audit": dict(audit),
        "controls": controls,
        "panels": panels,
        "summary": {
            "voynich_absolute_layout_profile_gain_bits_per_symbol": (
                real_summary[
                    "absolute_layout_profile_gain_bits_per_symbol"
                ]
            ),
            "voynich_absolute_layout_positive_test_folds": (
                real_summary["absolute_layout_positive_test_folds"]
            ),
            "voynich_independent_line_profile_gain_bits_per_symbol": (
                real_summary[
                    "independent_line_profile_gain_bits_per_symbol"
                ]
            ),
            "voynich_independent_word_profile_gain_bits_per_symbol": (
                real_summary[
                    "independent_word_profile_gain_bits_per_symbol"
                ]
            ),
            "voynich_residual_gain_bits_per_symbol": real_summary[
                "gain_bits_per_symbol"
            ],
            "voynich_positive_test_folds": real_summary[
                "positive_test_folds"
            ],
            "voynich_table_selected_folds": real_summary[
                "table_selected_folds"
            ],
            "voynich_selected_modulus_counts": real_summary[
                "selected_modulus_counts"
            ],
            "voynich_phase_origin_null_p_one_sided": real_summary[
                "phase_origin_null_p_one_sided"
            ],
            "controls_pass": controls_pass,
            "layout_residual_counter_gate_pass": (
                primary_pass and (controls_pass is not False)
            ),
        },
        "interpretation": (
            "If the first gate passes and this gate fails, its apparent "
            "periodicity is explained by stable absolute line/word position "
            "profiles rather than a repeating counter."
        ),
        "caveats": [
            (
                "The nuisance profiles are deliberately flexible and may "
                "absorb a short counter on pages with fewer than 12 lines. "
                "The planted page-line control tests available residual power."
            ),
            (
                "The word-in-line mod-12 survivor from the first gate is not "
                "separately identifiable from exact word position on typical "
                "line lengths and is therefore treated as closed here."
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
