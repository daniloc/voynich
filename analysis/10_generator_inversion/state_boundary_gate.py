#!/usr/bin/env python3
"""
Test where the causal transition state resets and whether it is page-specific.

The boundary tournament starts from the same train-only register-conditioned
character trigram as production_algorithm_gate.py.  Complete-quire folds select
a reset unit and prior concentration on validation, then score untouched test
quires.  Declared reset candidates are line, paragraph, page side, physical
leaf, repository quire, Currier/section register, Currier, and global.  A
second family layers page-side state over leaf, quire, register, Currier, or
global state.

The state-swap gate then uses a common page-side sample:

* pages must contain at least 64 prefix words plus 16 suffix words;
* validation selects a 16/32/64-word prefix and concentration;
* the scored suffix always starts at word 65, so candidates see identical
  target observations;
* the prefix state is frozen while scoring the suffix;
* every donor has the same repository quire, Currier, and section as its
  target;
* donor conditional counts are rescaled context by context to the target's
  evidence totals, preventing prefix length or context exposure from deciding
  the comparison;
* matched prefix states are permuted only within fold, repository quire,
  Currier, and section.

Controls are an independent character source, a source with planted page-side
state, and ordinary Latin reflowed into the Voynich layout.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import production_algorithm_gate as source


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_state_boundary.json"
)
SEED = 20260724
ALPHAS = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
BOUNDARIES = (
    "line",
    "paragraph",
    "side",
    "leaf",
    "quire",
    "register",
    "currier",
    "global",
)
OUTER_BOUNDARIES = ("leaf", "quire", "register", "currier", "global")
OUTER_ALPHAS = (16.0, 32.0, 64.0, 128.0)
SIDE_ALPHAS = (8.0, 16.0, 32.0, 64.0)
PREFIX_WORDS = (16, 32, 64)
SUFFIX_START = max(PREFIX_WORDS)
MIN_SUFFIX_WORDS = 16
MAX_SUFFIX_WORDS = 64
PERMUTATIONS = 500
PLANTED_ALPHA = 16.0


@dataclass(frozen=True)
class SymbolRow:
    event: source.Event
    symbols: tuple[tuple[tuple[str, ...], str, float], ...]


@dataclass(frozen=True)
class PageState:
    folio: str
    block: str
    currier: str
    section: str
    prefix_rows: tuple[SymbolRow, ...]
    suffix_rows: tuple[SymbolRow, ...]
    counts: Counter
    totals: Counter


def score_name(boundary: str, alpha: float) -> str:
    return f"{boundary}_alpha_{int(alpha)}"


def hierarchical_name(
    outer_boundary: str,
    outer_alpha: float,
    side_alpha: float,
) -> str:
    return (
        f"side_over_{outer_boundary}_outer{int(outer_alpha)}"
        f"_inner{int(side_alpha)}"
    )


def physical_leaf(folio: str) -> str:
    match = re.match(r"(f\d+)", folio)
    return match.group(1) if match else folio


def paragraph_map(path: Path) -> dict[tuple[str, str], str]:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for folio, lines in corpus["folios"].items():
        paragraph = 0
        for row in lines:
            locus = str(row["locus"]).strip()
            if locus.startswith("@"):
                paragraph += 1
            result[(folio, str(row["line"]))] = f"{folio}:{paragraph}"
    return result


def reset_unit(
    event: source.Event,
    boundary: str,
    paragraphs: dict[tuple[str, str], str],
) -> str:
    if boundary == "line":
        return f"{event.folio}:{event.line}"
    if boundary == "paragraph":
        return paragraphs.get(
            (event.folio, event.line),
            f"{event.folio}:0",
        )
    if boundary == "side":
        return event.folio
    if boundary == "leaf":
        return physical_leaf(event.folio)
    if boundary == "quire":
        return event.block
    if boundary == "register":
        return f"{event.currier}:{event.section}"
    if boundary == "currier":
        return event.currier
    if boundary == "global":
        return "<GLOBAL>"
    raise ValueError(boundary)


def symbol_rows(
    model: source.CharacterModel,
    events: Sequence[source.Event],
) -> list[SymbolRow]:
    rows = []
    for event in events:
        history = ["^"] * model.order
        symbols = []
        for symbol in model.symbols(event.word):
            context = tuple(history[-model.order:]) if model.order else ()
            symbols.append((
                context,
                symbol,
                model.probability(event, history, symbol),
            ))
            history.append(symbol)
        rows.append(SymbolRow(event, tuple(symbols)))
    return rows


def static_log_probabilities(rows: Sequence[SymbolRow]) -> list[float]:
    return [
        sum(math.log2(max(probability, 1e-300))
            for _context, _symbol, probability in row.symbols)
        for row in rows
    ]


def boundary_log_probabilities(
    rows: Sequence[SymbolRow],
    boundary: str,
    alpha: float,
    paragraphs: dict[tuple[str, str], str],
) -> list[float]:
    counts: Counter = Counter()
    totals: Counter = Counter()
    result = []
    for row in rows:
        unit = reset_unit(row.event, boundary, paragraphs)
        emitted = []
        total = 0.0
        for context, symbol, base_probability in row.symbols:
            key = (unit, context)
            probability = (
                counts[(key, symbol)] + alpha * base_probability
            ) / (totals[key] + alpha)
            total += math.log2(max(probability, 1e-300))
            emitted.append((key, symbol))
        result.append(total)
        for key, symbol in emitted:
            counts[(key, symbol)] += 1
            totals[key] += 1
    return result


def hierarchical_log_probabilities(
    rows: Sequence[SymbolRow],
    outer_boundary: str,
    outer_alpha: float,
    side_alpha: float,
    paragraphs: dict[tuple[str, str], str],
) -> list[float]:
    outer_counts: Counter = Counter()
    outer_totals: Counter = Counter()
    side_counts: Counter = Counter()
    side_totals: Counter = Counter()
    result = []
    for row in rows:
        outer_unit = reset_unit(
            row.event, outer_boundary, paragraphs
        )
        side_unit = row.event.folio
        emitted = []
        total = 0.0
        for context, symbol, base_probability in row.symbols:
            outer_key = (outer_unit, context)
            outer_probability = (
                outer_counts[(outer_key, symbol)]
                + outer_alpha * base_probability
            ) / (outer_totals[outer_key] + outer_alpha)
            side_key = (side_unit, context)
            probability = (
                side_counts[(side_key, symbol)]
                + side_alpha * outer_probability
            ) / (side_totals[side_key] + side_alpha)
            total += math.log2(max(probability, 1e-300))
            emitted.append((outer_key, side_key, symbol))
        result.append(total)
        for outer_key, side_key, symbol in emitted:
            outer_counts[(outer_key, symbol)] += 1
            outer_totals[outer_key] += 1
            side_counts[(side_key, symbol)] += 1
            side_totals[side_key] += 1
    return result


def bits_per_word(logs: Sequence[float]) -> float:
    return -sum(logs) / len(logs)


def state_counts(rows: Sequence[SymbolRow]) -> tuple[Counter, Counter]:
    counts: Counter = Counter()
    totals: Counter = Counter()
    for row in rows:
        for context, symbol, _base_probability in row.symbols:
            counts[(context, symbol)] += 1
            totals[context] += 1
    return counts, totals


def transplant_counts(
    donor_counts: Counter,
    donor_totals: Counter,
    target_totals: Counter,
) -> tuple[Counter, Counter]:
    result: Counter = Counter()
    result_totals: Counter = Counter()
    contexts = set(target_totals)
    for context in contexts:
        target_total = float(target_totals[context])
        donor_total = float(donor_totals[context])
        if donor_total <= 0 or target_total <= 0:
            continue
        scale = target_total / donor_total
        for key, count in donor_counts.items():
            key_context, symbol = key
            if key_context == context and count:
                result[(context, symbol)] = count * scale
        result_totals[context] = target_total
    return result, result_totals


def score_frozen_state(
    rows: Sequence[SymbolRow],
    counts: Counter,
    totals: Counter,
    alpha: float,
) -> float:
    total = 0.0
    for row in rows:
        for context, symbol, base_probability in row.symbols:
            probability = (
                counts[(context, symbol)] + alpha * base_probability
            ) / (totals[context] + alpha)
            total += math.log2(max(probability, 1e-300))
    return total


def group_page_rows(rows: Sequence[SymbolRow]) -> dict[str, list[SymbolRow]]:
    result: dict[str, list[SymbolRow]] = defaultdict(list)
    for row in rows:
        result[row.event.folio].append(row)
    return result


def page_states(
    rows: Sequence[SymbolRow],
    prefix_words: int,
) -> list[PageState]:
    result = []
    for folio, values in group_page_rows(rows).items():
        if len(values) < SUFFIX_START + MIN_SUFFIX_WORDS:
            continue
        prefix = tuple(values[:prefix_words])
        suffix = tuple(
            values[SUFFIX_START:SUFFIX_START + MAX_SUFFIX_WORDS]
        )
        counts, totals = state_counts(prefix)
        event = values[0].event
        result.append(PageState(
            folio=folio,
            block=event.block,
            currier=event.currier,
            section=event.section,
            prefix_rows=prefix,
            suffix_rows=suffix,
            counts=counts,
            totals=totals,
        ))
    return result


def select_prefix_candidate(
    rows: Sequence[SymbolRow],
) -> tuple[int, float, list[dict[str, object]]]:
    candidates = []
    for prefix_words in PREFIX_WORDS:
        pages = page_states(rows, prefix_words)
        observations = sum(len(page.suffix_rows) for page in pages)
        for alpha in ALPHAS:
            total = sum(
                score_frozen_state(
                    page.suffix_rows,
                    page.counts,
                    page.totals,
                    alpha,
                )
                for page in pages
            )
            candidates.append({
                "prefix_words": prefix_words,
                "alpha": alpha,
                "pages": len(pages),
                "suffix_words": observations,
                "bits_per_suffix_word": -total / observations,
            })
    winner = min(
        candidates,
        key=lambda row: (
            row["bits_per_suffix_word"],
            row["prefix_words"],
            row["alpha"],
        ),
    )
    return (
        int(winner["prefix_words"]),
        float(winner["alpha"]),
        candidates,
    )


def page_swap_fold(
    rows: Sequence[SymbolRow],
    prefix_words: int,
    alpha: float,
    seed: int,
) -> dict[str, object]:
    pages = page_states(rows, prefix_words)
    groups: dict[tuple[str, str, str], list[PageState]] = defaultdict(list)
    for page in pages:
        groups[(page.block, page.currier, page.section)].append(page)
    eligible = [
        page
        for group in groups.values()
        if len(group) >= 2
        for page in group
    ]
    own_total = static_total = donor_total = 0.0
    observations = 0
    target_rows = []
    matrices: dict[tuple[str, str, str], list[list[float]]] = {}
    matrix_pages: dict[tuple[str, str, str], list[PageState]] = {}
    for stratum, group in groups.items():
        if len(group) < 2:
            continue
        matrix = []
        for target in group:
            row_scores = []
            for donor in group:
                counts, totals = transplant_counts(
                    donor.counts,
                    donor.totals,
                    target.totals,
                )
                row_scores.append(score_frozen_state(
                    target.suffix_rows, counts, totals, alpha
                ))
            matrix.append(row_scores)
        matrices[stratum] = matrix
        matrix_pages[stratum] = group

    for stratum, matrix in matrices.items():
        group = matrix_pages[stratum]
        for target_index, target in enumerate(group):
            own = matrix[target_index][target_index]
            donors = [
                value
                for donor_index, value in enumerate(matrix[target_index])
                if donor_index != target_index
            ]
            static = sum(
                static_log_probabilities(target.suffix_rows)
            )
            own_total += own
            static_total += static
            donor_total += sum(donors) / len(donors)
            observations += len(target.suffix_rows)
            descending = sorted(
                range(len(group)),
                key=lambda index: (
                    matrix[target_index][index],
                    group[index].folio,
                ),
                reverse=True,
            )
            own_rank = descending.index(target_index) + 1
            best_index = descending[0]
            leaf_donors = [
                index
                for index, donor in enumerate(group)
                if (
                    index != target_index
                    and physical_leaf(donor.folio)
                    == physical_leaf(target.folio)
                )
            ]
            target_rows.append({
                "folio": target.folio,
                "block": target.block,
                "currier": target.currier,
                "section": target.section,
                "suffix_words": len(target.suffix_rows),
                "matched_states": len(group),
                "own_bits_per_word": -own / len(target.suffix_rows),
                "mean_donor_bits_per_word": (
                    -(sum(donors) / len(donors))
                    / len(target.suffix_rows)
                ),
                "own_rank": own_rank,
                "best_state_folio": group[best_index].folio,
                "opposite_side_bits_per_word": (
                    -matrix[target_index][leaf_donors[0]]
                    / len(target.suffix_rows)
                    if leaf_donors
                    else None
                ),
            })

    rng = random.Random(seed)
    null_totals = []
    for _replicate in range(PERMUTATIONS):
        total = 0.0
        for stratum, matrix in matrices.items():
            size = len(matrix)
            assignment = list(range(size))
            rng.shuffle(assignment)
            total += sum(
                matrix[target_index][assignment[target_index]]
                for target_index in range(size)
            )
        null_totals.append(total)
    p_upper = (
        1 + sum(value >= own_total - 1e-12 for value in null_totals)
    ) / (len(null_totals) + 1)
    null_mean = sum(null_totals) / len(null_totals)
    return {
        "eligible_pages": len(eligible),
        "eligible_strata": len(matrices),
        "suffix_words": observations,
        "own_bits_per_word": -own_total / observations,
        "static_bits_per_word": -static_total / observations,
        "mean_matched_donor_bits_per_word": -donor_total / observations,
        "own_vs_static_gain_bits_per_word": (
            own_total - static_total
        ) / observations,
        "own_vs_mean_donor_gain_bits_per_word": (
            own_total - donor_total
        ) / observations,
        "own_vs_permuted_gain_bits_per_word": (
            own_total - null_mean
        ) / observations,
        "matched_state_permutation": {
            "replicates": PERMUTATIONS,
            "upper_p": p_upper,
            "null_mean_bits_per_word": -null_mean / observations,
        },
        "own_state_rank1_pages": sum(
            row["own_rank"] == 1 for row in target_rows
        ),
        "mean_own_rank_fraction": sum(
            (float(row["own_rank"]) - 1)
            / max(1, int(row["matched_states"]) - 1)
            for row in target_rows
        ) / len(target_rows),
        "targets": target_rows,
    }


def aggregate_swap_folds(
    folds: Sequence[dict[str, object]],
) -> dict[str, object]:
    observations = sum(int(fold["suffix_words"]) for fold in folds)
    weighted_keys = (
        "own_bits_per_word",
        "static_bits_per_word",
        "mean_matched_donor_bits_per_word",
        "own_vs_static_gain_bits_per_word",
        "own_vs_mean_donor_gain_bits_per_word",
        "own_vs_permuted_gain_bits_per_word",
    )
    result = {
        key: sum(
            float(fold[key]) * int(fold["suffix_words"])
            for fold in folds
        ) / observations
        for key in weighted_keys
    }
    result.update({
        "eligible_pages": sum(int(fold["eligible_pages"]) for fold in folds),
        "eligible_strata": sum(
            int(fold["eligible_strata"]) for fold in folds
        ),
        "suffix_words": observations,
        "selected_prefix_words": dict(Counter(
            str(fold["selected_prefix_words"]) for fold in folds
        )),
        "selected_alpha": dict(Counter(
            str(int(float(fold["selected_alpha"]))) for fold in folds
        )),
        "own_state_rank1_pages": sum(
            int(fold["own_state_rank1_pages"]) for fold in folds
        ),
    })
    # Fold permutation scores are independent but only p-values are retained.
    # Fisher's method is unnecessary here; report the conservative maximum.
    result["fold_permutation_upper_p_values"] = [
        fold["matched_state_permutation"]["upper_p"] for fold in folds
    ]
    result["all_folds_permutation_pass_0_05"] = all(
        fold["matched_state_permutation"]["upper_p"] <= 0.05
        for fold in folds
    )
    return result


def run_panel(
    name: str,
    events: Sequence[source.Event],
    paragraphs: dict[tuple[str, str], str],
    progress: bool,
) -> dict[str, object]:
    assignment, loads = source.block_folds(events)
    fold_results = []
    pooled: dict[str, Counter] = defaultdict(Counter)
    swap_folds = []
    for fold in range(source.N_FOLDS):
        train, validation, test = source.split_events(
            events, assignment, fold
        )
        model = source.CharacterModel(train, 2, "register")
        validation_rows = symbol_rows(model, validation)
        test_rows = symbol_rows(model, test)
        validation_logs = {
            "static": static_log_probabilities(validation_rows)
        }
        test_logs = {"static": static_log_probabilities(test_rows)}
        for boundary in BOUNDARIES:
            for alpha in ALPHAS:
                candidate = score_name(boundary, alpha)
                validation_logs[candidate] = boundary_log_probabilities(
                    validation_rows, boundary, alpha, paragraphs
                )
                test_logs[candidate] = boundary_log_probabilities(
                    test_rows, boundary, alpha, paragraphs
                )
        for outer_boundary in OUTER_BOUNDARIES:
            for outer_alpha in OUTER_ALPHAS:
                for side_alpha in SIDE_ALPHAS:
                    candidate = hierarchical_name(
                        outer_boundary, outer_alpha, side_alpha
                    )
                    validation_logs[
                        candidate
                    ] = hierarchical_log_probabilities(
                        validation_rows,
                        outer_boundary,
                        outer_alpha,
                        side_alpha,
                        paragraphs,
                    )
                    test_logs[
                        candidate
                    ] = hierarchical_log_probabilities(
                        test_rows,
                        outer_boundary,
                        outer_alpha,
                        side_alpha,
                        paragraphs,
                    )
        validation_scores = {
            candidate: bits_per_word(logs)
            for candidate, logs in validation_logs.items()
        }
        winner = min(
            validation_scores,
            key=lambda candidate: (
                validation_scores[candidate], candidate
            ),
        )
        candidate_scores = []
        for candidate, logs in test_logs.items():
            score = bits_per_word(logs)
            pooled[candidate]["log2_probability"] += sum(logs)
            pooled[candidate]["events"] += len(test)
            candidate_scores.append({
                "candidate": candidate,
                "validation_bits_per_word": validation_scores[candidate],
                "test_bits_per_word": score,
            })

        prefix_words, alpha, prefix_candidates = select_prefix_candidate(
            validation_rows
        )
        swap = page_swap_fold(
            test_rows,
            prefix_words,
            alpha,
            SEED + 10_000 * (fold + 1) + sum(map(ord, name)),
        )
        swap.update({
            "fold": fold,
            "selected_prefix_words": prefix_words,
            "selected_alpha": alpha,
            "validation_candidates": prefix_candidates,
        })
        swap_folds.append(swap)
        fold_results.append({
            "fold": fold,
            "train_blocks": sorted({event.block for event in train}),
            "validation_blocks": sorted({
                event.block for event in validation
            }),
            "test_blocks": sorted({event.block for event in test}),
            "selected_boundary_candidate": winner,
            "selected_test_bits_per_word": bits_per_word(
                test_logs[winner]
            ),
            "candidate_scores": candidate_scores,
        })
        if progress:
            print(
                f"{name} fold={fold} boundary={winner} "
                f"swap=prefix{prefix_words}/alpha{int(alpha)} "
                f"gain={swap['own_vs_mean_donor_gain_bits_per_word']:+.4f} "
                f"p={swap['matched_state_permutation']['upper_p']:.4f}",
                flush=True,
            )

    candidate_summary = {
        candidate: {
            "test_events": int(values["events"]),
            "pooled_test_bits_per_word": (
                -values["log2_probability"] / values["events"]
            ),
        }
        for candidate, values in pooled.items()
    }
    best_by_boundary = {}
    for boundary in BOUNDARIES:
        candidates = [
            score_name(boundary, alpha) for alpha in ALPHAS
        ]
        best = min(
            candidates,
            key=lambda candidate: candidate_summary[candidate][
                "pooled_test_bits_per_word"
            ],
        )
        best_by_boundary[boundary] = {
            "descriptive_best_candidate": best,
            **candidate_summary[best],
        }
    hierarchical_candidates = [
        hierarchical_name(outer, outer_alpha, side_alpha)
        for outer in OUTER_BOUNDARIES
        for outer_alpha in OUTER_ALPHAS
        for side_alpha in SIDE_ALPHAS
    ]
    best_hierarchical = min(
        hierarchical_candidates,
        key=lambda candidate: candidate_summary[candidate][
            "pooled_test_bits_per_word"
        ],
    )
    best_single = min(
        (
            score_name(boundary, alpha)
            for boundary in BOUNDARIES
            for alpha in ALPHAS
        ),
        key=lambda candidate: candidate_summary[candidate][
            "pooled_test_bits_per_word"
        ],
    )
    return {
        "name": name,
        "fold_assignment": assignment,
        "fold_event_loads": loads,
        "folds": fold_results,
        "candidate_summary": candidate_summary,
        "best_pooled_candidate_by_boundary": best_by_boundary,
        "state_swap_folds": swap_folds,
        "summary": {
            "selected_boundary_candidates": dict(Counter(
                fold["selected_boundary_candidate"]
                for fold in fold_results
            )),
            "selected_boundary_bits_per_word": (
                sum(
                    float(fold["selected_test_bits_per_word"])
                    * sum(
                        1
                        for event in events
                        if assignment[event.block] == int(fold["fold"])
                    )
                    for fold in fold_results
                )
                / len(events)
            ),
            "best_pooled_candidate_by_boundary": best_by_boundary,
            "descriptive_best_single_boundary": {
                "candidate": best_single,
                **candidate_summary[best_single],
            },
            "descriptive_best_hierarchical": {
                "candidate": best_hierarchical,
                **candidate_summary[best_hierarchical],
            },
            "hierarchical_gain_over_best_single_bits_per_word": (
                candidate_summary[best_single][
                    "pooled_test_bits_per_word"
                ]
                - candidate_summary[best_hierarchical][
                    "pooled_test_bits_per_word"
                ]
            ),
            "state_swap": aggregate_swap_folds(swap_folds),
        },
    }


def sample_page_state_word(
    model: source.CharacterModel,
    event: source.Event,
    rng: random.Random,
    counts: Counter,
    totals: Counter,
    alpha: float,
) -> tuple[str, list[tuple[tuple[str, ...], str]]]:
    unit = event.folio
    for _attempt in range(50):
        history = ["^"] * model.order
        word = []
        updates = []
        for _ in range(source.MAX_WORD_LENGTH):
            context = tuple(history[-model.order:]) if model.order else ()
            key = (unit, context)
            probabilities = [
                (
                    counts[(key, symbol)]
                    + alpha * model.probability(event, history, symbol)
                )
                / (totals[key] + alpha)
                for symbol in model.alphabet
            ]
            symbol = rng.choices(
                model.alphabet, weights=probabilities, k=1
            )[0]
            updates.append((context, symbol))
            if symbol == model.eos:
                if len(word) >= 2:
                    return "".join(word), updates
                break
            if symbol == model.unknown:
                symbol = rng.choice(tuple(sorted(model.alphabet_set)))
                updates[-1] = (context, symbol)
            word.append(symbol)
            history.append(symbol)
    return "ol", [(("^", "^"), "o"), (("^", "o"), "l"), (("o", "l"), "$")]


def generated_page_state_events(
    template: Sequence[source.Event],
    model: source.CharacterModel,
    alpha: float,
    seed: int,
) -> list[source.Event]:
    rng = random.Random(seed)
    counts: Counter = Counter()
    totals: Counter = Counter()
    words = []
    for event in template:
        word, updates = sample_page_state_word(
            model, event, rng, counts, totals, alpha
        )
        words.append(word)
        unit = event.folio
        for context, symbol in updates:
            key = (unit, context)
            counts[(key, symbol)] += 1
            totals[key] += 1
    return source.rebuild_histories(template, words)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=source.CORPUS)
    parser.add_argument("--latin", type=Path, default=source.LATIN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events, audit = source.load_events(args.corpus)
    paragraphs = paragraph_map(args.corpus)
    panels = [run_panel("VOYNICH", events, paragraphs, args.progress)]
    if not args.skip_controls:
        base = source.CharacterModel(events, 2, "register")
        channel = source.EditChannel(events)
        independent, _truth = source.generated_events(
            events,
            base,
            channel,
            {"base": 1.0, "copy": 0.0, "edit": 0.0},
            SEED + 40_000,
        )
        planted = generated_page_state_events(
            events, base, PLANTED_ALPHA, SEED + 50_000
        )
        panels.extend([
            run_panel(
                "SYNTHETIC_INDEPENDENT",
                independent,
                paragraphs,
                args.progress,
            ),
            run_panel(
                "SYNTHETIC_PAGE_STATE",
                planted,
                paragraphs,
                args.progress,
            ),
            run_panel(
                "LATIN_REFLOW",
                source.latin_events(events, args.latin),
                paragraphs,
                args.progress,
            ),
        ])
    by_name = {panel["name"]: panel for panel in panels}
    controls = {}
    if not args.skip_controls:
        independent_swap = by_name["SYNTHETIC_INDEPENDENT"]["summary"][
            "state_swap"
        ]
        planted_panel = by_name["SYNTHETIC_PAGE_STATE"]
        planted_swap = planted_panel["summary"]["state_swap"]
        controls = {
            "independent_swap_gain_below_0_05_pass": (
                independent_swap[
                    "own_vs_permuted_gain_bits_per_word"
                ]
                < 0.05
            ),
            "planted_side_selected_at_least_3_folds_pass": (
                sum(
                    count
                    for candidate, count in planted_panel["summary"][
                        "selected_boundary_candidates"
                    ].items()
                    if candidate.startswith("side_")
                )
                >= 3
            ),
            "planted_swap_gain_above_0_10_pass": (
                planted_swap[
                    "own_vs_permuted_gain_bits_per_word"
                ]
                > 0.10
            ),
            "planted_swap_all_folds_significant_pass": (
                planted_swap["all_folds_permutation_pass_0_05"]
            ),
        }
    result = {
        "experiment": "state_reset_boundary_and_page_prefix_swap",
        "seed": SEED,
        "parameters": {
            "boundaries": list(BOUNDARIES),
            "alphas": list(ALPHAS),
            "hierarchical_outer_boundaries": list(OUTER_BOUNDARIES),
            "hierarchical_outer_alphas": list(OUTER_ALPHAS),
            "hierarchical_side_alphas": list(SIDE_ALPHAS),
            "prefix_words": list(PREFIX_WORDS),
            "suffix_start_word": SUFFIX_START + 1,
            "minimum_suffix_words": MIN_SUFFIX_WORDS,
            "maximum_scored_suffix_words": MAX_SUFFIX_WORDS,
            "matched_state_permutations": PERMUTATIONS,
            "planted_page_state_alpha": PLANTED_ALPHA,
            "split": (
                "test=fold f; validation=fold f+1; fit=other two folds; "
                "complete repository quire blocks"
            ),
        },
        "claim_boundary": (
            "A page-side reset and authentic prefix advantage establish a "
            "reusable page-specific text state beyond repository quire, "
            "Currier, and section. They do not determine whether that state "
            "represents topic, plaintext, or a content-free key."
        ),
        "metadata_boundary": (
            "Repository metadata identifies page sides, physical leaves, and "
            "repository quire codes. It does not safely identify true "
            "codicological bifolia, so no bifolio candidate is inferred."
        ),
        "corpus_audit": dict(audit),
        "assets": {
            source.asset_name(args.corpus): source.sha256(args.corpus),
            source.asset_name(args.latin): source.sha256(args.latin),
        },
        "controls": controls,
        "panels": panels,
        "summary": {
            "voynich": panels[0]["summary"],
            "controls_pass": all(controls.values()) if controls else None,
        },
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
