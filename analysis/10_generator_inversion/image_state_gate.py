#!/usr/bin/env python3
"""
Test whether page illustrations predict the page-specific transition state.

The source model and the 64-word frozen page state are inherited from
state_boundary_gate.py.  For each complete-quire fold, fit pages determine
visual feature scaling, validation pages select a guarded feature family,
neighbor count, and state concentration, and untouched test pages provide the
reported result.

An image can select donors only among other pages with the same repository
quire, Currier variety, and section.  The primary null relabels complete image
feature vectors within those exact strata.  Ordinary small text components are
excluded from every primary feature family and are evaluated separately as a
leakage diagnostic.

This is a cross-modal prediction gate.  Success would link illustrations to
the already established textual page state; it would not identify plaintext
or the semantic meaning of an illustration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import production_algorithm_gate as source
import state_boundary_gate as state


ROOT = Path(__file__).resolve().parents[2]
VISUAL = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_page_illustration_features.json"
)
OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_image_state_gate.json"
)
SEED = 20260724
PREFIX_WORDS = 64
ALPHAS = (8.0, 16.0, 32.0, 64.0)
NEIGHBORS = (1, 3, 5)
PERMUTATIONS = 500
PRIMARY_FAMILIES = (
    "global_guarded",
    "silhouette_guarded",
    "pigment_guarded",
    "combined_guarded",
)
LEAKAGE_FAMILY = "small_component_grid"


@dataclass(frozen=True)
class Scaler:
    mean: np.ndarray
    scale: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visual_records(path: Path) -> tuple[dict[str, dict], dict]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    duplicates = Counter()
    for record in artifact["records"]:
        folio = str(record["folio"])
        duplicates[folio] += 1
        # Composite foldout canvases do not map safely to transcription sides.
        # Exact labels are retained; repeated labels use the first view.
        records.setdefault(folio, record)
    audit = {
        "artifact_records": len(artifact["records"]),
        "unique_exact_folios": len(records),
        "duplicate_exact_folios": {
            folio: count
            for folio, count in duplicates.items()
            if count > 1
        },
        "eligible_unique_folios": sum(
            bool(record["eligible_illustration"])
            for record in records.values()
        ),
    }
    return records, audit


def feature_vector(record: dict, family: str) -> np.ndarray:
    if family == LEAKAGE_FAMILY:
        values = record["leakage_diagnostic"][family]
    else:
        values = record["features"][family]
    return np.asarray(values, dtype=np.float64)


def fit_scaler(
    records: dict[str, dict],
    folios: Sequence[str],
    family: str,
) -> Scaler:
    values = np.stack([
        feature_vector(records[folio], family)
        for folio in folios
        if folio in records
    ])
    mean = values.mean(axis=0)
    deviation = values.std(axis=0)
    scale = np.maximum(
        deviation,
        np.maximum(np.abs(mean) * 0.1, 1e-4),
    )
    return Scaler(mean, scale)


def transform(
    records: dict[str, dict],
    folio: str,
    family: str,
    scaler: Scaler,
) -> np.ndarray:
    return (feature_vector(records[folio], family) - scaler.mean) / scaler.scale


def matched_groups(
    pages: Sequence[state.PageState],
    records: dict[str, dict],
) -> dict[tuple[str, str, str], list[state.PageState]]:
    groups: dict[tuple[str, str, str], list[state.PageState]] = defaultdict(list)
    for page in pages:
        record = records.get(page.folio)
        if record is None or not record["eligible_illustration"]:
            continue
        groups[(page.block, page.currier, page.section)].append(page)
    return {
        stratum: sorted(group, key=lambda page: source.folio_key(page.folio))
        for stratum, group in groups.items()
        if len(group) >= 2
    }


def donor_score_matrices(
    groups: dict[tuple[str, str, str], list[state.PageState]],
    alpha: float,
) -> dict[tuple[str, str, str], np.ndarray]:
    matrices = {}
    for stratum, group in groups.items():
        matrix = np.full((len(group), len(group)), np.nan)
        for target_index, target in enumerate(group):
            for donor_index, donor in enumerate(group):
                if donor_index == target_index:
                    continue
                counts, totals = state.transplant_counts(
                    donor.counts,
                    donor.totals,
                    target.totals,
                )
                matrix[target_index, donor_index] = state.score_frozen_state(
                    target.suffix_rows,
                    counts,
                    totals,
                    alpha,
                )
        matrices[stratum] = matrix
    return matrices


def state_distance(
    target: state.PageState,
    donor: state.PageState,
) -> float:
    weighted = total_weight = 0.0
    target_symbols: dict[tuple[str, ...], set[str]] = defaultdict(set)
    donor_symbols: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for (context, symbol), count in target.counts.items():
        if count:
            target_symbols[context].add(symbol)
    for (context, symbol), count in donor.counts.items():
        if count:
            donor_symbols[context].add(symbol)
    for context, target_total in target.totals.items():
        donor_total = donor.totals[context]
        if target_total <= 0 or donor_total <= 0:
            weighted += float(target_total)
            total_weight += float(target_total)
            continue
        symbols = target_symbols[context] | donor_symbols[context]
        distance = sum(
            abs(
                target.counts[(context, symbol)] / target_total
                - donor.counts[(context, symbol)] / donor_total
            )
            for symbol in symbols
        ) / 2.0
        weighted += float(target_total) * distance
        total_weight += float(target_total)
    return weighted / max(total_weight, 1.0)


def feature_distances(vectors: np.ndarray) -> np.ndarray:
    difference = vectors[:, None, :] - vectors[None, :, :]
    distances = np.sqrt(np.mean(difference * difference, axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances


def selected_total(
    matrices: dict[tuple[str, str, str], np.ndarray],
    distances: dict[tuple[str, str, str], np.ndarray],
    neighbors: int,
) -> float:
    total = 0.0
    for stratum, matrix in matrices.items():
        distance = distances[stratum]
        use = min(neighbors, len(matrix) - 1)
        for target_index in range(len(matrix)):
            chosen = np.argsort(
                distance[target_index], kind="stable"
            )[:use]
            total += float(np.mean(matrix[target_index, chosen]))
    return total


def evaluate(
    pages: Sequence[state.PageState],
    records: dict[str, dict],
    family: str,
    scaler: Scaler,
    neighbors: int,
    alpha: float,
    permutations: int = 0,
    seed: int = SEED,
    keep_targets: bool = False,
) -> dict:
    groups = matched_groups(pages, records)
    matrices = donor_score_matrices(groups, alpha)
    distances = {}
    vectors = {}
    for stratum, group in groups.items():
        values = np.stack([
            transform(records, page.folio, family, scaler)
            for page in group
        ])
        vectors[stratum] = values
        distances[stratum] = feature_distances(values)

    visual_total = selected_total(matrices, distances, neighbors)
    mean_donor_total = own_total = static_total = 0.0
    oracle_total = ceiling_total = 0.0
    observations = 0
    target_rows = []
    for stratum, group in groups.items():
        matrix = matrices[stratum]
        distance = distances[stratum]
        use = min(neighbors, len(group) - 1)
        for target_index, target in enumerate(group):
            donor_indices = [
                index for index in range(len(group))
                if index != target_index
            ]
            chosen = np.argsort(
                distance[target_index], kind="stable"
            )[:use]
            oracle_index = min(
                donor_indices,
                key=lambda index: (
                    state_distance(target, group[index]),
                    group[index].folio,
                ),
            )
            own = state.score_frozen_state(
                target.suffix_rows,
                target.counts,
                target.totals,
                alpha,
            )
            static = sum(
                state.static_log_probabilities(target.suffix_rows)
            )
            mean_donor = float(np.mean(matrix[target_index, donor_indices]))
            oracle = float(matrix[target_index, oracle_index])
            ceiling = float(np.max(matrix[target_index, donor_indices]))
            mean_donor_total += mean_donor
            own_total += own
            static_total += static
            oracle_total += oracle
            ceiling_total += ceiling
            observations += len(target.suffix_rows)
            if keep_targets:
                target_rows.append({
                    "folio": target.folio,
                    "block": target.block,
                    "currier": target.currier,
                    "section": target.section,
                    "suffix_words": len(target.suffix_rows),
                    "matched_states": len(group),
                    "selected_image_donors": [
                        group[int(index)].folio for index in chosen
                    ],
                    "image_selected_bits_per_word": (
                        -float(np.mean(matrix[target_index, chosen]))
                        / len(target.suffix_rows)
                    ),
                    "mean_donor_bits_per_word": (
                        -mean_donor / len(target.suffix_rows)
                    ),
                    "prefix_state_oracle_folio": group[oracle_index].folio,
                })

    null_totals = []
    if permutations:
        rng = random.Random(seed)
        for _replicate in range(permutations):
            permuted_distances = {}
            for stratum, values in vectors.items():
                assignment = list(range(len(values)))
                rng.shuffle(assignment)
                permuted_distances[stratum] = feature_distances(
                    values[assignment]
                )
            null_totals.append(
                selected_total(matrices, permuted_distances, neighbors)
            )
    null_mean = (
        float(np.mean(null_totals)) if null_totals else float("nan")
    )
    upper_p = (
        (
            1
            + sum(value >= visual_total - 1e-12 for value in null_totals)
        )
        / (len(null_totals) + 1)
        if null_totals
        else None
    )
    result = {
        "eligible_pages": sum(len(group) for group in groups.values()),
        "eligible_strata": len(groups),
        "suffix_words": observations,
        "image_selected_bits_per_word": -visual_total / observations,
        "mean_matched_donor_bits_per_word": (
            -mean_donor_total / observations
        ),
        "own_prefix_bits_per_word": -own_total / observations,
        "static_bits_per_word": -static_total / observations,
        "prefix_state_oracle_bits_per_word": -oracle_total / observations,
        "posthoc_best_donor_ceiling_bits_per_word": (
            -ceiling_total / observations
        ),
        "image_gain_over_mean_donor_bits_per_word": (
            visual_total - mean_donor_total
        ) / observations,
        "own_gain_over_mean_donor_bits_per_word": (
            own_total - mean_donor_total
        ) / observations,
        "prefix_state_oracle_gain_over_mean_donor_bits_per_word": (
            oracle_total - mean_donor_total
        ) / observations,
        "image_relabeling_null": {
            "replicates": len(null_totals),
            "upper_p": upper_p,
            "null_mean_bits_per_word": (
                -null_mean / observations if null_totals else None
            ),
            "observed_gain_over_null_bits_per_word": (
                (visual_total - null_mean) / observations
                if null_totals
                else None
            ),
        },
        "targets": target_rows,
    }
    result["_raw"] = {
        "visual_total": visual_total,
        "mean_donor_total": mean_donor_total,
        "null_totals": null_totals,
    }
    return result


def candidate_table(
    pages: Sequence[state.PageState],
    records: dict[str, dict],
    scalers: dict[str, Scaler],
    families: Sequence[str],
) -> list[dict]:
    candidates = []
    for family in families:
        for neighbors in NEIGHBORS:
            for alpha in ALPHAS:
                result = evaluate(
                    pages,
                    records,
                    family,
                    scalers[family],
                    neighbors,
                    alpha,
                )
                candidates.append({
                    "family": family,
                    "neighbors": neighbors,
                    "alpha": alpha,
                    "eligible_pages": result["eligible_pages"],
                    "suffix_words": result["suffix_words"],
                    "image_selected_bits_per_word": (
                        result["image_selected_bits_per_word"]
                    ),
                    "mean_matched_donor_bits_per_word": (
                        result["mean_matched_donor_bits_per_word"]
                    ),
                    "image_gain_over_mean_donor_bits_per_word": (
                        result[
                            "image_gain_over_mean_donor_bits_per_word"
                        ]
                    ),
                })
    return candidates


def winner(candidates: Sequence[dict]) -> dict:
    return min(
        candidates,
        key=lambda row: (
            row["image_selected_bits_per_word"],
            row["family"],
            row["neighbors"],
            row["alpha"],
        ),
    )


def serializable(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "_raw"}


def aggregate(folds: Sequence[dict]) -> dict:
    observations = sum(int(fold["suffix_words"]) for fold in folds)
    weighted = (
        "image_selected_bits_per_word",
        "mean_matched_donor_bits_per_word",
        "own_prefix_bits_per_word",
        "static_bits_per_word",
        "prefix_state_oracle_bits_per_word",
        "posthoc_best_donor_ceiling_bits_per_word",
        "image_gain_over_mean_donor_bits_per_word",
        "own_gain_over_mean_donor_bits_per_word",
        "prefix_state_oracle_gain_over_mean_donor_bits_per_word",
    )
    result = {
        key: sum(
            float(fold[key]) * int(fold["suffix_words"])
            for fold in folds
        ) / observations
        for key in weighted
    }
    result.update({
        "eligible_pages": sum(int(fold["eligible_pages"]) for fold in folds),
        "eligible_strata": sum(
            int(fold["eligible_strata"]) for fold in folds
        ),
        "suffix_words": observations,
        "selected_families": dict(Counter(
            str(fold["selected_family"]) for fold in folds
        )),
        "selected_neighbors": dict(Counter(
            str(fold["selected_neighbors"]) for fold in folds
        )),
        "selected_alphas": dict(Counter(
            str(int(float(fold["selected_alpha"]))) for fold in folds
        )),
        "fold_relabeling_upper_p_values": [
            fold["image_relabeling_null"]["upper_p"] for fold in folds
        ],
    })
    pooled_null = [
        sum(
            fold["_raw"]["null_totals"][replicate]
            for fold in folds
        )
        for replicate in range(PERMUTATIONS)
    ]
    observed = sum(fold["_raw"]["visual_total"] for fold in folds)
    null_mean = float(np.mean(pooled_null))
    result["pooled_image_relabeling_null"] = {
        "replicates": PERMUTATIONS,
        "upper_p": (
            1 + sum(value >= observed - 1e-12 for value in pooled_null)
        ) / (PERMUTATIONS + 1),
        "null_mean_bits_per_word": -null_mean / observations,
        "observed_gain_over_null_bits_per_word": (
            observed - null_mean
        ) / observations,
    }
    return result


def run(
    corpus: Path,
    visual: Path,
    output: Path,
    progress: bool,
) -> dict:
    events, corpus_audit = source.load_events(corpus)
    records, visual_audit = visual_records(visual)
    assignment, loads = source.block_folds(events)
    fold_results = []
    leakage_results = []
    event_folios = {event.folio for event in events}
    for fold in range(source.N_FOLDS):
        train, validation, test = source.split_events(
            events, assignment, fold
        )
        model = source.CharacterModel(train, 2, "register")
        validation_pages = state.page_states(
            state.symbol_rows(model, validation), PREFIX_WORDS
        )
        test_pages = state.page_states(
            state.symbol_rows(model, test), PREFIX_WORDS
        )
        train_folios = sorted(
            {
                event.folio for event in train
                if event.folio in records
            },
            key=source.folio_key,
        )
        scalers = {
            family: fit_scaler(records, train_folios, family)
            for family in (*PRIMARY_FAMILIES, LEAKAGE_FAMILY)
        }
        validation_candidates = candidate_table(
            validation_pages,
            records,
            scalers,
            PRIMARY_FAMILIES,
        )
        selected = winner(validation_candidates)
        test_result = evaluate(
            test_pages,
            records,
            str(selected["family"]),
            scalers[str(selected["family"])],
            int(selected["neighbors"]),
            float(selected["alpha"]),
            permutations=PERMUTATIONS,
            seed=SEED + 10_000 * (fold + 1),
            keep_targets=True,
        )
        test_result.update({
            "fold": fold,
            "train_blocks": sorted({event.block for event in train}),
            "validation_blocks": sorted({
                event.block for event in validation
            }),
            "test_blocks": sorted({event.block for event in test}),
            "selected_family": selected["family"],
            "selected_neighbors": selected["neighbors"],
            "selected_alpha": selected["alpha"],
            "validation_winner": selected,
            "validation_candidates": validation_candidates,
        })
        fold_results.append(test_result)

        leakage_candidates = candidate_table(
            validation_pages,
            records,
            scalers,
            (LEAKAGE_FAMILY,),
        )
        leakage_selected = winner(leakage_candidates)
        leakage_test = evaluate(
            test_pages,
            records,
            LEAKAGE_FAMILY,
            scalers[LEAKAGE_FAMILY],
            int(leakage_selected["neighbors"]),
            float(leakage_selected["alpha"]),
            permutations=PERMUTATIONS,
            seed=SEED + 50_000 + 10_000 * (fold + 1),
        )
        leakage_test.update({
            "fold": fold,
            "selected_family": LEAKAGE_FAMILY,
            "selected_neighbors": leakage_selected["neighbors"],
            "selected_alpha": leakage_selected["alpha"],
            "validation_winner": leakage_selected,
        })
        leakage_results.append(leakage_test)
        if progress:
            print(
                f"fold={fold} family={selected['family']} "
                f"k={selected['neighbors']} alpha={int(selected['alpha'])} "
                f"pages={test_result['eligible_pages']} "
                f"gain={test_result['image_gain_over_mean_donor_bits_per_word']:+.4f} "
                f"p={test_result['image_relabeling_null']['upper_p']:.4f}",
                flush=True,
            )

    primary_summary = aggregate(fold_results)
    leakage_summary = aggregate(leakage_results)
    prefix_oracle_pass = (
        primary_summary[
            "prefix_state_oracle_gain_over_mean_donor_bits_per_word"
        ] > 0
    )
    primary_pass = (
        primary_summary[
            "image_gain_over_mean_donor_bits_per_word"
        ] > 0
        and primary_summary["pooled_image_relabeling_null"]["upper_p"]
        <= 0.05
    )
    result = {
        "experiment": "guarded_illustration_to_page_state_gate",
        "seed": SEED,
        "parameters": {
            "prefix_words": PREFIX_WORDS,
            "suffix_start_word": state.SUFFIX_START + 1,
            "maximum_scored_suffix_words": state.MAX_SUFFIX_WORDS,
            "minimum_suffix_words": state.MIN_SUFFIX_WORDS,
            "primary_feature_families": list(PRIMARY_FAMILIES),
            "excluded_text_layout_diagnostic": LEAKAGE_FAMILY,
            "neighbors": list(NEIGHBORS),
            "alphas": list(ALPHAS),
            "image_relabeling_permutations": PERMUTATIONS,
            "donor_match": "exact repository quire x Currier x section",
            "split": (
                "test=fold f; validation=fold f+1; fit=other two folds; "
                "complete repository quire blocks"
            ),
            "selection_objective": (
                "minimum validation image-selected bits per suffix word"
            ),
        },
        "assets": {
            str(corpus.relative_to(ROOT)): sha256(corpus),
            str(visual.relative_to(ROOT)): sha256(visual),
        },
        "corpus_audit": dict(corpus_audit),
        "visual_audit": {
            **visual_audit,
            "event_folios": len(event_folios),
            "exact_event_visual_overlap": len(event_folios & set(records)),
        },
        "fold_assignment": assignment,
        "fold_event_loads": loads,
        "folds": [serializable(fold) for fold in fold_results],
        "leakage_diagnostic_folds": [
            serializable(fold) for fold in leakage_results
        ],
        "summary": {
            "primary_guarded_features": primary_summary,
            "excluded_text_layout_diagnostic": leakage_summary,
            "controls": {
                "prefix_state_oracle_positive": prefix_oracle_pass,
                "prefix_state_oracle_description": (
                    "nearest other donor chosen from target's own 64-word "
                    "prefix state; positive gain establishes donor-pool power"
                ),
            },
            "primary_gate_pass": primary_pass,
        },
        "claim_boundary": (
            "A pass links guarded page-image structure to the frozen textual "
            "page state after exact metadata matching. It does not decode "
            "Voynichese, name an illustration, or establish semantics. A "
            "failure rejects only these coarse silhouette/pigment features "
            "and this nearest-donor rule."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=source.CORPUS)
    parser.add_argument("--visual", type=Path, default=VISUAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        args.corpus,
        args.visual,
        args.output,
        progress=not args.quiet,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
