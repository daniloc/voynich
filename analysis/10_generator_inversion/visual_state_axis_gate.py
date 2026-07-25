#!/usr/bin/env python3
"""
Predict low-dimensional page-state axes from guarded DINOv2 embeddings.

For each complete-quire fold:

1. fit the register-conditioned word-reset source on fit blocks;
2. learn PCA axes of 64-word page-prefix transition counts on fit pages;
3. fit kernel-ridge maps from frozen, text-masked visual embeddings to those
   text-state axes using fit pages only;
4. select visual representation, ridge, axis count, donor count, and state
   concentration on validation pages;
5. use the frozen map to select textual state donors for untouched test-page
   suffixes.

Every donor and every image-relabeling null is restricted to exact repository
quire x Currier x section strata.  Test prefix coordinates are used only for
declared axis/full-state oracles and reconstruction diagnostics, never for the
primary image prediction.
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

import image_state_gate as first_gate
import production_algorithm_gate as source
import state_boundary_gate as state


ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_guarded_dinov2_embeddings.json"
)
HANDCRAFTED = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_page_illustration_features.json"
)
OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_visual_state_axis_gate.json"
)
SEED = 20260724
PREFIX_WORDS = 64
MAX_AXES = 16
AXIS_COUNTS = (2, 4, 8, 16)
RIDGES = (0.01, 0.1, 1.0, 10.0)
NEIGHBORS = (1, 3, 5)
ALPHAS = (16.0, 32.0, 64.0)
PERMUTATIONS = 500
VISUAL_FAMILIES = (
    "dino_full_rgb_cls",
    "dino_tight_rgb_cls",
    "dino_full_silhouette_cls",
    "dino_tight_silhouette_cls",
    "dino_full_rgb_both",
    "dino_tight_rgb_both",
    "dino_full_tight_rgb_cls",
    "handcrafted_combined",
)
LEAKAGE_FAMILY = "small_component_grid"


@dataclass(frozen=True)
class AxisModel:
    features: tuple[tuple[tuple[str, ...], str], ...]
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    axis_scale: np.ndarray
    singular_values: np.ndarray
    total_variance: float

    def coordinates(
        self,
        pages: Sequence[state.PageState],
    ) -> np.ndarray:
        raw = np.zeros((len(pages), len(self.features)), dtype=np.float64)
        for row_index, page in enumerate(pages):
            for feature_index, key in enumerate(self.features):
                raw[row_index, feature_index] = math.log1p(
                    page.counts[key]
                )
        standardized = (raw - self.mean) / self.scale
        scores = standardized @ self.components.T
        return scores / self.axis_scale

    def explained_fraction(self, axes: int) -> float:
        kept = float(np.sum(self.singular_values[:axes] ** 2))
        return kept / max(self.total_variance, 1e-12)


@dataclass(frozen=True)
class VisualData:
    embeddings: dict[str, dict]
    handcrafted: dict[str, dict]

    def vector(self, folio: str, family: str) -> np.ndarray:
        if family == "handcrafted_combined":
            return np.asarray(
                self.handcrafted[folio]["features"]["combined_guarded"],
                dtype=np.float64,
            )
        if family == LEAKAGE_FAMILY:
            return np.asarray(
                self.handcrafted[folio]["leakage_diagnostic"][family],
                dtype=np.float64,
            )
        values = self.embeddings[folio]["features"]
        if family == "dino_full_rgb_cls":
            keys = ("full_rgb_cls",)
        elif family == "dino_tight_rgb_cls":
            keys = ("tight_rgb_cls",)
        elif family == "dino_full_silhouette_cls":
            keys = ("full_silhouette_cls",)
        elif family == "dino_tight_silhouette_cls":
            keys = ("tight_silhouette_cls",)
        elif family == "dino_full_rgb_both":
            keys = ("full_rgb_cls", "full_rgb_patch_mean")
        elif family == "dino_tight_rgb_both":
            keys = ("tight_rgb_cls", "tight_rgb_patch_mean")
        elif family == "dino_full_tight_rgb_cls":
            keys = ("full_rgb_cls", "tight_rgb_cls")
        else:
            raise ValueError(family)
        return np.concatenate([
            np.asarray(values[key], dtype=np.float64) for key in keys
        ])

    def has(self, folio: str) -> bool:
        return folio in self.embeddings and folio in self.handcrafted


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_records(path: Path) -> tuple[dict[str, dict], dict]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    duplicates = Counter()
    excluded = []
    for record in artifact["records"]:
        folio = str(record["folio"])
        if record.get("exclude_from_gate", False):
            excluded.append(folio)
            continue
        duplicates[folio] += 1
        records.setdefault(folio, record)
    return records, {
        "records": len(artifact["records"]),
        "excluded_from_gate": sorted(excluded),
        "unique_exact_folios": len(records),
        "duplicates": {
            folio: count for folio, count in duplicates.items()
            if count > 1
        },
    }


def load_visual(
    embeddings: Path,
    handcrafted: Path,
) -> tuple[VisualData, dict]:
    embedded, embedding_audit = unique_records(embeddings)
    manual, handcrafted_audit = unique_records(handcrafted)
    return VisualData(embedded, manual), {
        "embeddings": embedding_audit,
        "handcrafted": handcrafted_audit,
        "common_exact_folios": len(set(embedded) & set(manual)),
    }


def fit_axis_model(
    pages: Sequence[state.PageState],
    max_axes: int = MAX_AXES,
) -> AxisModel:
    totals: Counter = Counter()
    for page in pages:
        totals.update(page.counts)
    features = tuple(sorted(
        (key for key, count in totals.items() if count >= 2),
        key=lambda key: (key[0], key[1]),
    ))
    raw = np.zeros((len(pages), len(features)), dtype=np.float64)
    for row_index, page in enumerate(pages):
        for feature_index, key in enumerate(features):
            raw[row_index, feature_index] = math.log1p(page.counts[key])
    mean = raw.mean(axis=0)
    deviation = raw.std(axis=0)
    keep = deviation > 1e-9
    features = tuple(
        feature for feature, use in zip(features, keep) if use
    )
    raw = raw[:, keep]
    mean = mean[keep]
    scale = deviation[keep]
    standardized = (raw - mean) / scale
    _u, singular_values, components = np.linalg.svd(
        standardized,
        full_matrices=False,
    )
    axes = min(max_axes, len(pages) - 1, len(singular_values))
    components = components[:axes]
    axis_scale = (
        singular_values[:axes] / math.sqrt(max(1, len(pages) - 1))
    )
    axis_scale = np.maximum(axis_scale, 1e-9)
    return AxisModel(
        features=features,
        mean=mean,
        scale=scale,
        components=components,
        axis_scale=axis_scale,
        singular_values=singular_values,
        total_variance=float(np.sum(singular_values ** 2)),
    )


def axis_descriptions(model: AxisModel, axes: int = 8) -> list[dict]:
    rows = []
    for axis in range(min(axes, len(model.components))):
        weights = model.components[axis]
        positive = np.argsort(weights)[-8:][::-1]
        negative = np.argsort(weights)[:8]

        def label(index: int) -> str:
            context, symbol = model.features[int(index)]
            return f"{''.join(context)}>{symbol}"

        rows.append({
            "axis": axis + 1,
            "variance_fraction": (
                float(model.singular_values[axis] ** 2)
                / max(model.total_variance, 1e-12)
            ),
            "positive_transitions": [
                {
                    "transition": label(index),
                    "loading": float(weights[index]),
                }
                for index in positive
            ],
            "negative_transitions": [
                {
                    "transition": label(index),
                    "loading": float(weights[index]),
                }
                for index in negative
            ],
        })
    return rows


def eligible_pages(
    pages: Sequence[state.PageState],
    visual: VisualData,
) -> list[state.PageState]:
    return [page for page in pages if visual.has(page.folio)]


def matched_groups(
    pages: Sequence[state.PageState],
    visual: VisualData,
) -> dict[tuple[str, str, str], list[state.PageState]]:
    groups: dict[tuple[str, str, str], list[state.PageState]] = defaultdict(list)
    for page in eligible_pages(pages, visual):
        groups[(page.block, page.currier, page.section)].append(page)
    return {
        stratum: sorted(group, key=lambda page: source.folio_key(page.folio))
        for stratum, group in groups.items()
        if len(group) >= 2
    }


def visual_matrices(
    visual: VisualData,
    pages: Sequence[state.PageState],
    family: str,
) -> np.ndarray:
    return np.stack([
        visual.vector(page.folio, family) for page in pages
    ])


def ridge_predict(
    visual: VisualData,
    train_pages: Sequence[state.PageState],
    target_pages: Sequence[state.PageState],
    train_coordinates: np.ndarray,
    family: str,
    ridge: float,
) -> np.ndarray:
    train_x = visual_matrices(visual, train_pages, family)
    target_x = visual_matrices(visual, target_pages, family)
    mean = train_x.mean(axis=0)
    deviation = train_x.std(axis=0)
    scale = np.maximum(
        deviation,
        np.maximum(np.abs(mean) * 0.05, 1e-5),
    )
    train_x = (train_x - mean) / scale
    target_x = (target_x - mean) / scale
    dimensions = train_x.shape[1]
    kernel = train_x @ train_x.T / dimensions
    target_kernel = target_x @ train_x.T / dimensions
    coefficients = np.linalg.solve(
        kernel + ridge * np.eye(len(train_x)),
        train_coordinates,
    )
    return target_kernel @ coefficients


def page_map(
    pages: Sequence[state.PageState],
    coordinates: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        page.folio: coordinates[index]
        for index, page in enumerate(pages)
    }


def score_matrices(
    groups: dict[tuple[str, str, str], list[state.PageState]],
) -> dict[float, dict[tuple[str, str, str], np.ndarray]]:
    return {
        alpha: first_gate.donor_score_matrices(groups, alpha)
        for alpha in ALPHAS
    }


def choose_indices(
    target_index: int,
    predicted: np.ndarray,
    actual: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    distances = np.sqrt(np.mean(
        (actual - predicted[None, :]) ** 2,
        axis=1,
    ))
    distances[target_index] = np.inf
    use = min(neighbors, len(actual) - 1)
    return np.argsort(distances, kind="stable")[:use]


def selected_score(
    groups: dict[tuple[str, str, str], list[state.PageState]],
    matrices: dict[tuple[str, str, str], np.ndarray],
    predictions: dict[str, np.ndarray],
    actual_coordinates: dict[str, np.ndarray],
    axes: int,
    neighbors: int,
) -> float:
    total = 0.0
    for stratum, group in groups.items():
        matrix = matrices[stratum]
        actual = np.stack([
            actual_coordinates[page.folio][:axes] for page in group
        ])
        for target_index, target in enumerate(group):
            chosen = choose_indices(
                target_index,
                predictions[target.folio][:axes],
                actual,
                neighbors,
            )
            total += float(np.mean(matrix[target_index, chosen]))
    return total


def candidate_metrics(
    groups: dict[tuple[str, str, str], list[state.PageState]],
    matrices: dict[tuple[str, str, str], np.ndarray],
    predictions: dict[str, np.ndarray],
    actual_coordinates: dict[str, np.ndarray],
    axes: int,
    neighbors: int,
) -> dict:
    visual_total = selected_score(
        groups,
        matrices,
        predictions,
        actual_coordinates,
        axes,
        neighbors,
    )
    mean_total = 0.0
    observations = 0
    predicted_rows = []
    actual_rows = []
    for stratum, group in groups.items():
        matrix = matrices[stratum]
        for target_index, target in enumerate(group):
            donors = [
                index for index in range(len(group))
                if index != target_index
            ]
            mean_total += float(np.mean(matrix[target_index, donors]))
            observations += len(target.suffix_rows)
            predicted_rows.append(predictions[target.folio][:axes])
            actual_rows.append(actual_coordinates[target.folio][:axes])
    predicted_array = np.stack(predicted_rows)
    actual_array = np.stack(actual_rows)
    squared_error = float(np.sum(
        (predicted_array - actual_array) ** 2
    ))
    baseline_error = float(np.sum(actual_array ** 2))
    return {
        "eligible_pages": len(predicted_rows),
        "suffix_words": observations,
        "image_axis_r2": (
            1.0 - squared_error / max(baseline_error, 1e-12)
        ),
        "image_selected_bits_per_word": -visual_total / observations,
        "mean_matched_donor_bits_per_word": -mean_total / observations,
        "image_gain_over_mean_donor_bits_per_word": (
            visual_total - mean_total
        ) / observations,
    }


def evaluate(
    groups: dict[tuple[str, str, str], list[state.PageState]],
    matrices: dict[tuple[str, str, str], np.ndarray],
    predictions: dict[str, np.ndarray],
    actual_coordinates: dict[str, np.ndarray],
    axes: int,
    neighbors: int,
    alpha: float,
    permutations: int = 0,
    seed: int = SEED,
    keep_targets: bool = False,
) -> dict:
    visual_total = selected_score(
        groups,
        matrices,
        predictions,
        actual_coordinates,
        axes,
        neighbors,
    )
    mean_total = own_total = static_total = 0.0
    axis_oracle_total = full_oracle_total = ceiling_total = 0.0
    observations = 0
    target_rows = []
    predicted_rows = []
    actual_rows = []
    for stratum, group in groups.items():
        matrix = matrices[stratum]
        actual = np.stack([
            actual_coordinates[page.folio][:axes] for page in group
        ])
        for target_index, target in enumerate(group):
            donor_indices = [
                index for index in range(len(group))
                if index != target_index
            ]
            chosen = choose_indices(
                target_index,
                predictions[target.folio][:axes],
                actual,
                neighbors,
            )
            axis_oracle_index = choose_indices(
                target_index,
                actual[target_index],
                actual,
                1,
            )[0]
            full_oracle_index = min(
                donor_indices,
                key=lambda index: (
                    first_gate.state_distance(target, group[index]),
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
            mean_total += mean_donor
            own_total += own
            static_total += static
            axis_oracle_total += float(
                matrix[target_index, axis_oracle_index]
            )
            full_oracle_total += float(
                matrix[target_index, full_oracle_index]
            )
            ceiling_total += float(
                np.max(matrix[target_index, donor_indices])
            )
            observations += len(target.suffix_rows)
            predicted_rows.append(predictions[target.folio][:axes])
            actual_rows.append(actual[target_index])
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
                    "axis_oracle_donor": group[
                        int(axis_oracle_index)
                    ].folio,
                    "full_state_oracle_donor": group[
                        int(full_oracle_index)
                    ].folio,
                })

    predicted_array = np.stack(predicted_rows)
    actual_array = np.stack(actual_rows)
    squared_error = float(np.sum(
        (predicted_array - actual_array) ** 2
    ))
    baseline_error = float(np.sum(actual_array ** 2))
    axis_r2 = 1.0 - squared_error / max(baseline_error, 1e-12)

    null_totals = []
    if permutations:
        rng = random.Random(seed)
        for _replicate in range(permutations):
            permuted = dict(predictions)
            for group in groups.values():
                folios = [page.folio for page in group]
                assignment = list(range(len(folios)))
                rng.shuffle(assignment)
                for target_index, folio in enumerate(folios):
                    permuted[folio] = predictions[
                        folios[assignment[target_index]]
                    ]
            null_totals.append(selected_score(
                groups,
                matrices,
                permuted,
                actual_coordinates,
                axes,
                neighbors,
            ))
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
        "image_axis_r2": axis_r2,
        "image_selected_bits_per_word": -visual_total / observations,
        "mean_matched_donor_bits_per_word": -mean_total / observations,
        "own_prefix_bits_per_word": -own_total / observations,
        "static_bits_per_word": -static_total / observations,
        "axis_oracle_bits_per_word": -axis_oracle_total / observations,
        "full_state_oracle_bits_per_word": (
            -full_oracle_total / observations
        ),
        "posthoc_best_donor_ceiling_bits_per_word": (
            -ceiling_total / observations
        ),
        "image_gain_over_mean_donor_bits_per_word": (
            visual_total - mean_total
        ) / observations,
        "axis_oracle_gain_over_mean_donor_bits_per_word": (
            axis_oracle_total - mean_total
        ) / observations,
        "full_state_oracle_gain_over_mean_donor_bits_per_word": (
            full_oracle_total - mean_total
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
        "_raw": {
            "visual_total": visual_total,
            "mean_total": mean_total,
            "null_totals": null_totals,
        },
    }
    return result


def candidate_table(
    train_pages: Sequence[state.PageState],
    target_pages: Sequence[state.PageState],
    groups: dict[tuple[str, str, str], list[state.PageState]],
    matrices_by_alpha: dict[
        float, dict[tuple[str, str, str], np.ndarray]
    ],
    visual: VisualData,
    axis_model: AxisModel,
    families: Sequence[str],
) -> tuple[list[dict], dict[tuple[str, float], dict[str, np.ndarray]]]:
    train_coordinates = axis_model.coordinates(train_pages)
    target_coordinates = axis_model.coordinates(target_pages)
    actual = page_map(target_pages, target_coordinates)
    prediction_maps = {}
    candidates = []
    for family in families:
        for ridge in RIDGES:
            predicted = ridge_predict(
                visual,
                train_pages,
                target_pages,
                train_coordinates,
                family,
                ridge,
            )
            predictions = page_map(target_pages, predicted)
            prediction_maps[(family, ridge)] = predictions
            for axes in AXIS_COUNTS:
                if axes > train_coordinates.shape[1]:
                    continue
                for neighbors in NEIGHBORS:
                    for alpha in ALPHAS:
                        result = candidate_metrics(
                            groups,
                            matrices_by_alpha[alpha],
                            predictions,
                            actual,
                            axes,
                            neighbors,
                        )
                        candidates.append({
                            "family": family,
                            "ridge": ridge,
                            "axes": axes,
                            "axis_explained_fraction": (
                                axis_model.explained_fraction(axes)
                            ),
                            "neighbors": neighbors,
                            "alpha": alpha,
                            "eligible_pages": result["eligible_pages"],
                            "suffix_words": result["suffix_words"],
                            "image_axis_r2": result["image_axis_r2"],
                            "image_selected_bits_per_word": (
                                result["image_selected_bits_per_word"]
                            ),
                            "mean_matched_donor_bits_per_word": (
                                result[
                                    "mean_matched_donor_bits_per_word"
                                ]
                            ),
                            "image_gain_over_mean_donor_bits_per_word": (
                                result[
                                    "image_gain_over_mean_donor_bits_per_word"
                                ]
                            ),
                        })
    return candidates, prediction_maps


def winner(candidates: Sequence[dict]) -> dict:
    return min(
        candidates,
        key=lambda row: (
            row["image_selected_bits_per_word"],
            row["family"],
            row["ridge"],
            row["axes"],
            row["neighbors"],
            row["alpha"],
        ),
    )


def serializable(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "_raw"}


def aggregate(folds: Sequence[dict]) -> dict:
    observations = sum(int(fold["suffix_words"]) for fold in folds)
    weighted = (
        "image_axis_r2",
        "image_selected_bits_per_word",
        "mean_matched_donor_bits_per_word",
        "own_prefix_bits_per_word",
        "static_bits_per_word",
        "axis_oracle_bits_per_word",
        "full_state_oracle_bits_per_word",
        "posthoc_best_donor_ceiling_bits_per_word",
        "image_gain_over_mean_donor_bits_per_word",
        "axis_oracle_gain_over_mean_donor_bits_per_word",
        "full_state_oracle_gain_over_mean_donor_bits_per_word",
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
        "selected_ridges": dict(Counter(
            str(fold["selected_ridge"]) for fold in folds
        )),
        "selected_axes": dict(Counter(
            str(fold["selected_axes"]) for fold in folds
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
        "null_selection_gain_over_mean_donor_bits_per_word": (
            result["mean_matched_donor_bits_per_word"]
            + null_mean / observations
        ),
        "observed_gain_over_null_bits_per_word": (
            observed - null_mean
        ) / observations,
    }
    return result


def run(
    corpus: Path,
    embeddings: Path,
    handcrafted: Path,
    output: Path,
    progress: bool,
) -> dict:
    events, corpus_audit = source.load_events(corpus)
    visual, visual_audit = load_visual(embeddings, handcrafted)
    assignment, loads = source.block_folds(events)
    fold_results = []
    leakage_results = []
    for fold in range(source.N_FOLDS):
        train, validation, test = source.split_events(
            events, assignment, fold
        )
        model = source.CharacterModel(train, 2, "register")
        train_pages = eligible_pages(
            state.page_states(
                state.symbol_rows(model, train), PREFIX_WORDS
            ),
            visual,
        )
        validation_pages = eligible_pages(
            state.page_states(
                state.symbol_rows(model, validation), PREFIX_WORDS
            ),
            visual,
        )
        test_pages = eligible_pages(
            state.page_states(
                state.symbol_rows(model, test), PREFIX_WORDS
            ),
            visual,
        )
        axis_model = fit_axis_model(train_pages)
        validation_groups = matched_groups(validation_pages, visual)
        validation_matrices = score_matrices(validation_groups)
        validation_candidates, _validation_predictions = candidate_table(
            train_pages,
            validation_pages,
            validation_groups,
            validation_matrices,
            visual,
            axis_model,
            VISUAL_FAMILIES,
        )
        selected = winner(validation_candidates)

        train_coordinates = axis_model.coordinates(train_pages)
        test_coordinates = axis_model.coordinates(test_pages)
        test_actual = page_map(test_pages, test_coordinates)
        test_predicted = ridge_predict(
            visual,
            train_pages,
            test_pages,
            train_coordinates,
            str(selected["family"]),
            float(selected["ridge"]),
        )
        test_predictions = page_map(test_pages, test_predicted)
        test_groups = matched_groups(test_pages, visual)
        test_matrices = score_matrices(test_groups)
        test_result = evaluate(
            test_groups,
            test_matrices[float(selected["alpha"])],
            test_predictions,
            test_actual,
            int(selected["axes"]),
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
            "train_axis_pages": len(train_pages),
            "axis_features": len(axis_model.features),
            "axis_descriptions": axis_descriptions(axis_model),
            "selected_family": selected["family"],
            "selected_ridge": selected["ridge"],
            "selected_axes": selected["axes"],
            "selected_axis_explained_fraction": selected[
                "axis_explained_fraction"
            ],
            "selected_neighbors": selected["neighbors"],
            "selected_alpha": selected["alpha"],
            "validation_winner": selected,
            "validation_candidates": validation_candidates,
        })
        fold_results.append(test_result)

        leakage_candidates, _leakage_predictions = candidate_table(
            train_pages,
            validation_pages,
            validation_groups,
            validation_matrices,
            visual,
            axis_model,
            (LEAKAGE_FAMILY,),
        )
        leakage_selected = winner(leakage_candidates)
        leakage_predicted = ridge_predict(
            visual,
            train_pages,
            test_pages,
            train_coordinates,
            LEAKAGE_FAMILY,
            float(leakage_selected["ridge"]),
        )
        leakage_result = evaluate(
            test_groups,
            test_matrices[float(leakage_selected["alpha"])],
            page_map(test_pages, leakage_predicted),
            test_actual,
            int(leakage_selected["axes"]),
            int(leakage_selected["neighbors"]),
            float(leakage_selected["alpha"]),
            permutations=PERMUTATIONS,
            seed=SEED + 50_000 + 10_000 * (fold + 1),
        )
        leakage_result.update({
            "fold": fold,
            "selected_family": LEAKAGE_FAMILY,
            "selected_ridge": leakage_selected["ridge"],
            "selected_axes": leakage_selected["axes"],
            "selected_neighbors": leakage_selected["neighbors"],
            "selected_alpha": leakage_selected["alpha"],
            "validation_winner": leakage_selected,
        })
        leakage_results.append(leakage_result)
        if progress:
            print(
                f"fold={fold} family={selected['family']} "
                f"ridge={selected['ridge']} axes={selected['axes']} "
                f"k={selected['neighbors']} alpha={int(selected['alpha'])} "
                f"gain={test_result['image_gain_over_mean_donor_bits_per_word']:+.4f} "
                f"r2={test_result['image_axis_r2']:+.3f} "
                f"p={test_result['image_relabeling_null']['upper_p']:.4f}",
                flush=True,
            )

    primary_summary = aggregate(fold_results)
    leakage_summary = aggregate(leakage_results)
    primary_pass = (
        primary_summary[
            "image_gain_over_mean_donor_bits_per_word"
        ] > 0
        and primary_summary["pooled_image_relabeling_null"]["upper_p"]
        <= 0.05
    )
    control_pass = (
        primary_summary[
            "axis_oracle_gain_over_mean_donor_bits_per_word"
        ] > 0
        and primary_summary[
            "full_state_oracle_gain_over_mean_donor_bits_per_word"
        ] > 0
    )
    result = {
        "experiment": "guarded_visual_prediction_of_text_state_axes",
        "seed": SEED,
        "parameters": {
            "prefix_words": PREFIX_WORDS,
            "suffix_start_word": state.SUFFIX_START + 1,
            "maximum_scored_suffix_words": state.MAX_SUFFIX_WORDS,
            "state_axis_transform": (
                "train-page log1p character-transition counts; "
                "feature standardization; PCA; unit-variance axes"
            ),
            "maximum_state_axes": MAX_AXES,
            "axis_counts": list(AXIS_COUNTS),
            "ridge_values": list(RIDGES),
            "neighbors": list(NEIGHBORS),
            "alphas": list(ALPHAS),
            "primary_visual_families": list(VISUAL_FAMILIES),
            "excluded_text_layout_diagnostic": LEAKAGE_FAMILY,
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
            str(embeddings.relative_to(ROOT)): sha256(embeddings),
            str(handcrafted.relative_to(ROOT)): sha256(handcrafted),
        },
        "corpus_audit": dict(corpus_audit),
        "visual_audit": visual_audit,
        "fold_assignment": assignment,
        "fold_event_loads": loads,
        "folds": [serializable(fold) for fold in fold_results],
        "leakage_diagnostic_folds": [
            serializable(fold) for fold in leakage_results
        ],
        "summary": {
            "primary": primary_summary,
            "excluded_text_layout_diagnostic": leakage_summary,
            "controls": {
                "axis_and_full_state_oracles_positive": control_pass,
            },
            "primary_gate_pass": primary_pass,
        },
        "claim_boundary": (
            "A pass establishes out-of-quire prediction of frozen text-state "
            "axes from frozen, text-masked visual embeddings under exact "
            "metadata rematching. It does not identify plaintext or image "
            "semantics. A failure closes these guarded DINOv2/global "
            "representations and linear kernel map, not object graphs or "
            "manually aligned local correspondences."
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
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS)
    parser.add_argument("--handcrafted", type=Path, default=HANDCRAFTED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        args.corpus,
        args.embeddings,
        args.handcrafted,
        args.output,
        progress=not args.quiet,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
