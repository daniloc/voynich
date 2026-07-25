#!/usr/bin/env python3
"""
Full-zodiac and cross-domain multimodal discriminator.

This follows K21 without pooling incompatible annotation units:

1. NODE GATE: all 286 inner/outer zodiac labels are tested against guarded
   figure-core pixel features.  Candidate selection and scoring hold out whole
   folios.  The null cyclically rotates complete visual sequences within ring.
2. CALIBRATION GATE: on the four hand-anchored rings, the same pixel channels
   must predict at least some independently recorded semantic attributes.
3. PHASE SENSITIVITY: all eight geometry-only bindings are shifted by -1, 0,
   and +1 nodes.  No phase is selected from the text.
4. CROSS-DOMAIN GATE: page-level pigment/graph features select a relationship
   with text statistics in one domain and test its direction in the other.
   Complete text-statistic records are permuted for the null.

The primary node features use a small figure-core crop and large connected ink
components.  Raw ink, edge density, and entropy are excluded because the
adjacent Voynich label would make them direct proxies for the target word.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import stateful_line_program_search as stateful  # noqa: E402


ZODIAC_NODES = ROOT / "data" / "grounding" / "zodiac_all12_visual_nodes.json"
ZODIAC_ANCHORS = ROOT / "data" / "grounding" / "z10_bindings.json"
MULTIMODAL_GRAPHS = ROOT / "data" / "grounding" / "multimodal_visual_graphs.json"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "followups_multimodal_graph_transfer_gate.json"
)
SEED = 20260723

NODE_PIXEL_FEATURES = (
    "red_fraction",
    "green_fraction",
    "blue_fraction",
    "gold_fraction",
    "large_component_fraction",
    "largest_component_fraction",
    "large_component_count",
    "large_component_centroid_x",
    "large_component_centroid_y",
    "large_horizontal_symmetry",
    "large_vertical_symmetry",
)
RAW_LEAKAGE_FEATURES = (
    "ink_fraction",
    "edge_density",
    "contrast",
    "entropy",
    "ink_centroid_x",
    "ink_centroid_y",
)
TEXT_OPERATIONS = ("current", "delta", "lag_plus")
TEXT_MODULI = (2, 3, 5, 7)
SEMANTIC_ATTRIBUTES = (
    "body",
    "container",
    "star_tail",
    "star_hand",
    "arms",
    "headwear",
    "facing",
)


@dataclass(frozen=True)
class NodeCandidate:
    text_feature: str
    operation: str
    modulus: int
    visual_channel: str

    @property
    def name(self) -> str:
        return (
            f"{self.text_feature}|{self.operation}|mod{self.modulus}|"
            f"{self.visual_channel}"
        )


def clean_word(value: str) -> str:
    return "".join(char for char in value if char in stateful.GLYPH_VALUE)


def empirical_upper_p(observed: float, nulls: Sequence[float]) -> float:
    return (1 + sum(value >= observed - 1e-12 for value in nulls)) / (
        len(nulls) + 1
    )


def entropy(values: Sequence[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    return -sum(
        count / total * math.log2(count / total)
        for count in counts.values()
    ) if total else 0.0


def tertile_categories(values: Sequence[float]) -> list[str]:
    array = np.asarray(values, dtype=float)
    lower, upper = np.quantile(array, (1 / 3, 2 / 3))
    if math.isclose(float(lower), float(upper), abs_tol=1e-12):
        return ["same"] * len(array)
    return [
        "low" if value <= lower else ("mid" if value <= upper else "high")
        for value in array
    ]


def build_visual_channels(visuals: Sequence[dict]) -> dict[str, list[str]]:
    channels = {}
    for feature in NODE_PIXEL_FEATURES:
        values = [float(row[feature]) for row in visuals]
        channels[f"current:{feature}"] = tertile_categories(values)
        deltas = [
            values[index] - values[index - 1]
            for index in range(len(values))
        ]
        tolerance = max(np.std(values) * 0.02, 1e-10)
        channels[f"transition:{feature}"] = [
            "down" if value < -tolerance else (
                "up" if value > tolerance else "same"
            )
            for value in deltas
        ]
    return channels


def text_outcomes(
    words: Sequence[str],
    feature: str,
    operation: str,
    modulus: int,
) -> list[int]:
    values = [stateful.feature_value(word, feature) for word in words]
    outcomes = []
    for index, current in enumerate(values):
        previous = values[index - 1]
        if operation == "current":
            value = current
        elif operation == "delta":
            value = current - previous
        elif operation == "lag_plus":
            value = current + previous
        else:
            raise ValueError(operation)
        outcomes.append(value % modulus)
    return outcomes


def prepare_sequence(
    folio: str,
    tier: str,
    records: Sequence[dict],
    alignment_confidence: str,
) -> dict:
    words = [clean_word(record["label_primary"]) for record in records]
    visuals = [record["visual"] for record in records]
    outcomes = {
        (feature, operation, modulus): text_outcomes(
            words, feature, operation, modulus
        )
        for feature in stateful.FEATURES
        for operation in TEXT_OPERATIONS
        for modulus in TEXT_MODULI
    }
    return {
        "folio": folio,
        "tier": tier,
        "words": words,
        "records": list(records),
        "visuals": visuals,
        "channels": build_visual_channels(visuals),
        "outcomes": outcomes,
        "alignment_confidence": alignment_confidence,
    }


def load_node_sequences() -> dict[tuple[str, str], dict]:
    source = json.loads(ZODIAC_NODES.read_text(encoding="utf-8"))
    sequences = {}
    for folio, page in source["folios"].items():
        for tier in ("outer", "inner"):
            records = [
                record for record in page["records"]
                if record["tier"] == tier
            ]
            if not records:
                continue
            confidence = page["alignment"][tier]["confidence"]
            sequences[(folio, tier)] = prepare_sequence(
                folio, tier, records, confidence
            )
    return sequences


def node_candidates() -> list[NodeCandidate]:
    channels = [
        f"{kind}:{feature}"
        for feature in NODE_PIXEL_FEATURES
        for kind in ("current", "transition")
    ]
    return [
        NodeCandidate(feature, operation, modulus, channel)
        for feature in stateful.FEATURES
        for operation in TEXT_OPERATIONS
        for modulus in TEXT_MODULI
        for channel in channels
    ]


def categorical_gain(
    sequences: dict[tuple[str, str], dict],
    candidate: NodeCandidate,
    train_folios: set[str],
    score_folios: set[str],
    alpha: float = 4.0,
) -> tuple[float, int]:
    base: dict[str, Counter[int]] = defaultdict(Counter)
    conditional: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    outcome_key = (
        candidate.text_feature,
        candidate.operation,
        candidate.modulus,
    )
    for sequence in sequences.values():
        if sequence["folio"] not in train_folios:
            continue
        outcomes = sequence["outcomes"][outcome_key]
        visuals = sequence["channels"][candidate.visual_channel]
        for outcome, visual in zip(outcomes, visuals):
            base[sequence["tier"]][outcome] += 1
            conditional[(sequence["tier"], visual)][outcome] += 1

    gain = 0.0
    count = 0
    for sequence in sequences.values():
        if sequence["folio"] not in score_folios:
            continue
        tier = sequence["tier"]
        base_total = sum(base[tier].values())
        if base_total == 0:
            continue
        outcomes = sequence["outcomes"][outcome_key]
        visuals = sequence["channels"][candidate.visual_channel]
        for outcome, visual in zip(outcomes, visuals):
            p_base = (base[tier][outcome] + 1) / (
                base_total + candidate.modulus
            )
            bucket = conditional[(tier, visual)]
            p_visual = (bucket[outcome] + alpha * p_base) / (
                sum(bucket.values()) + alpha
            )
            gain += math.log2(p_visual / p_base)
            count += 1
    return (gain / count if count else float("-inf")), count


def run_node_search(
    sequences: dict[tuple[str, str], dict],
    candidates: Sequence[NodeCandidate],
) -> list[dict]:
    folios = sorted({sequence["folio"] for sequence in sequences.values()})
    folds = []
    for fold, test in enumerate(folios):
        validation = folios[(fold + 1) % len(folios)]
        train = set(folios) - {test, validation}
        ranked = []
        for candidate in candidates:
            score, n_validation = categorical_gain(
                sequences, candidate, train, {validation}
            )
            ranked.append((score, candidate.name, candidate, n_validation))
        validation_gain, _, winner, n_validation = max(ranked)
        test_gain, n_test = categorical_gain(
            sequences, winner, train, {test}
        )
        folds.append({
            "fold": fold,
            "train_folios": sorted(train),
            "validation_folio": validation,
            "test_folio": test,
            "candidate": winner.name,
            "text_feature": winner.text_feature,
            "operation": winner.operation,
            "modulus": winner.modulus,
            "visual_channel": winner.visual_channel,
            "validation_gain_bits_per_node": validation_gain,
            "test_gain_bits_per_node": test_gain,
            "n_validation": n_validation,
            "n_test": n_test,
        })
    return folds


def node_summary(folds: Sequence[dict]) -> dict:
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


def rotate_visual_sequences(
    sequences: dict[tuple[str, str], dict],
    offsets: dict[tuple[str, str], int],
) -> dict[tuple[str, str], dict]:
    result = {}
    for key, sequence in sequences.items():
        offset = offsets.get(key, 0) % len(sequence["visuals"])
        visuals = (
            sequence["visuals"][offset:] + sequence["visuals"][:offset]
            if offset else list(sequence["visuals"])
        )
        copy = dict(sequence)
        copy["visuals"] = visuals
        copy["channels"] = build_visual_channels(visuals)
        result[key] = copy
    return result


def random_rotation_offsets(
    sequences: dict[tuple[str, str], dict],
    rng: random.Random,
) -> dict[tuple[str, str], int]:
    return {
        key: rng.randrange(1, len(sequence["visuals"]))
        for key, sequence in sequences.items()
    }


def phase_scenarios(
    sequences: dict[tuple[str, str], dict],
    candidates: Sequence[NodeCandidate],
) -> list[dict]:
    inferred = {
        key for key, sequence in sequences.items()
        if sequence["alignment_confidence"] != "high"
    }
    scenarios = {
        "all_minus_one": {
            key: -1 for key in inferred
        },
        "declared_center": {},
        "all_plus_one": {
            key: 1 for key in inferred
        },
        "alternating": {
            key: (-1 if index % 2 == 0 else 1)
            for index, key in enumerate(sorted(inferred))
        },
    }
    reports = []
    for name, offsets in scenarios.items():
        shifted = rotate_visual_sequences(sequences, offsets)
        folds = run_node_search(shifted, candidates)
        reports.append({
            "scenario": name,
            "offsets": {"|".join(key): value for key, value in offsets.items()},
            "summary": node_summary(folds),
            "folds": folds,
        })
    return reports


def generic_visual_gain(
    sequences: dict[tuple[str, str], dict],
    outcomes: dict[tuple[str, str], list],
    channel: str,
    train_folios: set[str],
    score_folios: set[str],
    alpha: float = 4.0,
) -> tuple[float, int]:
    base: dict[str, Counter] = defaultdict(Counter)
    conditional: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for key, sequence in sequences.items():
        if sequence["folio"] not in train_folios:
            continue
        for outcome, visual in zip(
            outcomes[key], sequence["channels"][channel]
        ):
            base[sequence["tier"]][outcome] += 1
            conditional[(sequence["tier"], visual)][outcome] += 1

    gain = 0.0
    count = 0
    for key, sequence in sequences.items():
        if sequence["folio"] not in score_folios:
            continue
        tier = sequence["tier"]
        categories = len({
            outcome
            for other_key, values in outcomes.items()
            if sequences[other_key]["tier"] == tier
            for outcome in values
        })
        total = sum(base[tier].values())
        if not total:
            continue
        for outcome, visual in zip(
            outcomes[key], sequence["channels"][channel]
        ):
            p_base = (base[tier][outcome] + 1) / (
                total + max(categories, 1)
            )
            bucket = conditional[(tier, visual)]
            p_visual = (bucket[outcome] + alpha * p_base) / (
                sum(bucket.values()) + alpha
            )
            gain += math.log2(p_visual / p_base)
            count += 1
    return (gain / count if count else float("-inf")), count


def run_generic_search(
    sequences: dict[tuple[str, str], dict],
    outcomes: dict[tuple[str, str], list],
    candidates: Sequence[str],
) -> list[dict]:
    folios = sorted({sequence["folio"] for sequence in sequences.values()})
    folds = []
    for fold, test in enumerate(folios):
        validation = folios[(fold + 1) % len(folios)]
        train = set(folios) - {test, validation}
        ranked = [
            (
                generic_visual_gain(
                    sequences, outcomes, channel, train, {validation}
                )[0],
                channel,
            )
            for channel in candidates
        ]
        validation_gain, winner = max(ranked)
        test_gain, n_test = generic_visual_gain(
            sequences, outcomes, winner, train, {test}
        )
        folds.append({
            "fold": fold,
            "test_folio": test,
            "validation_folio": validation,
            "selected_visual_channel": winner,
            "validation_gain_bits_per_node": validation_gain,
            "test_gain_bits_per_node": test_gain,
            "n_test": n_test,
        })
    return folds


def synthetic_node_control(
    sequences: dict[tuple[str, str], dict],
) -> dict:
    true_channel = "current:large_component_fraction"
    rng = random.Random(SEED + 200)
    outcomes = {}
    for key, sequence in sequences.items():
        values = [
            int(value == "high")
            for value in sequence["channels"][true_channel]
        ]
        outcomes[key] = [
            1 - value if rng.random() < 0.10 else value
            for value in values
        ]
    channels = sorted(next(iter(sequences.values()))["channels"])
    folds = run_generic_search(sequences, outcomes, channels)
    summary = {
        "true_visual_channel": true_channel,
        "noise_rate": 0.10,
        "mean_test_gain_bits_per_node": mean(
            fold["test_gain_bits_per_node"] for fold in folds
        ),
        "correct_selection_rate": mean(
            fold["selected_visual_channel"] == true_channel
            for fold in folds
        ),
    }
    summary["passed"] = (
        summary["mean_test_gain_bits_per_node"] > 0.10
        and summary["correct_selection_rate"] >= 0.50
    )
    return {"summary": summary, "folds": folds}


def semantic_calibration_sequences(
    sequences: dict[tuple[str, str], dict],
) -> tuple[dict[tuple[str, str], dict], dict[str, dict[tuple[str, str], list]]]:
    anchors = json.loads(ZODIAC_ANCHORS.read_text(encoding="utf-8"))["folios"]
    calibrated = {
        key: sequence
        for key, sequence in sequences.items()
        if key[0] in anchors
    }
    semantic = {
        attribute: {}
        for attribute in SEMANTIC_ATTRIBUTES
    }
    for key, sequence in calibrated.items():
        anchor_map = {
            (record["tier"], int(record["locus"])): record
            for record in anchors[key[0]]["records"]
            if record["tier"] in {"outer", "inner"}
        }
        for attribute in SEMANTIC_ATTRIBUTES:
            semantic[attribute][key] = [
                str(anchor_map[(key[1], record["locus"])].get(attribute) or "none")
                for record in sequence["records"]
            ]
    semantic = {
        attribute: values
        for attribute, values in semantic.items()
        if len({
            value for sequence_values in values.values()
            for value in sequence_values
        }) > 1
    }
    return calibrated, semantic


def run_semantic_search(
    sequences: dict[tuple[str, str], dict],
    semantic: dict[str, dict[tuple[str, str], list]],
) -> list[dict]:
    folios = sorted({sequence["folio"] for sequence in sequences.values()})
    channels = sorted(next(iter(sequences.values()))["channels"])
    folds = []
    for fold, test in enumerate(folios):
        validation = folios[(fold + 1) % len(folios)]
        train = set(folios) - {test, validation}
        ranked = []
        for attribute, outcomes in semantic.items():
            for channel in channels:
                score, _ = generic_visual_gain(
                    sequences, outcomes, channel, train, {validation}
                )
                ranked.append((score, attribute, channel))
        validation_gain, attribute, channel = max(ranked)
        test_gain, n_test = generic_visual_gain(
            sequences, semantic[attribute], channel, train, {test}
        )
        folds.append({
            "fold": fold,
            "test_folio": test,
            "validation_folio": validation,
            "semantic_attribute": attribute,
            "visual_channel": channel,
            "validation_gain_bits_per_node": validation_gain,
            "test_gain_bits_per_node": test_gain,
            "n_test": n_test,
        })
    return folds


def semantic_summary(folds: Sequence[dict]) -> dict:
    return {
        "mean_test_gain_bits_per_node": mean(
            fold["test_gain_bits_per_node"] for fold in folds
        ),
        "positive_test_folds": sum(
            fold["test_gain_bits_per_node"] > 0 for fold in folds
        ),
        "selection_counts": dict(Counter(
            f"{fold['semantic_attribute']}|{fold['visual_channel']}"
            for fold in folds
        )),
    }


def standardized_within_sequence_correlation(
    sequences: dict[tuple[str, str], dict],
    feature: str,
    source: str,
    offsets: Optional[dict[tuple[str, str], int]] = None,
) -> float:
    xs = []
    ys = []
    offsets = offsets or {}
    for key, sequence in sequences.items():
        values = [
            float(record[source][feature])
            for record in sequence["records"]
        ]
        offset = offsets.get(key, 0) % len(values)
        if offset:
            values = values[offset:] + values[:offset]
        lengths = [len(word) for word in sequence["words"]]
        x = np.asarray(values, dtype=float)
        y = np.asarray(lengths, dtype=float)
        if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            continue
        xs.extend(((x - np.mean(x)) / np.std(x)).tolist())
        ys.extend(((y - np.mean(y)) / np.std(y)).tolist())
    return pearson(xs, ys)


def label_leakage_probe(
    sequences: dict[tuple[str, str], dict],
    null_count: int,
) -> dict:
    feature_sets = {
        "guarded_core": [
            (feature, "visual") for feature in NODE_PIXEL_FEATURES
        ],
        "raw_core": [
            (feature, "visual") for feature in RAW_LEAKAGE_FEATURES
        ],
        "raw_context": [
            (feature, "context_visual") for feature in RAW_LEAKAGE_FEATURES
        ],
    }
    observed = {}
    for name, features in feature_sets.items():
        ranked = [
            (
                abs(standardized_within_sequence_correlation(
                    sequences, feature, source
                )),
                feature,
                source,
            )
            for feature, source in features
        ]
        score, feature, source = max(ranked)
        observed[name] = {
            "max_abs_correlation_with_word_length": score,
            "selected_feature": feature,
            "source": source,
        }

    rng = random.Random(SEED + 500)
    nulls = {name: [] for name in feature_sets}
    for _ in range(null_count):
        offsets = random_rotation_offsets(sequences, rng)
        for name, features in feature_sets.items():
            nulls[name].append(max(
                abs(standardized_within_sequence_correlation(
                    sequences, feature, source, offsets
                ))
                for feature, source in features
            ))
    for name in feature_sets:
        observed[name]["p_cyclic_rotation"] = empirical_upper_p(
            observed[name]["max_abs_correlation_with_word_length"],
            nulls[name],
        )
        observed[name]["null_mean_max_abs_correlation"] = mean(nulls[name])
    return {
        "target": "label_primary_length",
        "design": (
            "maximum within-ring standardized correlation; null rotates each "
            "complete pixel sequence and repeats feature selection"
        ),
        "results": observed,
    }


def targeted_green_gallows_gate(
    sequences: dict[tuple[str, str], dict],
    null_count: int,
) -> dict:
    channel = "presence:green_fraction"
    enriched = {}
    for key, sequence in sequences.items():
        copy = dict(sequence)
        copy["channels"] = dict(sequence["channels"])
        copy["channels"][channel] = [
            "present" if float(visual["green_fraction"]) > 1e-5 else "absent"
            for visual in sequence["visuals"]
        ]
        enriched[key] = copy
    outcomes = {
        key: [
            int(any(glyph in word for glyph in "ktpf"))
            for word in sequence["words"]
        ]
        for key, sequence in enriched.items()
    }
    folds = run_generic_search(enriched, outcomes, [channel])
    observed = mean(
        fold["test_gain_bits_per_node"] for fold in folds
    )
    rng = random.Random(SEED + 550)
    nulls = []
    for _ in range(null_count):
        offsets = random_rotation_offsets(enriched, rng)
        rotated = rotate_visual_sequences(enriched, offsets)
        for sequence in rotated.values():
            sequence["channels"][channel] = [
                "present" if float(visual["green_fraction"]) > 1e-5 else "absent"
                for visual in sequence["visuals"]
            ]
        null_folds = run_generic_search(
            rotated, outcomes, [channel]
        )
        nulls.append(mean(
            fold["test_gain_bits_per_node"] for fold in null_folds
        ))
    return {
        "status": "post_hoc_localization_check",
        "page_level_relation": "green pigment versus gallows frequency",
        "node_outcome": "label contains any of k,t,p,f",
        "visual_channel": channel,
        "mean_test_gain_bits_per_node": observed,
        "positive_test_folds": sum(
            fold["test_gain_bits_per_node"] > 0 for fold in folds
        ),
        "p_cyclic_rotation": empirical_upper_p(observed, nulls),
        "null_mean_gain_bits_per_node": mean(nulls),
        "folds": folds,
    }


def text_statistics(words: Sequence[str]) -> dict[str, float]:
    lengths = [len(word) for word in words]
    glyphs = [glyph for word in words for glyph in word]
    types = len(set(words))
    return {
        "mean_length": float(np.mean(lengths)),
        "length_sd": float(np.std(lengths)),
        "type_token_ratio": types / len(words),
        "repeat_fraction": 1 - types / len(words),
        "glyph_entropy": entropy(glyphs),
        "q_initial_fraction": mean(word.startswith("q") for word in words),
        "o_initial_fraction": mean(word.startswith("o") for word in words),
        "gallows_fraction": mean(
            any(glyph in word for glyph in "ktpf") for word in words
        ),
        "bench_fraction": mean(
            "ch" in word or "sh" in word for word in words
        ),
        "suffix_y_fraction": mean(word.endswith("y") for word in words),
        "mean_core_fraction": mean(
            len(stateful.decompose(word)[1]) / len(word) for word in words
        ),
    }


def load_page_records() -> dict[str, list[dict]]:
    graphs = json.loads(MULTIMODAL_GRAPHS.read_text(encoding="utf-8"))[
        "records"
    ]
    word_lines, _ = stateful.load_word_lines()
    plant_words: dict[str, list[str]] = defaultdict(list)
    for line in word_lines:
        plant_words[line.folio].extend(line.words)
    zodiac = json.loads(ZODIAC_NODES.read_text(encoding="utf-8"))["folios"]
    zodiac_words = {
        folio: [
            clean_word(record["label_primary"])
            for record in page["records"]
        ]
        for folio, page in zodiac.items()
    }

    result: dict[str, list[dict]] = defaultdict(list)
    for row in graphs:
        words = (
            plant_words[row["folio"]]
            if row["domain"] == "herbal"
            else zodiac_words[row["folio"]]
        )
        visual = {
            "pixel.red_fraction": row["visual"]["red_fraction"],
            "pixel.green_fraction": row["visual"]["green_fraction"],
            "pixel.blue_fraction": row["visual"]["blue_fraction"],
            "pixel.pigment_total": (
                row["visual"]["red_fraction"]
                + row["visual"]["green_fraction"]
                + row["visual"]["blue_fraction"]
            ),
        }
        pigment_values = np.asarray([
            row["visual"]["red_fraction"],
            row["visual"]["green_fraction"],
            row["visual"]["blue_fraction"],
        ])
        pigment_total = float(pigment_values.sum())
        if pigment_total:
            probabilities = pigment_values[pigment_values > 0] / pigment_total
            visual["pixel.pigment_entropy"] = float(
                -np.sum(probabilities * np.log2(probabilities))
            )
        else:
            visual["pixel.pigment_entropy"] = 0.0
        visual.update({
            f"graph.{name}": float(value)
            for name, value in row["graph"].items()
        })
        result[row["domain"]].append({
            "id": row["id"],
            "folio": row["folio"],
            "source": row["source"],
            "source_group": (
                row["source"]
                if row["domain"] == "herbal"
                else zodiac[row["folio"]]["yale_canvas_id"]
            ),
            "n_words": len(words),
            "visual": visual,
            "text": text_statistics(words),
        })
    return dict(result)


def rank_values(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 3 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(rank_values(xs), rank_values(ys))


def transfer_direction(
    source: Sequence[dict],
    target: Sequence[dict],
    visual_features: Optional[Sequence[str]] = None,
) -> dict:
    available_visual = sorted(
        set(source[0]["visual"]) & set(target[0]["visual"])
    )
    visual_features = (
        sorted(set(visual_features) & set(available_visual))
        if visual_features is not None else available_visual
    )
    text_features = sorted(set(source[0]["text"]) & set(target[0]["text"]))
    ranked = []
    for visual_feature in visual_features:
        source_x = [row["visual"][visual_feature] for row in source]
        target_x = [row["visual"][visual_feature] for row in target]
        if np.std(source_x) <= 1e-12 or np.std(target_x) <= 1e-12:
            continue
        for text_feature in text_features:
            source_y = [row["text"][text_feature] for row in source]
            target_y = [row["text"][text_feature] for row in target]
            if np.std(source_y) <= 1e-12 or np.std(target_y) <= 1e-12:
                continue
            source_r = spearman(source_x, source_y)
            target_r = spearman(target_x, target_y)
            sign = 1 if source_r >= 0 else -1
            ranked.append((
                abs(source_r),
                visual_feature,
                text_feature,
                source_r,
                sign * target_r,
                target_r,
            ))
    (
        _,
        visual_feature,
        text_feature,
        source_r,
        aligned_target_r,
        target_r,
    ) = max(ranked)
    return {
        "visual_feature": visual_feature,
        "text_feature": text_feature,
        "source_spearman": source_r,
        "source_direction": 1 if source_r >= 0 else -1,
        "target_spearman": target_r,
        "aligned_target_spearman": aligned_target_r,
        "candidate_count": len(ranked),
    }


def permute_text_records(
    rows: Sequence[dict],
    rng: random.Random,
) -> list[dict]:
    text_rows = [(row["text"], row["n_words"]) for row in rows]
    rng.shuffle(text_rows)
    return [
        {**row, "text": text, "n_words": n_words}
        for row, (text, n_words) in zip(rows, text_rows)
    ]


def folio_number(value: str) -> float:
    match = re.search(r"\d+", value)
    return float(match.group()) if match else 0.0


def partial_spearman(
    rows: Sequence[dict],
    visual_feature: str,
    text_feature: str,
) -> float:
    x = rank_values([row["visual"][visual_feature] for row in rows])
    y = rank_values([row["text"][text_feature] for row in rows])
    controls = np.column_stack([
        np.ones(len(rows)),
        rank_values([folio_number(row["folio"]) for row in rows]),
        rank_values([row["n_words"] for row in rows]),
    ])
    x_residual = x - controls @ np.linalg.lstsq(
        controls, x, rcond=None
    )[0]
    y_residual = y - controls @ np.linalg.lstsq(
        controls, y, rcond=None
    )[0]
    return pearson(x_residual, y_residual)


def common_pair(
    domains: dict[str, list[dict]],
    visual_features: Sequence[str],
    controlled: bool,
) -> dict:
    herbal = domains["herbal"]
    zodiac = domains["zodiac"]
    text_features = sorted(set(herbal[0]["text"]) & set(zodiac[0]["text"]))
    ranked = []
    for visual_feature in visual_features:
        if any(
            np.std([row["visual"][visual_feature] for row in rows]) <= 1e-12
            for rows in (herbal, zodiac)
        ):
            continue
        for text_feature in text_features:
            correlations = {}
            for domain, rows in domains.items():
                correlations[domain] = (
                    partial_spearman(rows, visual_feature, text_feature)
                    if controlled else spearman(
                        [row["visual"][visual_feature] for row in rows],
                        [row["text"][text_feature] for row in rows],
                    )
                )
            same_direction = correlations["herbal"] * correlations["zodiac"] > 0
            score = min(abs(value) for value in correlations.values())
            if not same_direction:
                score = -score
            ranked.append((
                score,
                visual_feature,
                text_feature,
                correlations,
            ))
    score, visual_feature, text_feature, correlations = max(ranked)
    return {
        "visual_feature": visual_feature,
        "text_feature": text_feature,
        "correlations": correlations,
        "same_direction_min_abs_spearman": score,
        "controlled_for": (
            ["folio_number_rank", "text_sample_size_rank"]
            if controlled else []
        ),
        "candidate_count": len(ranked),
    }


def selected_pair_robustness(
    domains: dict[str, list[dict]],
    visual_feature: str,
    text_feature: str,
) -> dict:
    result = {}
    for domain, rows in domains.items():
        raw = []
        controlled = []
        for index in range(len(rows)):
            subset = rows[:index] + rows[index + 1:]
            raw.append(spearman(
                [row["visual"][visual_feature] for row in subset],
                [row["text"][text_feature] for row in subset],
            ))
            controlled.append(partial_spearman(
                subset, visual_feature, text_feature
            ))
        result[domain] = {
            "leave_one_out_raw_range": [min(raw), max(raw)],
            "leave_one_out_raw_mean": mean(raw),
            "leave_one_out_controlled_range": [
                min(controlled),
                max(controlled),
            ],
            "leave_one_out_controlled_mean": mean(controlled),
        }

    zodiac_groups: dict[str, list[dict]] = defaultdict(list)
    for row in domains["zodiac"]:
        zodiac_groups[row["source_group"]].append(row)
    group_x = [
        mean(row["visual"][visual_feature] for row in rows)
        for rows in zodiac_groups.values()
    ]
    group_y = [
        mean(row["text"][text_feature] for row in rows)
        for rows in zodiac_groups.values()
    ]
    result["zodiac_source_canvas_aggregation"] = {
        "groups": len(zodiac_groups),
        "spearman": spearman(group_x, group_y),
        "group_sizes": {
            source: len(rows) for source, rows in zodiac_groups.items()
        },
    }
    return result


def synthetic_transfer_control(pages: dict[str, list[dict]]) -> dict:
    rng = random.Random(SEED + 700)
    synthetic = {}
    for domain, rows in pages.items():
        x = np.asarray([
            row["visual"]["pixel.pigment_total"] for row in rows
        ])
        scale = max(float(np.std(x)), 1e-9)
        noise = np.asarray([
            rng.gauss(0, 0.15 * scale) for _ in rows
        ])
        synthetic[domain] = [
            {
                **row,
                "text": {"synthetic_pigment_channel": float(value)},
            }
            for row, value in zip(rows, x + noise)
        ]
    directions = {
        "herbal_to_zodiac": transfer_direction(
            synthetic["herbal"], synthetic["zodiac"]
        ),
        "zodiac_to_herbal": transfer_direction(
            synthetic["zodiac"], synthetic["herbal"]
        ),
    }
    passed = all(
        row["aligned_target_spearman"] > 0.50
        for row in directions.values()
    )
    return {
        "true_visual_feature": "pixel.pigment_total",
        "noise_sd_fraction": 0.15,
        "directions": directions,
        "passed": passed,
    }


def run_cross_domain_gate(
    pages: dict[str, list[dict]],
    null_count: int,
    progress: bool,
) -> dict:
    pigment_features = sorted(
        feature
        for feature in pages["herbal"][0]["visual"]
        if feature.startswith("pixel.")
    )
    all_features = sorted(pages["herbal"][0]["visual"])
    observed_directions = {
        "herbal_to_zodiac": transfer_direction(
            pages["herbal"], pages["zodiac"], pigment_features
        ),
        "zodiac_to_herbal": transfer_direction(
            pages["zodiac"], pages["herbal"], pigment_features
        ),
    }
    raw_common = common_pair(pages, pigment_features, controlled=False)
    controlled_common = common_pair(pages, pigment_features, controlled=True)
    exploratory_all_features = {
        "herbal_to_zodiac": transfer_direction(
            pages["herbal"], pages["zodiac"], all_features
        ),
        "zodiac_to_herbal": transfer_direction(
            pages["zodiac"], pages["herbal"], all_features
        ),
    }
    rng = random.Random(SEED + 800)
    nulls = []
    for replicate in range(null_count):
        permuted = {
            domain: permute_text_records(rows, rng)
            for domain, rows in pages.items()
        }
        directions = {
            "herbal_to_zodiac": transfer_direction(
                permuted["herbal"], permuted["zodiac"], pigment_features
            ),
            "zodiac_to_herbal": transfer_direction(
                permuted["zodiac"], permuted["herbal"], pigment_features
            ),
        }
        null_raw_common = common_pair(
            permuted, pigment_features, controlled=False
        )
        null_controlled_common = common_pair(
            permuted, pigment_features, controlled=True
        )
        nulls.append({
            "replicate": replicate + 1,
            "directions": directions,
            "conjunction": min(
                row["aligned_target_spearman"]
                for row in directions.values()
            ),
            "raw_common_score": (
                null_raw_common["same_direction_min_abs_spearman"]
            ),
            "controlled_common_score": (
                null_controlled_common["same_direction_min_abs_spearman"]
            ),
        })
        if progress and (replicate + 1) % 250 == 0:
            print(
                f"transfer null {replicate + 1}/{null_count}",
                flush=True,
            )
    for name, row in observed_directions.items():
        row["p_aligned_target"] = empirical_upper_p(
            row["aligned_target_spearman"],
            [
                null["directions"][name]["aligned_target_spearman"]
                for null in nulls
            ],
        )
    conjunction = min(
        row["aligned_target_spearman"]
        for row in observed_directions.values()
    )
    raw_common["p_selection_adjusted"] = empirical_upper_p(
        raw_common["same_direction_min_abs_spearman"],
        [row["raw_common_score"] for row in nulls],
    )
    controlled_common["p_selection_adjusted"] = empirical_upper_p(
        controlled_common["same_direction_min_abs_spearman"],
        [row["controlled_common_score"] for row in nulls],
    )
    controlled_common["robustness"] = selected_pair_robustness(
        pages,
        controlled_common["visual_feature"],
        controlled_common["text_feature"],
    )
    return {
        "design": {
            "domains": {
                domain: {
                    "records": len(rows),
                    "folios": [row["folio"] for row in rows],
                    "words": sum(row["n_words"] for row in rows),
                }
                for domain, rows in pages.items()
            },
            "primary_visual_features": pigment_features,
            "exploratory_visual_features": all_features,
            "text_features": sorted(pages["herbal"][0]["text"]),
            "selection": (
                "maximum absolute Spearman in source domain; frozen feature "
                "pair and direction scored in target domain"
            ),
            "null_unit": (
                "complete page text-statistic vector permuted within domain; "
                "feature selection repeated"
            ),
            "raw_ink_features_excluded": True,
            "primary_gate": (
                "same pigment/text pair must have the same direction in both "
                "domains after rank-residualizing folio order and text sample "
                "size; pair selection is repeated inside every null"
            ),
        },
        "observed": {
            "directional_pigment_transfer": observed_directions,
            "raw_common_pair": raw_common,
            "controlled_common_pair": controlled_common,
            "exploratory_all_features": exploratory_all_features,
        },
        "conjunction_aligned_spearman": conjunction,
        "p_conjunction": empirical_upper_p(
            conjunction, [row["conjunction"] for row in nulls]
        ),
        "synthetic_control": synthetic_transfer_control(pages),
        "nulls": nulls,
    }


def run_node_gate(args: argparse.Namespace) -> dict:
    sequences = load_node_sequences()
    candidates = node_candidates()
    folds = run_node_search(sequences, candidates)
    observed = node_summary(folds)
    if args.progress:
        print(
            "node observed: "
            f"gain={observed['mean_test_gain_bits_per_node']:+.4f} "
            f"positive={observed['positive_test_folds']}/12",
            flush=True,
        )
    synthetic = synthetic_node_control(sequences)
    if args.progress:
        row = synthetic["summary"]
        print(
            "node synthetic: "
            f"gain={row['mean_test_gain_bits_per_node']:+.4f} "
            f"selection={row['correct_selection_rate']:.2f} "
            f"pass={row['passed']}",
            flush=True,
        )

    rng = random.Random(SEED + 300)
    nulls = []
    for replicate in range(args.node_nulls):
        offsets = random_rotation_offsets(sequences, rng)
        rotated = rotate_visual_sequences(sequences, offsets)
        null_folds = run_node_search(rotated, candidates)
        summary = node_summary(null_folds)
        nulls.append({
            "replicate": replicate + 1,
            "offsets": {"|".join(key): value for key, value in offsets.items()},
            "summary": summary,
            "folds": null_folds,
        })
        if args.progress and (
            replicate < 3 or (replicate + 1) % 10 == 0
        ):
            print(
                f"node null {replicate + 1}/{args.node_nulls}: "
                f"gain={summary['mean_test_gain_bits_per_node']:+.4f}",
                flush=True,
            )

    phase = phase_scenarios(sequences, candidates)

    calibrated, semantic = semantic_calibration_sequences(sequences)
    semantic_folds = run_semantic_search(calibrated, semantic)
    semantic_observed = semantic_summary(semantic_folds)
    semantic_nulls = []
    semantic_rng = random.Random(SEED + 400)
    for replicate in range(args.semantic_nulls):
        offsets = random_rotation_offsets(calibrated, semantic_rng)
        rotated = rotate_visual_sequences(calibrated, offsets)
        null_folds = run_semantic_search(rotated, semantic)
        semantic_nulls.append({
            "replicate": replicate + 1,
            "summary": semantic_summary(null_folds),
        })
        if args.progress and (replicate + 1) % 20 == 0:
            print(
                f"semantic null {replicate + 1}/{args.semantic_nulls}",
                flush=True,
            )
    semantic_observed["p_gain"] = empirical_upper_p(
        semantic_observed["mean_test_gain_bits_per_node"],
        [
            row["summary"]["mean_test_gain_bits_per_node"]
            for row in semantic_nulls
        ],
    )

    null_gains = [
        row["summary"]["mean_test_gain_bits_per_node"] for row in nulls
    ]
    return {
        "design": {
            "folios": sorted({key[0] for key in sequences}),
            "sequences": [
                {
                    "folio": key[0],
                    "tier": key[1],
                    "nodes": len(sequence["words"]),
                    "alignment_confidence": sequence["alignment_confidence"],
                }
                for key, sequence in sorted(sequences.items())
            ],
            "n_nodes": sum(
                len(sequence["words"]) for sequence in sequences.values()
            ),
            "text_features": list(stateful.FEATURES),
            "text_operations": list(TEXT_OPERATIONS),
            "text_moduli": list(TEXT_MODULI),
            "visual_features": list(NODE_PIXEL_FEATURES),
            "visual_channels": sorted(
                next(iter(sequences.values()))["channels"]
            ),
            "candidate_count": len(candidates),
            "null_unit": (
                "nonzero cyclic visual rotation within folio and ring tier; "
                "complete nested candidate selection repeated"
            ),
            "label_leakage_guard": (
                "small core crop; glyph-sized connected components removed; "
                "raw ink/edge/entropy excluded from primary candidates"
            ),
        },
        "observed": {
            "summary": observed,
            "folds": folds,
            "p_gain": empirical_upper_p(
                observed["mean_test_gain_bits_per_node"], null_gains
            ),
        },
        "phase_sensitivity": phase,
        "synthetic_control": synthetic,
        "semantic_pixel_calibration": {
            "design": {
                "folios": sorted({key[0] for key in calibrated}),
                "attributes": sorted(semantic),
                "null_unit": (
                    "nonzero cyclic visual rotation within anchored ring"
                ),
            },
            "observed": {
                "summary": semantic_observed,
                "folds": semantic_folds,
            },
            "nulls": semantic_nulls,
        },
        "label_leakage_probe": label_leakage_probe(
            sequences, args.leakage_nulls
        ),
        "targeted_green_gallows_localization": targeted_green_gallows_gate(
            sequences, args.leakage_nulls
        ),
        "nulls": nulls,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-nulls", type=int, default=60)
    parser.add_argument("--semantic-nulls", type=int, default=60)
    parser.add_argument("--leakage-nulls", type=int, default=300)
    parser.add_argument("--transfer-nulls", type=int, default=1000)
    parser.add_argument("--skip-node", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.node_nulls,
        args.semantic_nulls,
        args.leakage_nulls,
        args.transfer_nulls,
    ) < 1:
        raise ValueError("null counts must be positive")
    report = {
        "experiment": "multimodal_graph_transfer_gate",
        "built": "2026-07-23",
        "seed": SEED,
        "node_gate": None,
        "cross_domain_gate": None,
    }
    if not args.skip_node:
        report["node_gate"] = run_node_gate(args)
    if not args.skip_transfer:
        pages = load_page_records()
        report["cross_domain_gate"] = run_cross_domain_gate(
            pages, args.transfer_nulls, args.progress
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
