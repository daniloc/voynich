#!/usr/bin/env python3
"""
Held-out Voynich gate for the table-constrained permutation decoder.

The successful Naibbe decoder knows each glyph's emission role and table, but
not its letter value.  Voynich has no published table labels, so this experiment
tests three fixed, observable channel hypotheses:

``exact_inventory``
    A strict known-family hypothesis.  A segmented Voynich surface is assigned
    only when it exactly matches a published Naibbe role/table surface.

``morph_nearest_k12`` and ``morph_nearest_k23``
    Per role, the 6*K most frequent fit-only surfaces are balanced across six
    channels by edit distance to the published table surface inventories.

``morph_cluster_k12``
    Per role, the 72 most frequent fit-only surfaces are balanced into six
    channels using only their character morphology.

Every channel is an injective subset of a 23-letter permutation.  The
context-rank initializer and deterministic LM refinement import the frozen
parameters from ``naibbe_permutation_decoder.py``.  Channel inventories and
keys are fit on complete fit quires, lambda is chosen on complete validation
quires, and the key is frozen for complete test quires.  Missing and unseen
symbols are hard breaks.  No n-gram crosses a physical line, exclusion, or
quire boundary.

Each hypothesis is also run blind on the untouched Naibbe ciphertext, using
only structural role/table labels and never the published letter values or
plaintext.  A Voynich hypothesis passes only if its Naibbe control activates in
all four folds, its Voynich validation activates in all four folds, and its
weakest Voynich held-out residual clears the weakest Naibbe held-out residual.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
FOLLOWUPS = HERE.parent / "07_followups"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FOLLOWUPS))

from naibbe_permutation_decoder import (  # noqa: E402
    ALPHABET,
    BLOCK_PASSES,
    CONTEXT_WEIGHT,
    DEV_SEEDS,
    NAIBBE_ROOT,
    SMOOTHING,
    SWAP_PASSES,
    TABLES,
    TetragramLM,
    hungarian_max,
    load_control_segments,
    load_structural_inventory,
)
from naibbe_style_attack import (  # noqa: E402
    SegmentModel,
    calibrate,
    structural_counts,
)


ROOT = HERE.parents[1]
CORPUS = ROOT / "data" / "corpus" / "corpus.json"
NAIBBE_CIPHER = (
    NAIBBE_ROOT / "encrypted" / "nathist_output_ciphertext.txt"
)
NAIBBE_RESPACED_CIPHER = (
    NAIBBE_ROOT / "encrypted" / "nathist_output_ciphertext_respaced.txt"
)
TABLE_CSV = NAIBBE_ROOT / "references" / "naibbe_tables.csv"
ROLE_TO_FULL = {"U": "unigram", "L": "prefix", "R": "suffix"}
FULL_TO_ROLE = {value: key for key, value in ROLE_TO_FULL.items()}
LAMBDAS = (0.25, 0.5, 0.75, 1.0)
BASELINE_ALPHA = 0.5
N_FOLDS = 4
HYPOTHESES = (
    "exact_inventory",
    "morph_nearest_k12",
    "morph_nearest_k23",
    "morph_cluster_k12",
)


@dataclass(frozen=True)
class SourceLine:
    block: str
    symbols: tuple[str, ...]
    slots: tuple[str, ...]


@dataclass(frozen=True)
class MappedLine:
    block: str
    channels: tuple[int, ...]
    symbols: tuple[int, ...]
    slots: tuple[str, ...]


@dataclass
class FitResult:
    key: list[list[int]]
    init_score: float
    final_score: float
    block_updates: int
    swap_updates: int


@dataclass
class FoldResult:
    fold: int
    fit_blocks: int
    validation_blocks: int
    test_blocks: int
    mapped_symbols: int
    fit_coverage: float
    validation_coverage: float
    test_coverage: float
    lambda_: float
    validation_gain: float
    validation_penalty: float
    validation_penalized: float
    active: bool
    test_gain_raw: float
    test_gain_reported: float
    test_lm: float
    test_observations: int
    init_lm: float
    fit_lm: float
    block_updates: int
    swap_updates: int
    sample: str


@dataclass
class FlatTraining:
    channels: list[int]
    symbols: list[int]
    windows: list[tuple[int, int, int, int]]
    line_ranges: list[tuple[int, int]]


def locus_type(locus: str) -> str:
    match = re.search(r"[A-Za-z]", locus)
    return match.group(0).upper() if match else "?"


def position_bucket(index: int, length: int) -> str:
    if index == 0:
        return "first"
    if index == 1:
        return "second"
    if index == length - 2:
        return "penult"
    if index == length - 1:
        return "last"
    return "interior"


def block_folds(blocks: Sequence[str]) -> dict[str, int]:
    return {
        block: index % N_FOLDS
        for index, block in enumerate(sorted(blocks))
    }


def valid_word_runs(
    words: Sequence[str | None],
) -> Iterable[list[tuple[int, str]]]:
    current: list[tuple[int, str]] = []
    for index, word in enumerate(words):
        if word is None:
            if current:
                yield current
                current = []
        else:
            current.append((index, word))
    if current:
        yield current


def load_voynich_raw() -> tuple[list[tuple[str, tuple[str | None, ...]]], list[str]]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    raw = []
    for folio, source_lines in corpus["folios"].items():
        meta = corpus["meta"].get(folio, {})
        block = str(meta.get("Q", "?"))
        for source in source_lines:
            if locus_type(source["locus"]) != "P":
                continue
            words = tuple(
                None
                if "?" in word or not word.isalpha() or len(word) < 2
                else word
                for word in source["words"]
            )
            if any(word is not None for word in words):
                raw.append((block, words))
    return raw, sorted({block for block, _words in raw})


def emit_voynich(
    raw: Sequence[tuple[str, tuple[str | None, ...]]],
    model: SegmentModel,
    fit_blocks: set[str],
) -> list[SourceLine]:
    training_words = [
        word
        for block, words in raw
        if block in fit_blocks
        for word in words
        if word is not None
    ]
    counts = structural_counts(training_words)
    emitted: list[SourceLine] = []
    for block, words in raw:
        length = len(words)
        for run in valid_word_runs(words):
            symbols: list[str] = []
            slots: list[str] = []
            for index, word in run:
                emission = model.segment(word, counts)
                position = position_bucket(index, length)
                symbols.extend(emission)
                slots.extend(
                    f"{symbol.split(':', 1)[0]}@{position}"
                    for symbol in emission
                )
            if symbols:
                emitted.append(
                    SourceLine(block, tuple(symbols), tuple(slots))
                )
    return emitted


def parse_naibbe_token(token: str, inventory) -> tuple[str, ...]:
    if token in inventory.unigram_surfaces:
        return ("U:" + token,)
    candidates = [
        (token[:split], token[split:])
        for split in range(1, len(token))
        if (
            token[:split] in inventory.prefix_surfaces
            and token[split:] in inventory.suffix_surfaces
        )
    ]
    if not candidates:
        return ()
    prefix, suffix = max(candidates, key=lambda pair: len(pair[0]))
    return ("L:" + prefix, "R:" + suffix)


def load_naibbe_source(inventory, n_blocks: int = 16) -> tuple[list[SourceLine], list[str]]:
    token_lines = [
        re.findall(r"[a-z]+", line.lower())
        for line in NAIBBE_CIPHER.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
    ]
    token_lines = [line for line in token_lines if line]
    total = sum(map(len, token_lines))
    boundaries = [total * index / n_blocks for index in range(1, n_blocks)]
    emitted: list[SourceLine] = []
    cumulative = 0
    block_index = 0
    for tokens in token_lines:
        while (
            block_index < n_blocks - 1
            and cumulative >= boundaries[block_index]
        ):
            block_index += 1
        block = f"N{block_index:02d}"
        symbols: list[str] = []
        slots: list[str] = []

        def flush() -> None:
            if symbols:
                emitted.append(
                    SourceLine(block, tuple(symbols), tuple(slots))
                )
                symbols.clear()
                slots.clear()

        for index, token in enumerate(tokens):
            parsed = parse_naibbe_token(token, inventory)
            if not parsed:
                flush()
                continue
            position = position_bucket(index, len(tokens))
            symbols.extend(parsed)
            slots.extend(
                f"{symbol.split(':', 1)[0]}@{position}"
                for symbol in parsed
            )
        flush()
        cumulative += len(tokens)
    return emitted, [f"N{index:02d}" for index in range(n_blocks)]


def calibration_segmenter() -> tuple[SegmentModel, dict]:
    lines = [
        re.findall(r"[a-z]+", line.lower())
        for line in NAIBBE_RESPACED_CIPHER.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
        if re.search(r"[a-z]", line.lower())
    ]
    return calibrate(lines, TABLE_CSV)


def fit_symbol_counts(
    lines: Sequence[SourceLine], fit_blocks: set[str]
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {
        role: Counter() for role in ROLE_TO_FULL
    }
    for line in lines:
        if line.block not in fit_blocks:
            continue
        for symbol in line.symbols:
            role = symbol.split(":", 1)[0]
            counts[role][symbol] += 1
    return counts


def inventory_table_surfaces(inventory) -> dict[str, dict[str, tuple[str, ...]]]:
    result: dict[str, dict[str, tuple[str, ...]]] = {
        role: {} for role in ROLE_TO_FULL
    }
    for (full_role, table), surfaces in zip(
        inventory.blocks, inventory.glyphs
    ):
        result[FULL_TO_ROLE[full_role]][table] = surfaces
    return result


def exact_inventory_map(
    counts: dict[str, Counter[str]], inventory
) -> dict[str, tuple[int, int]]:
    mapping: dict[str, tuple[int, int]] = {}
    for role, role_counts in counts.items():
        full_role = ROLE_TO_FULL[role]
        for symbol in role_counts:
            surface = symbol.split(":", 1)[1]
            locations = inventory.glyph_locations.get(
                (full_role, surface), ()
            )
            if locations:
                mapping[symbol] = locations[0]
    return mapping


def edit_distance(first: str, second: str) -> int:
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, 1):
        current = [i]
        for j, right in enumerate(second, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def normalized_edit(first: str, second: str) -> float:
    return edit_distance(first, second) / max(len(first), len(second), 1)


def balanced_assignment(
    names: Sequence[str],
    group_scores: Sequence[Sequence[float]],
    capacity: int,
) -> list[int]:
    size = 6 * capacity
    weights = [
        [
            group_scores[row][column // capacity]
            for column in range(size)
        ]
        for row in range(len(names))
    ]
    weights.extend(
        [[0.0 for _column in range(size)] for _row in range(size - len(names))]
    )
    assignment = hungarian_max(weights)
    return [column // capacity for column in assignment[: len(names)]]


def morph_nearest_map(
    counts: dict[str, Counter[str]],
    inventory,
    capacity: int,
) -> dict[str, tuple[int, int]]:
    tables = inventory_table_surfaces(inventory)
    mapping: dict[str, tuple[int, int]] = {}
    for role, role_counts in counts.items():
        names = [
            symbol
            for symbol, _count in role_counts.most_common(6 * capacity)
        ]
        if len(names) < 6:
            continue
        scores = []
        for symbol in names:
            surface = symbol.split(":", 1)[1]
            scores.append(
                [
                    -min(
                        normalized_edit(surface, reference)
                        for reference in tables[role][table]
                    )
                    for table in TABLES
                ]
            )
        groups = balanced_assignment(names, scores, capacity)
        grouped: dict[int, list[str]] = defaultdict(list)
        for symbol, group in zip(names, groups):
            grouped[group].append(symbol)
        for group, symbols in grouped.items():
            channel = inventory.block_index[
                (ROLE_TO_FULL[role], TABLES[group])
            ]
            for symbol_id, symbol in enumerate(sorted(symbols)):
                mapping[symbol] = (channel, symbol_id)
    return mapping


def morphology_features(names: Sequence[str]) -> list[list[float]]:
    characters = sorted(
        {
            char
            for symbol in names
            for char in symbol.split(":", 1)[1]
        }
    )
    features = []
    for symbol in names:
        surface = symbol.split(":", 1)[1]
        length = max(1, len(surface))
        row = [
            len(surface) / 12.0,
            len(set(surface)) / length,
            surface.count("o") / length,
            surface.count("e") / length,
        ]
        row.extend(surface.count(char) / length for char in characters)
        row.extend(
            1.0 if surface.startswith(char) else 0.0
            for char in characters
        )
        row.extend(
            1.0 if surface.endswith(char) else 0.0
            for char in characters
        )
        features.append(row)
    if not features:
        return features
    width = len(features[0])
    means = [
        sum(row[column] for row in features) / len(features)
        for column in range(width)
    ]
    scales = [
        math.sqrt(
            sum(
                (row[column] - means[column]) ** 2
                for row in features
            )
            / len(features)
        )
        + 1e-6
        for column in range(width)
    ]
    return [
        [
            (row[column] - means[column]) / scales[column]
            for column in range(width)
        ]
        for row in features
    ]


def squared_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return sum((left - right) ** 2 for left, right in zip(first, second))


def initial_centroids(features: Sequence[Sequence[float]]) -> list[list[float]]:
    selected = [0]
    while len(selected) < 6:
        candidate = max(
            (
                index
                for index in range(len(features))
                if index not in selected
            ),
            key=lambda index: min(
                squared_distance(features[index], features[chosen])
                for chosen in selected
            ),
        )
        selected.append(candidate)
    return [list(features[index]) for index in selected]


def morph_cluster_map(
    counts: dict[str, Counter[str]],
    inventory,
    capacity: int,
) -> dict[str, tuple[int, int]]:
    mapping: dict[str, tuple[int, int]] = {}
    for role, role_counts in counts.items():
        names = [
            symbol
            for symbol, _count in role_counts.most_common(6 * capacity)
        ]
        if len(names) < 6:
            continue
        features = morphology_features(names)
        centroids = initial_centroids(features)
        groups = [0] * len(names)
        for _iteration in range(12):
            scores = [
                [
                    -squared_distance(feature, centroid)
                    for centroid in centroids
                ]
                for feature in features
            ]
            updated = balanced_assignment(names, scores, capacity)
            if updated == groups and _iteration:
                break
            groups = updated
            for group in range(6):
                members = [
                    features[index]
                    for index, value in enumerate(groups)
                    if value == group
                ]
                centroids[group] = [
                    sum(row[column] for row in members) / len(members)
                    for column in range(len(features[0]))
                ]
        grouped: dict[int, list[str]] = defaultdict(list)
        for symbol, group in zip(names, groups):
            grouped[group].append(symbol)
        for group, symbols in grouped.items():
            channel = inventory.block_index[
                (ROLE_TO_FULL[role], TABLES[group])
            ]
            for symbol_id, symbol in enumerate(sorted(symbols)):
                mapping[symbol] = (channel, symbol_id)
    return mapping


def build_mapping(
    hypothesis: str,
    lines: Sequence[SourceLine],
    fit_blocks: set[str],
    inventory,
) -> dict[str, tuple[int, int]]:
    counts = fit_symbol_counts(lines, fit_blocks)
    if hypothesis == "exact_inventory":
        return exact_inventory_map(counts, inventory)
    if hypothesis.startswith("morph_nearest_k"):
        capacity = int(hypothesis.rsplit("k", 1)[1])
        return morph_nearest_map(counts, inventory, capacity)
    if hypothesis.startswith("morph_cluster_k"):
        capacity = int(hypothesis.rsplit("k", 1)[1])
        return morph_cluster_map(counts, inventory, capacity)
    raise ValueError(f"Unknown hypothesis {hypothesis!r}")


def map_source_lines(
    lines: Sequence[SourceLine],
    mapping: dict[str, tuple[int, int]],
) -> list[MappedLine]:
    result: list[MappedLine] = []
    for line in lines:
        channels: list[int] = []
        symbols: list[int] = []
        slots: list[str] = []

        def flush() -> None:
            if channels:
                result.append(
                    MappedLine(
                        line.block,
                        tuple(channels),
                        tuple(symbols),
                        tuple(slots),
                    )
                )
                channels.clear()
                symbols.clear()
                slots.clear()

        for source_symbol, slot in zip(line.symbols, line.slots):
            location = mapping.get(source_symbol)
            if location is None:
                flush()
                continue
            channel, symbol = location
            channels.append(channel)
            symbols.append(symbol)
            slots.append(slot)
        flush()
    return result


def mapping_coverage(
    source: Sequence[SourceLine],
    blocks: set[str],
    mapping: dict[str, tuple[int, int]],
) -> float:
    known = total = 0
    for line in source:
        if line.block not in blocks:
            continue
        total += len(line.symbols)
        known += sum(symbol in mapping for symbol in line.symbols)
    return known / max(total, 1)


def flatten_training(
    lines: Sequence[MappedLine], fit_blocks: set[str]
) -> FlatTraining:
    channels: list[int] = []
    symbols: list[int] = []
    windows: list[tuple[int, int, int, int]] = []
    line_ranges: list[tuple[int, int]] = []
    for line in lines:
        if line.block not in fit_blocks:
            continue
        start = len(channels)
        channels.extend(line.channels)
        symbols.extend(line.symbols)
        end = len(channels)
        line_ranges.append((start, end))
        windows.extend(
            (position, position + 1, position + 2, position + 3)
            for position in range(start, end - 3)
        )
    return FlatTraining(channels, symbols, windows, line_ranges)


def context_initializer(
    training: FlatTraining, lm: TetragramLM, block_count: int
) -> list[list[int]]:
    counts = [[0 for _ in ALPHABET] for _ in range(block_count)]
    for channel, symbol in zip(training.channels, training.symbols):
        counts[channel][symbol] += 1
    ranks: list[list[int]] = []
    for row in counts:
        order = sorted(range(len(ALPHABET)), key=lambda i: (-row[i], i))
        rank = [0] * len(ALPHABET)
        for position, symbol in enumerate(order):
            rank[symbol] = position
        ranks.append(rank)

    left_counts = [
        [[0 for _ in ALPHABET] for _ in ALPHABET]
        for _ in range(block_count)
    ]
    right_counts = [
        [[0 for _ in ALPHABET] for _ in ALPHABET]
        for _ in range(block_count)
    ]
    for start, end in training.line_ranges:
        for position in range(start, end):
            channel = training.channels[position]
            symbol = training.symbols[position]
            if position > start:
                left_counts[channel][symbol][
                    ranks[training.channels[position - 1]][
                        training.symbols[position - 1]
                    ]
                ] += 1
            if position + 1 < end:
                right_counts[channel][symbol][
                    ranks[training.channels[position + 1]][
                        training.symbols[position + 1]
                    ]
                ] += 1

    key = []
    for channel in range(block_count):
        weights = []
        for symbol in range(len(ALPHABET)):
            row = []
            for letter in range(len(ALPHABET)):
                frequency_score = counts[channel][symbol] * math.log(
                    lm.letter_prob[letter]
                )
                context_score = sum(
                    left_counts[channel][symbol][rank]
                    * lm.rank_left_logp[letter][rank]
                    + right_counts[channel][symbol][rank]
                    * lm.rank_right_logp[letter][rank]
                    for rank in range(len(ALPHABET))
                )
                row.append(
                    frequency_score + CONTEXT_WEIGHT * context_score
                )
            weights.append(row)
        key.append(hungarian_max(weights))
    return key


def window_score(
    decoded: Sequence[int],
    windows: Sequence[tuple[int, int, int, int]],
    lm: TetragramLM,
    selected: Iterable[int] | None = None,
) -> float:
    indices = range(len(windows)) if selected is None else selected
    return sum(
        lm.logp4(
            decoded[windows[index][0]],
            decoded[windows[index][1]],
            decoded[windows[index][2]],
            decoded[windows[index][3]],
        )
        for index in indices
    )


def fit_permutations(
    training: FlatTraining,
    lm: TetragramLM,
    initial_key: list[list[int]],
) -> FitResult:
    key = [row[:] for row in initial_key]
    decoded = [
        key[channel][symbol]
        for channel, symbol in zip(training.channels, training.symbols)
    ]
    block_count = len(key)
    positions = [
        [[] for _ in ALPHABET]
        for _ in range(block_count)
    ]
    for position, (channel, symbol) in enumerate(
        zip(training.channels, training.symbols)
    ):
        positions[channel][symbol].append(position)
    affected = [
        [set() for _ in ALPHABET]
        for _ in range(block_count)
    ]
    for window_id, window in enumerate(training.windows):
        for position in window:
            affected[training.channels[position]][
                training.symbols[position]
            ].add(window_id)
    affected_rows = [
        [tuple(sorted(row)) for row in block]
        for block in affected
    ]

    def replace_channel(channel: int, row: Sequence[int]) -> None:
        for symbol in range(len(ALPHABET)):
            for position in positions[channel][symbol]:
                decoded[position] = row[symbol]

    score = window_score(decoded, training.windows, lm)
    initial_score = score
    block_updates = 0
    swap_updates = 0
    for _pass in range(BLOCK_PASSES):
        changed = False
        for channel in range(block_count):
            weights = []
            for symbol in range(len(ALPHABET)):
                rows = affected_rows[channel][symbol]
                baseline = window_score(
                    decoded, training.windows, lm, rows
                )
                original = key[channel][symbol]
                candidate_scores = []
                for letter in range(len(ALPHABET)):
                    if letter == original:
                        candidate_scores.append(0.0)
                        continue
                    for position in positions[channel][symbol]:
                        decoded[position] = letter
                    candidate_scores.append(
                        window_score(
                            decoded, training.windows, lm, rows
                        )
                        - baseline
                    )
                    for position in positions[channel][symbol]:
                        decoded[position] = original
                weights.append(candidate_scores)
            proposal = hungarian_max(weights)
            if proposal == key[channel]:
                continue
            old_row = key[channel][:]
            replace_channel(channel, proposal)
            candidate = window_score(decoded, training.windows, lm)
            if candidate > score + 1e-8:
                key[channel] = proposal
                score = candidate
                block_updates += 1
                changed = True
            else:
                replace_channel(channel, old_row)

        for _swap_pass in range(SWAP_PASSES):
            swap_changed = False
            for channel in range(block_count):
                best_delta = 1e-8
                best_pair: tuple[int, int] | None = None
                for first in range(len(ALPHABET) - 1):
                    for second in range(first + 1, len(ALPHABET)):
                        rows = tuple(
                            sorted(
                                affected[channel][first]
                                | affected[channel][second]
                            )
                        )
                        if not rows:
                            continue
                        baseline = window_score(
                            decoded, training.windows, lm, rows
                        )
                        first_value = key[channel][first]
                        second_value = key[channel][second]
                        for position in positions[channel][first]:
                            decoded[position] = second_value
                        for position in positions[channel][second]:
                            decoded[position] = first_value
                        delta = (
                            window_score(
                                decoded, training.windows, lm, rows
                            )
                            - baseline
                        )
                        for position in positions[channel][first]:
                            decoded[position] = first_value
                        for position in positions[channel][second]:
                            decoded[position] = second_value
                        if delta > best_delta:
                            best_delta = delta
                            best_pair = (first, second)
                if best_pair is not None:
                    first, second = best_pair
                    key[channel][first], key[channel][second] = (
                        key[channel][second],
                        key[channel][first],
                    )
                    replace_channel(channel, key[channel])
                    score += best_delta
                    swap_updates += 1
                    swap_changed = True
                    changed = True
            if not swap_changed:
                break
        if not changed:
            break
    score = window_score(decoded, training.windows, lm)
    return FitResult(
        key=key,
        init_score=initial_score,
        final_score=score,
        block_updates=block_updates,
        swap_updates=swap_updates,
    )


def fit_baseline(
    lines: Sequence[MappedLine],
    fit_blocks: set[str],
    key: Sequence[Sequence[int]],
) -> tuple[Counter[tuple[str, int]], Counter[str]]:
    counts: Counter[tuple[str, int]] = Counter()
    totals: Counter[str] = Counter()
    for line in lines:
        if line.block not in fit_blocks:
            continue
        for channel, symbol, slot in zip(
            line.channels, line.symbols, line.slots
        ):
            character = key[channel][symbol]
            counts[(slot, character)] += 1
            totals[slot] += 1
    return counts, totals


def residual_score(
    lines: Sequence[MappedLine],
    blocks: set[str],
    fit_blocks: set[str],
    key: Sequence[Sequence[int]],
    lm: TetragramLM,
    lambda_: float,
) -> tuple[float, float, int]:
    baseline, totals = fit_baseline(lines, fit_blocks, key)
    residual = 0.0
    lm_score = 0.0
    observations = 0
    width = len(ALPHABET)
    for line in lines:
        if line.block not in blocks or len(line.symbols) < 4:
            continue
        decoded = [
            key[channel][symbol]
            for channel, symbol in zip(line.channels, line.symbols)
        ]
        for index in range(3, len(decoded)):
            character = decoded[index]
            slot = line.slots[index]
            base_probability = (
                baseline[(slot, character)] + BASELINE_ALPHA
            ) / (totals[slot] + BASELINE_ALPHA * width)
            language_probability = math.exp(
                lm.logp4(
                    decoded[index - 3],
                    decoded[index - 2],
                    decoded[index - 1],
                    character,
                )
            )
            mixture = (
                (1.0 - lambda_) * base_probability
                + lambda_ * language_probability
            )
            residual += math.log2(mixture / base_probability)
            lm_score += math.log2(language_probability)
            observations += 1
    if not observations:
        return 0.0, float("-inf"), 0
    return residual / observations, lm_score / observations, observations


def render_sample(
    lines: Sequence[MappedLine],
    blocks: set[str],
    key: Sequence[Sequence[int]],
    limit: int = 220,
) -> str:
    pieces = []
    for line in lines:
        if line.block not in blocks:
            continue
        pieces.append(
            "".join(
                ALPHABET[key[channel][symbol]]
                for channel, symbol in zip(
                    line.channels, line.symbols
                )
            )
        )
        if sum(map(len, pieces)) >= limit:
            break
    return " / ".join(pieces)[:limit]


def evaluate_fold(
    hypothesis: str,
    fold: int,
    source_lines: Sequence[SourceLine],
    blocks: Sequence[str],
    inventory,
    lm: TetragramLM,
) -> FoldResult:
    folds = block_folds(blocks)
    test = {block for block in blocks if folds[block] == fold}
    validation = {
        block
        for block in blocks
        if folds[block] == (fold + 1) % N_FOLDS
    }
    fit = set(blocks) - test - validation
    mapping = build_mapping(hypothesis, source_lines, fit, inventory)
    mapped = map_source_lines(source_lines, mapping)
    training = flatten_training(mapped, fit)
    initial_key = context_initializer(
        training, lm, len(inventory.blocks)
    )
    result = fit_permutations(training, lm, initial_key)
    validation_rows = [
        (lambda_,) + residual_score(
            mapped, validation, fit, result.key, lm, lambda_
        )
        for lambda_ in LAMBDAS
    ]
    selected = max(validation_rows, key=lambda row: row[1])
    lambda_, validation_gain, _validation_lm, validation_n = selected
    parameters = len(mapping)
    penalty = (
        parameters * math.log2(max(validation_n, 2))
        / (2 * max(validation_n, 1))
    )
    penalized = validation_gain - penalty
    active = penalized > 0
    test_gain, test_lm, test_n = residual_score(
        mapped, test, fit, result.key, lm, lambda_
    )
    return FoldResult(
        fold=fold,
        fit_blocks=len(fit),
        validation_blocks=len(validation),
        test_blocks=len(test),
        mapped_symbols=parameters,
        fit_coverage=mapping_coverage(source_lines, fit, mapping),
        validation_coverage=mapping_coverage(
            source_lines, validation, mapping
        ),
        test_coverage=mapping_coverage(source_lines, test, mapping),
        lambda_=lambda_,
        validation_gain=validation_gain,
        validation_penalty=penalty,
        validation_penalized=penalized,
        active=active,
        test_gain_raw=test_gain,
        test_gain_reported=test_gain if active else 0.0,
        test_lm=test_lm,
        test_observations=test_n,
        init_lm=(
            result.init_score / max(1, len(training.windows))
            / math.log(2)
        ),
        fit_lm=(
            result.final_score / max(1, len(training.windows))
            / math.log(2)
        ),
        block_updates=result.block_updates,
        swap_updates=result.swap_updates,
        sample=render_sample(mapped, test, result.key),
    )


def evaluate_hypothesis(
    hypothesis: str,
    source_lines: Sequence[SourceLine],
    blocks: Sequence[str],
    inventory,
    lm: TetragramLM,
) -> list[FoldResult]:
    return [
        evaluate_fold(
            hypothesis,
            fold,
            source_lines,
            blocks,
            inventory,
            lm,
        )
        for fold in range(N_FOLDS)
    ]


def summarize(
    rows: Sequence[FoldResult],
) -> dict[str, float | int]:
    return {
        "active_folds": sum(row.active for row in rows),
        "mean_validation_penalized": mean(
            row.validation_penalized for row in rows
        ),
        "mean_test_gain_raw": mean(row.test_gain_raw for row in rows),
        "min_test_gain_raw": min(row.test_gain_raw for row in rows),
        "mean_test_gain_reported": mean(
            row.test_gain_reported for row in rows
        ),
        "mean_test_coverage": mean(row.test_coverage for row in rows),
        "mean_test_lm": mean(row.test_lm for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hypotheses",
        default=",".join(HYPOTHESES),
        help="comma-separated fixed channel hypotheses",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data/intermediate/followups_voynich_latent_channel_gate.json",
    )
    args = parser.parse_args()
    hypotheses = tuple(args.hypotheses.split(","))
    unknown = set(hypotheses) - set(HYPOTHESES)
    if unknown:
        raise SystemExit(f"unknown hypotheses: {sorted(unknown)}")

    lm_text, _dev_plaintext = load_control_segments()
    lm = TetragramLM(lm_text)
    inventory = load_structural_inventory()
    segmenter, calibration = calibration_segmenter()
    naibbe_source, naibbe_blocks = load_naibbe_source(inventory)
    voynich_raw, voynich_blocks = load_voynich_raw()

    report: dict[str, object] = {
        "method": "latent-channel held-out permutation gate",
        "letter_values_loaded": False,
        "official_plaintext_loaded": False,
        "frozen_parameters": {
            "alphabet": ALPHABET,
            "context_weight": CONTEXT_WEIGHT,
            "block_passes": BLOCK_PASSES,
            "swap_passes": SWAP_PASSES,
            "smoothing": SMOOTHING,
            "synthetic_seeds": DEV_SEEDS,
            "lambdas": LAMBDAS,
        },
        "segmenter_calibration": calibration,
        "hypotheses": {},
    }
    for hypothesis in hypotheses:
        print(f"[{hypothesis}] Naibbe control", file=sys.stderr)
        naibbe_rows = evaluate_hypothesis(
            hypothesis,
            naibbe_source,
            naibbe_blocks,
            inventory,
            lm,
        )
        print(f"[{hypothesis}] Voynich gate", file=sys.stderr)
        voynich_rows = []
        folds = block_folds(voynich_blocks)
        for fold in range(N_FOLDS):
            test = {
                block
                for block in voynich_blocks
                if folds[block] == fold
            }
            validation = {
                block
                for block in voynich_blocks
                if folds[block] == (fold + 1) % N_FOLDS
            }
            fit = set(voynich_blocks) - test - validation
            emitted = emit_voynich(voynich_raw, segmenter, fit)
            voynich_rows.append(
                evaluate_fold(
                    hypothesis,
                    fold,
                    emitted,
                    voynich_blocks,
                    inventory,
                    lm,
                )
            )
        naibbe_summary = summarize(naibbe_rows)
        voynich_summary = summarize(voynich_rows)
        control_valid = naibbe_summary["active_folds"] == N_FOLDS
        control_floor = (
            naibbe_summary["min_test_gain_raw"]
            if control_valid
            else None
        )
        passed = bool(
            control_valid
            and voynich_summary["active_folds"] == N_FOLDS
            and control_floor is not None
            and voynich_summary["min_test_gain_raw"] >= control_floor
        )
        report["hypotheses"][hypothesis] = {
            "naibbe": {
                "folds": [asdict(row) for row in naibbe_rows],
                "summary": naibbe_summary,
            },
            "voynich": {
                "folds": [asdict(row) for row in voynich_rows],
                "summary": voynich_summary,
            },
            "control_valid": control_valid,
            "naibbe_calibrated_floor": control_floor,
            "passes_naibbe_calibrated_gate": passed,
        }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
