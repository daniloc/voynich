#!/usr/bin/env python3
"""
Synthetic-key learner for Naibbe using recurrence statistics only.

The learner never trains on the official Naibbe plaintext/key pairing.  It:

1. reads a Latin control corpus;
2. repeatedly randomizes every Naibbe role/table-to-letter assignment;
3. encrypts the Latin control under those synthetic keys;
4. learns a recurrence-only atomic/compound segmenter;
5. learns a recurrence-only emission-to-Latin-letter classifier; and
6. freezes both models before evaluating the official Naibbe ciphertext.

Features contain no glyph characters or glyph identities shared between keys.
They summarize counts, recurrence gaps, line positions, neighbour-role
distributions, neighbour-frequency distributions, and conditional recurrence
entropy.  The role-specific glyph inventories and table weights are cipher
specification priors; their official letter labels are shuffled before every
synthetic example.

Two official evaluations are reported:

* oracle structure: the recurrence key learner receives correct U/P/S emission
  boundaries, isolating key learning;
* blind structure: the synthetic-trained recurrence segmenter supplies those
  boundaries, measuring end-to-end token/plaintext recovery.

The official CSV is used after freezing for truth scoring.  This is a prototype,
not a claim about Voynich.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


SEED = 20260723
ALPHABET = tuple("abcdefghilmnopqrstuvxyz")
ROLE_NAMES = ("U", "P", "S")
TABLES = ("alpha", "beta1", "beta2", "beta3", "gamma1", "gamma2")
TABLE_WEIGHTS = {
    "alpha": 28,
    "beta1": 14,
    "beta2": 11,
    "beta3": 11,
    "gamma1": 7,
    "gamma2": 7,
}
UNIGRAM_PROBABILITY = 17 / 36
TRAIN_KEYS = 48
VALIDATION_KEYS = 12
STRUCTURE_KEYS = 10
RIDGE = 8.0


@dataclass(frozen=True)
class Slot:
    role: str
    table: str
    official_letter: str
    glyph: str


@dataclass
class SyntheticCorpus:
    token_lines: list[list[str]]
    emission_lines: list[list[tuple[str, str, str]]]
    token_truth: dict[str, tuple[tuple[str, str, str], ...]]


@dataclass
class StructuralCounts:
    word: Counter
    left_tokens: Counter
    right_tokens: Counter
    left_types: Counter
    right_types: Counter


@dataclass
class SegmentModel:
    split_weights: np.ndarray
    split_mean: np.ndarray
    split_scale: np.ndarray
    class_weights: np.ndarray
    class_mean: np.ndarray
    class_scale: np.ndarray

    def split_features(
        self, word: str, counts: StructuralCounts
    ) -> np.ndarray:
        return np.stack([
            raw_split_features(word, split, counts)
            for split in range(1, len(word))
        ])

    def best_split(
        self, word: str, counts: StructuralCounts
    ) -> tuple[Optional[tuple[str, str]], float, float]:
        if len(word) < 2:
            return None, -20.0, 20.0
        features = self.split_features(word, counts)
        scores = (
            (features - self.split_mean) / self.split_scale
        ) @ self.split_weights
        best = int(np.argmax(scores)) + 1
        ordered = np.sort(scores)
        margin = (
            float(ordered[-1] - ordered[-2])
            if len(ordered) > 1 else 20.0
        )
        return (word[:best], word[best:]), float(scores.max()), margin

    def class_features(
        self, word: str, counts: StructuralCounts
    ) -> np.ndarray:
        boundary, score, margin = self.best_split(word, counts)
        if boundary is None:
            return np.array([
                math.log1p(counts.word[word]),
                math.log1p(len(word)),
                -20.0, 20.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0,
            ])
        left, right = boundary
        own = max(1, counts.word[word])
        return np.array([
            math.log1p(counts.word[word]),
            math.log1p(len(word)),
            score,
            margin,
            math.log1p(max(0, counts.left_tokens[left] - own)),
            math.log1p(max(0, counts.right_tokens[right] - own)),
            math.log1p(max(0, counts.left_types[left] - 1)),
            math.log1p(max(0, counts.right_types[right] - 1)),
            math.log1p(counts.word[left]),
            math.log1p(counts.word[right]),
            len(left) / len(word),
            1.0,
        ])

    def segment(
        self, word: str, counts: StructuralCounts
    ) -> tuple[tuple[str, str], ...]:
        boundary, _, _ = self.best_split(word, counts)
        if boundary is None:
            return (("U", word),)
        features = (
            self.class_features(word, counts) - self.class_mean
        ) / self.class_scale
        score = float(features @ self.class_weights)
        if score <= 0:
            return (("U", word),)
        return (("P", boundary[0]), ("S", boundary[1]))


@dataclass
class KeyRow:
    role: str
    symbol: str
    letter: Optional[str]
    count: int
    features: np.ndarray


@dataclass
class RidgeClassifier:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def logits(self, features: np.ndarray) -> np.ndarray:
        normalized = (features - self.mean) / self.scale
        augmented = np.c_[normalized, np.ones(len(normalized))]
        return np.einsum(
            "ij,jk->ik", augmented, self.weights, optimize=False
        )


class Deck:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.cards: list[str] = []
        self.index = 0

    def draw(self) -> str:
        if self.index >= len(self.cards):
            self.cards = [
                table
                for table in TABLES
                for _ in range(TABLE_WEIGHTS[table])
            ]
            self.rng.shuffle(self.cards)
            self.index = 0
        value = self.cards[self.index]
        self.index += 1
        return value


def normalize_latin(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(
        character for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    replacements = {
        "æ": "ae", "œ": "oe", "ð": "d", "þ": "th", "ł": "l",
        "ß": "ss", "ø": "o",
    }
    normalized = "".join(
        replacements.get(character, character)
        for character in normalized.lower()
    )
    lines = []
    for raw in normalized.splitlines():
        cleaned = "".join(character for character in raw if character.isalpha())
        cleaned = cleaned.replace("w", "uu").replace("j", "i").replace("k", "c")
        cleaned = "".join(character for character in cleaned if character in ALPHABET)
        if cleaned:
            lines.append(cleaned)
    return lines


def load_slots(path: Path) -> list[Slot]:
    role_map = {"unigram": "U", "prefix": "P", "suffix": "S"}
    slots = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            state, table, letter = row["code"].split("_")
            slots.append(Slot(
                role=role_map[state],
                table=table,
                official_letter=letter,
                glyph=row["glyphs"],
            ))
    return slots


def randomized_key(
    slots: Sequence[Slot],
    rng: np.random.Generator,
) -> dict[tuple[str, str, str], str]:
    by_role_table: dict[tuple[str, str], list[Slot]] = defaultdict(list)
    for slot in slots:
        by_role_table[(slot.role, slot.table)].append(slot)

    result = {}
    forced_duplicate_letter: Optional[str] = None
    for role in ROLE_NAMES:
        for table in TABLES:
            group = sorted(
                by_role_table[(role, table)],
                key=lambda slot: slot.official_letter,
            )
            letters = list(ALPHABET)
            rng.shuffle(letters)
            if role == "U" and table == "beta2":
                forced_duplicate_letter = letters[
                    [slot.glyph for slot in group].index("dar")
                ]
            if role == "U" and table == "beta3":
                dar_index = [slot.glyph for slot in group].index("dar")
                desired = forced_duplicate_letter
                current = letters.index(desired)
                letters[dar_index], letters[current] = (
                    letters[current], letters[dar_index]
                )
            for slot, synthetic_letter in zip(group, letters):
                result[(role, table, synthetic_letter)] = slot.glyph
    return result


def valid_compound_catalog(
    slots: Sequence[Slot],
) -> tuple[set[str], dict[str, set[tuple[str, str]]]]:
    atomic = {slot.glyph for slot in slots if slot.role == "U"}
    left = {slot.glyph for slot in slots if slot.role == "P"}
    right = {slot.glyph for slot in slots if slot.role == "S"}
    catalog: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for prefix_glyph in left:
        for suffix_glyph in right:
            catalog[prefix_glyph + suffix_glyph].add(
                (prefix_glyph, suffix_glyph)
            )
    return atomic, catalog


def encrypt_synthetic(
    plain_lines: Sequence[str],
    slots: Sequence[Slot],
    rng: np.random.Generator,
) -> SyntheticCorpus:
    key = randomized_key(slots, rng)
    atomic, catalog = valid_compound_catalog(slots)
    token_lines: list[list[str]] = []
    emission_lines: list[list[tuple[str, str, str]]] = []
    token_truth: dict[str, tuple[tuple[str, str, str], ...]] = {}

    for plain in plain_lines:
        deck = Deck(rng)
        tokens = []
        emissions = []
        index = 0
        while index < len(plain):
            use_unigram = (
                index == len(plain) - 1
                or rng.random() < UNIGRAM_PROBABILITY
            )
            if use_unigram:
                letter = plain[index]
                table = deck.draw()
                glyph = key[("U", table, letter)]
                emitted = (("U", glyph, letter),)
                index += 1
            else:
                first, second = plain[index], plain[index + 1]
                emitted = ()
                for _ in range(10000):
                    prefix_glyph = key[("P", deck.draw(), first)]
                    suffix_glyph = key[("S", deck.draw(), second)]
                    combined = prefix_glyph + suffix_glyph
                    if (
                        combined not in atomic
                        and len(catalog[combined]) == 1
                    ):
                        emitted = (
                            ("P", prefix_glyph, first),
                            ("S", suffix_glyph, second),
                        )
                        break
                if not emitted:
                    raise RuntimeError("could not draw unambiguous compound")
                glyph = emitted[0][1] + emitted[1][1]
                index += 2
            previous = token_truth.get(glyph)
            if previous is not None and previous != emitted:
                raise RuntimeError("synthetic structural collision")
            token_truth[glyph] = emitted
            tokens.append(glyph)
            emissions.extend(emitted)
        token_lines.append(tokens)
        emission_lines.append(emissions)
    return SyntheticCorpus(token_lines, emission_lines, token_truth)


def structural_counts(words: Sequence[str]) -> StructuralCounts:
    word_counts = Counter(words)
    left_tokens: Counter = Counter()
    right_tokens: Counter = Counter()
    left_types: Counter = Counter()
    right_types: Counter = Counter()
    for word, count in word_counts.items():
        for split in range(1, len(word)):
            left, right = word[:split], word[split:]
            left_tokens[left] += count
            right_tokens[right] += count
            left_types[left] += 1
            right_types[right] += 1
    return StructuralCounts(
        word_counts, left_tokens, right_tokens, left_types, right_types
    )


def raw_split_features(
    word: str,
    split: int,
    counts: StructuralCounts,
) -> np.ndarray:
    left, right = word[:split], word[split:]
    own = max(1, counts.word[word])
    return np.array([
        math.log1p(max(0, counts.left_tokens[left] - own)),
        math.log1p(max(0, counts.right_tokens[right] - own)),
        math.log1p(max(0, counts.left_types[left] - 1)),
        math.log1p(max(0, counts.right_types[right] - 1)),
        math.log1p(counts.word[left]),
        math.log1p(counts.word[right]),
        len(left) / len(word),
        len(right) / len(word),
        abs(len(left) - len(right)) / len(word),
        math.log1p(len(word)),
        1.0,
    ])


def standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0) + 1e-6
    if np.allclose(features[:, -1], 1.0):
        mean[-1] = 0.0
        scale[-1] = 1.0
    return mean, scale


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    weights: Optional[np.ndarray] = None,
    ridge: float = RIDGE,
) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(features))
    weighted = features * np.sqrt(weights)[:, None]
    weighted_targets = targets * np.sqrt(weights)[:, None]
    penalty = np.eye(features.shape[1]) * ridge
    penalty[-1, -1] = ridge * 0.05
    # Chunking avoids an Accelerate/BLAS overflow warning seen on large, very
    # narrow matrices even though every input and partial product is finite.
    gram = np.zeros((features.shape[1], features.shape[1]), dtype=float)
    right = np.zeros(
        (features.shape[1], weighted_targets.shape[1]), dtype=float
    )
    for start in range(0, len(weighted), 8192):
        batch = weighted[start:start + 8192]
        batch_targets = weighted_targets[start:start + 8192]
        gram += np.einsum(
            "ni,nj->ij", batch, batch, optimize=False
        )
        right += np.einsum(
            "ni,nk->ik", batch, batch_targets, optimize=False
        )
    return np.linalg.solve(gram + penalty, right)


def train_segmenter(
    corpora: Sequence[SyntheticCorpus],
    rng: np.random.Generator,
) -> SegmentModel:
    pair_features = []
    pair_targets = []
    pair_weights = []
    class_raw = []

    for corpus in corpora:
        words = [word for line in corpus.token_lines for word in line]
        counts = structural_counts(words)
        for word, frequency in Counter(words).items():
            truth = corpus.token_truth[word]
            if len(truth) == 2 and len(word) > 1:
                boundary = len(truth[0][1])
                true_features = raw_split_features(word, boundary, counts)
                wrong_splits = [
                    split for split in range(1, len(word))
                    if split != boundary
                ]
                if wrong_splits:
                    chosen = rng.choice(
                        wrong_splits,
                        size=min(3, len(wrong_splits)),
                        replace=False,
                    )
                    for split in chosen:
                        wrong = raw_split_features(word, int(split), counts)
                        pair_features.extend((true_features - wrong, wrong - true_features))
                        pair_targets.extend((1.0, -1.0))
                        weight = min(frequency, 10)
                        pair_weights.extend((weight, weight))

    split_features = np.stack(pair_features)
    split_mean, split_scale = standardize(split_features)
    normalized = (split_features - split_mean) / split_scale
    split_weights = fit_ridge(
        normalized,
        np.asarray(pair_targets)[:, None],
        np.asarray(pair_weights),
    )[:, 0]

    partial = SegmentModel(
        split_weights=split_weights,
        split_mean=split_mean,
        split_scale=split_scale,
        class_weights=np.zeros(12),
        class_mean=np.zeros(12),
        class_scale=np.ones(12),
    )
    for corpus in corpora:
        words = [word for line in corpus.token_lines for word in line]
        counts = structural_counts(words)
        for word, frequency in Counter(words).items():
            class_raw.append((
                partial.class_features(word, counts),
                1.0 if len(corpus.token_truth[word]) == 2 else -1.0,
                min(frequency, 10),
            ))
    features = np.stack([row[0] for row in class_raw])
    mean, scale = standardize(features)
    normalized = (features - mean) / scale
    targets = np.asarray([row[1] for row in class_raw])[:, None]
    weights = np.asarray([row[2] for row in class_raw])
    class_weights = fit_ridge(
        normalized, targets, weights, ridge=4.0
    )[:, 0]
    return SegmentModel(
        split_weights, split_mean, split_scale,
        class_weights, mean, scale,
    )


def segment_metrics(
    model: SegmentModel,
    token_lines: Sequence[Sequence[str]],
    truth: dict[str, tuple[tuple[str, str, str], ...]],
) -> dict:
    words = [word for line in token_lines for word in line]
    counts = structural_counts(words)
    known = classified = joint = compounds = boundaries = 0
    for word in words:
        expected = truth.get(word)
        if expected is None:
            continue
        predicted = model.segment(word, counts)
        known += 1
        predicted_compound = len(predicted) == 2
        expected_compound = len(expected) == 2
        classified += predicted_compound == expected_compound
        if not expected_compound:
            joint += not predicted_compound
        else:
            compounds += 1
            correct = (
                predicted_compound
                and predicted[0][1] == expected[0][1]
                and predicted[1][1] == expected[1][1]
            )
            boundaries += correct
            joint += correct
    return {
        "tokens": known,
        "class_accuracy": classified / known,
        "joint_accuracy": joint / known,
        "compound_boundary_accuracy": boundaries / compounds,
    }


def role_index(role: str) -> int:
    return ROLE_NAMES.index(role)


def entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum(
        count / total * math.log2(count / total)
        for count in counter.values()
    )


def frequency_bin(value: float, quantiles: np.ndarray) -> int:
    return int(np.searchsorted(quantiles, value, side="right"))


def recurrence_rows(
    emission_lines: Sequence[Sequence[tuple[str, str, Optional[str]]]],
) -> list[KeyRow]:
    symbols = [
        (role, glyph)
        for line in emission_lines
        for role, glyph, _ in line
    ]
    counts = Counter(symbols)
    total = len(symbols)
    frequencies = {symbol: count / total for symbol, count in counts.items()}
    quantiles = np.quantile(
        list(frequencies.values()), np.linspace(0.1, 0.9, 9)
    )

    labels: dict[tuple[str, str], Counter] = defaultdict(Counter)
    gaps: dict[tuple[str, str], list[int]] = defaultdict(list)
    positions: dict[tuple[str, str], list[float]] = defaultdict(list)
    prev_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    next_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    prev_roles: dict[tuple[str, str], Counter] = defaultdict(Counter)
    next_roles: dict[tuple[str, str], Counter] = defaultdict(Counter)
    prev_bins: dict[tuple[str, str], Counter] = defaultdict(Counter)
    next_bins: dict[tuple[str, str], Counter] = defaultdict(Counter)
    line_counts: Counter = Counter()
    first_counts: Counter = Counter()
    last_counts: Counter = Counter()
    previous_global: dict[tuple[str, str], int] = {}
    clock = 0

    for line in emission_lines:
        for index, (role, glyph, letter) in enumerate(line):
            symbol = (role, glyph)
            if letter is not None:
                labels[symbol][letter] += 1
            line_counts[symbol] += 1
            if index == 0:
                first_counts[symbol] += 1
            if index == len(line) - 1:
                last_counts[symbol] += 1
            positions[symbol].append(
                index / max(1, len(line) - 1)
            )
            if symbol in previous_global:
                gaps[symbol].append(clock - previous_global[symbol])
            previous_global[symbol] = clock
            if index:
                previous = (line[index - 1][0], line[index - 1][1])
                prev_counts[symbol][previous] += 1
                prev_roles[symbol][previous[0]] += 1
                prev_bins[symbol][
                    frequency_bin(frequencies[previous], quantiles)
                ] += 1
            if index + 1 < len(line):
                following = (line[index + 1][0], line[index + 1][1])
                next_counts[symbol][following] += 1
                next_roles[symbol][following[0]] += 1
                next_bins[symbol][
                    frequency_bin(frequencies[following], quantiles)
                ] += 1
            clock += 1

    role_totals = Counter(role for role, _ in symbols)
    rows = []
    for symbol, count in counts.items():
        role, glyph = symbol
        gap_values = np.asarray(gaps[symbol], dtype=float)
        position_values = np.asarray(positions[symbol], dtype=float)

        def distribution(counter: Counter, keys: Sequence) -> list[float]:
            denominator = max(1, sum(counter.values()))
            return [counter[key] / denominator for key in keys]

        previous_entropy = entropy(prev_counts[symbol])
        following_entropy = entropy(next_counts[symbol])
        feature = [
            *(1.0 if role == name else 0.0 for name in ROLE_NAMES),
            math.log1p(count),
            math.log1p(count) - math.log1p(role_totals[role]),
            count / role_totals[role],
            math.log1p(line_counts[symbol]),
            first_counts[symbol] / count,
            last_counts[symbol] / count,
            float(position_values.mean()),
            float(position_values.std()),
            math.log1p(float(gap_values.mean())) if len(gap_values) else 0.0,
            math.log1p(float(np.median(gap_values))) if len(gap_values) else 0.0,
            float(np.std(np.log1p(gap_values))) if len(gap_values) else 0.0,
            float(np.mean(gap_values <= 2)) if len(gap_values) else 0.0,
            float(np.mean(gap_values <= 5)) if len(gap_values) else 0.0,
            float(np.mean(gap_values <= 20)) if len(gap_values) else 0.0,
            previous_entropy,
            following_entropy,
            previous_entropy / math.log2(max(2, len(prev_counts[symbol]))),
            following_entropy / math.log2(max(2, len(next_counts[symbol]))),
            max(prev_counts[symbol].values(), default=0) / max(
                1, sum(prev_counts[symbol].values())
            ),
            max(next_counts[symbol].values(), default=0) / max(
                1, sum(next_counts[symbol].values())
            ),
            math.log1p(len(prev_counts[symbol])),
            math.log1p(len(next_counts[symbol])),
            *distribution(prev_roles[symbol], ROLE_NAMES),
            *distribution(next_roles[symbol], ROLE_NAMES),
            *distribution(prev_bins[symbol], tuple(range(10))),
            *distribution(next_bins[symbol], tuple(range(10))),
        ]
        label = (
            labels[symbol].most_common(1)[0][0]
            if labels[symbol] else None
        )
        rows.append(KeyRow(
            role=role,
            symbol=glyph,
            letter=label,
            count=count,
            features=np.asarray(feature, dtype=float),
        ))
    return rows


def train_key_classifier(rows: Sequence[KeyRow]) -> RidgeClassifier:
    features = np.stack([row.features for row in rows])
    mean, scale = standardize(features)
    normalized = (features - mean) / scale
    augmented = np.c_[normalized, np.ones(len(normalized))]
    targets = np.zeros((len(rows), len(ALPHABET)))
    for index, row in enumerate(rows):
        targets[index, ALPHABET.index(row.letter)] = 1.0
    example_weights = np.sqrt(
        np.asarray([row.count for row in rows], dtype=float)
    )
    weights = fit_ridge(
        augmented, targets, example_weights, ridge=RIDGE
    )
    return RidgeClassifier(mean, scale, weights)


def hungarian_min(cost: np.ndarray) -> np.ndarray:
    """Rectangular Hungarian algorithm for rows <= columns."""
    rows, columns = cost.shape
    if rows > columns:
        raise ValueError("Hungarian implementation requires rows <= columns")
    u = np.zeros(rows + 1)
    v = np.zeros(columns + 1)
    p = np.zeros(columns + 1, dtype=int)
    way = np.zeros(columns + 1, dtype=int)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minimum = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, columns + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minimum[j]:
                    minimum[j] = current
                    way[j] = j0
                if minimum[j] < delta:
                    delta = minimum[j]
                    j1 = j
            for j in range(columns + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = np.full(rows, -1, dtype=int)
    for column in range(1, columns + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    return assignment


def predict_key(
    model: RidgeClassifier,
    rows: Sequence[KeyRow],
    balanced: bool,
) -> dict[tuple[str, str], str]:
    features = np.stack([row.features for row in rows])
    logits = model.logits(features)
    predictions = np.argmax(logits, axis=1)
    if balanced:
        for role in ROLE_NAMES:
            indices = [
                index for index, row in enumerate(rows) if row.role == role
            ]
            if len(indices) > len(ALPHABET) * 6:
                continue
            slots = np.repeat(np.arange(len(ALPHABET)), 6)
            assignment = hungarian_min(-logits[indices][:, slots])
            predictions[indices] = slots[assignment]
    return {
        (row.role, row.symbol): ALPHABET[predictions[index]]
        for index, row in enumerate(rows)
    }


def key_metrics(
    rows: Sequence[KeyRow],
    predictions: dict[tuple[str, str], str],
) -> dict:
    known = [row for row in rows if row.letter is not None]
    correct_types = sum(
        predictions[(row.role, row.symbol)] == row.letter
        for row in known
    )
    correct_tokens = sum(
        row.count
        for row in known
        if predictions[(row.role, row.symbol)] == row.letter
    )
    return {
        "types": len(known),
        "type_accuracy": correct_types / len(known),
        "emissions": sum(row.count for row in known),
        "weighted_accuracy": correct_tokens / sum(row.count for row in known),
    }


def official_truth(
    slots: Sequence[Slot],
    token_lines: Sequence[Sequence[str]],
) -> tuple[
    list[list[tuple[str, str, str]]],
    dict[str, tuple[tuple[str, str, str], ...]],
]:
    maps: dict[str, dict[str, str]] = {
        role: {} for role in ROLE_NAMES
    }
    for slot in slots:
        previous = maps[slot.role].get(slot.glyph)
        if previous is not None and previous != slot.official_letter:
            raise ValueError("official glyph has conflicting letters")
        maps[slot.role][slot.glyph] = slot.official_letter

    truth = {}
    emission_lines = []
    for line in token_lines:
        emitted_line = []
        for word in line:
            if word in maps["U"]:
                # The official decryptor gives atomic codes precedence over
                # accidental substring parses.
                emitted = (
                    ("U", word, maps["U"][word]),
                )
            else:
                candidates = []
                for split in range(1, len(word)):
                    left, right = word[:split], word[split:]
                    if left in maps["P"] and right in maps["S"]:
                        candidates.append((
                            ("P", left, maps["P"][left]),
                            ("S", right, maps["S"][right]),
                        ))
                if len(candidates) != 1:
                    continue
                emitted = candidates[0]
            truth[word] = emitted
            emitted_line.extend(emitted)
        emission_lines.append(emitted_line)
    return emission_lines, truth


def predicted_emissions(
    model: SegmentModel,
    token_lines: Sequence[Sequence[str]],
) -> list[list[tuple[str, str, None]]]:
    words = [word for line in token_lines for word in line]
    counts = structural_counts(words)
    return [
        [
            (role, glyph, None)
            for word in line
            for role, glyph in model.segment(word, counts)
        ]
        for line in token_lines
    ]


def label_predicted_emissions(
    predicted: Sequence[Sequence[tuple[str, str, None]]],
    truth: Sequence[Sequence[tuple[str, str, str]]],
) -> list[list[tuple[str, str, Optional[str]]]]:
    key_truth: dict[tuple[str, str], str] = {}
    for line in truth:
        for role, glyph, letter in line:
            key_truth[(role, glyph)] = letter
    return [
        [
            (role, glyph, key_truth.get((role, glyph)))
            for role, glyph, _ in line
        ]
        for line in predicted
    ]


def decode_metrics(
    token_lines: Sequence[Sequence[str]],
    truth: dict[str, tuple[tuple[str, str, str], ...]],
    segmenter: Optional[SegmentModel],
    key: dict[tuple[str, str], str],
) -> dict:
    words = [word for line in token_lines for word in line]
    counts = structural_counts(words)
    tokens = exact_tokens = true_characters = correct_characters = 0
    structural_tokens = 0
    samples = []
    for line in token_lines:
        predicted_line = []
        true_line = []
        for word in line:
            expected = truth.get(word)
            if expected is None:
                continue
            if segmenter is None:
                predicted_structure = tuple(
                    (role, glyph) for role, glyph, _ in expected
                )
            else:
                predicted_structure = segmenter.segment(word, counts)
            expected_structure = tuple(
                (role, glyph) for role, glyph, _ in expected
            )
            structural_tokens += predicted_structure == expected_structure
            predicted_text = "".join(
                key.get((role, glyph), "?")
                for role, glyph in predicted_structure
            )
            true_text = "".join(letter for _, _, letter in expected)
            tokens += 1
            exact_tokens += predicted_text == true_text
            true_characters += len(true_text)
            correct_characters += sum(
                left == right
                for left, right in zip(predicted_text, true_text)
            )
            predicted_line.append(predicted_text)
            true_line.append(true_text)
        if predicted_line and len(samples) < 2:
            samples.append((
                " ".join(predicted_line[:40]),
                " ".join(true_line[:40]),
            ))
    return {
        "tokens": tokens,
        "structural_accuracy": structural_tokens / tokens,
        "token_accuracy": exact_tokens / tokens,
        "character_accuracy": correct_characters / true_characters,
        "samples": samples,
    }


def read_ciphertext(path: Path) -> list[list[str]]:
    return [
        re.findall(r"[a-z]+", line.lower())
        for line in path.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines()
        if re.search(r"[a-z]", line.lower())
    ]


def synthetic_validation(
    model: RidgeClassifier,
    corpora: Sequence[SyntheticCorpus],
) -> tuple[dict, dict]:
    raw_totals = Counter()
    balanced_totals = Counter()
    for corpus in corpora:
        rows = recurrence_rows(corpus.emission_lines)
        for balanced, totals in (
            (False, raw_totals), (True, balanced_totals)
        ):
            prediction = predict_key(model, rows, balanced)
            metrics = key_metrics(rows, prediction)
            totals["types"] += metrics["types"]
            totals["correct_types"] += (
                metrics["type_accuracy"] * metrics["types"]
            )
            totals["emissions"] += metrics["emissions"]
            totals["correct_emissions"] += (
                metrics["weighted_accuracy"] * metrics["emissions"]
            )

    def finish(totals: Counter) -> dict:
        return {
            "type_accuracy": totals["correct_types"] / totals["types"],
            "weighted_accuracy": (
                totals["correct_emissions"] / totals["emissions"]
            ),
            "types": int(totals["types"]),
            "emissions": int(totals["emissions"]),
        }

    return finish(raw_totals), finish(balanced_totals)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--latin",
        type=Path,
        default=Path("data/controls/latin.txt"),
    )
    parser.add_argument(
        "--naibbe-dir",
        type=Path,
        default=Path("/tmp/naibbe-cipher"),
    )
    parser.add_argument("--train-keys", type=int, default=TRAIN_KEYS)
    parser.add_argument(
        "--validation-keys", type=int, default=VALIDATION_KEYS
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_path = args.naibbe_dir / "references/naibbe_tables.csv"
    ciphertext_path = (
        args.naibbe_dir / "encrypted/nathist_output_ciphertext.txt"
    )
    plaintext_path = (
        args.naibbe_dir
        / "respaced_plaintext/nathist_pre_encryption_respaced_plaintext.txt"
    )
    for path in (args.latin, table_path, ciphertext_path, plaintext_path):
        if not path.exists():
            raise FileNotFoundError(path)

    rng = np.random.default_rng(SEED)
    slots = load_slots(table_path)
    latin_lines = normalize_latin(
        args.latin.read_text(encoding="utf-8", errors="ignore")
    )
    official_lines = read_ciphertext(ciphertext_path)

    print("=" * 96)
    print("RECURRENCE-ONLY SYNTHETIC-KEY LEARNER")
    print("=" * 96)
    print(
        f"seed={SEED} Latin lines/chars={len(latin_lines)}/"
        f"{sum(map(len, latin_lines))}; train keys={args.train_keys}; "
        f"validation keys={args.validation_keys}"
    )
    print(
        "Official key/plaintext labels are inaccessible until final scoring; "
        "synthetic keys randomize all role/table letter assignments."
    )

    synthetic = [
        encrypt_synthetic(latin_lines, slots, rng)
        for _ in range(args.train_keys + args.validation_keys)
    ]
    train_corpora = synthetic[:args.train_keys]
    validation_corpora = synthetic[args.train_keys:]

    segmenter = train_segmenter(
        train_corpora[:min(STRUCTURE_KEYS, len(train_corpora))],
        rng,
    )
    structure_validation = [
        segment_metrics(
            segmenter, corpus.token_lines, corpus.token_truth
        )
        for corpus in validation_corpora
    ]
    print("\nSYNTHETIC STRUCTURE VALIDATION")
    print(
        f"  joint emission mean="
        f"{np.mean([row['joint_accuracy'] for row in structure_validation]):.4f}; "
        f"class mean="
        f"{np.mean([row['class_accuracy'] for row in structure_validation]):.4f}; "
        f"compound boundary mean="
        f"{np.mean([row['compound_boundary_accuracy'] for row in structure_validation]):.4f}"
    )

    oracle_training_rows = [
        row
        for corpus in train_corpora
        for row in recurrence_rows(corpus.emission_lines)
        if row.letter is not None
    ]
    noisy_training_rows = []
    for corpus in train_corpora:
        predicted = predicted_emissions(segmenter, corpus.token_lines)
        labelled = label_predicted_emissions(
            predicted, corpus.emission_lines
        )
        noisy_training_rows.extend(
            row for row in recurrence_rows(labelled)
            if row.letter is not None
        )
    oracle_classifier = train_key_classifier(oracle_training_rows)
    robust_classifier = train_key_classifier(
        oracle_training_rows + noisy_training_rows
    )
    raw_validation, balanced_validation = synthetic_validation(
        oracle_classifier, validation_corpora
    )
    print("\nSYNTHETIC KEY VALIDATION")
    print(
        f"  raw type={raw_validation['type_accuracy']:.4f}; "
        f"emission-weighted={raw_validation['weighted_accuracy']:.4f}; "
        f"types={raw_validation['types']}; "
        f"emissions={raw_validation['emissions']}"
    )
    print(
        f"  balanced type={balanced_validation['type_accuracy']:.4f}; "
        f"emission-weighted={balanced_validation['weighted_accuracy']:.4f}"
    )

    official_emissions, official_truth_map = official_truth(
        slots, official_lines
    )
    official_structure = segment_metrics(
        segmenter, official_lines, official_truth_map
    )
    print("\nOFFICIAL NAIBBE STRUCTURE")
    print(
        f"  tokens={official_structure['tokens']}; "
        f"class={official_structure['class_accuracy']:.4f}; "
        f"joint={official_structure['joint_accuracy']:.4f}; "
        f"compound boundary="
        f"{official_structure['compound_boundary_accuracy']:.4f}"
    )

    oracle_rows = recurrence_rows(official_emissions)
    oracle_raw_key = predict_key(oracle_classifier, oracle_rows, False)
    oracle_balanced_key = predict_key(
        oracle_classifier, oracle_rows, True
    )
    print("\nOFFICIAL NAIBBE KEY RECOVERY (ORACLE STRUCTURE)")
    for name, key in (
        ("raw", oracle_raw_key),
        ("balanced", oracle_balanced_key),
    ):
        metrics = key_metrics(oracle_rows, key)
        decoding = decode_metrics(
            official_lines, official_truth_map, None, key
        )
        print(
            f"  {name:8s} observed-key type={metrics['type_accuracy']:.4f} "
            f"({metrics['types']} types); weighted="
            f"{metrics['weighted_accuracy']:.4f} "
            f"({metrics['emissions']} emissions); plaintext chars="
            f"{decoding['character_accuracy']:.4f}; tokens="
            f"{decoding['token_accuracy']:.4f}"
        )

    blind_emissions = predicted_emissions(segmenter, official_lines)
    blind_rows = recurrence_rows(blind_emissions)
    blind_key = predict_key(robust_classifier, blind_rows, False)
    blind_metrics = decode_metrics(
        official_lines, official_truth_map, segmenter, blind_key
    )
    print("\nOFFICIAL NAIBBE END-TO-END (RECURRENCE STRUCTURE + RAW KEY)")
    print(
        f"  inferred emission types={len(blind_rows)}; structural tokens="
        f"{blind_metrics['structural_accuracy']:.4f}; plaintext chars="
        f"{blind_metrics['character_accuracy']:.4f}; exact tokens="
        f"{blind_metrics['token_accuracy']:.4f}"
    )
    for predicted, truth in blind_metrics["samples"]:
        print(f"  predicted: {predicted}")
        print(f"  truth:     {truth}")

    official_plain_lines = read_ciphertext(plaintext_path)
    audited_tokens = matching_tokens = audited_characters = 0
    matching_characters = 0
    for cipher_line, plain_line in zip(official_lines, official_plain_lines):
        for cipher_token, plain_token in zip(cipher_line, plain_line):
            expected = official_truth_map.get(cipher_token)
            if expected is None:
                continue
            decoded = "".join(letter for _, _, letter in expected)
            audited_tokens += 1
            matching_tokens += decoded == plain_token
            audited_characters += len(plain_token)
            matching_characters += sum(
                left == right
                for left, right in zip(decoded, plain_token)
            )
    print("\nTRUTH AUDIT")
    print(
        f"  known aligned tokens={audited_tokens}; table-vs-bundled "
        f"plaintext token agreement={matching_tokens / audited_tokens:.4f}; "
        f"character agreement="
        f"{matching_characters / audited_characters:.4f}"
    )
    print(
        "  Chance is 1/23=0.0435 before frequency imbalance. Recovery above "
        "chance is a calibration result; it is not a transferable Voynich key."
    )


if __name__ == "__main__":
    main()
