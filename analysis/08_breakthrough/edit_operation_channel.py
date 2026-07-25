#!/usr/bin/env python3
"""
Decode the local edit operations between Voynich word variants.

The visible word sequence has strong local self-citation: a word is often close
to one of the preceding words.  Previous repository attacks tested word forms,
prefix/suffix slots, fixed strides, and binary word properties, but not the
ordered operation that transforms the selected local source into its target.

This experiment predeclares a small family of edit-event representations.  For
each eligible target, the source is either the immediately preceding word or
the closest of the preceding eight words in the same uninterrupted line run.
Distance-at-most-one, distance-at-most-two, and complete local-source streams
are tested.  Ineligible transitions in a restricted stream are hard sequence
breaks.

A many-to-one homophonic key is fitted on complete fit quires.  Candidate and
language selection uses disjoint validation quires; the selected key is frozen
before scoring complete test quires.  A positive control embeds held-out
English or Latin into the observed line/block layout.  Matched nulls shuffle
events only within quire, line-position, and source-lag strata and repeat the
entire candidate-selection procedure.

The experiment is intentionally a gate.  It can detect a stable monoalphabetic
payload in edit choices; it does not test arbitrary keyed or stateful ciphers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "corpus.json"
CONTROLS = {
    "english": ROOT / "data" / "controls" / "english.txt",
    "latin": ROOT / "data" / "controls" / "latin.txt",
}
SEED = 20260723
N_FOLDS = 4
ALPHA = 0.1
PENALTY_WEIGHT = 3.0
MIN_SYMBOL_COUNT = 3
MIN_HELDOUT_QUADS = 100
MIN_CHANNEL_QUADS = 1000


@dataclass(frozen=True)
class RawLine:
    block: str
    folio: str
    line: str
    words: tuple[str | None, ...]


@dataclass(frozen=True)
class EditOp:
    kind: str
    source_pos: int
    target_pos: int
    source_char: str
    target_char: str


@dataclass(frozen=True)
class EditEvent:
    block: str
    line_id: str
    target_index: int
    line_length: int
    lag: int
    distance: int
    source: str
    target: str
    operations: tuple[EditOp, ...]


@dataclass(frozen=True)
class SymbolLine:
    block: str
    line_id: str
    symbols: tuple[str | None, ...]
    strata: tuple[str | None, ...]


@dataclass
class KeyFit:
    names: list[str]
    key: np.ndarray
    train_score: float
    train_penalized: float
    train_kl: float
    train_quads: int


@dataclass
class Score:
    lm: float
    quads: int
    coverage: float
    sample: str


@dataclass
class CandidateResult:
    candidate: str
    language: str
    fold: int
    validation_lm: float
    validation_gain: float
    validation_quads: int
    validation_coverage: float
    test_lm: float
    test_gain: float
    test_quads: int
    test_coverage: float
    ceiling: float
    test_gap_to_ceiling: float
    sample: str


@dataclass
class FoldSelection:
    fold: int
    candidate: str
    language: str
    validation_gain: float
    test_gain: float
    test_lm: float
    ceiling: float
    test_gap_to_ceiling: float
    test_quads: int
    sample: str


def folio_key(folio: str) -> tuple[int, int, int]:
    match = re.match(r"f(\d+)([rv])(\d*)", folio)
    if not match:
        return (10**9, 0, 0)
    return (
        int(match.group(1)),
        0 if match.group(2) == "r" else 1,
        int(match.group(3) or 0),
    )


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


def load_lines() -> tuple[list[RawLine], list[str], Counter[str]]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    lines: list[RawLine] = []
    audit: Counter[str] = Counter()
    for folio in sorted(corpus["folios"], key=folio_key):
        meta = corpus["meta"].get(folio, {})
        block = str(meta.get("Q", "?"))
        for raw in corpus["folios"][folio]:
            if locus_type(raw["locus"]) != "P":
                continue
            words: list[str | None] = []
            for word in raw["words"]:
                audit["tokens"] += 1
                if "?" in word or not word.isalpha() or len(word) < 2:
                    words.append(None)
                    audit["hard_breaks"] += 1
                else:
                    words.append(word)
                    audit["eligible_words"] += 1
            if any(word is not None for word in words):
                lines.append(
                    RawLine(
                        block=block,
                        folio=folio,
                        line=str(raw["line"]),
                        words=tuple(words),
                    )
                )
                audit["prose_lines"] += 1
    blocks = sorted({line.block for line in lines})
    return lines, blocks, audit


@lru_cache(maxsize=None)
def align(source: str, target: str) -> tuple[int, tuple[EditOp, ...]]:
    """Return a deterministic minimum Levenshtein alignment."""
    n, m = len(source), len(target)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + (source[i - 1] != target[j - 1]),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )

    i, j = n, m
    reverse: list[EditOp] = []
    while i or j:
        if (
            i
            and j
            and source[i - 1] == target[j - 1]
            and cost[i][j] == cost[i - 1][j - 1]
        ):
            reverse.append(
                EditOp("M", i - 1, j - 1, source[i - 1], target[j - 1])
            )
            i -= 1
            j -= 1
        elif (
            i
            and j
            and cost[i][j] == cost[i - 1][j - 1] + 1
        ):
            reverse.append(
                EditOp("S", i - 1, j - 1, source[i - 1], target[j - 1])
            )
            i -= 1
            j -= 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            reverse.append(EditOp("D", i - 1, j, source[i - 1], ""))
            i -= 1
        else:
            reverse.append(EditOp("I", i, j - 1, "", target[j - 1]))
            j -= 1
    return cost[n][m], tuple(reversed(reverse))


def valid_runs(
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


def source_for(
    run: Sequence[tuple[int, str]],
    target_offset: int,
    mode: str,
) -> tuple[int, str]:
    if mode == "adjacent":
        return 1, run[target_offset - 1][1]
    if mode != "nearest8":
        raise ValueError(mode)
    target = run[target_offset][1]
    candidates = []
    for lag in range(1, min(8, target_offset) + 1):
        source = run[target_offset - lag][1]
        distance, _operations = align(source, target)
        candidates.append(
            (
                distance / max(len(source), len(target)),
                distance,
                lag,
                source,
            )
        )
    _ratio, _distance, lag, source = min(candidates)
    return lag, source


def extract_events(
    lines: Sequence[RawLine],
    mode: str,
) -> list[list[EditEvent]]:
    event_runs: list[list[EditEvent]] = []
    for line in lines:
        for run in valid_runs(line.words):
            events: list[EditEvent] = []
            for offset in range(1, len(run)):
                target_index, target = run[offset]
                lag, source = source_for(run, offset, mode)
                distance, operations = align(source, target)
                events.append(
                    EditEvent(
                        block=line.block,
                        line_id=f"{line.folio}:{line.line}",
                        target_index=target_index,
                        line_length=len(line.words),
                        lag=lag,
                        distance=distance,
                        source=source,
                        target=target,
                        operations=operations,
                    )
                )
            if events:
                event_runs.append(events)
    return event_runs


def position_bin(operation: EditOp, event: EditEvent) -> str:
    index = max(operation.source_pos, operation.target_pos)
    width = max(len(event.source), len(event.target))
    if index <= 0:
        return "0"
    if index >= width - 1:
        return "4"
    return str(min(3, 1 + (3 * index) // max(1, width - 1)))


def changes(event: EditEvent) -> tuple[EditOp, ...]:
    return tuple(operation for operation in event.operations if operation.kind != "M")


def render_symbol(event: EditEvent, representation: str) -> str:
    edited = changes(event)
    if not edited:
        operation = "="
        positions = "="
        target_delta = "="
        source_delta = "="
        signed = "="
        full = "="
    else:
        operation = "".join(item.kind for item in edited)
        positions = ".".join(position_bin(item, event) for item in edited)
        target_delta = "".join(
            item.target_char if item.target_char else "-"
            for item in edited
        )
        source_delta = "".join(
            item.source_char if item.source_char else "-"
            for item in edited
        )
        signed = ".".join(
            item.kind
            + ":"
            + (item.target_char if item.target_char else item.source_char)
            for item in edited
        )
        full = ".".join(
            item.kind
            + ":"
            + position_bin(item, event)
            + ":"
            + (item.source_char or "-")
            + ">"
            + (item.target_char or "-")
            for item in edited
        )

    if representation == "operation":
        return operation
    if representation == "operation_position":
        return operation + "@" + positions
    if representation == "target_delta":
        return target_delta
    if representation == "source_delta":
        return source_delta
    if representation == "signed_delta":
        return signed
    if representation == "full_delta":
        return full
    if representation == "lag_operation":
        return f"L{event.lag}:{operation}"
    raise ValueError(representation)


REPRESENTATIONS = (
    "operation",
    "operation_position",
    "target_delta",
    "source_delta",
    "signed_delta",
    "full_delta",
    "lag_operation",
)


def event_stratum(event: EditEvent, representation: str) -> str:
    position = position_bucket(event.target_index, event.line_length)
    if representation == "lag_operation":
        return position
    lag = str(event.lag) if event.lag <= 3 else "4-8"
    return position + "|L" + lag


def make_channel(
    event_runs: Sequence[Sequence[EditEvent]],
    source_mode: str,
    max_distance: int | None,
    representation: str,
) -> tuple[str, list[SymbolLine]]:
    result: list[SymbolLine] = []
    for run in event_runs:
        symbols: list[str | None] = []
        strata: list[str | None] = []
        for event in run:
            if max_distance is None or event.distance <= max_distance:
                symbols.append(render_symbol(event, representation))
                strata.append(event_stratum(event, representation))
            else:
                symbols.append(None)
                strata.append(None)
        if any(symbol is not None for symbol in symbols):
            result.append(
                SymbolLine(
                    block=run[0].block,
                    line_id=run[0].line_id,
                    symbols=tuple(symbols),
                    strata=tuple(strata),
                )
            )
    distance_name = "all" if max_distance is None else str(max_distance)
    name = f"{source_mode}/d{distance_name}/{representation}"
    return name, result


def clean_control(path: Path, language: str) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    start = re.search(r"\*\*\* start of.*?\*\*\*", text, re.I | re.S)
    if start:
        text = text[start.end() :]
    end = re.search(r"\*\*\* end of", text, re.I | re.S)
    if end:
        text = text[: end.start()]
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    if language == "latin":
        normalized = (
            normalized.replace("j", "i")
            .replace("v", "u")
            .replace("w", "u")
        )
    return re.sub(r"[^a-z]+", "", normalized)


class TetragramLM:
    def __init__(self, text: str):
        self.alphabet = sorted(set(text))
        self.char_to_id = {
            character: index for index, character in enumerate(self.alphabet)
        }
        self.id_to_char = self.alphabet
        self.width = len(self.alphabet)
        ids = [self.char_to_id[character] for character in text]
        unigram = Counter(ids)
        trigram = Counter(zip(ids, ids[1:], ids[2:]))
        quad = Counter(zip(ids, ids[1:], ids[2:], ids[3:]))
        self.logp = np.empty(
            (self.width, self.width, self.width, self.width),
            dtype=np.float64,
        )
        for a in range(self.width):
            for b in range(self.width):
                for c in range(self.width):
                    denominator = trigram[(a, b, c)] + ALPHA * self.width
                    for d in range(self.width):
                        self.logp[a, b, c, d] = math.log(
                            (quad[(a, b, c, d)] + ALPHA) / denominator
                        )
        frequency_order = sorted(
            range(self.width), key=lambda item: (-unigram[item], item)
        )
        self.target_ids = np.asarray(frequency_order, dtype=np.int32)
        weights = np.asarray(
            [unigram[index] for index in self.target_ids], dtype=float
        )
        self.target_probabilities = weights / weights.sum()
        self.all_probabilities = np.asarray(
            [unigram[index] for index in range(self.width)], dtype=float
        )
        self.all_probabilities /= self.all_probabilities.sum()

    def mean_score(self, text: str) -> float:
        ids = [self.char_to_id[character] for character in text]
        if len(ids) < 4:
            return float("-inf")
        values = self.logp[
            ids[:-3],
            ids[1:-2],
            ids[2:-1],
            ids[3:],
        ]
        return float(np.mean(values))


def block_folds(blocks: Sequence[str]) -> dict[str, int]:
    return {block: index % N_FOLDS for index, block in enumerate(blocks)}


def split_runs(
    symbols: Sequence[str | None],
) -> Iterable[tuple[str, ...]]:
    current: list[str] = []
    for symbol in symbols:
        if symbol is None:
            if current:
                yield tuple(current)
                current = []
        else:
            current.append(symbol)
    if current:
        yield tuple(current)


def symbol_vocabulary(
    lines: Sequence[SymbolLine],
    blocks: set[str],
) -> tuple[list[str], Counter[str]]:
    counts = Counter(
        symbol
        for line in lines
        if line.block in blocks
        for symbol in line.symbols
        if symbol is not None
    )
    names = sorted(
        symbol for symbol, count in counts.items() if count >= MIN_SYMBOL_COUNT
    )
    return names, counts


def ngram_table(
    lines: Sequence[SymbolLine],
    blocks: set[str],
    name_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], int, int]:
    counts: Counter[tuple[int, int, int, int]] = Counter()
    possible = 0
    mapped = 0
    for line in lines:
        if line.block not in blocks:
            continue
        for run in split_runs(line.symbols):
            possible += max(0, len(run) - 3)
            current: list[int] = []
            for symbol in run:
                symbol_id = name_to_id.get(symbol)
                if symbol_id is None:
                    if len(current) >= 4:
                        counts.update(
                            zip(
                                current,
                                current[1:],
                                current[2:],
                                current[3:],
                            )
                        )
                    current = []
                else:
                    current.append(symbol_id)
            if len(current) >= 4:
                counts.update(
                    zip(
                        current,
                        current[1:],
                        current[2:],
                        current[3:],
                    )
                )
    if not counts:
        return (
            np.empty((0, 4), dtype=np.int32),
            np.empty(0, dtype=np.float64),
            {},
            0,
            possible,
        )
    quads = np.asarray(list(counts), dtype=np.int32)
    weights = np.asarray(list(counts.values()), dtype=np.float64)
    affected: dict[int, list[int]] = defaultdict(list)
    for row_index, row in enumerate(quads):
        for symbol in set(map(int, row)):
            affected[symbol].append(row_index)
    mapped = int(weights.sum())
    return (
        quads,
        weights,
        {
            symbol: np.asarray(indices, dtype=np.int32)
            for symbol, indices in affected.items()
        },
        mapped,
        possible,
    )


def contributions(
    quads: np.ndarray,
    weights: np.ndarray,
    key: np.ndarray,
    lm: TetragramLM,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    rows = quads if indices is None else quads[indices]
    selected_weights = weights if indices is None else weights[indices]
    if not len(rows):
        return np.empty(0, dtype=float)
    return selected_weights * lm.logp[
        key[rows[:, 0]],
        key[rows[:, 1]],
        key[rows[:, 2]],
        key[rows[:, 3]],
    ]


def unigram_kl(
    key: np.ndarray,
    frequencies: np.ndarray,
    lm: TetragramLM,
) -> float:
    decoded = np.bincount(
        key, weights=frequencies, minlength=lm.width
    ).astype(float)
    decoded /= decoded.sum()
    mask = decoded > 0
    return float(
        np.sum(
            decoded[mask]
            * np.log(decoded[mask] / lm.all_probabilities[mask])
        )
    )


def baseline_key(
    names: Sequence[str],
    counts: Counter[str],
    lm: TetragramLM,
) -> np.ndarray:
    order = sorted(
        range(len(names)), key=lambda index: (-counts[names[index]], names[index])
    )
    key = np.empty(len(names), dtype=np.int32)
    cumulative_source = 0.0
    total_source = sum(counts[name] for name in names)
    target_cumulative = np.cumsum(lm.target_probabilities)
    for symbol_index in order:
        midpoint = (
            cumulative_source + 0.5 * counts[names[symbol_index]]
        ) / total_source
        target_rank = int(np.searchsorted(target_cumulative, midpoint))
        target_rank = min(target_rank, len(lm.target_ids) - 1)
        key[symbol_index] = lm.target_ids[target_rank]
        cumulative_source += counts[names[symbol_index]]
    return key


def score_quads(
    quads: np.ndarray,
    weights: np.ndarray,
    key: np.ndarray,
    lm: TetragramLM,
) -> float:
    if not len(weights):
        return float("-inf")
    return float(contributions(quads, weights, key, lm).sum() / weights.sum())


def fit_key(
    lines: Sequence[SymbolLine],
    blocks: set[str],
    lm: TetragramLM,
    rng: np.random.Generator,
    steps: int,
    restarts: int,
) -> KeyFit | None:
    names, counts = symbol_vocabulary(lines, blocks)
    if len(names) < 4:
        return None
    name_to_id = {name: index for index, name in enumerate(names)}
    quads, weights, affected, mapped, _possible = ngram_table(
        lines, blocks, name_to_id
    )
    if mapped < 40:
        return None
    frequencies = np.asarray([counts[name] for name in names], dtype=float)
    total = float(weights.sum())
    best_key: np.ndarray | None = None
    best_penalized = float("-inf")

    starts = [baseline_key(names, counts, lm)]
    starts.extend(
        rng.choice(
            lm.target_ids,
            size=len(names),
            p=lm.target_probabilities,
        ).astype(np.int32)
        for _ in range(max(0, restarts - 1))
    )

    for initial in starts:
        key = initial.copy()
        current_contributions = contributions(quads, weights, key, lm)
        decoded_frequency = np.bincount(
            key, weights=frequencies, minlength=lm.width
        ).astype(float)

        def frequency_kl(values: np.ndarray) -> float:
            probabilities = values / values.sum()
            mask = probabilities > 0
            return float(
                np.sum(
                    probabilities[mask]
                    * np.log(
                        probabilities[mask]
                        / lm.all_probabilities[mask]
                    )
                )
            )

        current_kl = frequency_kl(decoded_frequency)
        current = (
            float(current_contributions.sum())
            - PENALTY_WEIGHT * total * current_kl
        )
        local_best = current
        local_key = key.copy()
        for step in range(steps):
            symbol = int(rng.integers(len(names)))
            if symbol not in affected:
                continue
            old_target = int(key[symbol])
            swap_symbol: int | None = None
            if rng.random() < 0.45:
                swap_symbol = int(rng.integers(len(names)))
                if swap_symbol == symbol or swap_symbol not in affected:
                    continue
                new_target = int(key[swap_symbol])
            else:
                new_target = int(
                    rng.choice(
                        lm.target_ids, p=lm.target_probabilities
                    )
                )
            if new_target == old_target:
                continue
            indices = affected[symbol]
            if swap_symbol is not None:
                indices = np.union1d(indices, affected[swap_symbol])
            before = float(current_contributions[indices].sum())
            proposed_frequency = decoded_frequency.copy()
            proposed_frequency[old_target] -= frequencies[symbol]
            proposed_frequency[new_target] += frequencies[symbol]
            if swap_symbol is None:
                key[symbol] = new_target
            else:
                proposed_frequency[new_target] -= frequencies[swap_symbol]
                proposed_frequency[old_target] += frequencies[swap_symbol]
                key[symbol], key[swap_symbol] = (
                    key[swap_symbol],
                    key[symbol],
                )
            proposed_kl = frequency_kl(proposed_frequency)
            proposed_values = contributions(
                quads, weights, key, lm, indices
            )
            delta = (
                float(proposed_values.sum())
                - before
                - PENALTY_WEIGHT * total * (proposed_kl - current_kl)
            )
            temperature = 20.0 * (1.0 - step / max(1, steps)) + 0.05
            if delta >= 0 or rng.random() < math.exp(delta / temperature):
                current_contributions[indices] = proposed_values
                current += delta
                current_kl = proposed_kl
                decoded_frequency = proposed_frequency
                if current > local_best:
                    local_best = current
                    local_key = key.copy()
            else:
                if swap_symbol is None:
                    key[symbol] = old_target
                else:
                    key[symbol], key[swap_symbol] = (
                        key[swap_symbol],
                        key[symbol],
                    )
        if local_best > best_penalized:
            best_penalized = local_best
            best_key = local_key

    assert best_key is not None
    return KeyFit(
        names=names,
        key=best_key,
        train_score=score_quads(quads, weights, best_key, lm),
        train_penalized=best_penalized / total,
        train_kl=unigram_kl(best_key, frequencies, lm),
        train_quads=mapped,
    )


def score_lines(
    lines: Sequence[SymbolLine],
    blocks: set[str],
    fit: KeyFit,
    lm: TetragramLM,
) -> Score:
    name_to_id = {name: index for index, name in enumerate(fit.names)}
    quads, weights, _affected, mapped, possible = ngram_table(
        lines, blocks, name_to_id
    )
    lm_score = score_quads(quads, weights, fit.key, lm)
    samples: list[str] = []
    remaining = 360
    for line in lines:
        if line.block not in blocks or remaining <= 0:
            continue
        rendered = "".join(
            "?"
            if symbol is None or symbol not in name_to_id
            else lm.id_to_char[fit.key[name_to_id[symbol]]]
            for symbol in line.symbols
        )
        if rendered:
            selected = rendered[:remaining]
            samples.append(selected)
            remaining -= len(selected)
    return Score(
        lm=lm_score,
        quads=mapped,
        coverage=mapped / max(1, possible),
        sample="/".join(samples),
    )


def score_baseline(
    lines: Sequence[SymbolLine],
    fit_blocks: set[str],
    score_blocks: set[str],
    lm: TetragramLM,
) -> Score:
    names, counts = symbol_vocabulary(lines, fit_blocks)
    if len(names) < 4:
        return Score(float("-inf"), 0, 0.0, "")
    fit = KeyFit(
        names=names,
        key=baseline_key(names, counts, lm),
        train_score=float("nan"),
        train_penalized=float("nan"),
        train_kl=float("nan"),
        train_quads=0,
    )
    return score_lines(lines, score_blocks, fit, lm)


def analyze_candidate(
    candidate: str,
    lines: Sequence[SymbolLine],
    language: str,
    lm: TetragramLM,
    ceiling: float,
    fold: int,
    folds: dict[str, int],
    steps: int,
    restarts: int,
    seed: int,
) -> CandidateResult | None:
    test_blocks = {block for block, value in folds.items() if value == fold}
    validation_blocks = {
        block for block, value in folds.items() if value == (fold + 1) % N_FOLDS
    }
    fit_blocks = set(folds) - test_blocks - validation_blocks
    rng = np.random.default_rng(seed)
    fit = fit_key(lines, fit_blocks, lm, rng, steps, restarts)
    if fit is None:
        return None
    validation = score_lines(lines, validation_blocks, fit, lm)
    test = score_lines(lines, test_blocks, fit, lm)
    validation_baseline = score_baseline(
        lines, fit_blocks, validation_blocks, lm
    )
    test_baseline = score_baseline(lines, fit_blocks, test_blocks, lm)
    if (
        validation.quads < MIN_HELDOUT_QUADS
        or test.quads < MIN_HELDOUT_QUADS
    ):
        return None
    return CandidateResult(
        candidate=candidate,
        language=language,
        fold=fold,
        validation_lm=validation.lm,
        validation_gain=validation.lm - validation_baseline.lm,
        validation_quads=validation.quads,
        validation_coverage=validation.coverage,
        test_lm=test.lm,
        test_gain=test.lm - test_baseline.lm,
        test_quads=test.quads,
        test_coverage=test.coverage,
        ceiling=ceiling,
        test_gap_to_ceiling=test.lm - ceiling,
        sample=test.sample,
    )


def select_fold(rows: Sequence[CandidateResult], fold: int) -> FoldSelection:
    selected = max(
        (row for row in rows if row.fold == fold),
        key=lambda row: (row.validation_gain, row.validation_lm),
    )
    return FoldSelection(
        fold=fold,
        candidate=selected.candidate,
        language=selected.language,
        validation_gain=selected.validation_gain,
        test_gain=selected.test_gain,
        test_lm=selected.test_lm,
        ceiling=selected.ceiling,
        test_gap_to_ceiling=selected.test_gap_to_ceiling,
        test_quads=selected.test_quads,
        sample=selected.sample,
    )


def shuffle_channel(
    lines: Sequence[SymbolLine],
    rng: np.random.Generator,
) -> list[SymbolLine]:
    """Shuffle symbols within block and layout/source-choice strata."""
    pools: dict[tuple[str, str], list[str]] = defaultdict(list)
    positions: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for line_index, line in enumerate(lines):
        for symbol_index, (symbol, stratum) in enumerate(
            zip(line.symbols, line.strata)
        ):
            if symbol is None or stratum is None:
                continue
            key = (line.block, stratum)
            pools[key].append(symbol)
            positions[key].append((line_index, symbol_index))
    mutable = [list(line.symbols) for line in lines]
    for key, values in pools.items():
        shuffled = values.copy()
        rng.shuffle(shuffled)
        for (line_index, symbol_index), symbol in zip(positions[key], shuffled):
            mutable[line_index][symbol_index] = symbol
    return [
        SymbolLine(
            block=line.block,
            line_id=line.line_id,
            symbols=tuple(symbols),
            strata=line.strata,
        )
        for line, symbols in zip(lines, mutable)
    ]


def positive_channel(
    template: Sequence[SymbolLine],
    plaintext: str,
    rng: np.random.Generator,
) -> list[SymbolLine]:
    cursor = 0
    result = []
    for line in template:
        symbols: list[str | None] = []
        for symbol in line.symbols:
            if symbol is None:
                symbols.append(None)
                continue
            character = plaintext[cursor % len(plaintext)]
            homophone = int(rng.integers(4))
            symbols.append(f"H:{character}:{homophone}")
            cursor += 1
        result.append(
            SymbolLine(
                block=line.block,
                line_id=line.line_id,
                symbols=tuple(symbols),
                strata=line.strata,
            )
        )
    return result


def candidate_audit(lines: Sequence[SymbolLine]) -> dict[str, object]:
    counts = Counter(
        symbol
        for line in lines
        for symbol in line.symbols
        if symbol is not None
    )
    possible = sum(
        max(0, len(run) - 3)
        for line in lines
        for run in split_runs(line.symbols)
    )
    return {
        "events": sum(counts.values()),
        "symbol_types": len(counts),
        "types_count_ge_3": sum(count >= 3 for count in counts.values()),
        "possible_quads": possible,
        "top_symbols": counts.most_common(12),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_candidates(
    raw_lines: Sequence[RawLine],
) -> dict[str, list[SymbolLine]]:
    candidates = {}
    for source_mode in ("adjacent", "nearest8"):
        event_runs = extract_events(raw_lines, source_mode)
        for max_distance in (1, 2, None):
            for representation in REPRESENTATIONS:
                name, lines = make_channel(
                    event_runs,
                    source_mode,
                    max_distance,
                    representation,
                )
                candidates[name] = lines
    return candidates


def make_lms() -> tuple[dict[str, TetragramLM], dict[str, str], dict[str, float]]:
    lms = {}
    holdouts = {}
    ceilings = {}
    for language, path in CONTROLS.items():
        control = clean_control(path, language)
        cut = len(control) * 2 // 3
        lms[language] = TetragramLM(control[:cut])
        holdouts[language] = control[cut:]
        ceilings[language] = lms[language].mean_score(holdouts[language])
    return lms, holdouts, ceilings


def run_panel(
    candidates: dict[str, list[SymbolLine]],
    lms: dict[str, TetragramLM],
    ceilings: dict[str, float],
    folds: dict[str, int],
    steps: int,
    restarts: int,
    seed_offset: int,
) -> tuple[list[CandidateResult], list[FoldSelection]]:
    rows: list[CandidateResult] = []
    tasks = [
        (candidate, language, fold)
        for candidate in sorted(candidates)
        for language in sorted(lms)
        for fold in range(N_FOLDS)
    ]
    for task_index, (candidate, language, fold) in enumerate(tasks):
        row = analyze_candidate(
            candidate,
            candidates[candidate],
            language,
            lms[language],
            ceilings[language],
            fold,
            folds,
            steps,
            restarts,
            SEED + seed_offset + task_index * 1009,
        )
        if row is not None:
            rows.append(row)
    selections = [select_fold(rows, fold) for fold in range(N_FOLDS)]
    return rows, selections


def print_selections(label: str, selections: Sequence[FoldSelection]) -> None:
    print("\n" + "=" * 108)
    print(label)
    print("=" * 108)
    for selected in selections:
        print(
            f"fold={selected.fold} {selected.language:<7} "
            f"{selected.candidate:<42} "
            f"val_gain={selected.validation_gain:+.4f} "
            f"test_gain={selected.test_gain:+.4f} "
            f"test_lm={selected.test_lm:.4f} "
            f"gap={selected.test_gap_to_ceiling:+.4f} "
            f"n4={selected.test_quads}"
        )
        print(" ", selected.sample[:220])
    print(
        "mean selected test gain="
        f"{mean(item.test_gain for item in selections):+.4f}; "
        "mean gap to ceiling="
        f"{mean(item.test_gap_to_ceiling for item in selections):+.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--positive-steps", type=int, default=12000)
    parser.add_argument("--positive-restarts", type=int, default=4)
    parser.add_argument("--nulls", type=int, default=8)
    parser.add_argument("--shortlist", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    raw_lines, blocks, corpus_audit = load_lines()
    folds = block_folds(blocks)
    candidates = build_candidates(raw_lines)
    lms, holdouts, ceilings = make_lms()

    audits = {
        candidate: candidate_audit(lines)
        for candidate, lines in candidates.items()
    }
    viable = {
        candidate: lines
        for candidate, lines in candidates.items()
        if audits[candidate]["possible_quads"] >= MIN_CHANNEL_QUADS
        and audits[candidate]["types_count_ge_3"] >= 4
    }
    print(
        f"corpus lines={corpus_audit['prose_lines']} "
        f"words={corpus_audit['eligible_words']} "
        f"breaks={corpus_audit['hard_breaks']} blocks={len(blocks)}"
    )
    print(
        f"candidate channels={len(candidates)} viable={len(viable)} "
        f"folds={folds}"
    )
    for candidate in sorted(viable):
        audit = audits[candidate]
        print(
            f"  {candidate:<42} events={audit['events']:>5} "
            f"types={audit['symbol_types']:>4} "
            f"ge3={audit['types_count_ge_3']:>3} "
            f"n4={audit['possible_quads']:>4}"
        )

    # The real panel supplies a validation-only shortlist.  Nulls rerun
    # selection over exactly that frozen shortlist.
    real_rows, real_selections = run_panel(
        viable,
        lms,
        ceilings,
        folds,
        args.steps,
        args.restarts,
        0,
    )
    print_selections("REAL EDIT-CHANNEL SELECTION", real_selections)

    validation_rank = Counter()
    validation_count = Counter()
    for row in real_rows:
        validation_rank[row.candidate] += row.validation_gain
        validation_count[row.candidate] += 1
    ranked_candidates = sorted(
        validation_rank,
        key=lambda candidate: (
            validation_rank[candidate] / validation_count[candidate],
            candidate,
        ),
        reverse=True,
    )
    shortlist_names = ranked_candidates[: args.shortlist]
    shortlist = {name: viable[name] for name in shortlist_names}
    print("\nFrozen validation shortlist:")
    for name in shortlist_names:
        print(
            f"  {name:<42} "
            f"mean_val_gain="
            f"{validation_rank[name] / validation_count[name]:+.4f}"
        )

    # Positive controls use the highest-capacity shortlisted layout.
    template_name = max(
        viable,
        key=lambda name: audits[name]["possible_quads"],
    )
    positive_rows = []
    positive_selections = []
    for language in sorted(lms):
        rng = np.random.default_rng(
            SEED + 700_000 + (0 if language == "english" else 1)
        )
        positive = positive_channel(
            viable[template_name], holdouts[language], rng
        )
        rows, selections = run_panel(
            {f"positive/{language}": positive},
            {language: lms[language]},
            {language: ceilings[language]},
            folds,
            args.positive_steps,
            args.positive_restarts,
            800_000 + (0 if language == "english" else 10_000),
        )
        positive_rows.extend(rows)
        positive_selections.extend(selections)
    print_selections("EMBEDDED PLAINTEXT POSITIVE CONTROL", positive_selections)

    null_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(SEED + 900_000)
    for replicate in range(args.nulls):
        shuffled = {
            name: shuffle_channel(lines, rng)
            for name, lines in shortlist.items()
        }
        _rows, selections = run_panel(
            shuffled,
            lms,
            ceilings,
            folds,
            args.steps,
            args.restarts,
            1_000_000 + replicate * 100_000,
        )
        row = {
            "replicate": replicate,
            "mean_selected_test_gain": mean(
                item.test_gain for item in selections
            ),
            "max_selected_test_gain": max(
                item.test_gain for item in selections
            ),
            "mean_gap_to_ceiling": mean(
                item.test_gap_to_ceiling for item in selections
            ),
            "selections": [asdict(item) for item in selections],
        }
        null_rows.append(row)
        if args.progress:
            print(
                f"null {replicate + 1}/{args.nulls}: "
                f"mean_gain={row['mean_selected_test_gain']:+.4f} "
                f"max_gain={row['max_selected_test_gain']:+.4f}"
            )

    real_mean_gain = mean(item.test_gain for item in real_selections)
    null_mean_gains = [
        float(row["mean_selected_test_gain"]) for row in null_rows
    ]
    null_p = (
        1 + sum(value >= real_mean_gain for value in null_mean_gains)
    ) / (1 + len(null_mean_gains))
    positive_mean_gain = mean(
        item.test_gain for item in positive_selections
    )
    positive_mean_gap = mean(
        item.test_gap_to_ceiling for item in positive_selections
    )
    summary = {
        "real_mean_selected_test_gain": real_mean_gain,
        "real_mean_gap_to_ceiling": mean(
            item.test_gap_to_ceiling for item in real_selections
        ),
        "null_mean_gain_mean": mean(null_mean_gains)
        if null_mean_gains
        else float("nan"),
        "null_mean_gain_max": max(null_mean_gains)
        if null_mean_gains
        else float("nan"),
        "null_empirical_p": null_p,
        "positive_mean_selected_test_gain": positive_mean_gain,
        "positive_mean_gap_to_ceiling": positive_mean_gap,
        "positive_pass": positive_mean_gap > -0.35,
        "real_pass": (
            real_mean_gain > max(null_mean_gains, default=float("inf"))
            and mean(
                item.test_gap_to_ceiling for item in real_selections
            )
            > -0.50
        ),
    }
    print("\n" + "=" * 108)
    print("GATE SUMMARY")
    print("=" * 108)
    for key, value in summary.items():
        print(f"{key}: {value}")

    payload = {
        "experiment": "edit_operation_channel",
        "seed": SEED,
        "parameters": {
            "steps": args.steps,
            "restarts": args.restarts,
            "positive_steps": args.positive_steps,
            "positive_restarts": args.positive_restarts,
            "nulls": args.nulls,
            "shortlist": args.shortlist,
            "minimum_symbol_count": MIN_SYMBOL_COUNT,
            "minimum_heldout_quads": MIN_HELDOUT_QUADS,
            "minimum_channel_quads": MIN_CHANNEL_QUADS,
            "representations": list(REPRESENTATIONS),
            "source_modes": ["adjacent", "nearest8"],
            "distance_thresholds": [1, 2, "all"],
        },
        "assets": {
            str(CORPUS.relative_to(ROOT)): sha256(CORPUS),
            **{
                str(path.relative_to(ROOT)): sha256(path)
                for path in CONTROLS.values()
            },
        },
        "corpus_audit": dict(corpus_audit),
        "blocks": blocks,
        "folds": folds,
        "ceilings": ceilings,
        "candidate_audit": audits,
        "viable_candidates": sorted(viable),
        "shortlist": shortlist_names,
        "real_rows": [asdict(row) for row in real_rows],
        "real_selections": [asdict(row) for row in real_selections],
        "positive_rows": [asdict(row) for row in positive_rows],
        "positive_selections": [
            asdict(row) for row in positive_selections
        ],
        "nulls": null_rows,
        "summary": summary,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
