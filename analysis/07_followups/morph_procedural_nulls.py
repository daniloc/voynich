"""
Procedural-null benchmark for the held-out morphotactic signal.

The positive result in sequence_parallel.py is only interesting if simple
content-free production models fail to reproduce it.  This script compares
Voynich with three reproducible controls:

1. Exact whole-word exchange within
   section x Currier x hand x quire x line-position.
2. The schema/copy generator called "matched" in
   analysis/03_null_models/residual.py (p_copy=.12, window=40, p_mut=.4).
3. The aggressive local self-citation generator in
   analysis/03_null_models/null_and_morph.py (window=40,
   p_verbatim=.12, p_mutate=.55).  That implementation otherwise copies the
   selected recent form, so its total exact-copy probability is .45.

Every dataset is scored leave-one-quire-out.  Counts, class vocabularies, and
probabilities exclude the test quire.  The aligned primary score is

    mean log2 P(y | predecessor, target-position)
       - log2 P(y | target-position)

on held-out transitions.  The second score compares the bigram probability of
the observed line with a version whose deep interior is reversed; the first,
second, penultimate, and last slots remain fixed.  Alpha is fixed at .5.
Uncertain and one-character tokens remain in their original slots as adjacency
breaks; generated controls never bridge across or move those breaks.

The inference unit for stability is the quire, never the line.  Independent
parametric/permutation replicates provide empirical 95% intervals.  Schema
distributions are estimated globally, as in the repository controls; copy
hyperparameters are inherited rather than refitted.  This is therefore a
parametric-bootstrap challenge, not nested refitting.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Callable, Hashable, Iterable, Optional, Sequence

import numpy as np


DEFAULT_REPLICATES = 60
DEFAULT_SEED = 20260722
ALPHA = 0.5

# Expanded analysis representation, matching sequence_parallel.py.
SCORE_PREFIXES = tuple(sorted(
    (
        "qok", "qot", "qo", "ok", "ot", "o", "y", "ch", "sh", "d",
        "cth", "ckh", "cph", "cfh",
    ),
    key=len,
    reverse=True,
))
SCORE_SUFFIXES = tuple(sorted(
    (
        "eedy", "eody", "edy", "aiin", "aiir", "ain", "iin", "dy",
        "ol", "or", "ar", "al", "am", "dam", "ey", "eey", "y",
    ),
    key=len,
    reverse=True,
))

# Generator inventory copied from residual.py, not expanded after the fact.
GEN_PREFIXES = tuple(sorted(
    ("qok", "qot", "qo", "ok", "ot", "o", "y", "ch", "sh", "d",
     "cth", "ckh", "cph"),
    key=len,
    reverse=True,
))
GEN_SUFFIXES = tuple(sorted(
    ("eedy", "eody", "edy", "aiin", "ain", "iin", "dy", "ol", "or",
     "ar", "al", "am", "y"),
    key=len,
    reverse=True,
))

Position = str
ClassValue = Hashable
Transition = tuple[ClassValue, ClassValue]


@dataclass(frozen=True)
class Line:
    folio: str
    quire: str
    section: str
    currier: str
    hand: str
    words: tuple[Optional[str], ...]


@dataclass
class BlockScore:
    gain_sum: float = 0.0
    direction_sum: float = 0.0
    gain_transitions: int = 0
    direction_transitions: int = 0

    @property
    def gain(self) -> float:
        return (
            self.gain_sum / self.gain_transitions
            if self.gain_transitions
            else 0.0
        )

    @property
    def direction(self) -> float:
        return (
            self.direction_sum / self.direction_transitions
            if self.direction_transitions
            else 0.0
        )


@dataclass(frozen=True)
class DatasetScore:
    gain: float
    direction: float
    gain_positive: float
    direction_positive: float
    gain_median: float
    direction_median: float
    block_gain: tuple[float, ...]
    block_direction: tuple[float, ...]
    profile: dict[Transition, float]


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


def load_lines(path: Path) -> tuple[list[Line], Counter]:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    lines: list[Line] = []
    audit: Counter = Counter()
    for folio in sorted(corpus["folios"], key=folio_key):
        meta = corpus["meta"].get(folio, {})
        for raw in corpus["folios"][folio]:
            if locus_type(raw["locus"]) != "P":
                continue
            audit["prose_lines"] += 1
            words = []
            for word in raw["words"]:
                audit["source_tokens"] += 1
                if "?" in word:
                    words.append(None)
                    audit["uncertain_breaks"] += 1
                elif len(word) < 2:
                    words.append(None)
                    audit["one_char_breaks"] += 1
                else:
                    words.append(word)
                    audit["eligible_tokens"] += 1
            lines.append(Line(
                folio=folio,
                quire=str(meta.get("Q", "?")),
                section=str(meta.get("I", "?")),
                currier=str(meta.get("L", "?")),
                hand=str(meta.get("H", "?")),
                words=tuple(words),
            ))
    return lines, audit


@lru_cache(maxsize=None)
def score_decompose(word: str) -> tuple[str, str, str]:
    prefix = next(
        (value for value in SCORE_PREFIXES if word.startswith(value)),
        "",
    )
    residual = word[len(prefix):]
    suffix = next(
        (
            value for value in SCORE_SUFFIXES
            if residual.endswith(value) and len(residual) > len(value)
        ),
        "",
    )
    core = residual[:-len(suffix)] if suffix else residual
    return prefix, core, suffix


@lru_cache(maxsize=None)
def generator_decompose(word: str) -> tuple[str, str, str]:
    prefix = next(
        (value for value in GEN_PREFIXES if word.startswith(value)),
        "",
    )
    residual = word[len(prefix):]
    suffix = next(
        (
            value for value in GEN_SUFFIXES
            if residual.endswith(value) and len(residual) > len(value)
        ),
        "",
    )
    core = residual[:-len(suffix)] if suffix else residual
    return prefix, core, suffix


def shape(word: str) -> tuple[bool, bool, int, bool]:
    prefix, _, suffix = score_decompose(word)
    return (
        bool(prefix),
        bool(suffix),
        min(len(word), 8),
        any(character in word for character in "ktpf"),
    )


REPRESENTATIONS: dict[str, Callable[[str], ClassValue]] = {
    "prefix": lambda word: score_decompose(word)[0] or "none",
    "suffix": lambda word: score_decompose(word)[2] or "none",
    "affix_pair": lambda word: (
        score_decompose(word)[0] or "none",
        score_decompose(word)[2] or "none",
    ),
    "shape": shape,
}


def position_bucket(index: int, length: int) -> Position:
    if index == 0:
        return "first"
    if index == 1:
        return "second"
    if index == length - 2:
        return "penult"
    if index == length - 1:
        return "last"
    return "interior"


def valid_transitions(
    values: Sequence[Optional[ClassValue]],
) -> Iterable[tuple[int, ClassValue, ClassValue]]:
    for index in range(1, len(values)):
        left, right = values[index - 1], values[index]
        if left is not None and right is not None:
            yield index, left, right


def reverse_deep_interior(
    values: Sequence[Optional[ClassValue]],
) -> tuple[Optional[ClassValue], ...]:
    result = list(values)
    indices = [
        index for index in range(2, len(values) - 2)
        if values[index] is not None
    ]
    reversed_values = [values[index] for index in reversed(indices)]
    for index, value in zip(indices, reversed_values):
        result[index] = value
    return tuple(result)


def subtract(total: Counter, held_out: Counter, key: Hashable) -> int:
    return total[key] - held_out[key]


def score_dataset(
    all_lines: Sequence[Line],
    transform: Callable[[str], ClassValue],
    diagnostic_pairs: set[Transition] | None = None,
) -> DatasetScore:
    """Score one real or generated dataset with leave-one-quire-out models."""
    by_quire: dict[
        str,
        list[tuple[Optional[ClassValue], ...]],
    ] = defaultdict(list)
    for line in all_lines:
        values = tuple(
            transform(word) if word is not None else None
            for word in line.words
        )
        if any(True for _ in valid_transitions(values)):
            by_quire[line.quire].append(values)

    left_token_by_block: dict[str, Counter] = {}
    right_token_by_block: dict[str, Counter] = {}
    unigram_by_block: dict[str, Counter] = {}
    position_total_by_block: dict[str, Counter] = {}
    bigram_by_block: dict[str, Counter] = {}
    context_by_block: dict[str, Counter] = {}

    left_token_total: Counter = Counter()
    right_token_total: Counter = Counter()
    unigram_total: Counter = Counter()
    position_total: Counter = Counter()
    bigram_total: Counter = Counter()
    context_total: Counter = Counter()
    profile_counts: Counter = Counter()
    profile_total = 0

    for quire, block_lines in by_quire.items():
        left_tokens: Counter = Counter()
        right_tokens: Counter = Counter()
        unigrams: Counter = Counter()
        positions: Counter = Counter()
        bigrams: Counter = Counter()
        contexts: Counter = Counter()
        for values in block_lines:
            length = len(values)
            for index, left, right in valid_transitions(values):
                position = position_bucket(index, length)
                left_tokens[left] += 1
                right_tokens[right] += 1
                unigrams[(position, right)] += 1
                positions[position] += 1
                bigrams[(position, left, right)] += 1
                contexts[(position, left)] += 1
                profile_total += 1
                if diagnostic_pairs is None or (left, right) in diagnostic_pairs:
                    profile_counts[(left, right)] += 1
        left_token_by_block[quire] = left_tokens
        right_token_by_block[quire] = right_tokens
        unigram_by_block[quire] = unigrams
        position_total_by_block[quire] = positions
        bigram_by_block[quire] = bigrams
        context_by_block[quire] = contexts
        left_token_total.update(left_tokens)
        right_token_total.update(right_tokens)
        unigram_total.update(unigrams)
        position_total.update(positions)
        bigram_total.update(bigrams)
        context_total.update(contexts)

    block_scores: dict[str, BlockScore] = {}
    unknown = ("<unknown>", id(transform))
    for quire in sorted(by_quire):
        held_left_tokens = left_token_by_block[quire]
        held_right_tokens = right_token_by_block[quire]
        held_unigrams = unigram_by_block[quire]
        held_positions = position_total_by_block[quire]
        held_bigrams = bigram_by_block[quire]
        held_contexts = context_by_block[quire]
        train_left_alphabet = {
            value for value, count in left_token_total.items()
            if count - held_left_tokens[value] > 0
        }
        train_right_alphabet = {
            value for value, count in right_token_total.items()
            if count - held_right_tokens[value] > 0
        }
        vocabulary_size = len(train_right_alphabet) + 1
        score = BlockScore()

        def map_left(value: ClassValue) -> ClassValue:
            return value if value in train_left_alphabet else unknown

        def map_right(value: ClassValue) -> ClassValue:
            return value if value in train_right_alphabet else unknown

        def log_bigram(
            left: ClassValue,
            right: ClassValue,
            position: Position,
        ) -> float:
            mapped_left = map_left(left)
            mapped_right = map_right(right)
            count = subtract(
                bigram_total,
                held_bigrams,
                (position, mapped_left, mapped_right),
            )
            denominator = subtract(
                context_total,
                held_contexts,
                (position, mapped_left),
            )
            return math.log2(
                (count + ALPHA)
                / (denominator + ALPHA * vocabulary_size)
            )

        def log_unigram(right: ClassValue, position: Position) -> float:
            mapped_right = map_right(right)
            count = subtract(
                unigram_total,
                held_unigrams,
                (position, mapped_right),
            )
            denominator = subtract(
                position_total,
                held_positions,
                position,
            )
            return math.log2(
                (count + ALPHA)
                / (denominator + ALPHA * vocabulary_size)
            )

        for values in by_quire[quire]:
            length = len(values)
            for index, left, right in valid_transitions(values):
                position = position_bucket(index, length)
                forward = log_bigram(left, right, position)
                baseline = log_unigram(right, position)
                score.gain_sum += forward - baseline
                score.gain_transitions += 1
            if length < 7:
                continue
            reversed_values = reverse_deep_interior(values)
            for index, left, right in valid_transitions(values):
                reversed_left = reversed_values[index - 1]
                reversed_right = reversed_values[index]
                # Break locations are fixed, so these should be jointly valid.
                if reversed_left is None or reversed_right is None:
                    continue
                position = position_bucket(index, length)
                score.direction_sum += (
                    log_bigram(left, right, position)
                    - log_bigram(reversed_left, reversed_right, position)
                )
                score.direction_transitions += 1
        block_scores[quire] = score

    blocks = list(block_scores.values())
    gain_values = tuple(block.gain for block in blocks)
    direction_values = tuple(block.direction for block in blocks)
    gain_transitions = sum(block.gain_transitions for block in blocks)
    direction_transitions = sum(
        block.direction_transitions for block in blocks
    )
    profile = {
        pair: count / profile_total
        for pair, count in profile_counts.items()
    } if profile_total else {}
    return DatasetScore(
        gain=sum(block.gain_sum for block in blocks) / gain_transitions,
        direction=(
            sum(block.direction_sum for block in blocks)
            / direction_transitions
        ),
        gain_positive=mean(value > 0 for value in gain_values),
        direction_positive=mean(value > 0 for value in direction_values),
        gain_median=median(gain_values),
        direction_median=median(direction_values),
        block_gain=gain_values,
        block_direction=direction_values,
        profile=profile,
    )


def exchange_groups(lines: Sequence[Line]) -> dict[tuple[str, ...], list[tuple[int, int]]]:
    groups: dict[tuple[str, ...], list[tuple[int, int]]] = defaultdict(list)
    for line_index, line in enumerate(lines):
        length = len(line.words)
        for word_index, word in enumerate(line.words):
            if word is None:
                continue
            groups[(
                line.section,
                line.currier,
                line.hand,
                line.quire,
                position_bucket(word_index, length),
            )].append((line_index, word_index))
    return groups


def layout_exchange(
    source: Sequence[Line],
    groups: dict[tuple[str, ...], list[tuple[int, int]]],
    rng: random.Random,
) -> list[Line]:
    words = [list(line.words) for line in source]
    for locations in groups.values():
        values = [source[i].words[j] for i, j in locations]
        rng.shuffle(values)
        for (line_index, word_index), value in zip(locations, values):
            words[line_index][word_index] = value
    return [
        Line(
            folio=line.folio,
            quire=line.quire,
            section=line.section,
            currier=line.currier,
            hand=line.hand,
            words=tuple(line_words),
        )
        for line, line_words in zip(source, words)
    ]


class SchemaCopyGenerator:
    """Exact residual.py schema and copy parameterization."""

    def __init__(self, words: Sequence[str]) -> None:
        core_counts: Counter[str] = Counter()
        prefixes: dict[str, Counter[str]] = defaultdict(Counter)
        suffixes: dict[str, Counter[str]] = defaultdict(Counter)
        for word in words:
            prefix, core, suffix = generator_decompose(word)
            core_counts[core] += 1
            prefixes[core][prefix] += 1
            suffixes[core][suffix] += 1
        self.core_values = tuple(core_counts)
        self.core_weights = tuple(core_counts.values())
        self.prefix_tables = {
            core: (tuple(counts), tuple(counts.values()))
            for core, counts in prefixes.items()
        }
        self.suffix_tables = {
            core: (tuple(counts), tuple(counts.values()))
            for core, counts in suffixes.items()
        }
        self.glyphs = tuple(sorted(set("".join(words))))

    def sample_word(self, rng: random.Random) -> str:
        core = rng.choices(
            self.core_values,
            weights=self.core_weights,
            k=1,
        )[0]
        prefix_values, prefix_weights = self.prefix_tables[core]
        suffix_values, suffix_weights = self.suffix_tables[core]
        return (
            rng.choices(prefix_values, weights=prefix_weights, k=1)[0]
            + core
            + rng.choices(suffix_values, weights=suffix_weights, k=1)[0]
        )

    def generate(self, n: int, rng: random.Random) -> list[str]:
        generated: list[str] = []
        for _ in range(n):
            if generated and rng.random() < 0.12:
                window = min(40, len(generated))
                word = generated[-rng.randint(1, window)]
                if len(word) > 1 and rng.random() < 0.4:
                    index = rng.randrange(len(word))
                    word = (
                        word[:index]
                        + rng.choice(self.glyphs)
                        + word[index + 1:]
                    )
                generated.append(word)
            else:
                generated.append(self.sample_word(rng))
        return generated


def self_citation(
    source_words: Sequence[str],
    n: int,
    rng: random.Random,
) -> list[str]:
    """Exact branching behavior of null_and_morph.py:self_citation."""
    generated = [rng.choice(source_words[:50])]
    glyphs = tuple(sorted(set("".join(source_words))))
    for _ in range(n - 1):
        source = rng.choice(generated[-40:])
        draw = rng.random()
        if draw < 0.12:
            generated.append(source)
        elif draw < 0.12 + 0.55 and len(source) > 1:
            index = rng.randrange(len(source))
            generated.append(
                source[:index] + rng.choice(glyphs) + source[index + 1:]
            )
        else:
            generated.append(source)
    return generated


def reflow(template: Sequence[Line], stream: Sequence[str]) -> list[Line]:
    result: list[Line] = []
    offset = 0
    for line in template:
        line_words: list[Optional[str]] = []
        for word in line.words:
            if word is None:
                line_words.append(None)
            else:
                line_words.append(stream[offset])
                offset += 1
        result.append(Line(
            folio=line.folio,
            quire=line.quire,
            section=line.section,
            currier=line.currier,
            hand=line.hand,
            words=tuple(line_words),
        ))
    if offset != len(stream):
        raise ValueError(f"template used {offset} of {len(stream)} words")
    return result


def interval(values: Sequence[float]) -> tuple[float, float]:
    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def empirical_upper(real: float, null: Sequence[float]) -> float:
    return (1 + sum(value >= real for value in null)) / (len(null) + 1)


def exact_sign_flip(values: Sequence[float]) -> float:
    """Exact one-sided sign-flip test on equally weighted block effects."""
    observed = sum(values)
    exceed = 0
    total = 1 << len(values)
    for mask in range(total):
        permuted = sum(
            -value if mask & (1 << index) else value
            for index, value in enumerate(values)
        )
        if permuted >= observed - 1e-15:
            exceed += 1
    return exceed / total


def format_class(value: ClassValue) -> str:
    if isinstance(value, tuple):
        return "(" + ",".join(format_class(item) for item in value) + ")"
    return str(value)


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def say(self, value: str = "") -> None:
        print(value)
        self.lines.append(value)

    def write(self, path: Path | None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def report_family(
    reporter: Reporter,
    name: str,
    real: dict[str, DatasetScore],
    null: dict[str, list[DatasetScore]],
) -> None:
    reporter.say("\n" + "=" * 108)
    reporter.say(name)
    reporter.say("=" * 108)
    reporter.say(
        f"{'repr':12s} {'real gain':>10s} {'null gain 95%':>23s} {'p':>7s} "
        f"{'real dir':>10s} {'null dir 95%':>23s} {'p':>7s}"
    )
    for representation in REPRESENTATIONS:
        observed = real[representation]
        samples = null[representation]
        gains = [sample.gain for sample in samples]
        directions = [sample.direction for sample in samples]
        gain_lo, gain_hi = interval(gains)
        dir_lo, dir_hi = interval(directions)
        reporter.say(
            f"{representation:12s} {observed.gain:+10.4f} "
            f"[{gain_lo:+.4f},{gain_hi:+.4f}] "
            f"{empirical_upper(observed.gain, gains):7.4f} "
            f"{observed.direction:+10.4f} "
            f"[{dir_lo:+.4f},{dir_hi:+.4f}] "
            f"{empirical_upper(observed.direction, directions):7.4f}"
        )

    reporter.say("\nCross-quire stability (fraction of 16 held-out quires > 0):")
    reporter.say(
        f"{'repr':12s} {'real G':>8s} {'null G 95%':>19s} "
        f"{'real D':>8s} {'null D 95%':>19s} {'match?':>10s}"
    )
    all_match = True
    for representation in REPRESENTATIONS:
        observed = real[representation]
        samples = null[representation]
        gain_stability = [sample.gain_positive for sample in samples]
        dir_stability = [sample.direction_positive for sample in samples]
        gain_lo, gain_hi = interval(gain_stability)
        dir_lo, dir_hi = interval(dir_stability)
        magnitude_match = (
            interval([sample.gain for sample in samples])[0]
            <= observed.gain
            <= interval([sample.gain for sample in samples])[1]
            and interval([sample.direction for sample in samples])[0]
            <= observed.direction
            <= interval([sample.direction for sample in samples])[1]
        )
        stability_match = (
            gain_lo <= observed.gain_positive <= gain_hi
            and dir_lo <= observed.direction_positive <= dir_hi
        )
        matched = magnitude_match and stability_match
        all_match = all_match and matched
        reporter.say(
            f"{representation:12s} {observed.gain_positive:8.3f} "
            f"[{gain_lo:.3f},{gain_hi:.3f}] "
            f"{observed.direction_positive:8.3f} "
            f"[{dir_lo:.3f},{dir_hi:.3f}] "
            f"{'YES' if matched else 'NO':>10s}"
        )
    reporter.say(
        "Family verdict: "
        + (
            "reproduces magnitude and stability for every representation."
            if all_match
            else "does NOT reproduce both magnitude and stability across all "
                 "representations."
        )
    )


def report_diagnostics(
    reporter: Reporter,
    family: str,
    real: dict[str, DatasetScore],
    samples: dict[str, list[DatasetScore]],
    candidates: dict[str, set[Transition]],
) -> None:
    reporter.say(f"\nDiagnostic transition residuals: {family}")
    reporter.say(
        "Largest real-minus-null probability differences among the 30 most "
        "frequent real transitions."
    )
    for representation in REPRESENTATIONS:
        rows = []
        for pair in candidates[representation]:
            real_rate = real[representation].profile.get(pair, 0.0)
            rates = [
                sample.profile.get(pair, 0.0)
                for sample in samples[representation]
            ]
            null_mean = mean(rates)
            low, high = interval(rates)
            rows.append((
                abs(real_rate - null_mean),
                real_rate - null_mean,
                pair,
                real_rate,
                null_mean,
                low,
                high,
            ))
        rows.sort(reverse=True, key=lambda row: row[0])
        reporter.say(f"  {representation}:")
        for _, difference, pair, real_rate, null_mean, low, high in rows[:4]:
            transition = (
                f"{format_class(pair[0])} -> {format_class(pair[1])}"
            )
            reporter.say(
                f"    {transition:47.47s} real={100 * real_rate:6.3f}% "
                f"null={100 * null_mean:6.3f}% "
                f"95%=[{100 * low:6.3f},{100 * high:6.3f}] "
                f"diff={100 * difference:+6.3f}pp"
            )


def real_transition_candidates(
    lines: Sequence[Line],
    transform: Callable[[str], ClassValue],
    limit: int = 30,
) -> set[Transition]:
    counts: Counter = Counter()
    for line in lines:
        values = [
            transform(word) if word is not None else None
            for word in line.words
        ]
        counts.update(
            (left, right)
            for _, left, right in valid_transitions(values)
        )
    return {pair for pair, _ in counts.most_common(limit)}


def independent_rng(seed: int, family: int, replicate: int) -> random.Random:
    # Large coprime offsets make the three streams stable if loop order changes.
    return random.Random(seed + family * 1_000_003 + replicate * 10_007)


_WORKER_STATE: dict[str, object] = {}


def initialize_worker(
    lines: list[Line],
    words: list[str],
    groups: dict[tuple[str, ...], list[tuple[int, int]]],
    schema: SchemaCopyGenerator,
    candidates: dict[str, set[Transition]],
    seed: int,
) -> None:
    _WORKER_STATE.update({
        "lines": lines,
        "words": words,
        "groups": groups,
        "schema": schema,
        "candidates": candidates,
        "seed": seed,
    })


def score_replicate(
    replicate: int,
) -> tuple[dict[str, DatasetScore], ...]:
    lines = _WORKER_STATE["lines"]
    words = _WORKER_STATE["words"]
    groups = _WORKER_STATE["groups"]
    schema = _WORKER_STATE["schema"]
    candidates = _WORKER_STATE["candidates"]
    seed = _WORKER_STATE["seed"]
    assert isinstance(lines, list)
    assert isinstance(words, list)
    assert isinstance(groups, dict)
    assert isinstance(schema, SchemaCopyGenerator)
    assert isinstance(candidates, dict)
    assert isinstance(seed, int)

    generated_sets = (
        layout_exchange(
            lines,
            groups,
            independent_rng(seed, 1, replicate),
        ),
        reflow(
            lines,
            schema.generate(
                len(words),
                independent_rng(seed, 2, replicate),
            ),
        ),
        reflow(
            lines,
            self_citation(
                words,
                len(words),
                independent_rng(seed, 3, replicate),
            ),
        ),
    )
    result = tuple(
        {
            representation: score_dataset(
                generated,
                transform,
                candidates[representation],
            )
            for representation, transform in REPRESENTATIONS.items()
        }
        for generated in generated_sets
    )
    # Synthetic one-glyph variants are mostly replicate-specific. Retaining them
    # in the morphology cache across a long run only increases memory pressure.
    score_decompose.cache_clear()
    return result


def run(args: argparse.Namespace) -> Reporter:
    reporter = Reporter()
    lines, audit = load_lines(args.corpus)
    words = [
        word for line in lines for word in line.words
        if word is not None
    ]
    quires = sorted({line.quire for line in lines})
    primary_transitions = sum(
        1
        for line in lines
        for _ in valid_transitions(line.words)
    )
    direction_transitions = sum(
        1
        for line in lines
        if len(line.words) >= 7
        for _ in valid_transitions(line.words)
    )

    reporter.say("MORPHOTACTIC PROCEDURAL-NULL BENCHMARK")
    reporter.say(
        f"seed={args.seed} independent replicates={args.replicates} alpha={ALPHA}"
    )
    reporter.say(
        f"prose lines={len(lines)} quires={len(quires)} "
        f"source tokens={audit['source_tokens']} eligible tokens={len(words)} "
        f"uncertain breaks={audit['uncertain_breaks']} "
        f"one-char breaks={audit['one_char_breaks']}"
    )
    reporter.say(
        f"primary adjacent transitions={primary_transitions}; "
        f"direction transitions on original lines >=7 slots="
        f"{direction_transitions}"
    )
    reporter.say(
        "Primary: held-out log2 P(y|predecessor,target-position) - "
        "log2 P(y|target-position)."
    )
    reporter.say(
        "Direction: held-out forward minus deep-interior-reversed bigram "
        "log probability."
    )

    candidates = {
        name: real_transition_candidates(lines, transform)
        for name, transform in REPRESENTATIONS.items()
    }
    real = {
        name: score_dataset(lines, transform, candidates[name])
        for name, transform in REPRESENTATIONS.items()
    }

    reporter.say("\nObserved Voynich block inference:")
    reporter.say(
        f"{'repr':12s} {'gain':>10s} {'G median':>10s} {'G +blocks':>10s} "
        f"{'G sign-p':>10s} {'direction':>11s} {'D median':>10s} "
        f"{'D +blocks':>10s} {'D sign-p':>10s}"
    )
    for representation, score in real.items():
        reporter.say(
            f"{representation:12s} {score.gain:+10.4f} "
            f"{score.gain_median:+10.4f} {score.gain_positive:10.3f} "
            f"{exact_sign_flip(score.block_gain):10.4f} "
            f"{score.direction:+11.4f} {score.direction_median:+10.4f} "
            f"{score.direction_positive:10.3f} "
            f"{exact_sign_flip(score.block_direction):10.4f}"
        )

    controls: dict[str, dict[str, list[DatasetScore]]] = {
        "LAYOUT EXCHANGE (section x Currier x hand x quire x position)": {
            name: [] for name in REPRESENTATIONS
        },
        "SCHEMA/COPY (residual.py matched generator)": {
            name: [] for name in REPRESENTATIONS
        },
        "LOCAL SELF-CITATION (null_and_morph.py stress control)": {
            name: [] for name in REPRESENTATIONS
        },
    }
    groups = exchange_groups(lines)
    schema = SchemaCopyGenerator(words)
    families = list(controls)

    worker_arguments = (lines, words, groups, schema, candidates, args.seed)
    if args.jobs == 1:
        initialize_worker(*worker_arguments)
        replicate_results = map(score_replicate, range(args.replicates))
    else:
        context = multiprocessing.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=args.jobs,
            mp_context=context,
            initializer=initialize_worker,
            initargs=worker_arguments,
        )
        replicate_results = executor.map(
            score_replicate,
            range(args.replicates),
            chunksize=1,
        )

    try:
        for replicate, generated_scores in enumerate(replicate_results):
            for family, scores in zip(families, generated_scores):
                for representation, score in scores.items():
                    controls[family][representation].append(score)
            if args.progress and (
                replicate == 0
                or (replicate + 1) % max(1, args.replicates // 10) == 0
            ):
                print(
                    f"[progress] {replicate + 1}/{args.replicates} replicates",
                    flush=True,
                )
    finally:
        if args.jobs != 1:
            executor.shutdown()

    for family in families:
        report_family(reporter, family, real, controls[family])
        report_diagnostics(
            reporter,
            family,
            real,
            controls[family],
            candidates,
        )

    reporter.say("\n" + "=" * 108)
    reporter.say("INTERPRETATION BOUNDARY")
    reporter.say("=" * 108)
    reporter.say(
        "A failed null says the observed directional morphotactics require more "
        "than that specific content-free process. It does not by itself imply "
        "syntax, semantics, or plaintext."
    )
    reporter.say(
        "Schema tables were estimated once on the full Voynich stream. The copy "
        "hyperparameters are inherited hard-coded values from residual.py, not "
        "a demonstrated fit. Held-out scoring excludes each generated quire, "
        "but schema estimation is not nested."
    )
    reporter.say(
        "Empirical comparison p-values are raw. With 60 replicates their minimum "
        "is .0164, so they do not by themselves resolve a four-representation "
        "Bonferroni threshold; use the effect intervals and block tests jointly."
    )
    reporter.say(
        "The local self-citation control is intentionally severe and "
        "single-lineage: it preserves length from one seed and has .45 total "
        "exact copying. A failure is informative; a match would not validate "
        "that model as a realistic manuscript generator."
    )
    return reporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/corpus.json"),
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "parallel replicate workers; use up to "
            f"{min(3, os.cpu_count() or 1)} where OS semaphores are available"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.replicates < 20:
        print(
            "warning: fewer than 20 replicates gives unstable empirical intervals",
        )
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    output = run(arguments)
    output.write(arguments.output)
