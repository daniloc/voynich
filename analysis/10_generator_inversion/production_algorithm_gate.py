#!/usr/bin/env python3
"""
Identify the smallest causal production model that predicts held-out Voynich.

This is a source-model experiment, not a plaintext decoder.  Each candidate
assigns a probability to the next complete word using only prior words and
known page/line metadata.  The declared ladder is:

    unigram characters
    global character bigram/trigram word generators
    position/register-conditioned trigram word generator
    + a causal folio-level transition cache
    + exact copying from the previous eight words
    + a normalized recent-source edit channel

Complete quire blocks provide fit, validation, and test folds.  No held-out
word, character count, edit pair, mixture weight, or candidate selection enters
training.  The edit channel uses a prefix-decodable canonical edit script:
geometric insertions at source gaps followed by delete/keep/substitute choices
for every source character.

Controls:

* SYNTHETIC_COPY_EDIT is generated from the fitted family with planted source
  operations.  It tests model selection and operation recovery.
* SYNTHETIC_BASE_ONLY is generated from the character model alone.  It tests
  whether the larger operation family manufactures a false gain.
* LATIN_REFLOW places ordinary Latin words in the exact Voynich block/layout
  template.  It shows which gains are generic to meaningful running language.

The output reports held-out code length, selected operation weights, operation
posterior assignments, and residual operation-order dependence.  A compact
model is an algorithmic description of production, not evidence of meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "corpus.json"
LATIN = ROOT / "data" / "controls" / "latin.txt"
OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_production_algorithm.json"
)

SEED = 20260724
N_FOLDS = 4
HISTORY = 8
CHAR_ALPHA = 0.1
CHAR_BACKOFF = 24.0
CELL_BACKOFF = 48.0
ADAPT_BACKOFF = 32.0
MIXTURE_PRIOR = 1.5
EDIT_ALPHA = 0.5
MAX_WORD_LENGTH = 24
OPS = ("base", "copy", "edit")


@dataclass(frozen=True)
class Event:
    sequence: int
    segment: int
    block: str
    folio: str
    line: str
    position: str
    section: str
    currier: str
    hand: str
    word: str
    history: tuple[str, ...]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    order: int
    cell_mode: str
    operations: tuple[str, ...]


CANDIDATES = (
    CandidateSpec("char_unigram", 0, "global", ("base",)),
    CandidateSpec("char_bigram", 1, "global", ("base",)),
    CandidateSpec("char_trigram", 2, "global", ("base",)),
    CandidateSpec("register_trigram", 2, "register", ("base",)),
    CandidateSpec("context_trigram", 2, "context", ("base",)),
    CandidateSpec("surface_trigram", 2, "surface", ("base",)),
    CandidateSpec(
        "adaptive_register_trigram", 2, "adaptive_register", ("base",)
    ),
    CandidateSpec(
        "register_trigram_copy", 2, "register", ("base", "copy")
    ),
    CandidateSpec(
        "register_trigram_edit", 2, "register", ("base", "edit")
    ),
    CandidateSpec(
        "register_trigram_copy_edit",
        2,
        "register",
        ("base", "copy", "edit"),
    ),
    CandidateSpec(
        "surface_trigram_copy", 2, "surface", ("base", "copy")
    ),
    CandidateSpec(
        "surface_trigram_copy_edit",
        2,
        "surface",
        ("base", "copy", "edit"),
    ),
    CandidateSpec(
        "adaptive_register_trigram_copy_edit",
        2,
        "adaptive_register",
        ("base", "copy", "edit"),
    ),
)

STRUCTURAL_BASE = "adaptive_register_trigram"
FULL_MODEL = "adaptive_register_trigram_copy_edit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


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


def length_bucket(word: str) -> str:
    if len(word) <= 3:
        return "short"
    if len(word) <= 5:
        return "medium"
    if len(word) <= 7:
        return "long"
    return "very_long"


def word_shape(word: str) -> tuple[str, str, str]:
    return (word[0], word[-1], length_bucket(word))


def load_events(path: Path) -> tuple[list[Event], Counter[str]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    audit: Counter[str] = Counter()
    histories: dict[str, list[str]] = defaultdict(list)
    segments: Counter = Counter()
    events: list[Event] = []
    sequence = 0
    for folio in sorted(source["folios"], key=folio_key):
        meta = source["meta"].get(folio, {})
        block = str(meta.get("Q", "?"))
        for raw in source["folios"][folio]:
            if locus_type(raw["locus"]) != "P":
                continue
            audit["prose_lines"] += 1
            raw_words = raw["words"]
            for index, word in enumerate(raw_words):
                audit["source_tokens"] += 1
                if (
                    "?" in word
                    or not word.isalpha()
                    or len(word) < 2
                ):
                    audit["hard_breaks"] += 1
                    histories[block].clear()
                    segments[block] += 1
                    continue
                history = tuple(reversed(histories[block][-HISTORY:]))
                events.append(Event(
                    sequence=sequence,
                    segment=segments[block],
                    block=block,
                    folio=folio,
                    line=str(raw["line"]),
                    position=position_bucket(index, len(raw_words)),
                    section=str(meta.get("I", "?")),
                    currier=str(meta.get("L", "?")),
                    hand=str(meta.get("H", "?")),
                    word=word,
                    history=history,
                ))
                histories[block].append(word)
                sequence += 1
                audit["eligible_words"] += 1
    return events, audit


def rebuild_histories(
    template: Sequence[Event],
    words: Sequence[str],
) -> list[Event]:
    histories: dict[str, list[str]] = defaultdict(list)
    previous_segments: dict[str, int] = {}
    result = []
    for event, word in zip(template, words):
        if previous_segments.get(event.block, event.segment) != event.segment:
            histories[event.block].clear()
        history = tuple(reversed(histories[event.block][-HISTORY:]))
        result.append(Event(
            sequence=event.sequence,
            segment=event.segment,
            block=event.block,
            folio=event.folio,
            line=event.line,
            position=event.position,
            section=event.section,
            currier=event.currier,
            hand=event.hand,
            word=word,
            history=history,
        ))
        histories[event.block].append(word)
        previous_segments[event.block] = event.segment
    return result


def logsum2(values: Iterable[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("-inf")
    maximum = max(finite)
    return maximum + math.log2(
        sum(2.0 ** (value - maximum) for value in finite)
    )


class CharacterModel:
    """Hierarchically smoothed, normalized word-level character model."""

    def __init__(
        self,
        events: Sequence[Event],
        order: int,
        cell_mode: str,
    ) -> None:
        self.order = order
        self.cell_mode = cell_mode
        self.previous_counts = Counter(
            event.history[0] for event in events if event.history
        )
        alphabet = sorted({character for event in events for character in event.word})
        self.unknown = "<UNK>"
        self.eos = "$"
        self.alphabet = tuple(alphabet + [self.unknown, self.eos])
        self.alphabet_set = set(alphabet)
        self.global_counts: list[Counter] = [
            Counter() for _ in range(order + 1)
        ]
        self.global_totals: list[Counter] = [
            Counter() for _ in range(order + 1)
        ]
        self.cell_counts: Counter = Counter()
        self.cell_totals: Counter = Counter()
        for event in events:
            self._add(event)

    def cells(self, event: Event) -> tuple[tuple[str, ...], ...]:
        if self.cell_mode == "global":
            return ()
        register = (
            "register",
            event.currier,
            event.section,
            event.position,
        )
        if self.cell_mode == "position":
            return (("position", event.position),)
        if self.cell_mode in {"register", "adaptive_register"}:
            return (register,)
        if self.cell_mode in {"context", "surface"}:
            if not event.history:
                return (register, ("shape", *register, "<START>"))
            previous = event.history[0]
            shape = ("shape", *register, *word_shape(previous))
            if self.cell_mode == "context":
                return (register, shape)
            if self.previous_counts[previous] >= 5:
                return (
                    register,
                    shape,
                    ("surface", *register, previous),
                )
            return (register, shape)
        raise ValueError(self.cell_mode)

    def symbols(self, word: str) -> list[str]:
        return [
            character if character in self.alphabet_set else self.unknown
            for character in word
        ] + [self.eos]

    def _add(self, event: Event) -> None:
        history = ["^"] * self.order
        cells = self.cells(event)
        for symbol in self.symbols(event.word):
            for depth in range(self.order + 1):
                context = tuple(history[-depth:]) if depth else ()
                self.global_counts[depth][(context, symbol)] += 1
                self.global_totals[depth][context] += 1
            top_context = tuple(history[-self.order:]) if self.order else ()
            for cell in cells:
                self.cell_counts[(cell, top_context, symbol)] += 1
                self.cell_totals[(cell, top_context)] += 1
            history.append(symbol)

    def global_probability(
        self,
        history: Sequence[str],
        symbol: str,
        depth: int,
    ) -> float:
        if depth == 0:
            context: tuple[str, ...] = ()
            return (
                self.global_counts[0][(context, symbol)] + CHAR_ALPHA
            ) / (
                self.global_totals[0][context]
                + CHAR_ALPHA * len(self.alphabet)
            )
        context = tuple(history[-depth:])
        lower = self.global_probability(history, symbol, depth - 1)
        return (
            self.global_counts[depth][(context, symbol)]
            + CHAR_BACKOFF * lower
        ) / (
            self.global_totals[depth][context] + CHAR_BACKOFF
        )

    def probability(
        self,
        event: Event,
        history: Sequence[str],
        symbol: str,
    ) -> float:
        global_probability = self.global_probability(
            history, symbol, self.order
        )
        cells = self.cells(event)
        if not cells:
            return global_probability
        context = tuple(history[-self.order:]) if self.order else ()
        probability = global_probability
        for cell in cells:
            probability = (
                self.cell_counts[(cell, context, symbol)]
                + CELL_BACKOFF * probability
            ) / (
                self.cell_totals[(cell, context)] + CELL_BACKOFF
            )
        return probability

    def log_probability(self, event: Event) -> float:
        history = ["^"] * self.order
        total = 0.0
        for symbol in self.symbols(event.word):
            probability = self.probability(event, history, symbol)
            total += math.log2(max(probability, 1e-300))
            history.append(symbol)
        return total

    def log_probabilities(self, events: Sequence[Event]) -> list[float]:
        if self.cell_mode != "adaptive_register":
            return [self.log_probability(event) for event in events]
        adaptive_counts: Counter = Counter()
        adaptive_totals: Counter = Counter()
        result = []
        for event in events:
            history = ["^"] * self.order
            total = 0.0
            emitted = self.symbols(event.word)
            for symbol in emitted:
                context = tuple(history[-self.order:]) if self.order else ()
                base_probability = self.probability(event, history, symbol)
                key = (event.folio, context)
                probability = (
                    adaptive_counts[(key, symbol)]
                    + ADAPT_BACKOFF * base_probability
                ) / (
                    adaptive_totals[key] + ADAPT_BACKOFF
                )
                total += math.log2(max(probability, 1e-300))
                history.append(symbol)
            result.append(total)
            history = ["^"] * self.order
            for symbol in emitted:
                context = tuple(history[-self.order:]) if self.order else ()
                key = (event.folio, context)
                adaptive_counts[(key, symbol)] += 1
                adaptive_totals[key] += 1
                history.append(symbol)
        return result

    def sample(self, event: Event, rng: random.Random) -> str:
        for _attempt in range(50):
            history = ["^"] * self.order
            result = []
            for _ in range(MAX_WORD_LENGTH):
                probabilities = [
                    self.probability(event, history, symbol)
                    for symbol in self.alphabet
                ]
                symbol = rng.choices(
                    self.alphabet, weights=probabilities, k=1
                )[0]
                if symbol == self.eos:
                    if len(result) >= 2:
                        return "".join(result)
                    break
                if symbol == self.unknown:
                    symbol = rng.choice(tuple(sorted(self.alphabet_set)))
                result.append(symbol)
                history.append(symbol)
        return "ol"


@dataclass(frozen=True)
class EditStep:
    kind: str
    source_pos: int
    target_pos: int
    source_char: str
    target_char: str


@lru_cache(maxsize=250_000)
def align(source: str, target: str) -> tuple[int, tuple[EditStep, ...]]:
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
    reverse = []
    while i or j:
        if (
            i
            and j
            and source[i - 1] == target[j - 1]
            and cost[i][j] == cost[i - 1][j - 1]
        ):
            reverse.append(EditStep(
                "M", i - 1, j - 1, source[i - 1], target[j - 1]
            ))
            i -= 1
            j -= 1
        elif (
            i
            and j
            and cost[i][j] == cost[i - 1][j - 1] + 1
        ):
            reverse.append(EditStep(
                "S", i - 1, j - 1, source[i - 1], target[j - 1]
            ))
            i -= 1
            j -= 1
        elif i and cost[i][j] == cost[i - 1][j] + 1:
            reverse.append(EditStep("D", i - 1, j, source[i - 1], ""))
            i -= 1
        else:
            reverse.append(EditStep("I", i, j - 1, "", target[j - 1]))
            j -= 1
    return cost[n][m], tuple(reversed(reverse))


class EditChannel:
    """Normalized edit-script code plus a learned recent-source prior."""

    def __init__(self, events: Sequence[Event]) -> None:
        lag_counts = Counter()
        glyph_counts = Counter(
            character for event in events for character in event.word
        )
        insertions = deletions = substitutions = matches = gaps = 0
        for event in events:
            if not event.history:
                continue
            candidates = []
            for lag, source in enumerate(event.history, 1):
                distance, steps = align(source, event.word)
                candidates.append((
                    distance / max(len(source), len(event.word)),
                    distance,
                    lag,
                    source,
                    steps,
                ))
            _ratio, _distance, lag, source, steps = min(candidates)
            lag_counts[lag] += 1
            gaps += len(source) + 1
            for step in steps:
                if step.kind == "I":
                    insertions += 1
                elif step.kind == "D":
                    deletions += 1
                elif step.kind == "S":
                    substitutions += 1
                else:
                    matches += 1
        self.alphabet = tuple(sorted(glyph_counts))
        self.glyph_counts = glyph_counts
        self.glyph_total = sum(glyph_counts.values())
        self.p_insert = (insertions + EDIT_ALPHA) / (
            insertions + gaps + 2 * EDIT_ALPHA
        )
        source_actions = deletions + substitutions + matches
        self.p_delete = (deletions + EDIT_ALPHA) / (
            source_actions + 2 * EDIT_ALPHA
        )
        emitted = substitutions + matches
        self.p_substitute = (substitutions + EDIT_ALPHA) / (
            emitted + 2 * EDIT_ALPHA
        )
        self.lag_probabilities = {
            lag: (lag_counts[lag] + EDIT_ALPHA)
            / (sum(lag_counts.values()) + EDIT_ALPHA * HISTORY)
            for lag in range(1, HISTORY + 1)
        }
        self.audit = {
            "nearest_pairs": sum(lag_counts.values()),
            "insertions": insertions,
            "deletions": deletions,
            "substitutions": substitutions,
            "matches": matches,
            "p_insert": self.p_insert,
            "p_delete": self.p_delete,
            "p_substitute": self.p_substitute,
            "lag_probabilities": self.lag_probabilities,
        }

    def glyph_probability(self, character: str) -> float:
        return (
            self.glyph_counts[character] + EDIT_ALPHA
        ) / (
            self.glyph_total + EDIT_ALPHA * (len(self.alphabet) + 1)
        )

    def substitution_probability(
        self, source: str, target: str
    ) -> float:
        target_probability = self.glyph_probability(target)
        source_probability = self.glyph_probability(source)
        return target_probability / max(1.0 - source_probability, 1e-12)

    def log_probability(self, source: str, target: str) -> float:
        _distance, steps = align(source, target)
        insertions = [step for step in steps if step.kind == "I"]
        total = (
            len(insertions) * math.log2(max(self.p_insert, 1e-300))
            + (len(source) + 1)
            * math.log2(max(1.0 - self.p_insert, 1e-300))
        )
        for step in insertions:
            total += math.log2(
                max(self.glyph_probability(step.target_char), 1e-300)
            )
        for step in steps:
            if step.kind == "I":
                continue
            if step.kind == "D":
                total += math.log2(max(self.p_delete, 1e-300))
            elif step.kind == "M":
                total += math.log2(max(1.0 - self.p_delete, 1e-300))
                total += math.log2(max(1.0 - self.p_substitute, 1e-300))
            else:
                total += math.log2(max(1.0 - self.p_delete, 1e-300))
                total += math.log2(max(self.p_substitute, 1e-300))
                total += math.log2(max(
                    self.substitution_probability(
                        step.source_char, step.target_char
                    ),
                    1e-300,
                ))
        return total

    def source_log_probability(self, lag: int) -> float:
        return math.log2(max(self.lag_probabilities[lag], 1e-300))

    def copy_log_probability(self, event: Event) -> float:
        return logsum2(
            self.source_log_probability(lag)
            for lag, source in enumerate(event.history, 1)
            if source == event.word
        )

    def edit_log_probability(self, event: Event) -> float:
        return logsum2(
            self.source_log_probability(lag)
            + self.log_probability(source, event.word)
            for lag, source in enumerate(event.history, 1)
        )

    def draw_lag(self, history_size: int, rng: random.Random) -> int:
        lags = list(range(1, history_size + 1))
        weights = [self.lag_probabilities[lag] for lag in lags]
        return rng.choices(lags, weights=weights, k=1)[0]

    def mutate(self, source: str, rng: random.Random) -> str:
        result = []
        for index in range(len(source) + 1):
            insert_count = 0
            while (
                insert_count < 3
                and rng.random() < self.p_insert
            ):
                result.append(rng.choices(
                    self.alphabet,
                    weights=[
                        self.glyph_probability(character)
                        for character in self.alphabet
                    ],
                    k=1,
                )[0])
                insert_count += 1
            if index == len(source):
                break
            character = source[index]
            if rng.random() < self.p_delete:
                continue
            if rng.random() < self.p_substitute:
                choices = [value for value in self.alphabet if value != character]
                weights = [self.glyph_probability(value) for value in choices]
                character = rng.choices(choices, weights=weights, k=1)[0]
            result.append(character)
        return "".join(result)


def component_logs(
    event: Event,
    base: CharacterModel,
    channel: Optional[EditChannel],
    base_log_probability: Optional[float] = None,
) -> dict[str, float]:
    result = {
        "base": (
            base.log_probability(event)
            if base_log_probability is None
            else base_log_probability
        )
    }
    if channel is not None:
        result["copy"] = channel.copy_log_probability(event)
        result["edit"] = channel.edit_log_probability(event)
    return result


def fit_mixture(
    rows: Sequence[dict[str, float]],
    operations: Sequence[str],
    iterations: int = 60,
) -> dict[str, float]:
    if len(operations) == 1:
        return {operations[0]: 1.0}
    weights = {operation: 1.0 / len(operations) for operation in operations}
    for _ in range(iterations):
        totals = {operation: MIXTURE_PRIOR - 1.0 for operation in operations}
        for row in rows:
            denominator = logsum2(
                math.log2(weights[operation]) + row[operation]
                for operation in operations
            )
            for operation in operations:
                value = row[operation]
                if math.isfinite(value):
                    totals[operation] += 2.0 ** (
                        math.log2(weights[operation]) + value - denominator
                    )
        normalizer = sum(totals.values())
        updated = {
            operation: max(totals[operation] / normalizer, 1e-12)
            for operation in operations
        }
        scale = sum(updated.values())
        updated = {key: value / scale for key, value in updated.items()}
        if max(
            abs(updated[key] - weights[key]) for key in operations
        ) < 1e-9:
            weights = updated
            break
        weights = updated
    return weights


@dataclass
class FittedCandidate:
    spec: CandidateSpec
    base: CharacterModel
    channel: Optional[EditChannel]
    weights: dict[str, float]

    def rows(self, events: Sequence[Event]) -> list[dict[str, float]]:
        base_logs = self.base.log_probabilities(events)
        return [
            component_logs(event, self.base, self.channel, base_log)
            for event, base_log in zip(events, base_logs)
        ]

    def row_log_probability(self, row: dict[str, float]) -> float:
        return logsum2(
            math.log2(self.weights[operation]) + row[operation]
            for operation in self.spec.operations
        )

    def row_posterior(self, row: dict[str, float]) -> dict[str, float]:
        denominator = self.row_log_probability(row)
        return {
            operation: (
                2.0 ** (
                    math.log2(self.weights[operation])
                    + row[operation]
                    - denominator
                )
                if math.isfinite(row[operation])
                else 0.0
            )
            for operation in self.spec.operations
        }

    def classifications(self, events: Sequence[Event]) -> list[str]:
        result = []
        for row in self.rows(events):
            posterior = self.row_posterior(row)
            result.append(max(
                posterior, key=lambda value: (posterior[value], value)
            ))
        return result

    def classify(self, event: Event) -> str:
        row = self.rows([event])[0]
        posterior = self.row_posterior(row)
        return max(posterior, key=lambda value: (posterior[value], value))


def fit_candidate(
    spec: CandidateSpec,
    train: Sequence[Event],
    base_cache: dict[tuple[int, str], CharacterModel],
    channel: EditChannel,
) -> FittedCandidate:
    key = (spec.order, spec.cell_mode)
    if key not in base_cache:
        base_cache[key] = CharacterModel(train, spec.order, spec.cell_mode)
    base = base_cache[key]
    candidate_channel = channel if len(spec.operations) > 1 else None
    base_logs = base.log_probabilities(train)
    rows = [
        component_logs(event, base, candidate_channel, base_log)
        for event, base_log in zip(train, base_logs)
    ]
    weights = fit_mixture(rows, spec.operations)
    return FittedCandidate(spec, base, candidate_channel, weights)


def score_candidate(
    candidate: FittedCandidate,
    events: Sequence[Event],
    include_assignments: bool = False,
) -> dict[str, object]:
    total = 0.0
    classifications: Counter[str] = Counter()
    posterior_totals: Counter[str] = Counter()
    posterior_entropy = 0.0
    per_event = []
    rows = candidate.rows(events)
    for event, row in zip(events, rows):
        value = candidate.row_log_probability(row)
        total += value
        if include_assignments:
            posterior = candidate.row_posterior(row)
            classification = max(
                posterior, key=lambda name: (posterior[name], name)
            )
            classifications[classification] += 1
            for operation, probability in posterior.items():
                posterior_totals[operation] += probability
                if probability > 0:
                    posterior_entropy -= probability * math.log2(probability)
            per_event.append({
                "sequence": event.sequence,
                "segment": event.segment,
                "block": event.block,
                "position": event.position,
                "word_length": len(event.word),
                "operation": classification,
                "posterior": posterior,
                "surprisal": -value,
            })
    count = len(events)
    result: dict[str, object] = {
        "events": count,
        "log2_probability": total,
        "bits_per_word": -total / count,
    }
    if include_assignments:
        result.update({
            "argmax_operation_counts": dict(classifications),
            "mean_posterior_operation_share": {
                operation: posterior_totals[operation] / count
                for operation in candidate.spec.operations
            },
            "mean_operation_posterior_entropy": posterior_entropy / count,
            "per_event": per_event,
        })
    return result


def block_folds(events: Sequence[Event]) -> tuple[dict[str, int], list[int]]:
    counts = Counter(event.block for event in events)
    loads = [0] * N_FOLDS
    assignment = {}
    for block in sorted(counts, key=lambda value: (-counts[value], value)):
        fold = min(range(N_FOLDS), key=lambda value: (loads[value], value))
        assignment[block] = fold
        loads[fold] += counts[block]
    return assignment, loads


def split_events(
    events: Sequence[Event],
    assignment: dict[str, int],
    fold: int,
) -> tuple[list[Event], list[Event], list[Event]]:
    test = [event for event in events if assignment[event.block] == fold]
    validation = [
        event
        for event in events
        if assignment[event.block] == (fold + 1) % N_FOLDS
    ]
    train = [
        event
        for event in events
        if assignment[event.block] not in {fold, (fold + 1) % N_FOLDS}
    ]
    return train, validation, test


def operation_transition_gain(
    candidate: FittedCandidate,
    train: Sequence[Event],
    test: Sequence[Event],
) -> dict[str, object]:
    operations = candidate.spec.operations
    base_counts: Counter = Counter()
    base_totals: Counter = Counter()
    transition_counts: Counter = Counter()
    transition_totals: Counter = Counter()

    def labels(events: Sequence[Event]) -> list[tuple[Event, str]]:
        return list(zip(events, candidate.classifications(events)))

    previous: dict[str, tuple[int, int, str]] = {}
    for event, operation in labels(train):
        base_counts[(event.position, operation)] += 1
        base_totals[event.position] += 1
        prior = previous.get(event.block)
        if (
            prior is not None
            and event.segment == prior[0]
            and event.sequence == prior[1] + 1
        ):
            transition_counts[
                (event.position, prior[2], operation)
            ] += 1
            transition_totals[(event.position, prior[2])] += 1
        previous[event.block] = (event.segment, event.sequence, operation)

    base_bits = transition_bits = 0.0
    transitions = 0
    previous.clear()
    for event, operation in labels(test):
        prior = previous.get(event.block)
        if (
            prior is not None
            and event.segment == prior[0]
            and event.sequence == prior[1] + 1
        ):
            alpha = 0.5
            base_probability = (
                base_counts[(event.position, operation)] + alpha
            ) / (
                base_totals[event.position] + alpha * len(operations)
            )
            transition_probability = (
                transition_counts[
                    (event.position, prior[2], operation)
                ]
                + 8.0 * base_probability
            ) / (
                transition_totals[(event.position, prior[2])] + 8.0
            )
            base_bits += math.log2(base_probability)
            transition_bits += math.log2(transition_probability)
            transitions += 1
        previous[event.block] = (event.segment, event.sequence, operation)
    return {
        "transitions": transitions,
        "gain_bits_per_operation": (
            (transition_bits - base_bits) / transitions
            if transitions
            else None
        ),
    }


def residual_autocorrelation(
    per_event: Sequence[dict[str, object]],
    seed: int,
    replicates: int = 400,
) -> dict[str, object]:
    by_stratum: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in per_event:
        stratum = (str(row["position"]), int(row["word_length"]))
        by_stratum[stratum].append(
            float(row["surprisal"]) / (int(row["word_length"]) + 1)
        )
    stratum_mean = {
        stratum: sum(values) / len(values)
        for stratum, values in by_stratum.items()
    }
    rows = [
        {
            **row,
            "residual": (
                float(row["surprisal"])
                / (int(row["word_length"]) + 1)
                - stratum_mean[
                    (str(row["position"]), int(row["word_length"]))
                ]
            ),
        }
        for row in per_event
    ]

    def correlation(values: Sequence[dict[str, object]]) -> float:
        pairs = []
        previous: dict[str, tuple[int, int, float]] = {}
        for row in values:
            block = str(row["block"])
            sequence = int(row["sequence"])
            segment = int(row["segment"])
            residual = float(row["residual"])
            prior = previous.get(block)
            if (
                prior is not None
                and segment == prior[0]
                and sequence == prior[1] + 1
            ):
                pairs.append((prior[2], residual))
            previous[block] = (segment, sequence, residual)
        if not pairs:
            return float("nan")
        left_mean = sum(left for left, _ in pairs) / len(pairs)
        right_mean = sum(right for _, right in pairs) / len(pairs)
        numerator = sum(
            (left - left_mean) * (right - right_mean)
            for left, right in pairs
        )
        left_square = sum((left - left_mean) ** 2 for left, _ in pairs)
        right_square = sum((right - right_mean) ** 2 for _, right in pairs)
        denominator = math.sqrt(left_square * right_square)
        return numerator / denominator if denominator else 0.0

    observed = correlation(rows)
    rng = random.Random(seed)
    nulls = []
    groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        groups[(
            str(row["block"]),
            str(row["position"]),
            int(row["word_length"]),
        )].append(
            float(row["residual"])
        )
    for _ in range(replicates):
        pools = {key: list(values) for key, values in groups.items()}
        for values in pools.values():
            rng.shuffle(values)
        cursors: Counter = Counter()
        shuffled = []
        for row in rows:
            key = (
                str(row["block"]),
                str(row["position"]),
                int(row["word_length"]),
            )
            value = pools[key][cursors[key]]
            cursors[key] += 1
            shuffled.append({**row, "residual": value})
        nulls.append(correlation(shuffled))
    p_two_sided = (
        1 + sum(abs(value) >= abs(observed) - 1e-12 for value in nulls)
    ) / (len(nulls) + 1)
    return {
        "metric": (
            "lag-1 correlation of bits/character residualized by exact word "
            "length and line position"
        ),
        "lag1_residual_correlation": observed,
        "stratified_shuffle_replicates": replicates,
        "two_sided_p": p_two_sided,
        "null_mean": sum(nulls) / len(nulls),
    }


def run_dataset(
    name: str,
    events: Sequence[Event],
    truth: Optional[dict[int, str]] = None,
    progress: bool = False,
) -> dict[str, object]:
    assignment, loads = block_folds(events)
    fold_results = []
    pooled: dict[str, Counter] = {
        spec.name: Counter() for spec in CANDIDATES
    }
    selected_test_total = 0.0
    selected_test_events = 0
    truth_correct = truth_total = 0
    for fold in range(N_FOLDS):
        train, validation, test = split_events(events, assignment, fold)
        base_cache: dict[tuple[int, str], CharacterModel] = {}
        channel = EditChannel(train)
        candidates = {}
        rows = []
        for spec in CANDIDATES:
            fitted = fit_candidate(
                spec, train, base_cache, channel
            )
            validation_score = score_candidate(fitted, validation)
            test_score = score_candidate(
                fitted,
                test,
                include_assignments=(
                    spec.name == FULL_MODEL
                ),
            )
            candidates[spec.name] = fitted
            pooled[spec.name]["log2_probability"] += float(
                test_score["log2_probability"]
            )
            pooled[spec.name]["events"] += int(test_score["events"])
            rows.append({
                "candidate": spec.name,
                "validation_bits_per_word": validation_score["bits_per_word"],
                "test_bits_per_word": test_score["bits_per_word"],
                "test_log2_probability": test_score["log2_probability"],
                "weights": fitted.weights,
                "test_assignments": (
                    {
                        key: value
                        for key, value in test_score.items()
                        if key
                        in {
                            "argmax_operation_counts",
                            "mean_posterior_operation_share",
                            "mean_operation_posterior_entropy",
                        }
                    }
                    if spec.name == FULL_MODEL
                    else None
                ),
            })
        winner_row = min(
            rows,
            key=lambda row: (
                row["validation_bits_per_word"],
                row["candidate"],
            ),
        )
        winner = candidates[str(winner_row["candidate"])]
        winner_test = score_candidate(
            winner, test, include_assignments=True
        )
        selected_test_total += float(winner_test["log2_probability"])
        selected_test_events += len(test)

        full = candidates[FULL_MODEL]
        full_test = score_candidate(full, test, include_assignments=True)
        transition = operation_transition_gain(full, train, test)
        operation_accuracy = None
        if truth is not None:
            correct = total = 0
            classifications = dict(zip(
                (event.sequence for event in test),
                full.classifications(test),
            ))
            for event in test:
                if event.sequence not in truth:
                    continue
                total += 1
                correct += int(
                    classifications[event.sequence] == truth[event.sequence]
                )
            truth_correct += correct
            truth_total += total
            operation_accuracy = correct / total if total else None
        fold_results.append({
            "fold": fold,
            "train_blocks": sorted({
                event.block for event in train
            }),
            "validation_blocks": sorted({
                event.block for event in validation
            }),
            "test_blocks": sorted({
                event.block for event in test
            }),
            "train_events": len(train),
            "validation_events": len(validation),
            "test_events": len(test),
            "candidate_scores": rows,
            "selected_candidate": winner.spec.name,
            "selected_test_bits_per_word": winner_test["bits_per_word"],
            "full_model_operation_transition_residual": transition,
            "planted_operation_accuracy": operation_accuracy,
            "full_model_channel": full.channel.audit if full.channel else None,
            "full_model_per_event": full_test["per_event"],
        })
        if progress:
            print(
                f"{name} fold={fold} selected={winner.spec.name} "
                f"test={float(winner_test['bits_per_word']):.4f}",
                flush=True,
            )

    candidate_summary = {}
    for spec in CANDIDATES:
        total = float(pooled[spec.name]["log2_probability"])
        count = int(pooled[spec.name]["events"])
        candidate_summary[spec.name] = {
            "test_events": count,
            "pooled_test_log2_probability": total,
            "pooled_test_bits_per_word": -total / count,
        }
    full_rows = [
        row
        for fold in fold_results
        for row in fold["full_model_per_event"]
    ]
    baseline = candidate_summary[STRUCTURAL_BASE]
    full = candidate_summary[FULL_MODEL]
    increments = {
        "trigram_to_register": (
            candidate_summary["char_trigram"]["pooled_test_bits_per_word"]
            - candidate_summary["register_trigram"]["pooled_test_bits_per_word"]
        ),
        "register_to_context": (
            candidate_summary["register_trigram"]["pooled_test_bits_per_word"]
            - candidate_summary["context_trigram"]["pooled_test_bits_per_word"]
        ),
        "context_to_surface": (
            candidate_summary["context_trigram"]["pooled_test_bits_per_word"]
            - candidate_summary["surface_trigram"]["pooled_test_bits_per_word"]
        ),
        "register_to_adaptive": (
            candidate_summary["register_trigram"]["pooled_test_bits_per_word"]
            - candidate_summary[
                "adaptive_register_trigram"
            ]["pooled_test_bits_per_word"]
        ),
        "adaptive_to_copy_edit": (
            candidate_summary[
                "adaptive_register_trigram"
            ]["pooled_test_bits_per_word"]
            - candidate_summary[FULL_MODEL]["pooled_test_bits_per_word"]
        ),
    }
    summary = {
        "selected_candidate_counts": dict(Counter(
            fold["selected_candidate"] for fold in fold_results
        )),
        "selected_pooled_test_bits_per_word": (
            -selected_test_total / selected_test_events
        ),
        "full_vs_structural_base_gain_bits_per_word": (
            float(baseline["pooled_test_bits_per_word"])
            - float(full["pooled_test_bits_per_word"])
        ),
        "algorithmic_increments_bits_per_word": increments,
        "full_model_positive_folds": sum(
            next(
                row["test_bits_per_word"]
                for row in fold["candidate_scores"]
                if row["candidate"] == STRUCTURAL_BASE
            )
            > next(
                row["test_bits_per_word"]
                for row in fold["candidate_scores"]
                if row["candidate"] == FULL_MODEL
            )
            for fold in fold_results
        ),
        "planted_operation_accuracy": (
            truth_correct / truth_total if truth_total else None
        ),
        "residual_surprisal_order": residual_autocorrelation(
            full_rows, SEED + sum(map(ord, name))
        ),
        "operation_transition_gain_bits_per_operation": (
            sum(
                fold["full_model_operation_transition_residual"][
                    "gain_bits_per_operation"
                ]
                * fold["full_model_operation_transition_residual"][
                    "transitions"
                ]
                for fold in fold_results
                if fold["full_model_operation_transition_residual"][
                    "gain_bits_per_operation"
                ]
                is not None
            )
            / sum(
                fold["full_model_operation_transition_residual"]["transitions"]
                for fold in fold_results
            )
        ),
    }
    for fold in fold_results:
        del fold["full_model_per_event"]
    return {
        "name": name,
        "fold_assignment": assignment,
        "fold_event_loads": loads,
        "candidate_summary": candidate_summary,
        "folds": fold_results,
        "summary": summary,
    }


def generated_events(
    template: Sequence[Event],
    base: CharacterModel,
    channel: EditChannel,
    weights: dict[str, float],
    seed: int,
) -> tuple[list[Event], dict[int, str]]:
    rng = random.Random(seed)
    histories: dict[str, list[str]] = defaultdict(list)
    previous_segments: dict[str, int] = {}
    words = []
    truth = {}
    for event in template:
        if previous_segments.get(event.block, event.segment) != event.segment:
            histories[event.block].clear()
        history = list(reversed(histories[event.block][-HISTORY:]))
        available = ["base"]
        if history:
            available.extend(
                operation for operation in ("copy", "edit")
                if weights.get(operation, 0) > 0
            )
        draw_weights = [weights.get(operation, 0.0) for operation in available]
        operation = rng.choices(available, weights=draw_weights, k=1)[0]
        if operation == "base":
            word = base.sample(event, rng)
        else:
            lag = channel.draw_lag(len(history), rng)
            source = history[lag - 1]
            if operation == "copy":
                word = source
            else:
                word = source
                for _ in range(20):
                    candidate = channel.mutate(source, rng)
                    if (
                        candidate != source
                        and candidate.isalpha()
                        and 2 <= len(candidate) <= MAX_WORD_LENGTH
                    ):
                        word = candidate
                        break
                if word == source:
                    operation = "copy"
        words.append(word)
        truth[event.sequence] = operation
        histories[event.block].append(word)
        previous_segments[event.block] = event.segment
    return rebuild_histories(template, words), truth


def latin_events(template: Sequence[Event], path: Path) -> list[Event]:
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    text = (
        text.replace("j", "i")
        .replace("v", "u")
        .replace("w", "u")
        .replace("k", "c")
    )
    tokens = [
        token for token in re.findall(r"[a-z]+", text) if len(token) >= 2
    ]
    if not tokens:
        raise ValueError("Latin control has no usable tokens")
    words = [tokens[index % len(tokens)] for index in range(len(template))]
    return rebuild_histories(template, words)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--latin", type=Path, default=LATIN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events, audit = load_events(args.corpus)
    real = run_dataset("VOYNICH", events, progress=args.progress)
    panels = [real]

    controls = {}
    if not args.skip_controls:
        full_base = CharacterModel(events, 2, "register")
        full_channel = EditChannel(events)
        synthetic, truth = generated_events(
            events,
            full_base,
            full_channel,
            {"base": 0.68, "copy": 0.14, "edit": 0.18},
            SEED + 10_000,
        )
        synthetic_panel = run_dataset(
            "SYNTHETIC_COPY_EDIT",
            synthetic,
            truth=truth,
            progress=args.progress,
        )
        panels.append(synthetic_panel)
        base_only, base_truth = generated_events(
            events,
            full_base,
            full_channel,
            {"base": 1.0, "copy": 0.0, "edit": 0.0},
            SEED + 20_000,
        )
        base_panel = run_dataset(
            "SYNTHETIC_BASE_ONLY",
            base_only,
            truth=base_truth,
            progress=args.progress,
        )
        panels.append(base_panel)
        latin_panel = run_dataset(
            "LATIN_REFLOW",
            latin_events(events, args.latin),
            progress=args.progress,
        )
        panels.append(latin_panel)
        controls = {
            "copy_edit_planted_weights": {
                "base": 0.68,
                "copy": 0.14,
                "edit": 0.18,
            },
            "copy_edit_operation_recovery_pass": (
                synthetic_panel["summary"]["planted_operation_accuracy"] >= 0.75
            ),
            "copy_edit_model_gain_pass": (
                synthetic_panel["summary"][
                    "full_vs_structural_base_gain_bits_per_word"
                ]
                > 0.10
            ),
            "base_only_false_positive_pass": (
                base_panel["summary"][
                    "full_vs_structural_base_gain_bits_per_word"
                ]
                < 0.02
            ),
        }

    result = {
        "experiment": "causal_production_algorithm_inversion",
        "seed": SEED,
        "claim_boundary": (
            "The winning source model describes production choices. It does "
            "not identify plaintext or prove that the residual is asemic."
        ),
        "parameters": {
            "folds": N_FOLDS,
            "split": (
                "test=fold f; validation=fold f+1; fit=other two folds; "
                "complete repository quire blocks"
            ),
            "history_words": HISTORY,
            "candidate_specs": [asdict(spec) for spec in CANDIDATES],
            "character_smoothing": {
                "alpha": CHAR_ALPHA,
                "history_backoff": CHAR_BACKOFF,
                "cell_backoff": CELL_BACKOFF,
                "adaptive_backoff": ADAPT_BACKOFF,
            },
            "mixture_prior": MIXTURE_PRIOR,
            "edit_alpha": EDIT_ALPHA,
        },
        "assets": {
            asset_name(args.corpus): sha256(args.corpus),
            asset_name(args.latin): sha256(args.latin),
        },
        "corpus_audit": dict(audit),
        "controls": controls,
        "panels": panels,
        "summary": {
            "voynich_selected_candidates": real["summary"][
                "selected_candidate_counts"
            ],
            "voynich_full_vs_structural_base_gain_bits_per_word": real["summary"][
                "full_vs_structural_base_gain_bits_per_word"
            ],
            "voynich_algorithmic_increments_bits_per_word": real["summary"][
                "algorithmic_increments_bits_per_word"
            ],
            "voynich_full_model_positive_folds": real["summary"][
                "full_model_positive_folds"
            ],
            "voynich_operation_transition_gain_bits_per_operation": (
                real["summary"][
                    "operation_transition_gain_bits_per_operation"
                ]
            ),
            "voynich_residual_surprisal_order": real["summary"][
                "residual_surprisal_order"
            ],
            "synthetic_controls_pass": (
                all(
                    value
                    for key, value in controls.items()
                    if key.endswith("_pass")
                )
                if controls
                else None
            ),
        },
        "caveats": [
            (
                "The algorithm is a probabilistic source description, not a "
                "deterministic reconstruction of each scribal decision."
            ),
            (
                "Canonical Levenshtein scripts give a valid decodable edit "
                "code but do not sum probability over alternative alignments."
            ),
            (
                "Register conditioning uses known Currier, section, and word "
                "position metadata; it deliberately excludes hand identity."
            ),
            (
                "Residual operation labels are posterior classifications, "
                "not observed ground truth outside the synthetic control."
            ),
            (
                "The adaptive predictor updates only after each observed "
                "word. Its gain can express folio-level language or topic "
                "variation and is not intrinsically evidence of a "
                "content-free copying procedure."
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
