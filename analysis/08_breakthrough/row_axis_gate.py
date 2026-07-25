#!/usr/bin/env python3
"""
Test whether Voynich prose has residual same-column coherence across rows.

The earlier Contact experiment tested whether word position inside a line
reduces vocabulary entropy.  It did not test the other table-like claim:
whether a cell is linked specifically to the cell above it.

For every target cell in adjacent prose rows on the same folio, this gate fits
three nested held-out models on complete quires:

``position``
    Target form from its numeric column bucket.

``previous_row_bag``
    A mixture of the position model and copying from anywhere in the preceding
    row.  This absorbs page-local self-citation without assuming alignment.

``previous_row_bag_plus_diagonal``
    Adds a point-mass copy component for the cell at the same numeric column.

Mixture weights are fitted by EM on fit quires and frozen for test quires.
Row-order shuffles within each folio retain page vocabulary, row shapes, and
column distributions while destroying adjacency.  They repeat the full fit
and family-wise max selection.  A weak same-column copy injection supplies a
positive control.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Callable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "corpus.json"
SEED = 20260723
ALPHA = 0.5
N_FOLDS = 4
EM_ITERATIONS = 120
UNKNOWN = "<TRAIN-ONLY-UNKNOWN>"

PREFIXES = tuple(
    sorted(
        (
            "qok",
            "qot",
            "qo",
            "ok",
            "ot",
            "o",
            "y",
            "ch",
            "sh",
            "d",
            "cth",
            "ckh",
            "cph",
            "cfh",
        ),
        key=len,
        reverse=True,
    )
)
SUFFIXES = tuple(
    sorted(
        (
            "eedy",
            "eody",
            "edy",
            "aiin",
            "aiir",
            "ain",
            "iin",
            "dy",
            "ol",
            "or",
            "ar",
            "al",
            "am",
            "dam",
            "ey",
            "eey",
            "y",
        ),
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True)
class Page:
    folio: str
    block: str
    section: str
    currier: str
    hand: str
    rows: tuple[tuple[str | None, ...], ...]


@dataclass(frozen=True)
class CellEvent:
    block: str
    position: str
    target: str
    diagonal: str | None
    previous_row: tuple[str, ...]


@dataclass
class FoldResult:
    representation: str
    fold: int
    fit_events: int
    test_events: int
    bag_weights: tuple[float, float]
    full_weights: tuple[float, float, float]
    baseline_bits: float
    bag_bits: float
    full_bits: float
    bag_gain: float
    diagonal_gain: float
    block_diagonal_gains: dict[str, float]


@dataclass
class RepresentationResult:
    representation: str
    events: int
    diagonal_match: float
    neighbor_match: float
    diagonal_excess: float
    fold_results: list[FoldResult]
    heldout_bag_gain: float
    heldout_diagonal_gain: float
    event_weighted_diagonal_gain: float
    median_block_diagonal_gain: float
    positive_blocks: int
    exact_sign_p: float
    mean_diagonal_weight: float


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


@lru_cache(maxsize=None)
def decompose(word: str) -> tuple[str, str, str]:
    prefix = next((item for item in PREFIXES if word.startswith(item)), "")
    residual = word[len(prefix) :]
    suffix = next(
        (
            item
            for item in SUFFIXES
            if residual.endswith(item) and len(residual) > len(item)
        ),
        "",
    )
    core = residual[: -len(suffix)] if suffix else residual
    return prefix, core or "<empty>", suffix


REPRESENTATIONS: dict[str, Callable[[str], str]] = {
    "word": lambda word: word,
    "core": lambda word: decompose(word)[1],
    "prefix": lambda word: decompose(word)[0] or "<none>",
    "suffix": lambda word: decompose(word)[2] or "<none>",
    "first_glyph": lambda word: word[0],
    "last_glyph": lambda word: word[-1],
}


def position_bucket(index: int) -> str:
    return f"c{index}" if index < 10 else "c10+"


def load_pages() -> tuple[list[Page], Counter[str]]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    pages = []
    audit: Counter[str] = Counter()
    for folio in sorted(corpus["folios"], key=folio_key):
        meta = corpus["meta"].get(folio, {})
        rows = []
        for raw in corpus["folios"][folio]:
            if locus_type(raw["locus"]) != "P":
                continue
            words = []
            for word in raw["words"]:
                audit["tokens"] += 1
                if "?" in word or not word.isalpha() or len(word) < 2:
                    words.append(None)
                    audit["hard_breaks"] += 1
                else:
                    words.append(word)
                    audit["eligible_words"] += 1
            if sum(word is not None for word in words) >= 3:
                rows.append(tuple(words))
                audit["eligible_rows"] += 1
        if len(rows) >= 2:
            pages.append(
                Page(
                    folio=folio,
                    block=str(meta.get("Q", "?")),
                    section=str(meta.get("I", "?")),
                    currier=str(meta.get("L", "?")),
                    hand=str(meta.get("H", "?")),
                    rows=tuple(rows),
                )
            )
            audit["eligible_pages"] += 1
    return pages, audit


def block_folds(blocks: Sequence[str]) -> dict[str, int]:
    return {
        block: index % N_FOLDS for index, block in enumerate(sorted(blocks))
    }


def make_events(
    pages: Sequence[Page],
    transform: Callable[[str], str],
) -> list[CellEvent]:
    events = []
    for page in pages:
        for source, target in zip(page.rows, page.rows[1:]):
            source_values = tuple(
                transform(word) for word in source if word is not None
            )
            if not source_values:
                continue
            for index, word in enumerate(target):
                if word is None:
                    continue
                diagonal_word = (
                    source[index]
                    if index < len(source) and source[index] is not None
                    else None
                )
                events.append(
                    CellEvent(
                        block=page.block,
                        position=position_bucket(index),
                        target=transform(word),
                        diagonal=(
                            transform(diagonal_word)
                            if diagonal_word is not None
                            else None
                        ),
                        previous_row=source_values,
                    )
                )
    return events


def raw_alignment(
    pages: Sequence[Page],
    transform: Callable[[str], str],
) -> tuple[int, float, float, float]:
    diagonal_matches = 0
    diagonal_total = 0
    neighbor_matches = 0
    neighbor_total = 0
    for page in pages:
        for source, target in zip(page.rows, page.rows[1:]):
            for index, target_word in enumerate(target):
                if target_word is None:
                    continue
                target_value = transform(target_word)
                if index < len(source) and source[index] is not None:
                    diagonal_matches += (
                        transform(source[index]) == target_value
                    )
                    diagonal_total += 1
                for source_index in (index - 1, index + 1):
                    if (
                        0 <= source_index < len(source)
                        and source[source_index] is not None
                    ):
                        neighbor_matches += (
                            transform(source[source_index]) == target_value
                        )
                        neighbor_total += 1
    diagonal = diagonal_matches / max(1, diagonal_total)
    neighbor = neighbor_matches / max(1, neighbor_total)
    return diagonal_total, diagonal, neighbor, diagonal - neighbor


@dataclass
class PositionModel:
    vocabulary: set[str]
    counts: dict[str, Counter[str]]
    totals: Counter[str]
    width: int

    def probability(self, value: str, position: str) -> float:
        mapped = value if value in self.vocabulary else UNKNOWN
        row = self.counts.get(position)
        if not row:
            row = self.counts["<global>"]
            position = "<global>"
        return (
            row[mapped] + ALPHA
        ) / (
            self.totals[position] + ALPHA * self.width
        )


def fit_position(events: Sequence[CellEvent]) -> PositionModel:
    vocabulary = {event.target for event in events}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for event in events:
        counts[event.position][event.target] += 1
        totals[event.position] += 1
        counts["<global>"][event.target] += 1
        totals["<global>"] += 1
    return PositionModel(
        vocabulary=vocabulary,
        counts=dict(counts),
        totals=totals,
        width=len(vocabulary) + 1,
    )


def component_probabilities(
    events: Sequence[CellEvent],
    base: PositionModel,
    diagonal: bool,
) -> np.ndarray:
    columns = 3 if diagonal else 2
    probabilities = np.zeros((len(events), columns), dtype=float)
    for index, event in enumerate(events):
        probabilities[index, 0] = base.probability(
            event.target, event.position
        )
        previous_counts = Counter(event.previous_row)
        probabilities[index, 1] = (
            previous_counts[event.target] / len(event.previous_row)
        )
        if diagonal:
            probabilities[index, 2] = float(
                event.diagonal is not None
                and event.diagonal == event.target
            )
    return probabilities


def fit_mixture(probabilities: np.ndarray) -> np.ndarray:
    width = probabilities.shape[1]
    if width == 2:
        weights = np.asarray((0.92, 0.08), dtype=float)
    elif width == 3:
        weights = np.asarray((0.90, 0.08, 0.02), dtype=float)
    else:
        raise ValueError(width)
    for _ in range(EM_ITERATIONS):
        weighted = probabilities * weights
        denominator = weighted.sum(axis=1)
        denominator[denominator <= 0] = 1e-300
        responsibilities = weighted / denominator[:, None]
        updated = responsibilities.mean(axis=0)
        if np.max(np.abs(updated - weights)) < 1e-10:
            weights = updated
            break
        weights = updated
    return weights


def bits_per_event(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.einsum("ij,j->i", probabilities, weights)
    return float(np.mean(np.log2(np.maximum(values, 1e-300))))


def exact_sign_flip(values: Sequence[float]) -> float:
    observed = sum(values)
    exceed = 0
    total = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        total += 1
        if (
            sum(value * sign for value, sign in zip(values, signs))
            >= observed - 1e-15
        ):
            exceed += 1
    return exceed / total


def cross_validate(
    events: Sequence[CellEvent],
    representation: str,
    folds: dict[str, int],
) -> list[FoldResult]:
    results = []
    for fold in range(N_FOLDS):
        train = [event for event in events if folds[event.block] != fold]
        test = [event for event in events if folds[event.block] == fold]
        base = fit_position(train)
        train_bag = component_probabilities(train, base, False)
        train_full = component_probabilities(train, base, True)
        bag_weights = fit_mixture(train_bag)
        full_weights = fit_mixture(train_full)
        test_bag = component_probabilities(test, base, False)
        test_full = component_probabilities(test, base, True)
        base_bits = bits_per_event(
            test_bag[:, :1], np.asarray((1.0,))
        )
        bag_bits = bits_per_event(test_bag, bag_weights)
        full_bits = bits_per_event(test_full, full_weights)
        block_gains = {}
        for block in sorted(
            {event.block for event in test}
        ):
            indices = np.asarray(
                [
                    index
                    for index, event in enumerate(test)
                    if event.block == block
                ],
                dtype=np.int32,
            )
            block_gains[block] = (
                bits_per_event(test_full[indices], full_weights)
                - bits_per_event(test_bag[indices], bag_weights)
            )
        results.append(
            FoldResult(
                representation=representation,
                fold=fold,
                fit_events=len(train),
                test_events=len(test),
                bag_weights=tuple(map(float, bag_weights)),
                full_weights=tuple(map(float, full_weights)),
                baseline_bits=base_bits,
                bag_bits=bag_bits,
                full_bits=full_bits,
                bag_gain=bag_bits - base_bits,
                diagonal_gain=full_bits - bag_bits,
                block_diagonal_gains=block_gains,
            )
        )
    return results


def analyze_representation(
    pages: Sequence[Page],
    name: str,
    transform: Callable[[str], str],
    folds: dict[str, int],
) -> RepresentationResult:
    events = make_events(pages, transform)
    n, diagonal, neighbor, excess = raw_alignment(pages, transform)
    fold_results = cross_validate(events, name, folds)
    total_events = sum(result.test_events for result in fold_results)
    bag_gain = sum(
        result.bag_gain * result.test_events for result in fold_results
    ) / total_events
    event_weighted_diagonal_gain = sum(
        result.diagonal_gain * result.test_events
        for result in fold_results
    ) / total_events
    block_gains = {
        block: gain
        for result in fold_results
        for block, gain in result.block_diagonal_gains.items()
    }
    diagonal_gain = mean(block_gains.values())
    return RepresentationResult(
        representation=name,
        events=n,
        diagonal_match=diagonal,
        neighbor_match=neighbor,
        diagonal_excess=excess,
        fold_results=fold_results,
        heldout_bag_gain=bag_gain,
        heldout_diagonal_gain=diagonal_gain,
        event_weighted_diagonal_gain=event_weighted_diagonal_gain,
        median_block_diagonal_gain=median(block_gains.values()),
        positive_blocks=sum(value > 0 for value in block_gains.values()),
        exact_sign_p=exact_sign_flip(list(block_gains.values())),
        mean_diagonal_weight=mean(
            result.full_weights[2] for result in fold_results
        ),
    )


def analyze(
    pages: Sequence[Page],
    folds: dict[str, int],
) -> dict[str, RepresentationResult]:
    return {
        name: analyze_representation(pages, name, transform, folds)
        for name, transform in REPRESENTATIONS.items()
    }


def shuffle_rows(
    pages: Sequence[Page],
    rng: random.Random,
) -> list[Page]:
    result = []
    for page in pages:
        rows = list(page.rows)
        rng.shuffle(rows)
        result.append(
            Page(
                folio=page.folio,
                block=page.block,
                section=page.section,
                currier=page.currier,
                hand=page.hand,
                rows=tuple(rows),
            )
        )
    return result


def inject_diagonal_copy(
    pages: Sequence[Page],
    probability: float,
    rng: random.Random,
) -> list[Page]:
    result = []
    for page in pages:
        rows: list[list[str | None]] = []
        for raw_row in page.rows:
            row = list(raw_row)
            if rows:
                previous = rows[-1]
                for index, word in enumerate(row):
                    if (
                        word is not None
                        and index < len(previous)
                        and previous[index] is not None
                        and rng.random() < probability
                    ):
                        row[index] = previous[index]
            rows.append(row)
        result.append(
            Page(
                folio=page.folio,
                block=page.block,
                section=page.section,
                currier=page.currier,
                hand=page.hand,
                rows=tuple(tuple(row) for row in rows),
            )
        )
    return result


def print_results(
    title: str,
    results: dict[str, RepresentationResult],
) -> None:
    print("\n" + "=" * 126)
    print(title)
    print("=" * 126)
    print(
        f"{'representation':13s} {'n':>7s} {'diag':>8s} "
        f"{'neighbor':>9s} {'excess':>9s} {'bag gain':>10s} "
        f"{'diag gain':>11s} {'diag wt':>9s} {'+blocks':>9s} "
        f"{'sign p':>9s}"
    )
    for result in results.values():
        print(
            f"{result.representation:13s} {result.events:7d} "
            f"{result.diagonal_match:8.4f} "
            f"{result.neighbor_match:9.4f} "
            f"{result.diagonal_excess:+9.4f} "
            f"{result.heldout_bag_gain:+10.5f} "
            f"{result.heldout_diagonal_gain:+11.5f} "
            f"{result.mean_diagonal_weight:9.4f} "
            f"{result.positive_blocks:3d}/16 "
            f"{result.exact_sign_p:9.5f}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nulls", type=int, default=100)
    parser.add_argument("--injection", type=float, default=0.08)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    pages, audit = load_pages()
    blocks = sorted({page.block for page in pages})
    folds = block_folds(blocks)
    real = analyze(pages, folds)
    print(
        f"pages={audit['eligible_pages']} rows={audit['eligible_rows']} "
        f"words={audit['eligible_words']} blocks={len(blocks)} folds={folds}"
    )
    print_results("REAL ROW-AXIS COHERENCE", real)

    positive_pages = inject_diagonal_copy(
        pages, args.injection, random.Random(SEED + 100_000)
    )
    positive = analyze(positive_pages, folds)
    print_results(
        f"POSITIVE CONTROL: {args.injection:.1%} SAME-COLUMN COPY INJECTION",
        positive,
    )

    null_rows = []
    rng = random.Random(SEED + 200_000)
    for replicate in range(args.nulls):
        shuffled = shuffle_rows(pages, rng)
        results = analyze(shuffled, folds)
        row = {
            "replicate": replicate,
            "diagonal_gains": {
                name: result.heldout_diagonal_gain
                for name, result in results.items()
            },
            "max_diagonal_gain": max(
                result.heldout_diagonal_gain
                for result in results.values()
            ),
            "diagonal_excesses": {
                name: result.diagonal_excess
                for name, result in results.items()
            },
        }
        null_rows.append(row)
        if args.progress and (
            replicate < 4 or (replicate + 1) % 10 == 0
        ):
            print(
                f"null {replicate + 1}/{args.nulls}: "
                f"max_diag_gain={row['max_diagonal_gain']:+.5f}"
            )

    selected = max(
        real.values(), key=lambda result: result.heldout_diagonal_gain
    )
    null_maxima = [
        float(row["max_diagonal_gain"]) for row in null_rows
    ]
    familywise_p = (
        1
        + sum(
            value >= selected.heldout_diagonal_gain
            for value in null_maxima
        )
    ) / (1 + len(null_maxima))
    positive_selected = max(
        positive.values(), key=lambda result: result.heldout_diagonal_gain
    )
    summary = {
        "selected_representation": selected.representation,
        "selected_heldout_diagonal_gain": selected.heldout_diagonal_gain,
        "selected_positive_blocks": selected.positive_blocks,
        "selected_exact_sign_p": selected.exact_sign_p,
        "null_max_gain_mean": mean(null_maxima),
        "null_max_gain_max": max(null_maxima),
        "familywise_empirical_p": familywise_p,
        "positive_selected_representation": positive_selected.representation,
        "positive_selected_gain": positive_selected.heldout_diagonal_gain,
        "positive_pass": (
            positive_selected.heldout_diagonal_gain
            > max(null_maxima)
        ),
        "real_pass": (
            selected.heldout_diagonal_gain > max(null_maxima)
            and selected.exact_sign_p < 0.05
        ),
    }
    print("\n" + "=" * 126)
    print("GATE SUMMARY")
    print("=" * 126)
    for key, value in summary.items():
        print(f"{key}: {value}")

    payload = {
        "experiment": "row_axis_gate",
        "seed": SEED,
        "parameters": {
            "nulls": args.nulls,
            "injection": args.injection,
            "alpha": ALPHA,
            "em_iterations": EM_ITERATIONS,
            "representations": list(REPRESENTATIONS),
        },
        "assets": {
            str(CORPUS.relative_to(ROOT)): sha256(CORPUS),
        },
        "audit": dict(audit),
        "blocks": blocks,
        "folds": folds,
        "real": {
            name: asdict(result) for name, result in real.items()
        },
        "positive": {
            name: asdict(result) for name, result in positive.items()
        },
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
