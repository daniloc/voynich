#!/usr/bin/env python3
"""
Calibrate a solver-facing prefix-pooling residual against meaningful and
procedural controls.

Previous prefix is a deterministic coarsening of previous exact word. It cannot
contain population information beyond an ideal exact-word model. This script
therefore tests a narrower finite-sample question: does pooling rare exact-word
contexts through their prefix class improve held-out prediction?

The target is the next word's generic first-glyph prefix. A fixed hierarchical
Dirichlet predictor is evaluated leave-one-quire-block-out:

    root: target position + local-copy state
    surface: previous word shape + root
    exact baseline: previous exact word + surface
    prefix augmented: previous exact word + previous prefix + surface

The requested residual is the proper log-score difference between prefix
augmented and exact baseline. A preceding surface-recovery score compares the
exact baseline with the surface model. It is a calibration gate: a pipeline
that cannot recover known meaningful sequential predictability should not be
used to interpret the Voynich residual.

All hyperparameters and the two-character sensitivity are fixed before data
loading. No held-out block is used for vocabulary fitting, count fitting, or
model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Hashable, Iterable, Optional, Sequence

import numpy as np


SEED = 20260723
ALPHA = 0.5
TAU_SHAPE = 16.0
TAU_PREFIX = 16.0
TAU_WORD = 8.0
COPY_WINDOW = 20
BOOTSTRAPS = 20_000
PREFIX_WIDTHS = (1, 2)
UNKNOWN_TARGET = "<TRAIN_ONLY_UNKNOWN_TARGET>"
ROOT = Path(__file__).resolve().parents[2]

GEN_PREFIXES = tuple(sorted(
    (
        "qok", "qot", "qo", "ok", "ot", "o", "y", "ch", "sh", "d",
        "cth", "ckh", "cph",
    ),
    key=len,
    reverse=True,
))
GEN_SUFFIXES = tuple(sorted(
    (
        "eedy", "eody", "edy", "aiin", "ain", "iin", "dy", "ol",
        "or", "ar", "al", "am", "y",
    ),
    key=len,
    reverse=True,
))


def asset_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class Line:
    folio: str
    quire: str
    section: str
    currier: str
    hand: str
    words: tuple[Optional[str], ...]


@dataclass(frozen=True)
class Event:
    block: str
    target: str
    previous_word: str
    target_position: str
    previous_shape: tuple[Hashable, ...]
    copy_state: str
    previous_prefix: str


@dataclass
class BlockScore:
    events: int = 0
    novel_events: int = 0
    surface_bits: float = 0.0
    residual_bits: float = 0.0
    novel_residual_bits: float = 0.0
    surface_correct_delta: int = 0
    residual_correct_delta: int = 0

    @property
    def surface_gain(self) -> float:
        return self.surface_bits / self.events

    @property
    def residual_gain(self) -> float:
        return self.residual_bits / self.events

    @property
    def novel_residual_gain(self) -> float:
        return (
            self.novel_residual_bits / self.novel_events
            if self.novel_events
            else float("nan")
        )

    @property
    def surface_accuracy_delta(self) -> float:
        return self.surface_correct_delta / self.events

    @property
    def residual_accuracy_delta(self) -> float:
        return self.residual_correct_delta / self.events


@dataclass(frozen=True)
class DatasetResult:
    blocks: dict[str, BlockScore]

    def micro(self, numerator: str, denominator: str) -> float:
        top = sum(getattr(value, numerator) for value in self.blocks.values())
        bottom = sum(
            getattr(value, denominator) for value in self.blocks.values()
        )
        return top / bottom if bottom else float("nan")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=root / "data" / "corpus" / "corpus.json",
    )
    parser.add_argument(
        "--naibbe-dir",
        type=Path,
        default=Path(os.environ.get("NAIBBE_REPO", "/tmp/naibbe-cipher")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "data"
        / "intermediate"
        / "followups_solver_residual_controls.txt",
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


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


def load_voynich(path: Path) -> tuple[list[Line], Counter[str]]:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    audit: Counter[str] = Counter()
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
                    audit["one_character_breaks"] += 1
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


def alpha_tokens(path: Path) -> list[str]:
    return re.findall(
        r"[a-z]+",
        path.read_text(encoding="utf-8", errors="ignore").lower(),
    )


def from_last_book16(tokens: list[str]) -> tuple[list[str], int]:
    incipit = ("pomiferae", "arbores", "quaeque", "mitioribus", "sucis")
    matches = [
        index
        for index in range(len(tokens) - len(incipit) + 1)
        if tuple(tokens[index : index + len(incipit)]) == incipit
    ]
    if not matches:
        raise ValueError("Book 16 incipit not found in Latin asset")
    offset = matches[-1]
    return tokens[offset:], offset


def reflow(template: Sequence[Line], tokens: Sequence[str]) -> list[Line]:
    needed = sum(
        word is not None for line in template for word in line.words
    )
    if len(tokens) < needed:
        raise ValueError(f"need {needed} tokens, found {len(tokens)}")
    offset = 0
    result = []
    for line in template:
        words: list[Optional[str]] = []
        for word in line.words:
            if word is None:
                words.append(None)
            else:
                words.append(tokens[offset])
                offset += 1
        result.append(Line(
            folio=line.folio,
            quire=line.quire,
            section=line.section,
            currier=line.currier,
            hand=line.hand,
            words=tuple(words),
        ))
    if offset != needed:
        raise AssertionError("reflow did not consume matched token count")
    return result


def simple_substitution(tokens: Sequence[str]) -> list[str]:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return [word.translate(str.maketrans(alphabet, alphabet[::-1]))
            for word in tokens]


def exchange_groups(
    lines: Sequence[Line],
) -> dict[tuple[str, ...], list[tuple[int, int]]]:
    groups: dict[tuple[str, ...], list[tuple[int, int]]] = defaultdict(list)
    for line_index, line in enumerate(lines):
        for word_index, word in enumerate(line.words):
            if word is None:
                continue
            groups[(
                line.section,
                line.currier,
                line.hand,
                line.quire,
                position_bucket(word_index, len(line.words)),
            )].append((line_index, word_index))
    return groups


def layout_exchange(lines: Sequence[Line], rng: random.Random) -> list[Line]:
    words = [list(line.words) for line in lines]
    for locations in exchange_groups(lines).values():
        values = [lines[i].words[j] for i, j in locations]
        rng.shuffle(values)
        for (line_index, word_index), value in zip(locations, values):
            words[line_index][word_index] = value
    return [
        Line(
            line.folio,
            line.quire,
            line.section,
            line.currier,
            line.hand,
            tuple(values),
        )
        for line, values in zip(lines, words)
    ]


@lru_cache(maxsize=None)
def generator_decompose(word: str) -> tuple[str, str, str]:
    prefix = next(
        (value for value in GEN_PREFIXES if word.startswith(value)),
        "",
    )
    residual = word[len(prefix) :]
    suffix = next(
        (
            value
            for value in GEN_SUFFIXES
            if residual.endswith(value) and len(residual) > len(value)
        ),
        "",
    )
    core = residual[: -len(suffix)] if suffix else residual
    return prefix, core, suffix


class SchemaCopyGenerator:
    """The residual.py schema with p_copy=.12, window=40, p_mut=.4."""

    def __init__(self, words: Sequence[str]) -> None:
        core_counts: Counter[str] = Counter()
        prefix_tables: dict[str, Counter[str]] = defaultdict(Counter)
        suffix_tables: dict[str, Counter[str]] = defaultdict(Counter)
        for word in words:
            prefix, core, suffix = generator_decompose(word)
            core_counts[core] += 1
            prefix_tables[core][prefix] += 1
            suffix_tables[core][suffix] += 1
        self.cores = tuple(core_counts)
        self.core_weights = tuple(core_counts.values())
        self.prefix_tables = {
            core: (tuple(counts), tuple(counts.values()))
            for core, counts in prefix_tables.items()
        }
        self.suffix_tables = {
            core: (tuple(counts), tuple(counts.values()))
            for core, counts in suffix_tables.items()
        }
        self.glyphs = tuple(sorted(set("".join(words))))

    def sample(self, rng: random.Random) -> str:
        core = rng.choices(self.cores, weights=self.core_weights, k=1)[0]
        prefixes, prefix_weights = self.prefix_tables[core]
        suffixes, suffix_weights = self.suffix_tables[core]
        return (
            rng.choices(prefixes, weights=prefix_weights, k=1)[0]
            + core
            + rng.choices(suffixes, weights=suffix_weights, k=1)[0]
        )

    def generate(self, count: int, rng: random.Random) -> list[str]:
        result = []
        for _ in range(count):
            if result and rng.random() < 0.12:
                word = result[-rng.randint(1, min(40, len(result)))]
                if len(word) > 1 and rng.random() < 0.4:
                    index = rng.randrange(len(word))
                    word = (
                        word[:index]
                        + rng.choice(self.glyphs)
                        + word[index + 1 :]
                    )
                result.append(word)
            else:
                result.append(self.sample(rng))
        return result


def self_citation(
    source: Sequence[str], count: int, rng: random.Random
) -> list[str]:
    """The null_and_morph.py p_verbatim=.12, p_mutate=.55 lineage."""
    result = [rng.choice(source[:50])]
    glyphs = tuple(sorted(set("".join(source))))
    for _ in range(count - 1):
        word = rng.choice(result[-40:])
        draw = rng.random()
        if draw < 0.12:
            result.append(word)
        elif draw < 0.67 and len(word) > 1:
            index = rng.randrange(len(word))
            result.append(
                word[:index] + rng.choice(glyphs) + word[index + 1 :]
            )
        else:
            result.append(word)
    return result


def word_shape(word: str) -> tuple[Hashable, ...]:
    return (
        min(len(word), 8),
        word[0] == word[-1],
        any(left == right for left, right in zip(word, word[1:])),
        len(set(word)) < len(word),
    )


def event_stream(lines: Sequence[Line], prefix_width: int) -> list[Event]:
    events = []
    for line in lines:
        history: list[str] = []
        for index, word in enumerate(line.words):
            if word is None:
                history = []
                continue
            if index + 1 < len(line.words) and line.words[index + 1] is not None:
                lag = next(
                    (
                        distance
                        for distance, old in enumerate(
                            reversed(history[-COPY_WINDOW:]), 1
                        )
                        if old == word
                    ),
                    None,
                )
                if lag is None:
                    copy_state = "no_local_copy"
                elif lag == 1:
                    copy_state = "lag_1"
                elif lag <= 5:
                    copy_state = "lag_2_5"
                else:
                    copy_state = "lag_6_20"
                target = line.words[index + 1]
                assert target is not None
                events.append(Event(
                    block=line.quire,
                    target=target[:prefix_width],
                    previous_word=word,
                    target_position=position_bucket(
                        index + 1, len(line.words)
                    ),
                    previous_shape=word_shape(word),
                    copy_state=copy_state,
                    previous_prefix=word[:prefix_width],
                ))
            history.append(word)
    return events


class HierarchicalPredictor:
    """Fixed train-only hierarchical categorical predictor."""

    def __init__(self, events: Sequence[Event]) -> None:
        observed = sorted({event.target for event in events})
        self.classes = tuple(observed) + (UNKNOWN_TARGET,)
        self.class_set = set(observed)
        self.vocabulary_size = len(self.classes)
        self.word_types = {event.previous_word for event in events}
        self.root: Counter[tuple[Hashable, ...]] = Counter()
        self.root_n: Counter[tuple[Hashable, ...]] = Counter()
        self.shape: Counter[tuple[Hashable, ...]] = Counter()
        self.shape_n: Counter[tuple[Hashable, ...]] = Counter()
        self.prefix: Counter[tuple[Hashable, ...]] = Counter()
        self.prefix_n: Counter[tuple[Hashable, ...]] = Counter()
        self.word: Counter[tuple[Hashable, ...]] = Counter()
        self.word_n: Counter[tuple[Hashable, ...]] = Counter()
        for event in events:
            root = self.root_context(event)
            shape = self.shape_context(event)
            prefix = self.prefix_context(event)
            word = self.word_context(event)
            self.root[(*root, event.target)] += 1
            self.root_n[root] += 1
            self.shape[(*shape, event.target)] += 1
            self.shape_n[shape] += 1
            self.prefix[(*prefix, event.target)] += 1
            self.prefix_n[prefix] += 1
            self.word[(*word, event.target)] += 1
            self.word_n[word] += 1

    @staticmethod
    def root_context(event: Event) -> tuple[Hashable, ...]:
        return event.target_position, event.copy_state

    @staticmethod
    def shape_context(event: Event) -> tuple[Hashable, ...]:
        return (
            event.previous_shape,
            event.target_position,
            event.copy_state,
        )

    @staticmethod
    def prefix_context(event: Event) -> tuple[Hashable, ...]:
        return (
            event.previous_prefix,
            event.previous_shape,
            event.target_position,
            event.copy_state,
        )

    @staticmethod
    def word_context(event: Event) -> tuple[Hashable, ...]:
        return (
            event.previous_word,
            event.target_position,
            event.copy_state,
        )

    def mapped_target(self, target: str) -> str:
        return target if target in self.class_set else UNKNOWN_TARGET

    def probabilities(
        self, event: Event, target: str
    ) -> tuple[float, float, float]:
        target = self.mapped_target(target)
        root = self.root_context(event)
        shape = self.shape_context(event)
        prefix = self.prefix_context(event)
        word = self.word_context(event)
        root_probability = (
            self.root[(*root, target)] + ALPHA
        ) / (
            self.root_n[root] + ALPHA * self.vocabulary_size
        )
        shape_probability = (
            self.shape[(*shape, target)] + TAU_SHAPE * root_probability
        ) / (
            self.shape_n[shape] + TAU_SHAPE
        )
        prefix_probability = (
            self.prefix[(*prefix, target)]
            + TAU_PREFIX * shape_probability
        ) / (
            self.prefix_n[prefix] + TAU_PREFIX
        )
        exact_count = self.word[(*word, target)]
        exact_total = self.word_n[word]
        exact_probability = (
            exact_count + TAU_WORD * shape_probability
        ) / (
            exact_total + TAU_WORD
        )
        augmented_probability = (
            exact_count + TAU_WORD * prefix_probability
        ) / (
            exact_total + TAU_WORD
        )
        return shape_probability, exact_probability, augmented_probability

    def predictions(self, event: Event) -> tuple[str, str, str]:
        surface = []
        exact = []
        augmented = []
        for target in self.classes:
            values = self.probabilities(event, target)
            surface.append(values[0])
            exact.append(values[1])
            augmented.append(values[2])
        return (
            self.classes[max(range(len(surface)), key=surface.__getitem__)],
            self.classes[max(range(len(exact)), key=exact.__getitem__)],
            self.classes[
                max(range(len(augmented)), key=augmented.__getitem__)
            ],
        )


def score_dataset(
    lines: Sequence[Line],
    prefix_width: int,
    include_accuracy: bool = True,
) -> DatasetResult:
    events = event_stream(lines, prefix_width)
    blocks = sorted({event.block for event in events})
    result = {}
    for held_out in blocks:
        train = [event for event in events if event.block != held_out]
        test = [event for event in events if event.block == held_out]
        model = HierarchicalPredictor(train)
        score = BlockScore()
        for event in test:
            surface, exact, augmented = model.probabilities(
                event, event.target
            )
            score.events += 1
            score.surface_bits += math.log2(exact / surface)
            score.residual_bits += math.log2(augmented / exact)
            if include_accuracy:
                surface_prediction, exact_prediction, augmented_prediction = (
                    model.predictions(event)
                )
                mapped_target = model.mapped_target(event.target)
                score.surface_correct_delta += (
                    int(exact_prediction == mapped_target)
                    - int(surface_prediction == mapped_target)
                )
                score.residual_correct_delta += (
                    int(augmented_prediction == mapped_target)
                    - int(exact_prediction == mapped_target)
                )
            if event.previous_word not in model.word_types:
                score.novel_events += 1
                score.novel_residual_bits += math.log2(augmented / exact)
        result[held_out] = score
    return DatasetResult(result)


def percentile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def bootstrap_ci(
    values: Sequence[float], replicates: int, key: str
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    seed = SEED ^ int.from_bytes(
        hashlib.sha256(key.encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    return percentile(means, 0.025), percentile(means, 0.975)


def exact_upper_sign_flip(values: Sequence[float]) -> float:
    observed = sum(values)
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = sum(
            value * sign for value, sign in zip(values, signs)
        )
        exceed += permuted >= observed - 1e-15
        total += 1
    return exceed / total


def exact_two_sided_sign_flip(values: Sequence[float]) -> float:
    observed = abs(sum(values))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(sum(
            value * sign for value, sign in zip(values, signs)
        ))
        exceed += permuted >= observed - 1e-15
        total += 1
    return exceed / total


def block_values(
    result: DatasetResult, attribute: str
) -> list[float]:
    return [
        getattr(result.blocks[block], attribute)
        for block in sorted(result.blocks)
    ]


def summary(
    result: DatasetResult,
    attribute: str,
    numerator: str,
    denominator: str,
    bootstrap: int,
    key: str,
) -> tuple[float, float, float, float, int]:
    values = block_values(result, attribute)
    mean = sum(values) / len(values)
    low, high = bootstrap_ci(values, bootstrap, key)
    micro = result.micro(numerator, denominator)
    positive = sum(value > 0 for value in values)
    return micro, mean, low, high, positive


def validate_inputs(
    corpus: Path, naibbe_dir: Path
) -> tuple[Path, Path, Path]:
    cipher = (
        naibbe_dir
        / "encrypted"
        / "nathist_output_ciphertext_respaced.txt"
    )
    latin = (
        naibbe_dir
        / "figure_utils"
        / "gaskell_bowern_2022"
        / "data"
        / "meaningful"
        / "texts"
        / "Historical - Latin - Technical - Pliny's Natural History.txt"
    )
    english = (
        naibbe_dir
        / "figure_utils"
        / "gaskell_bowern_2022"
        / "data"
        / "meaningful"
        / "texts"
        / "Modern - English - Literary - NT.txt"
    )
    for path in (corpus, cipher, latin, english):
        if not path.is_file():
            raise FileNotFoundError(path)
    return cipher, latin, english


def build_datasets(
    args: argparse.Namespace,
) -> tuple[dict[str, list[Line]], dict[str, Path], Counter[str], int]:
    cipher_path, latin_path, english_path = validate_inputs(
        args.corpus, args.naibbe_dir
    )
    lines, audit = load_voynich(args.corpus)
    words = [
        word for line in lines for word in line.words if word is not None
    ]
    token_count = len(words)
    cipher = alpha_tokens(cipher_path)
    all_latin = alpha_tokens(latin_path)
    latin, latin_offset = from_last_book16(all_latin)
    english = alpha_tokens(english_path)
    schema = SchemaCopyGenerator(words)
    datasets = {
        "VOYNICH": lines,
        "NAIBBE_MEANINGFUL": reflow(lines, cipher),
        "LATIN_PLAIN": reflow(lines, latin),
        "LATIN_SIMPLE_SUB": reflow(lines, simple_substitution(latin)),
        "ENGLISH_PLAIN": reflow(lines, english),
        "LAYOUT_EXCHANGE": layout_exchange(
            lines, random.Random(SEED + 1_000_003)
        ),
        "SCHEMA_COPY": reflow(
            lines,
            schema.generate(
                token_count, random.Random(SEED + 2_000_006)
            ),
        ),
        "SELF_CITATION": reflow(
            lines,
            self_citation(
                words, token_count, random.Random(SEED + 3_000_009)
            ),
        ),
    }
    audit["latin_book16_offset"] = latin_offset
    paths = {
        "corpus": args.corpus,
        "cipher": cipher_path,
        "latin": latin_path,
        "english": english_path,
    }
    return datasets, paths, audit, token_count


def build_report(args: argparse.Namespace) -> str:
    datasets, paths, audit, token_count = build_datasets(args)
    results = {
        width: {
            name: score_dataset(
                lines, width, include_accuracy=(width == 1)
            )
            for name, lines in datasets.items()
        }
        for width in PREFIX_WIDTHS
    }
    for width in PREFIX_WIDTHS:
        latin = results[width]["LATIN_PLAIN"]
        substituted = results[width]["LATIN_SIMPLE_SUB"]
        for block in latin.blocks:
            left = latin.blocks[block]
            right = substituted.blocks[block]
            for attribute in (
                "surface_bits",
                "residual_bits",
                "novel_residual_bits",
            ):
                if not math.isclose(
                    getattr(left, attribute),
                    getattr(right, attribute),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise AssertionError(
                        f"substitution invariance failed: width={width} "
                        f"block={block} attribute={attribute}"
                    )

    output = []

    def emit(value: str = "") -> None:
        output.append(value)

    emit("=" * 110)
    emit("SOLVER-FACING PREFIX-POOLING RESIDUAL CALIBRATION")
    emit("=" * 110)
    emit(
        f"seed={SEED} alpha={ALPHA} tau_shape={TAU_SHAPE:g} "
        f"tau_prefix={TAU_PREFIX:g} tau_word={TAU_WORD:g}"
    )
    emit(
        f"template={audit['prose_lines']} prose lines, {token_count} eligible "
        f"tokens, {audit['uncertain_breaks']} uncertain and "
        f"{audit['one_character_breaks']} one-character break slots"
    )
    emit(
        "split=leave one complete matched quire block out; train-only classes, "
        "counts, exact-word vocabulary, and hierarchy"
    )
    emit(
        "surface recovery=log2 P(next prefix|exact previous word,surface) "
        "- log2 P(next prefix|shape,copy,position)"
    )
    emit(
        "prefix residual=log2 P(next prefix|exact word,prefix,surface) "
        "- log2 P(next prefix|exact word,surface)"
    )
    emit(
        "Interpretation boundary: previous prefix is determined by previous "
        "exact word. Residual gain is finite-sample pooling value, not "
        "independent population information."
    )
    emit()
    emit("PROVENANCE")
    emit(
        f"  Naibbe checkout: {git_revision(args.naibbe_dir)} "
        f"(https://github.com/greshko/naibbe-cipher)"
    )
    for label, path in paths.items():
        emit(
            f"  {label:8s} SHA256={sha256(path)} "
            f"path={asset_name(path)}"
        )
    emit(
        f"  Latin starts at last duplicated Book 16 incipit, token offset "
        f"{audit['latin_book16_offset']}."
    )
    emit(
        "  Procedural arms: strict section+Currier+hand+quire+position "
        "exchange; residual.py schema/copy; null_and_morph.py self-citation."
    )

    for width in PREFIX_WIDTHS:
        label = "PRIMARY" if width == 1 else "FIXED SENSITIVITY"
        emit()
        emit("=" * 110)
        emit(f"{label}: GENERIC PREFIX WIDTH {width}")
        emit("=" * 110)
        emit()
        emit("SURFACE EXACT-WORD RECOVERY GATE")
        emit(
            f"  {'corpus':20s} {'events':>7s} {'micro bits':>11s} "
            f"{'block mean [95% CI]':>32s} {'+blocks':>8s} "
            f"{'p(>0)':>9s} {'accuracy':>10s}"
        )
        for name, result in results[width].items():
            micro, mean, low, high, positive = summary(
                result,
                "surface_gain",
                "surface_bits",
                "events",
                args.bootstrap,
                f"surface:{width}:{name}",
            )
            accuracy = (
                f"{100 * result.micro('surface_correct_delta', 'events'):+9.3f}pp"
                if width == 1
                else f"{'NA':>11s}"
            )
            p_value = exact_upper_sign_flip(
                block_values(result, "surface_gain")
            )
            events = sum(block.events for block in result.blocks.values())
            emit(
                f"  {name:20s} {events:7d} {micro:+11.5f} "
                f"{mean:+9.5f} [{low:+9.5f},{high:+9.5f}] "
                f"{positive:2d}/16 {p_value:9.5f} {accuracy}"
            )

        emit()
        emit("REQUESTED PREFIX-POOLING RESIDUAL")
        emit(
            f"  {'corpus':20s} {'micro bits':>11s} "
            f"{'block mean [95% CI]':>32s} {'+blocks':>8s} "
            f"{'p(>0)':>9s} {'accuracy':>10s} "
            f"{'novel-word bits':>16s}"
        )
        for name, result in results[width].items():
            micro, mean, low, high, positive = summary(
                result,
                "residual_gain",
                "residual_bits",
                "events",
                args.bootstrap,
                f"residual:{width}:{name}",
            )
            accuracy = (
                f"{100 * result.micro('residual_correct_delta', 'events'):+9.3f}pp"
                if width == 1
                else f"{'NA':>11s}"
            )
            novel = result.micro(
                "novel_residual_bits", "novel_events"
            )
            p_value = exact_upper_sign_flip(
                block_values(result, "residual_gain")
            )
            emit(
                f"  {name:20s} {micro:+11.5f} "
                f"{mean:+9.5f} [{low:+9.5f},{high:+9.5f}] "
                f"{positive:2d}/16 {p_value:9.5f} "
                f"{accuracy} {novel:+16.5f}"
            )

        emit()
        emit("DISCRIMINATION: VOYNICH minus corpus residual, paired blocks")
        emit(
            f"  {'corpus':20s} {'mean difference [95% CI]':>36s} "
            f"{'raw p':>9s} {'Bonf p':>9s}"
        )
        reference = results[width]["VOYNICH"]
        comparisons = []
        for name, result in results[width].items():
            if name in ("VOYNICH", "LATIN_SIMPLE_SUB"):
                continue
            differences = [
                reference.blocks[block].residual_gain
                - result.blocks[block].residual_gain
                for block in sorted(reference.blocks)
            ]
            low, high = bootstrap_ci(
                differences,
                args.bootstrap,
                f"contrast:{width}:{name}",
            )
            comparisons.append((
                name,
                sum(differences) / len(differences),
                low,
                high,
                exact_two_sided_sign_flip(differences),
            ))
        correction = len(comparisons)
        for name, mean, low, high, p_value in comparisons:
            emit(
                f"  {name:20s} {mean:+9.5f} "
                f"[{low:+9.5f},{high:+9.5f}] "
                f"{p_value:9.5f} "
                f"{min(1.0, correction * p_value):9.5f}"
            )
        emit(f"  Bonferroni family={correction} nonredundant contrasts")

    primary = results[1]
    known = ("NAIBBE_MEANINGFUL", "LATIN_PLAIN", "ENGLISH_PLAIN")
    known_positive = [
        name
        for name in known
        if primary[name].micro("residual_bits", "events") > 0
    ]
    english_surface_p = exact_upper_sign_flip(
        block_values(primary["ENGLISH_PLAIN"], "surface_gain")
    )
    self_citation_p = exact_upper_sign_flip(
        block_values(primary["SELF_CITATION"], "residual_gain")
    )
    voynich_p = exact_upper_sign_flip(
        block_values(primary["VOYNICH"], "residual_gain")
    )
    emit()
    emit("=" * 110)
    emit("CALIBRATION GATES")
    emit("=" * 110)
    emit(
        "  Surface recovery in held-out English: "
        + ("PASS" if english_surface_p < 0.05 else "FAIL")
        + f" (one-sided block p={english_surface_p:.5f})."
    )
    emit(
        "  Positive prefix residual in known-meaningful arms: "
        + (
            "PASS (" + ", ".join(known_positive) + ")"
            if known_positive
            else "FAIL (none of Naibbe, Latin, or English)"
        )
        + "."
    )
    emit(
        "  Aggressive self-citation detection: "
        + ("PASS" if self_citation_p < 0.05 else "FAIL")
        + f" (one-sided block p={self_citation_p:.5f})."
    )
    emit(
        "  Voynich positive prefix residual: "
        + ("YES" if voynich_p < 0.05 else "NO")
        + f" (one-sided block p={voynich_p:.5f})."
    )
    emit(
        "  Use decision: reject prefix-pooling residual as a meaning detector "
        "if known-meaningful recovery fails; retain it only as a diagnostic "
        "for prefix-preserving local production."
    )
    emit()
    emit("LIMITATIONS")
    emit(
        "  1. Prefix pooling is representational backoff, not additional "
        "information beyond exact word in the population."
    )
    emit(
        "  2. Latin, English, Naibbe, and generated controls use artificial "
        "Voynich line/quire blocks; source sentence and page boundaries are lost."
    )
    emit(
        "  3. Schema tables are estimated globally from Voynich and generator "
        "hyperparameters are inherited, not nested or refitted."
    )
    emit(
        "  4. The one-replicate procedural arms are deterministic calibration "
        "points, not empirical procedural-null distributions."
    )
    emit(
        "  5. Accuracy is secondary; proper held-out log score is primary. "
        "The two-character analysis was declared as a sensitivity, not selected."
    )
    return "\n".join(output) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"report written: {args.output}")


if __name__ == "__main__":
    main()
