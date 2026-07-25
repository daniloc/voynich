#!/usr/bin/env python3
"""
Bounded search for a line-reset stateful payload in Voynich prose.

This experiment addresses the main family left open by the stationary channel
attacks: a small, human-operable state machine whose output depends on a word's
numeric morphology, the previous word, the first word of the line, or a running
register.

The search is deliberately finite.  Eight deterministic word coordinates are
crossed with fifteen declared programs and two frozen language models.  For
each outer fold:

1. a global monoalphabetic output key is fitted on complete fit quires;
2. coordinate, program, language, and key refinement are selected on disjoint
   validation quires;
3. the selected program and key are frozen before complete test quires.

Every strict null replicate preserves quire, section, Currier dialect, and
physical word-position buckets, then repeats the complete candidate/language/
key search.  Synthetic controls hide Latin under unknown programs and unknown
permutations from the exact searched family.

The family emits one Latin-alphabet character per retained word.  Arbitrary
lookup tables, changing per-page keys, multi-character emissions, and programs
outside the declared grammar are not tested.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Optional, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from naibbe_permutation_decoder import (  # noqa: E402
    ALPHABET,
    TetragramLM,
    clean_latin,
    hungarian_max,
)


CORPUS = ROOT / "data" / "corpus" / "corpus.json"
CONTROL_PATHS = {
    "latin": ROOT / "data" / "controls" / "latin.txt",
    "english": ROOT / "data" / "controls" / "english.txt",
}
SEED = 20260723
MODULUS = len(ALPHABET)
LM_TRAIN_CHARS = 52_000
LM_CEILING_CHARS = 22_000
CONTEXT_WEIGHT = 0.55

GLYPHS = "acdefghiklmnopqrstxyz"
GLYPH_VALUE = {glyph: index + 1 for index, glyph in enumerate(GLYPHS)}

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
class WordLine:
    block: str
    folio: str
    line: int
    run: int
    section: str
    currier: str
    words: tuple[str, ...]


@dataclass(frozen=True)
class NumericLine:
    block: str
    values: tuple[int, ...]
    truth: tuple[Optional[int], ...]


@dataclass(frozen=True)
class SymbolLine:
    block: str
    symbols: tuple[int, ...]
    truth: tuple[Optional[int], ...]


@dataclass(frozen=True)
class Program:
    name: str
    kind: str
    a: int = 0
    b: int = 0
    c: int = 0
    skip_first: bool = False
    complexity: int = 1


@dataclass
class CandidateFit:
    feature: str
    program: str
    language: str
    key: list[int]
    initial_validation: float
    validation_score: float
    validation_penalized: float
    fit_initial: float
    fit_final: float
    swaps: int


@dataclass
class FoldResult:
    fold: int
    fit_blocks: list[str]
    validation_blocks: list[str]
    test_blocks: list[str]
    feature: str
    program: str
    language: str
    program_complexity: int
    initial_validation: float
    validation_score: float
    validation_penalized: float
    test_initial: float
    test_final: float
    test_residual: float
    gap_to_language_ceiling: float
    swaps: int
    plaintext_accuracy: Optional[float]
    known_plaintext: int
    sample: str


@dataclass
class LanguageSpec:
    name: str
    lm: TetragramLM
    table: np.ndarray
    ceiling: float


PROGRAMS = (
    Program("direct", "affine", complexity=1),
    Program("position_plus", "affine", c=1, complexity=2),
    Program("position_minus", "affine", c=-1, complexity=2),
    Program("lag_minus", "affine", a=-1, skip_first=True, complexity=2),
    Program("lag_plus", "affine", a=1, skip_first=True, complexity=2),
    Program("seed_minus", "affine", b=-1, skip_first=True, complexity=2),
    Program("seed_plus", "affine", b=1, skip_first=True, complexity=2),
    Program(
        "lag_minus_seed_plus",
        "affine",
        a=-1,
        b=1,
        skip_first=True,
        complexity=3,
    ),
    Program(
        "lag_minus_seed_minus",
        "affine",
        a=-1,
        b=-1,
        skip_first=True,
        complexity=3,
    ),
    Program(
        "lag_plus_seed_minus",
        "affine",
        a=1,
        b=-1,
        skip_first=True,
        complexity=3,
    ),
    Program(
        "lag_minus_position_plus",
        "affine",
        a=-1,
        c=1,
        skip_first=True,
        complexity=3,
    ),
    Program(
        "lag_minus_position_minus",
        "affine",
        a=-1,
        c=-1,
        skip_first=True,
        complexity=3,
    ),
    Program("running_plus", "running_plus", complexity=3),
    Program("running_minus", "running_minus", complexity=3),
    Program(
        "seeded_running_plus",
        "seeded_running_plus",
        skip_first=True,
        complexity=4,
    ),
)

FEATURES = (
    "length",
    "first",
    "last",
    "glyph_sum",
    "weighted_sum",
    "polynomial",
    "core_polynomial",
    "affix_polynomial",
)


def locus_type(value: str) -> str:
    match = re.search(r"[A-Za-z]", value)
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
    return (
        prefix,
        residual[: -len(suffix)] if suffix else residual,
        suffix,
    )


def polynomial(value: str, base: int = 3) -> int:
    result = 0
    for glyph in value:
        result = (base * result + GLYPH_VALUE[glyph]) % MODULUS
    return result


def feature_value(word: str, name: str) -> int:
    prefix, core, suffix = decompose(word)
    if name == "length":
        return len(word) % MODULUS
    if name == "first":
        return GLYPH_VALUE[word[0]] % MODULUS
    if name == "last":
        return GLYPH_VALUE[word[-1]] % MODULUS
    if name == "glyph_sum":
        return sum(GLYPH_VALUE[glyph] for glyph in word) % MODULUS
    if name == "weighted_sum":
        return sum(
            (index + 1) * GLYPH_VALUE[glyph]
            for index, glyph in enumerate(word)
        ) % MODULUS
    if name == "polynomial":
        return polynomial(word)
    if name == "core_polynomial":
        return polynomial(core)
    if name == "affix_polynomial":
        return (polynomial(prefix) * 5 + polynomial(suffix)) % MODULUS
    raise ValueError(f"unknown feature {name}")


def load_word_lines() -> tuple[list[WordLine], list[str]]:
    source = json.loads(CORPUS.read_text(encoding="utf-8"))
    lines = []
    for folio, records in source["folios"].items():
        meta = source["meta"].get(folio, {})
        block = str(meta.get("Q", "?"))
        for record in records:
            if locus_type(record["locus"]) != "P":
                continue
            runs: list[list[str]] = []
            current: list[str] = []
            for word in record["words"]:
                if (
                    "?" in word
                    or not word.isalpha()
                    or len(word) < 2
                    or any(glyph not in GLYPH_VALUE for glyph in word)
                ):
                    if current:
                        runs.append(current)
                        current = []
                else:
                    current.append(word)
            if current:
                runs.append(current)
            for run_index, words in enumerate(runs):
                if len(words) < 5:
                    continue
                lines.append(
                    WordLine(
                        block=block,
                        folio=folio,
                        line=int(record["line"]),
                        run=run_index,
                        section=str(meta.get("I", "?")),
                        currier=str(meta.get("L", "?")),
                        words=tuple(words),
                    )
                )
    return lines, sorted({line.block for line in lines})


def feature_lines(
    lines: Sequence[WordLine], feature: str
) -> list[NumericLine]:
    return [
        NumericLine(
            line.block,
            tuple(feature_value(word, feature) for word in line.words),
            tuple(None for _ in line.words),
        )
        for line in lines
    ]


def strict_shuffle(
    lines: Sequence[WordLine], rng: random.Random
) -> list[WordLine]:
    slots: dict[tuple[str, str, str, str], list[tuple[int, int]]] = defaultdict(
        list
    )
    values: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    mutable = [list(line.words) for line in lines]
    for line_index, line in enumerate(lines):
        for index, word in enumerate(line.words):
            key = (
                line.block,
                line.section,
                line.currier,
                position_bucket(index, len(line.words)),
            )
            slots[key].append((line_index, index))
            values[key].append(word)
    for key, positions in slots.items():
        shuffled = values[key][:]
        rng.shuffle(shuffled)
        for (line_index, index), word in zip(positions, shuffled):
            mutable[line_index][index] = word
    return [
        WordLine(
            line.block,
            line.folio,
            line.line,
            line.run,
            line.section,
            line.currier,
            tuple(words),
        )
        for line, words in zip(lines, mutable)
    ]


def apply_program(
    line: NumericLine, program: Program
) -> SymbolLine:
    values = line.values
    if not values:
        return SymbolLine(line.block, (), ())
    start = 1 if program.skip_first else 0
    symbols = []
    if program.kind == "affine":
        seed = values[0]
        for index in range(start, len(values)):
            previous = values[index - 1] if index else 0
            symbols.append(
                (
                    values[index]
                    + program.a * previous
                    + program.b * seed
                    + program.c * index
                )
                % MODULUS
            )
    elif program.kind in {"running_plus", "running_minus"}:
        state = 0
        sign = 1 if program.kind == "running_plus" else -1
        for value in values:
            state = (state + sign * value) % MODULUS
            symbols.append(state)
    elif program.kind == "seeded_running_plus":
        state = values[0]
        for value in values[1:]:
            state = (state + value) % MODULUS
            symbols.append(state)
    else:
        raise ValueError(f"unknown program kind {program.kind}")
    truth = line.truth[start:]
    return SymbolLine(line.block, tuple(symbols), tuple(truth))


def dense_lm_table(lm: TetragramLM) -> np.ndarray:
    # Classical substitution solvers use an unconditional tetragram score.
    # The conditional backoff score in TetragramLM is appropriate for held-out
    # prediction, but its normalization creates severe key-search local optima:
    # a wrong key can improve denominators as well as numerators.  Keep that
    # model for the context initializer and use a single normalized tetragram
    # distribution for the permutation objective.
    table = np.empty((MODULUS,) * 4, dtype=np.float64)
    alpha = 0.01
    denominator = sum(lm.quads.values()) + alpha * MODULUS**4
    for first in range(MODULUS):
        for second in range(MODULUS):
            for third in range(MODULUS):
                table[first, second, third, :] = [
                    math.log(
                        (
                            lm.quads[(first, second, third, fourth)]
                            + alpha
                        )
                        / denominator
                    )
                    for fourth in range(MODULUS)
                ]
    return table


def load_languages(names: Sequence[str]) -> dict[str, LanguageSpec]:
    result = {}
    for name in names:
        cleaned = clean_latin(
            CONTROL_PATHS[name].read_text(encoding="utf-8", errors="ignore")
        )
        if len(cleaned) < LM_TRAIN_CHARS + LM_CEILING_CHARS:
            raise RuntimeError(f"{name} control is too short")
        train = cleaned[:LM_TRAIN_CHARS]
        ceiling_text = cleaned[-LM_CEILING_CHARS:]
        lm = TetragramLM(train)
        table = dense_lm_table(lm)
        ceiling_ids = np.asarray(
            [lm.char_id[char] for char in ceiling_text], dtype=np.int16
        )
        ceiling_quads = np.column_stack(
            (
                ceiling_ids[:-3],
                ceiling_ids[1:-2],
                ceiling_ids[2:-1],
                ceiling_ids[3:],
            )
        )
        result[name] = LanguageSpec(
            name,
            lm,
            table,
            float(
                np.mean(
                    table[
                        ceiling_quads[:, 0],
                        ceiling_quads[:, 1],
                        ceiling_quads[:, 2],
                        ceiling_quads[:, 3],
                    ]
                )
            ),
        )
    return result


def quad_counter(
    lines: Sequence[SymbolLine], blocks: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    counts: Counter[tuple[int, int, int, int]] = Counter()
    for line in lines:
        if line.block not in blocks:
            continue
        counts.update(zip(
            line.symbols,
            line.symbols[1:],
            line.symbols[2:],
            line.symbols[3:],
        ))
    if not counts:
        return (
            np.empty((0, 4), dtype=np.int16),
            np.empty(0, dtype=np.float64),
        )
    quads = np.asarray(list(counts), dtype=np.int16)
    weights = np.asarray(list(counts.values()), dtype=np.float64)
    return quads, weights


def score_key(
    quads: np.ndarray,
    weights: np.ndarray,
    key: Sequence[int],
    language: LanguageSpec,
) -> float:
    if not len(weights):
        return float("-inf")
    mapping = np.asarray(key, dtype=np.int16)
    decoded = mapping[quads]
    values = language.table[
        decoded[:, 0],
        decoded[:, 1],
        decoded[:, 2],
        decoded[:, 3],
    ]
    return float(np.dot(values, weights) / weights.sum())


def initialize_key(
    lines: Sequence[SymbolLine],
    blocks: set[str],
    language: LanguageSpec,
) -> list[int]:
    counts = [0] * MODULUS
    left = [[0] * MODULUS for _ in range(MODULUS)]
    right = [[0] * MODULUS for _ in range(MODULUS)]
    for line in lines:
        if line.block not in blocks:
            continue
        for symbol in line.symbols:
            counts[symbol] += 1
    order = sorted(range(MODULUS), key=lambda item: (-counts[item], item))
    rank = [0] * MODULUS
    for index, symbol in enumerate(order):
        rank[symbol] = index
    for line in lines:
        if line.block not in blocks:
            continue
        for index, symbol in enumerate(line.symbols):
            if index:
                left[symbol][rank[line.symbols[index - 1]]] += 1
            if index + 1 < len(line.symbols):
                right[symbol][rank[line.symbols[index + 1]]] += 1

    weights = []
    for symbol in range(MODULUS):
        row = []
        for letter in range(MODULUS):
            score = counts[symbol] * math.log(language.lm.letter_prob[letter])
            context = sum(
                left[symbol][neighbor_rank]
                * language.lm.rank_left_logp[letter][neighbor_rank]
                + right[symbol][neighbor_rank]
                * language.lm.rank_right_logp[letter][neighbor_rank]
                for neighbor_rank in range(MODULUS)
            )
            row.append(score + CONTEXT_WEIGHT * context)
        weights.append(row)
    return hungarian_max(weights)


def refine_key(
    lines: Sequence[SymbolLine],
    blocks: set[str],
    language: LanguageSpec,
    initial_key: Sequence[int],
    maximum_swaps: int,
) -> tuple[list[int], float, float, int]:
    quads, weights = quad_counter(lines, blocks)
    key = list(initial_key)
    initial = score_key(quads, weights, key, language)
    if not len(weights):
        return key, initial, initial, 0

    affected = []
    for symbol in range(MODULUS):
        affected.append(np.flatnonzero(np.any(quads == symbol, axis=1)))
    mapped = np.asarray(key, dtype=np.int16)[quads]
    parts = language.table[
        mapped[:, 0], mapped[:, 1], mapped[:, 2], mapped[:, 3]
    ] * weights
    total_weight = float(weights.sum())
    swaps = 0

    for _ in range(maximum_swaps):
        best_delta = 1e-9
        best_pair: Optional[tuple[int, int]] = None
        best_indices: Optional[np.ndarray] = None
        best_parts: Optional[np.ndarray] = None
        for first in range(MODULUS - 1):
            for second in range(first + 1, MODULUS):
                indices = np.union1d(affected[first], affected[second])
                if not len(indices):
                    continue
                trial = np.asarray(key, dtype=np.int16)
                trial[first], trial[second] = trial[second], trial[first]
                decoded = trial[quads[indices]]
                candidate_parts = language.table[
                    decoded[:, 0],
                    decoded[:, 1],
                    decoded[:, 2],
                    decoded[:, 3],
                ] * weights[indices]
                delta = float(candidate_parts.sum() - parts[indices].sum())
                if delta > best_delta:
                    best_delta = delta
                    best_pair = (first, second)
                    best_indices = indices
                    best_parts = candidate_parts
        if best_pair is None or best_indices is None or best_parts is None:
            break
        first, second = best_pair
        key[first], key[second] = key[second], key[first]
        parts[best_indices] = best_parts
        swaps += 1
    return key, initial, float(parts.sum() / total_weight), swaps


def render_sample(
    lines: Sequence[SymbolLine],
    blocks: set[str],
    key: Sequence[int],
    limit: int = 360,
) -> str:
    parts = []
    length = 0
    for line in lines:
        if line.block not in blocks:
            continue
        text = "".join(ALPHABET[key[symbol]] for symbol in line.symbols)
        if text:
            parts.append(text)
            length += len(text) + 1
        if length >= limit:
            break
    return " ".join(parts)[:limit]


def plaintext_metrics(
    lines: Sequence[SymbolLine],
    blocks: set[str],
    key: Sequence[int],
) -> tuple[Optional[float], int]:
    correct = known = 0
    for line in lines:
        if line.block not in blocks:
            continue
        for symbol, truth in zip(line.symbols, line.truth):
            if truth is None:
                continue
            known += 1
            correct += key[symbol] == truth
    return (correct / known if known else None), known


def block_roles(
    blocks: Sequence[str], fold: int
) -> tuple[set[str], set[str], set[str]]:
    test = {block for index, block in enumerate(blocks) if index % 4 == fold}
    validation = {
        block
        for index, block in enumerate(blocks)
        if index % 4 == (fold + 1) % 4
    }
    fit = set(blocks) - test - validation
    return fit, validation, test


def search_fold(
    numeric_by_feature: dict[str, Sequence[NumericLine]],
    programs: Sequence[Program],
    languages: dict[str, LanguageSpec],
    blocks: Sequence[str],
    fold: int,
    shortlist: int,
    maximum_swaps: int,
) -> FoldResult:
    fit_blocks, validation_blocks, test_blocks = block_roles(blocks, fold)
    transformed = {
        (feature, program.name): [
            apply_program(line, program) for line in numeric
        ]
        for feature, numeric in numeric_by_feature.items()
        for program in programs
    }
    program_lookup = {program.name: program for program in programs}

    initial_rows = []
    for (feature, program_name), lines in transformed.items():
        program = program_lookup[program_name]
        for language_name, language in languages.items():
            key = initialize_key(lines, fit_blocks, language)
            validation_quads, validation_weights = quad_counter(
                lines, validation_blocks
            )
            validation_score = score_key(
                validation_quads, validation_weights, key, language
            )
            initial_rows.append(
                (
                    validation_score,
                    feature,
                    program,
                    language_name,
                    key,
                    lines,
                )
            )
    initial_rows.sort(key=lambda item: item[0], reverse=True)

    refined: list[CandidateFit] = []
    for initial_validation, feature, program, language_name, key, lines in (
        initial_rows[:shortlist]
    ):
        language = languages[language_name]
        key, fit_initial, fit_final, swaps = refine_key(
            lines, fit_blocks, language, key, maximum_swaps
        )
        validation_quads, validation_weights = quad_counter(
            lines, validation_blocks
        )
        validation_score = score_key(
            validation_quads, validation_weights, key, language
        )
        sample_size = max(1.0, float(validation_weights.sum()))
        penalty = program.complexity * math.log(max(sample_size, 2.0)) / (
            2.0 * sample_size
        )
        refined.append(
            CandidateFit(
                feature,
                program.name,
                language_name,
                key,
                initial_validation,
                validation_score,
                validation_score - penalty,
                fit_initial,
                fit_final,
                swaps,
            )
        )
    winner = max(refined, key=lambda item: item.validation_penalized)
    winner_program = program_lookup[winner.program]
    winner_lines = transformed[(winner.feature, winner.program)]
    language = languages[winner.language]
    test_quads, test_weights = quad_counter(winner_lines, test_blocks)
    initial_key = initialize_key(winner_lines, fit_blocks, language)
    test_initial = score_key(
        test_quads, test_weights, initial_key, language
    )
    test_final = score_key(
        test_quads, test_weights, winner.key, language
    )
    accuracy, known = plaintext_metrics(
        winner_lines, test_blocks, winner.key
    )
    return FoldResult(
        fold=fold,
        fit_blocks=sorted(fit_blocks),
        validation_blocks=sorted(validation_blocks),
        test_blocks=sorted(test_blocks),
        feature=winner.feature,
        program=winner.program,
        language=winner.language,
        program_complexity=winner_program.complexity,
        initial_validation=winner.initial_validation,
        validation_score=winner.validation_score,
        validation_penalized=winner.validation_penalized,
        test_initial=test_initial,
        test_final=test_final,
        test_residual=test_final - test_initial,
        gap_to_language_ceiling=test_final - language.ceiling,
        swaps=winner.swaps,
        plaintext_accuracy=accuracy,
        known_plaintext=known,
        sample=render_sample(
            winner_lines, test_blocks, winner.key
        ),
    )


def run_search(
    numeric_by_feature: dict[str, Sequence[NumericLine]],
    programs: Sequence[Program],
    languages: dict[str, LanguageSpec],
    blocks: Sequence[str],
    shortlist: int,
    maximum_swaps: int,
) -> list[FoldResult]:
    return [
        search_fold(
            numeric_by_feature,
            programs,
            languages,
            blocks,
            fold,
            shortlist,
            maximum_swaps,
        )
        for fold in range(4)
    ]


def invert_program(
    program: Program,
    cipher: Sequence[Optional[int]],
    rng: random.Random,
) -> tuple[int, ...]:
    values = [0] * len(cipher)
    if program.skip_first:
        values[0] = rng.randrange(MODULUS)
    if program.kind == "affine":
        start = 1 if program.skip_first else 0
        seed = values[0]
        for index in range(start, len(values)):
            if cipher[index] is None:
                raise ValueError("missing synthetic cipher output")
            previous = values[index - 1] if index else 0
            values[index] = (
                int(cipher[index])
                - program.a * previous
                - program.b * seed
                - program.c * index
            ) % MODULUS
    elif program.kind in {"running_plus", "running_minus"}:
        state = 0
        for index in range(len(values)):
            target = int(cipher[index])
            if program.kind == "running_plus":
                values[index] = (target - state) % MODULUS
            else:
                values[index] = (state - target) % MODULUS
            state = target
    elif program.kind == "seeded_running_plus":
        state = values[0]
        for index in range(1, len(values)):
            target = int(cipher[index])
            values[index] = (target - state) % MODULUS
            state = target
    else:
        raise ValueError(program.kind)
    return tuple(values)


def synthetic_numeric_lines(
    template: Sequence[WordLine],
    plaintext: str,
    program: Program,
    rng: random.Random,
) -> tuple[list[NumericLine], list[int]]:
    key = list(range(MODULUS))
    rng.shuffle(key)
    inverse_key = [0] * MODULUS
    for cipher_symbol, plain_letter in enumerate(key):
        inverse_key[plain_letter] = cipher_symbol

    plain_ids = [ALPHABET.index(character) for character in plaintext]
    cursor = 0
    result = []
    for line in template:
        truth: list[Optional[int]] = [None] * len(line.words)
        cipher: list[Optional[int]] = [None] * len(line.words)
        start = 1 if program.skip_first else 0
        for index in range(start, len(line.words)):
            letter = plain_ids[cursor % len(plain_ids)]
            cursor += 1
            truth[index] = letter
            cipher[index] = inverse_key[letter]
        values = invert_program(program, cipher, rng)
        result.append(NumericLine(line.block, values, tuple(truth)))
    return result, key


def summarize_folds(folds: Sequence[FoldResult]) -> dict:
    selections = Counter(
        (fold.feature, fold.program, fold.language) for fold in folds
    )
    return {
        "mean_test_final": mean(fold.test_final for fold in folds),
        "mean_test_residual": mean(fold.test_residual for fold in folds),
        "mean_gap_to_language_ceiling": mean(
            fold.gap_to_language_ceiling for fold in folds
        ),
        "mean_plaintext_accuracy": (
            mean(
                fold.plaintext_accuracy
                for fold in folds
                if fold.plaintext_accuracy is not None
            )
            if any(fold.plaintext_accuracy is not None for fold in folds)
            else None
        ),
        "selection_consistency": max(selections.values()) / len(folds),
        "selection_counts": {
            "|".join(key): value for key, value in sorted(selections.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nulls", type=int, default=8)
    parser.add_argument("--shortlist", type=int, default=15)
    parser.add_argument("--maximum-swaps", type=int, default=64)
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=tuple(CONTROL_PATHS),
        default=("latin", "english"),
    )
    parser.add_argument(
        "--synthetic-programs",
        nargs="+",
        default=("lag_minus", "seed_minus", "running_plus"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data"
        / "intermediate"
        / "followups_stateful_line_program_search.json",
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nulls < 1 or args.shortlist < 1 or args.maximum_swaps < 0:
        raise ValueError("invalid search limits")
    if "latin" not in args.languages:
        raise ValueError("Latin is required for the synthetic recovery gates")
    program_lookup = {program.name: program for program in PROGRAMS}
    unknown = set(args.synthetic_programs) - set(program_lookup)
    if unknown:
        raise ValueError(f"unknown synthetic programs: {sorted(unknown)}")

    word_lines, blocks = load_word_lines()
    languages = load_languages(args.languages)
    latin_clean = clean_latin(
        CONTROL_PATHS["latin"].read_text(encoding="utf-8", errors="ignore")
    )
    control_plaintext = latin_clean[-LM_CEILING_CHARS:]

    synthetic_reports = []
    for offset, program_name in enumerate(args.synthetic_programs):
        true_program = program_lookup[program_name]
        numeric, _truth_key = synthetic_numeric_lines(
            word_lines,
            control_plaintext,
            true_program,
            random.Random(SEED + 1000 + offset),
        )
        folds = run_search(
            {"synthetic": numeric},
            PROGRAMS,
            {"latin": languages["latin"]},
            blocks,
            min(args.shortlist, len(PROGRAMS)),
            args.maximum_swaps,
        )
        summary = summarize_folds(folds)
        summary["true_program"] = program_name
        summary["passed"] = (
            summary["mean_plaintext_accuracy"] is not None
            and summary["mean_plaintext_accuracy"] >= 0.85
            and summary["mean_gap_to_language_ceiling"] >= -0.50
        )
        synthetic_reports.append(
            {
                "summary": summary,
                "folds": [asdict(fold) for fold in folds],
            }
        )
        if args.progress:
            print(
                f"synthetic {program_name}: "
                f"accuracy={summary['mean_plaintext_accuracy']:.4f} "
                f"gap={summary['mean_gap_to_language_ceiling']:+.4f} "
                f"pass={summary['passed']}",
                flush=True,
            )

    observed_numeric = {
        feature: feature_lines(word_lines, feature) for feature in FEATURES
    }
    observed_folds = run_search(
        observed_numeric,
        PROGRAMS,
        languages,
        blocks,
        args.shortlist,
        args.maximum_swaps,
    )
    observed_summary = summarize_folds(observed_folds)
    if args.progress:
        print(
            f"Voynich: final={observed_summary['mean_test_final']:+.4f} "
            f"residual={observed_summary['mean_test_residual']:+.4f} "
            f"gap={observed_summary['mean_gap_to_language_ceiling']:+.4f}",
            flush=True,
        )

    rng = random.Random(SEED + 5000)
    null_reports = []
    for replicate in range(args.nulls):
        shuffled = strict_shuffle(word_lines, rng)
        numeric = {
            feature: feature_lines(shuffled, feature) for feature in FEATURES
        }
        folds = run_search(
            numeric,
            PROGRAMS,
            languages,
            blocks,
            args.shortlist,
            args.maximum_swaps,
        )
        summary = summarize_folds(folds)
        null_reports.append(
            {
                "replicate": replicate + 1,
                "summary": summary,
                "folds": [asdict(fold) for fold in folds],
            }
        )
        if args.progress:
            print(
                f"null {replicate + 1:02d}/{args.nulls}: "
                f"final={summary['mean_test_final']:+.4f} "
                f"residual={summary['mean_test_residual']:+.4f} "
                f"gap={summary['mean_gap_to_language_ceiling']:+.4f}",
                flush=True,
            )

    null_final = [
        report["summary"]["mean_test_final"] for report in null_reports
    ]
    null_residual = [
        report["summary"]["mean_test_residual"] for report in null_reports
    ]
    exceeds_final = sum(
        value >= observed_summary["mean_test_final"] for value in null_final
    )
    exceeds_residual = sum(
        value >= observed_summary["mean_test_residual"]
        for value in null_residual
    )
    synthetic_pass = all(
        report["summary"]["passed"] for report in synthetic_reports
    )
    real_pass = (
        synthetic_pass
        and observed_summary["mean_test_final"] > max(null_final)
        and observed_summary["mean_test_residual"] > max(null_residual)
        and observed_summary["mean_gap_to_language_ceiling"] >= -0.50
        and observed_summary["selection_consistency"] >= 0.75
    )
    gate = {
        "synthetic_controls_pass": synthetic_pass,
        "observed_exceeds_null_final_maximum": (
            observed_summary["mean_test_final"] > max(null_final)
        ),
        "observed_exceeds_null_residual_maximum": (
            observed_summary["mean_test_residual"] > max(null_residual)
        ),
        "observed_final_empirical_p": (
            exceeds_final + 1
        ) / (args.nulls + 1),
        "observed_residual_empirical_p": (
            exceeds_residual + 1
        ) / (args.nulls + 1),
        "language_ceiling_gap_required": -0.50,
        "selection_consistency_required": 0.75,
        "real_pass": real_pass,
    }
    report = {
        "method": (
            "nested complete-quire search over deterministic word coordinates, "
            "line-reset modular programs, frozen languages, and global "
            "monoalphabetic output keys"
        ),
        "scope": {
            "modulus": MODULUS,
            "alphabet": ALPHABET,
            "features": list(FEATURES),
            "programs": [asdict(program) for program in PROGRAMS],
            "languages": list(languages),
            "candidate_count": len(FEATURES) * len(PROGRAMS) * len(languages),
            "shortlist": args.shortlist,
            "maximum_key_swaps": args.maximum_swaps,
            "nulls": args.nulls,
            "null_search_repeated_in_full": True,
        },
        "corpus": {
            "lines": len(word_lines),
            "words": sum(len(line.words) for line in word_lines),
            "blocks": blocks,
        },
        "synthetic_controls": synthetic_reports,
        "observed": {
            "summary": observed_summary,
            "folds": [asdict(fold) for fold in observed_folds],
        },
        "nulls": null_reports,
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("=" * 96)
    print("STATEFUL LINE-PROGRAM SEARCH")
    print("=" * 96)
    print(json.dumps({
        "synthetic": [
            item["summary"] for item in synthetic_reports
        ],
        "observed": observed_summary,
        "null_final_maximum": max(null_final),
        "null_residual_maximum": max(null_residual),
        "gate": gate,
    }, indent=2))
    for fold in observed_folds:
        print(
            f"fold {fold.fold}: {fold.feature}/{fold.program}/"
            f"{fold.language} final={fold.test_final:+.5f} "
            f"residual={fold.test_residual:+.5f} "
            f"gap={fold.gap_to_language_ceiling:+.5f}"
        )
        print(f"  {fold.sample}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
