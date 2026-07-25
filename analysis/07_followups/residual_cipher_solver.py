#!/usr/bin/env python3
"""
Residualized held-out attack on a Naibbe-style generative cipher.

This is a stricter successor to naibbe_style_attack.py and
stego_slot_cipher.py.  It retains their defensible positive control and
homophonic search, but changes the evidence criterion:

* outer test sets contain complete Voynich quires or contiguous Naibbe blocks;
* key size and sequential-mixture weight are selected on an inner validation
  fold, never on the outer test fold;
* uncertain, one-character, unresolved, and out-of-training-inventory
  emissions are retained as hard breaks, never deleted in a way that joins
  formerly non-adjacent material;
* candidate evidence is held-out likelihood gain over
  P(character | emission-role, original word-position);
* lambda=0 is the exact position-only baseline and has score zero;
* an active key pays a BIC/MDL penalty for every independently mapped symbol.

The official Naibbe table supplies two distinct controls.  Structural labels
calibrate the frozen segmenter without exposing plaintext values.  Separately,
the official value-bearing key gives an oracle meaningful-cipher ceiling.  The
blind solver never reads those values; they are used only after fitting for
Naibbe recovery accuracy.

Positive residual likelihood means that a proposed mapping uses sequential
Latin constraints beyond the word-position baseline.  It is not a translation,
and generic forward directionality contributes exactly zero by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional, Sequence

import numpy as np

from naibbe_style_attack import SegmentModel, calibrate, structural_counts


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "corpus.json"
LATIN = ROOT / "data" / "controls" / "latin.txt"
SEED = 20260723
ALPHA = 0.5
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class RawLine:
    block: str
    words: tuple[Optional[str], ...]


@dataclass(frozen=True)
class EmissionLine:
    block: str
    symbols: tuple[str, ...]
    slots: tuple[str, ...]
    truth: tuple[Optional[str], ...]


@dataclass(frozen=True)
class EncodedLine:
    block: str
    symbols: np.ndarray
    slots: tuple[str, ...]
    truth: tuple[Optional[str], ...]


@dataclass
class LanguageModel:
    alphabet: tuple[str, ...]
    char_to_id: dict[str, int]
    logp: np.ndarray
    unigram: np.ndarray

    @classmethod
    def from_text(cls, text: str) -> "LanguageModel":
        alphabet = tuple(sorted(set(text)))
        lookup = {character: index for index, character in enumerate(alphabet)}
        width = len(alphabet)
        context = Counter(zip(text, text[1:]))
        triple = Counter(zip(text, text[1:], text[2:]))
        logp = np.empty((width, width, width), dtype=np.float64)
        for first in alphabet:
            for second in alphabet:
                denominator = context[(first, second)] + 0.1 * width
                for third in alphabet:
                    logp[
                        lookup[first], lookup[second], lookup[third]
                    ] = math.log(
                        (triple[(first, second, third)] + 0.1)
                        / denominator
                    )
        frequencies = Counter(text)
        unigram = np.array(
            [frequencies[character] + 0.1 for character in alphabet],
            dtype=float,
        )
        unigram /= unigram.sum()
        return cls(alphabet, lookup, logp, unigram)


@dataclass
class KeyCandidate:
    key: np.ndarray
    codebook: int
    parameters: int
    lambda_: float
    validation_gain: float
    validation_penalized: float
    validation_lm: float
    fit_score: float
    lines: list[EncodedLine]
    names: tuple[str, ...]


@dataclass
class FoldResult:
    corpus: str
    fold: int
    fit_blocks: int
    validation_blocks: int
    test_blocks: int
    codebook: int
    parameters: int
    lambda_: float
    active: bool
    validation_gain: float
    validation_penalized: float
    test_gain: float
    test_lm: float
    test_accuracy: Optional[float]
    test_known: int
    sample: str


def normalize_latin(text: str) -> str:
    text = text.lower().replace("j", "i").replace("v", "u").replace("w", "u")
    return re.sub(r"[^a-z]+", "", text)


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


def load_voynich() -> tuple[list[RawLine], list[str]]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    lines = []
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
                lines.append(RawLine(block, words))
    blocks = sorted({line.block for line in lines})
    return lines, blocks


def load_naibbe_raw(path: Path, n_blocks: int = 16) -> tuple[list[RawLine], list[str]]:
    token_lines = [
        re.findall(r"[a-z]+", line.lower())
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
    ]
    token_lines = [line for line in token_lines if line]
    total = sum(map(len, token_lines))
    boundaries = [total * index / n_blocks for index in range(1, n_blocks)]
    lines = []
    cumulative = 0
    block_index = 0
    for words in token_lines:
        while (
            block_index < n_blocks - 1
            and cumulative >= boundaries[block_index]
        ):
            block_index += 1
        block = f"N{block_index:02d}"
        lines.append(RawLine(block, tuple(words)))
        cumulative += len(words)
    blocks = [f"N{index:02d}" for index in range(n_blocks)]
    return lines, blocks


def table_maps(
    path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    maps = ({}, {}, {})
    for row in rows:
        code, glyph = row["code"], row["glyphs"]
        letter = normalize_latin(code.split("_")[-1])
        if code.startswith("unigram_"):
            maps[0][glyph] = letter
        elif code.startswith("prefix_"):
            maps[1][glyph] = letter
        elif code.startswith("suffix_"):
            maps[2][glyph] = letter
    return maps


def naibbe_emission(
    word: str,
    unigram: dict[str, str],
    prefix: dict[str, str],
    suffix: dict[str, str],
) -> tuple[tuple[str, ...], tuple[Optional[str], ...]]:
    if word in unigram:
        return ("U:" + word,), (unigram[word],)
    candidates = [
        (word[:index], word[index:])
        for index in range(1, len(word))
        if word[:index] in prefix and word[index:] in suffix
    ]
    readings = {
        prefix[left] + suffix[right] for left, right in candidates
    }
    if len(readings) == 1 and candidates:
        left, right = candidates[0]
        reading = next(iter(readings))
        return ("L:" + left, "R:" + right), (reading[0], reading[1])
    # Preserve the token in the blind stream but mark oracle truth unavailable.
    return ("U?:" + word,), (None,)


def emit_naibbe(
    raw: Sequence[RawLine],
    table_path: Path,
) -> tuple[list[EmissionLine], float]:
    unigram, prefix, suffix = table_maps(table_path)
    lines = []
    known = total = 0
    for line in raw:
        symbols, slots, truth = [], [], []
        length = len(line.words)
        for index, word in enumerate(line.words):
            if word is None:
                continue
            emitted, letters = naibbe_emission(word, unigram, prefix, suffix)
            position = position_bucket(index, length)
            symbols.extend(emitted)
            slots.extend(
                f"{symbol.split(':', 1)[0]}@{position}" for symbol in emitted
            )
            truth.extend(letters)
            total += len(letters)
            known += sum(letter is not None for letter in letters)
        lines.append(
            EmissionLine(line.block, tuple(symbols), tuple(slots), tuple(truth))
        )
    return lines, known / total


def valid_runs(line: RawLine) -> Iterable[list[tuple[int, str]]]:
    current = []
    for index, word in enumerate(line.words):
        if word is None:
            if current:
                yield current
                current = []
        else:
            current.append((index, word))
    if current:
        yield current


def emit_voynich(
    raw: Sequence[RawLine],
    model: SegmentModel,
    fit_blocks: set[str],
) -> list[EmissionLine]:
    training_words = [
        word
        for line in raw
        if line.block in fit_blocks
        for word in line.words
        if word is not None
    ]
    counts = structural_counts(training_words)
    emitted_lines = []
    for line in raw:
        length = len(line.words)
        for run in valid_runs(line):
            symbols, slots = [], []
            for index, word in run:
                emission = model.segment(word, counts)
                position = position_bucket(index, length)
                symbols.extend(emission)
                slots.extend(
                    f"{symbol.split(':', 1)[0]}@{position}"
                    for symbol in emission
                )
            emitted_lines.append(
                EmissionLine(
                    line.block,
                    tuple(symbols),
                    tuple(slots),
                    tuple(None for _ in symbols),
                )
            )
    return emitted_lines


def block_folds(blocks: Sequence[str]) -> dict[str, int]:
    return {block: index % 4 for index, block in enumerate(sorted(blocks))}


def collapse_and_encode(
    lines: Sequence[EmissionLine],
    fit_blocks: set[str],
    codebook: int,
) -> tuple[list[EncodedLine], tuple[str, ...]]:
    by_role: dict[str, Counter] = defaultdict(Counter)
    for line in lines:
        if line.block not in fit_blocks:
            continue
        for symbol in line.symbols:
            if symbol.startswith("U?:"):
                continue
            role = symbol.split(":", 1)[0]
            by_role[role][symbol] += 1
    retained = {
        symbol
        for role_counts in by_role.values()
        for symbol, _ in role_counts.most_common(codebook)
    }
    ordered = tuple(sorted(retained))
    lookup = {name: index for index, name in enumerate(ordered)}
    encoded = []
    for line in lines:
        run_symbols, run_slots, run_truth = [], [], []

        def flush() -> None:
            if run_symbols:
                encoded.append(EncodedLine(
                    line.block,
                    np.array(
                        [lookup[symbol] for symbol in run_symbols],
                        dtype=np.int32,
                    ),
                    tuple(run_slots),
                    tuple(run_truth),
                ))
                run_symbols.clear()
                run_slots.clear()
                run_truth.clear()

        for symbol, slot, truth in zip(
            line.symbols, line.slots, line.truth
        ):
            if symbol not in retained or symbol.startswith("U?:"):
                flush()
                continue
            run_symbols.append(symbol)
            run_slots.append(slot)
            run_truth.append(truth)
        flush()
    return encoded, ordered


def ngram_table(
    lines: Sequence[EncodedLine],
    blocks: set[str],
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    counts = Counter()
    for line in lines:
        if line.block not in blocks or len(line.symbols) < 3:
            continue
        counts.update(
            zip(line.symbols[:-2], line.symbols[1:-1], line.symbols[2:])
        )
    triples = np.array(list(counts), dtype=np.int32)
    weights = np.array(list(counts.values()), dtype=np.float64)
    affected: dict[int, list[int]] = defaultdict(list)
    for row, triple in enumerate(triples):
        for symbol in set(map(int, triple)):
            affected[symbol].append(row)
    return triples, weights, {
        symbol: np.array(indices, dtype=np.int32)
        for symbol, indices in affected.items()
    }


def raw_lm_score(
    triples: np.ndarray,
    weights: np.ndarray,
    key: np.ndarray,
    lm: LanguageModel,
    indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    rows = triples if indices is None else triples[indices]
    selected_weights = weights if indices is None else weights[indices]
    return selected_weights * lm.logp[
        key[rows[:, 0]], key[rows[:, 1]], key[rows[:, 2]]
    ]


def search_keys(
    lines: Sequence[EncodedLine],
    fit_blocks: set[str],
    names: Sequence[str],
    lm: LanguageModel,
    rng: np.random.Generator,
    steps: int,
    restarts: int,
) -> list[tuple[np.ndarray, float]]:
    triples, weights, affected = ngram_table(lines, fit_blocks)
    if not len(triples):
        raise ValueError("no fit trigrams")
    total = float(weights.sum())
    symbol_frequency = np.zeros(len(names), dtype=float)
    for line in lines:
        if line.block in fit_blocks:
            symbol_frequency += np.bincount(
                line.symbols, minlength=len(names)
            )

    def unigram_kl(key: np.ndarray) -> float:
        decoded = np.bincount(
            key,
            weights=symbol_frequency,
            minlength=len(lm.alphabet),
        )
        decoded /= decoded.sum()
        mask = decoded > 0
        return float(np.sum(
            decoded[mask] * np.log(decoded[mask] / lm.unigram[mask])
        ))

    candidates = []
    for _ in range(restarts):
        key = rng.choice(
            len(lm.alphabet), size=len(names), p=lm.unigram
        ).astype(np.int32)
        contribution = raw_lm_score(triples, weights, key, lm)
        current_kl = unigram_kl(key)
        current = contribution.sum() - 2.0 * total * current_kl
        best = current
        best_key = key.copy()
        for step in range(steps):
            symbol = int(rng.integers(len(names)))
            if symbol not in affected:
                continue
            old = int(key[symbol])
            replacement = int(rng.choice(len(lm.alphabet), p=lm.unigram))
            if replacement == old:
                continue
            indices = affected[symbol]
            before = contribution[indices].sum()
            key[symbol] = replacement
            proposed = raw_lm_score(
                triples, weights, key, lm, indices
            )
            proposed_kl = unigram_kl(key)
            delta = (
                proposed.sum() - before
                - 2.0 * total * (proposed_kl - current_kl)
            )
            temperature = 12.0 * (1.0 - step / steps) + 0.05
            if delta >= 0 or rng.random() < math.exp(delta / temperature):
                contribution[indices] = proposed
                current += delta
                current_kl = proposed_kl
                if current > best:
                    best = current
                    best_key = key.copy()
            else:
                key[symbol] = old
        candidates.append((best_key, best / total / math.log(2)))
    return candidates


def fit_position_baseline(
    lines: Sequence[EncodedLine],
    blocks: set[str],
    key: np.ndarray,
    width: int,
) -> tuple[Counter, Counter]:
    counts, totals = Counter(), Counter()
    for line in lines:
        if line.block not in blocks:
            continue
        for symbol, slot in zip(line.symbols, line.slots):
            character = int(key[symbol])
            counts[(slot, character)] += 1
            totals[slot] += 1
    return counts, totals


def residual_score(
    lines: Sequence[EncodedLine],
    blocks: set[str],
    fit_blocks: set[str],
    key: np.ndarray,
    lm: LanguageModel,
    lambda_: float,
) -> tuple[float, float, int]:
    baseline, totals = fit_position_baseline(
        lines, fit_blocks, key, len(lm.alphabet)
    )
    residual = lm_score = 0.0
    observations = 0
    width = len(lm.alphabet)
    for line in lines:
        if line.block not in blocks or len(line.symbols) < 3:
            continue
        decoded = key[line.symbols]
        for index in range(2, len(decoded)):
            character = int(decoded[index])
            slot = line.slots[index]
            base_probability = (
                baseline[(slot, character)] + ALPHA
            ) / (totals[slot] + ALPHA * width)
            language_probability = math.exp(
                lm.logp[
                    int(decoded[index - 2]),
                    int(decoded[index - 1]),
                    character,
                ]
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


def recovery_accuracy(
    lines: Sequence[EncodedLine],
    blocks: set[str],
    key: np.ndarray,
    lm: LanguageModel,
) -> tuple[Optional[float], int]:
    correct = known = 0
    for line in lines:
        if line.block not in blocks:
            continue
        for symbol, truth in zip(line.symbols, line.truth):
            if truth is None or truth not in lm.char_to_id:
                continue
            known += 1
            correct += int(key[symbol] == lm.char_to_id[truth])
    return (correct / known if known else None), known


def render(
    lines: Sequence[EncodedLine],
    blocks: set[str],
    key: np.ndarray,
    lm: LanguageModel,
    limit: int = 180,
) -> str:
    pieces = []
    for line in lines:
        if line.block in blocks:
            pieces.append("".join(lm.alphabet[index] for index in key[line.symbols]))
            if sum(map(len, pieces)) >= limit:
                break
    return " / ".join(pieces)[:limit]


def select_candidate(
    emission_lines: Sequence[EmissionLine],
    fit_blocks: set[str],
    validation_blocks: set[str],
    codebooks: Sequence[int],
    lm: LanguageModel,
    rng: np.random.Generator,
    steps: int,
    restarts: int,
) -> KeyCandidate:
    best: Optional[KeyCandidate] = None
    for codebook in codebooks:
        lines, names = collapse_and_encode(
            emission_lines, fit_blocks, codebook
        )
        for key, fit_score in search_keys(
            lines, fit_blocks, names, lm, rng, steps, restarts
        ):
            for lambda_ in LAMBDAS:
                if lambda_ == 0:
                    continue
                gain, lm_score, n_validation = residual_score(
                    lines,
                    validation_blocks,
                    fit_blocks,
                    key,
                    lm,
                    lambda_,
                )
                penalty = (
                    len(names) * math.log2(max(n_validation, 2))
                    / (2 * max(n_validation, 1))
                )
                penalized = gain - penalty
                candidate = KeyCandidate(
                    key=key.copy(),
                    codebook=codebook,
                    parameters=len(names),
                    lambda_=lambda_,
                    validation_gain=gain,
                    validation_penalized=penalized,
                    validation_lm=lm_score,
                    fit_score=fit_score,
                    lines=lines,
                    names=names,
                )
                if (
                    best is None
                    or candidate.validation_penalized
                    > best.validation_penalized
                ):
                    best = candidate
    assert best is not None
    return best


def evaluate_corpus(
    name: str,
    raw_lines: Sequence[RawLine],
    blocks: Sequence[str],
    static_emissions: Optional[Sequence[EmissionLine]],
    segmenter: Optional[SegmentModel],
    codebooks: Sequence[int],
    lm: LanguageModel,
    steps: int,
    restarts: int,
) -> list[FoldResult]:
    folds = block_folds(blocks)
    results = []
    for outer in range(4):
        test_blocks = {block for block in blocks if folds[block] == outer}
        validation_fold = (outer + 1) % 4
        validation_blocks = {
            block for block in blocks if folds[block] == validation_fold
        }
        fit_blocks = set(blocks) - test_blocks - validation_blocks
        if static_emissions is None:
            assert segmenter is not None
            emissions = emit_voynich(raw_lines, segmenter, fit_blocks)
        else:
            emissions = list(static_emissions)
        rng = np.random.default_rng(
            SEED + 100_003 * outer + (0 if name == "NAIBBE_BLIND" else 1_000_003)
        )
        candidate = select_candidate(
            emissions,
            fit_blocks,
            validation_blocks,
            codebooks,
            lm,
            rng,
            steps,
            restarts,
        )
        active = candidate.validation_penalized > 0
        if active:
            test_gain, test_lm, _ = residual_score(
                candidate.lines,
                test_blocks,
                fit_blocks,
                candidate.key,
                lm,
                candidate.lambda_,
            )
        else:
            test_gain = 0.0
            _, test_lm, _ = residual_score(
                candidate.lines,
                test_blocks,
                fit_blocks,
                candidate.key,
                lm,
                0.0,
            )
        accuracy, known = recovery_accuracy(
            candidate.lines, test_blocks, candidate.key, lm
        )
        results.append(FoldResult(
            corpus=name,
            fold=outer,
            fit_blocks=len(fit_blocks),
            validation_blocks=len(validation_blocks),
            test_blocks=len(test_blocks),
            codebook=candidate.codebook,
            parameters=candidate.parameters,
            lambda_=candidate.lambda_,
            active=active,
            validation_gain=candidate.validation_gain,
            validation_penalized=candidate.validation_penalized,
            test_gain=test_gain,
            test_lm=test_lm,
            test_accuracy=accuracy if name == "NAIBBE_BLIND" else None,
            test_known=known if name == "NAIBBE_BLIND" else 0,
            sample=render(
                candidate.lines,
                test_blocks,
                candidate.key,
                lm,
            ),
        ))
    return results


def oracle_naibbe(
    emissions: Sequence[EmissionLine],
    blocks: Sequence[str],
    lm: LanguageModel,
) -> list[dict]:
    folds = block_folds(blocks)
    results = []
    for outer in range(4):
        test = {block for block in blocks if folds[block] == outer}
        validation = {
            block for block in blocks if folds[block] == (outer + 1) % 4
        }
        fit = set(blocks) - test - validation
        # Encode true letters directly. Unknown oracle emissions are hard breaks.
        converted = []
        for line in emissions:
            symbols, slots, truth = [], [], []
            for letter, slot in zip(line.truth, line.slots):
                if letter is None or letter not in lm.char_to_id:
                    if symbols:
                        converted.append(
                            (line.block, tuple(symbols), tuple(slots), tuple(truth))
                        )
                        symbols, slots, truth = [], [], []
                    continue
                symbols.append(letter)
                slots.append(slot)
                truth.append(letter)
            if symbols:
                converted.append(
                    (line.block, tuple(symbols), tuple(slots), tuple(truth))
                )
        names = lm.alphabet
        lookup = lm.char_to_id
        encoded = [
            EncodedLine(
                block,
                np.array([lookup[value] for value in symbols], dtype=np.int32),
                slots,
                truth,
            )
            for block, symbols, slots, truth in converted
        ]
        key = np.arange(len(names), dtype=np.int32)
        validation_rows = [
            (lambda_,) + residual_score(
                encoded, validation, fit, key, lm, lambda_
            )
            for lambda_ in LAMBDAS
        ]
        selected = max(validation_rows, key=lambda row: row[1])
        test_gain, test_lm, n_test = residual_score(
            encoded, test, fit, key, lm, selected[0]
        )
        results.append({
            "fold": outer,
            "lambda": selected[0],
            "validation_gain": selected[1],
            "test_gain": test_gain,
            "test_lm": test_lm,
            "test_n": n_test,
        })
    return results


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def say(self, value: str = "") -> None:
        print(value)
        self.lines.append(value)

    def write(self, path: Optional[Path]) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def report(
    calibration: dict,
    coverage: float,
    oracle: list[dict],
    blind: Sequence[FoldResult],
    voynich: Sequence[FoldResult],
    args: argparse.Namespace,
) -> Reporter:
    out = Reporter()
    out.say("RESIDUALIZED GENERATIVE-CIPHER SOLVER")
    out.say("=" * 100)
    out.say(
        f"seed={SEED} outer folds=4 steps={args.steps} "
        f"restarts={args.restarts} codebooks={args.codebook_sizes}"
    )
    out.say(
        "Evidence = held-out log2 likelihood gain of sequential Latin mixture "
        "over P(character|U/L/R role, original word-position)."
    )
    out.say(
        "lambda=0 is the inactive baseline with exactly 0 gain; active keys pay "
        "M*log2(n)/(2n) validation bits for M mapped symbols."
    )
    out.say(
        "No generic directionality term is present. Uncertain, one-character, "
        "unresolved, and out-of-fit-inventory emissions are hard breaks."
    )
    out.say(
        "Invariant: no n-gram crosses an exclusion, physical line boundary, "
        "held-out quire/block boundary, or outer-fold boundary."
    )
    out.say(
        f"Naibbe structural calibration held-out joint emission accuracy="
        f"{calibration['test']['joint_emission_accuracy']:.4f}; "
        f"official oracle emission coverage={coverage:.4f}"
    )

    out.say("\nNAIBBE ORACLE POSITIVE CONTROL")
    out.say(
        f"{'fold':>4s} {'lambda':>7s} {'val gain':>10s} "
        f"{'test gain':>10s} {'Latin LM':>10s} {'n':>8s}"
    )
    for row in oracle:
        out.say(
            f"{row['fold']:4d} {row['lambda']:7.2f} "
            f"{row['validation_gain']:+10.4f} {row['test_gain']:+10.4f} "
            f"{row['test_lm']:+10.4f} {row['test_n']:8d}"
        )
    out.say(
        f"  mean oracle test residual={mean(row['test_gain'] for row in oracle):+.4f}"
    )

    for title, rows in (
        ("NAIBBE BLIND SOLVER", blind),
        ("VOYNICH BLIND SOLVER", voynich),
    ):
        out.say("\n" + title)
        out.say(
            f"{'fold':>4s} {'blocks':>7s} {'K':>4s} {'M':>4s} "
            f"{'lam':>5s} {'on':>3s} "
            f"{'val raw':>9s} {'val pen':>9s} {'test res':>9s} "
            f"{'Latin LM':>9s} {'accuracy':>9s}"
        )
        for row in rows:
            accuracy = (
                f"{row.test_accuracy:.3f}"
                if row.test_accuracy is not None
                else "-"
            )
            partition = (
                f"{row.fit_blocks}/{row.validation_blocks}/{row.test_blocks}"
            )
            out.say(
                f"{row.fold:4d} {partition:>7s} "
                f"{row.codebook:4d} {row.parameters:4d} "
                f"{row.lambda_:5.2f} {int(row.active):3d} "
                f"{row.validation_gain:+9.4f} "
                f"{row.validation_penalized:+9.4f} "
                f"{row.test_gain:+9.4f} {row.test_lm:+9.4f} "
                f"{accuracy:>9s}"
            )
            out.say(f"     test sample: {row.sample}")
        out.say(
            f"  active folds={sum(row.active for row in rows)}/4; "
            f"mean test residual={mean(row.test_gain for row in rows):+.4f}; "
            f"mean Latin LM={mean(row.test_lm for row in rows):+.4f}"
        )
        if title.startswith("NAIBBE"):
            weighted_correct = sum(
                (row.test_accuracy or 0.0) * row.test_known for row in rows
            )
            known = sum(row.test_known for row in rows)
            out.say(
                f"  weighted known-key recovery="
                f"{weighted_correct / known if known else float('nan'):.4f} "
                f"over {known} held-out emissions "
                f"(uniform {23}-letter chance={1 / 23:.4f})"
            )

    out.say("\nINTERPRETATION")
    oracle_mean = mean(row["test_gain"] for row in oracle)
    naibbe_active = sum(row.active for row in blind)
    voynich_active = sum(row.active for row in voynich)
    out.say(
        f"  Oracle Naibbe must be positive for the residual instrument to be "
        f"valid: observed mean={oracle_mean:+.4f} bits/emission."
    )
    out.say(
        f"  Blind activation: Naibbe {naibbe_active}/4 folds; "
        f"Voynich {voynich_active}/4 folds."
    )
    out.say(
        "  Because the blind optimizer does not activate on known-meaningful "
        "Naibbe, the Voynich null is underpowered and excludes nothing beyond "
        "the particular recovered keys. The oracle result validates the "
        "residual score, not blind key recovery."
    )
    out.say(
        "  A positive result is a lead, not plaintext: it must also recover "
        "stable held-out character assignments and beat the Naibbe-calibrated "
        "positive-control threshold."
    )
    return out


def parse_args() -> argparse.Namespace:
    default_naibbe = Path(os.environ.get("NAIBBE_REPO", "/tmp/naibbe-cipher"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--naibbe-dir", type=Path, default=default_naibbe)
    parser.add_argument("--codebook-sizes", default="40,80")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/intermediate/followups_residual_cipher_solver.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.codebook_sizes = tuple(
        int(value) for value in args.codebook_sizes.split(",")
    )
    ciphertext = (
        args.naibbe_dir / "encrypted/nathist_output_ciphertext_respaced.txt"
    )
    table = args.naibbe_dir / "references/naibbe_tables.csv"
    for path in (ciphertext, table, LATIN, CORPUS):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    latin = normalize_latin(LATIN.read_text(encoding="utf-8", errors="ignore"))
    lm = LanguageModel.from_text(latin[: len(latin) * 2 // 3])
    naibbe_raw, naibbe_blocks = load_naibbe_raw(ciphertext)
    naibbe_emissions, coverage = emit_naibbe(naibbe_raw, table)
    segmenter, calibration = calibrate(
        [[word for word in line.words if word is not None] for line in naibbe_raw],
        table,
    )
    voynich_raw, voynich_blocks = load_voynich()

    oracle = oracle_naibbe(naibbe_emissions, naibbe_blocks, lm)
    blind = evaluate_corpus(
        "NAIBBE_BLIND",
        naibbe_raw,
        naibbe_blocks,
        naibbe_emissions,
        None,
        args.codebook_sizes,
        lm,
        args.steps,
        args.restarts,
    )
    voynich = evaluate_corpus(
        "VOYNICH",
        voynich_raw,
        voynich_blocks,
        None,
        segmenter,
        args.codebook_sizes,
        lm,
        args.steps,
        args.restarts,
    )
    output = report(calibration, coverage, oracle, blind, voynich, args)
    output.write(args.output)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
