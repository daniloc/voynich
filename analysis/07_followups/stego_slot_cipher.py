#!/usr/bin/env python3
"""
Treat Voynich prefix/suffix choices as homophonic payload symbols.

The visible word engine has small closed prefix and suffix domains.  If it is
covertext, a writer could choose among those forms to encode characters while
leaving the open core to maintain the surface texture.  This script attacks
four fixed symbol streams with a many-to-one homophonic substitution:

  prefix, suffix, prefix/suffix pair, and interleaved prefix then suffix.

Keys are fit on the first half and frozen on the second.  Both uninterrupted
streams and a model in which each physical line break is a plaintext space are
tested.  Samples from both halves are printed so fluent-looking train overfit
cannot pass unnoticed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "corpus.json"
CONTROLS = {
    "english": ROOT / "data" / "controls" / "english.txt",
    "latin": ROOT / "data" / "controls" / "latin.txt",
}
SEED = 20260723
BOUNDARY = "<SPACE>"

PREFIXES = (
    "qok", "qot", "cth", "ckh", "cph", "cfh", "qo", "ok", "ot",
    "ch", "sh", "o", "y", "d",
)
SUFFIXES = (
    "eedy", "eody", "aiin", "aiir", "edy", "dam", "ain", "iin",
    "dy", "ol", "or", "ar", "al", "am", "eey", "ey", "y",
)


@dataclass(frozen=True)
class Stream:
    name: str
    symbols: tuple[str, ...]
    fixed_space: bool


def decompose(word: str) -> tuple[str, str, str]:
    prefix = next((p for p in PREFIXES if word.startswith(p)), "")
    remainder = word[len(prefix):]
    suffix = next(
        (
            s for s in SUFFIXES
            if remainder.endswith(s) and len(remainder) > len(s)
        ),
        "",
    )
    core = remainder[:-len(suffix)] if suffix else remainder
    return prefix, core, suffix


def clean_control(path: Path, language: str, spaces: bool) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    start = re.search(r"\*\*\* start of.*?\*\*\*", text, re.S)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* end of", text, re.S)
    if end:
        text = text[:end.start()]
    if language == "latin":
        text = text.replace("j", "i").replace("v", "u").replace("w", "u")
    if spaces:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", text)).strip()
    return re.sub(r"[^a-z]+", "", text)


def load_lines() -> list[list[tuple[str, str, str]]]:
    data = json.loads(CORPUS.read_text())
    lines = []
    for folio_lines in data["folios"].values():
        for line in folio_lines:
            words = [
                decompose(word) for word in line["words"]
                if "?" not in word and word.isalpha()
            ]
            if words:
                lines.append(words)
    return lines


def make_streams(lines: list[list[tuple[str, str, str]]]) -> list[Stream]:
    out = []
    for spaces in (False, True):
        suffix = "/line_spaces" if spaces else "/continuous"
        builders = {
            "prefix": [],
            "suffix": [],
            "pair": [],
            "prefix_suffix": [],
            "suffix_prefix": [],
        }
        for line in lines:
            for prefix, _, ending in line:
                builders["prefix"].append("P:" + prefix)
                builders["suffix"].append("S:" + ending)
                builders["pair"].append("X:" + prefix + "|" + ending)
                builders["prefix_suffix"].extend(
                    ("P:" + prefix, "S:" + ending)
                )
                builders["suffix_prefix"].extend(
                    ("S:" + ending, "P:" + prefix)
                )
            if spaces:
                for symbols in builders.values():
                    symbols.append(BOUNDARY)
        out.extend(
            Stream(name + suffix, tuple(symbols), spaces)
            for name, symbols in builders.items()
        )
    return out


class LM:
    def __init__(self, text: str):
        alphabet = sorted(set(text))
        self.char_to_id = {char: i for i, char in enumerate(alphabet)}
        self.id_to_char = alphabet
        width = len(alphabet)
        quad = Counter(zip(text, text[1:], text[2:], text[3:]))
        tri = Counter(zip(text, text[1:], text[2:]))
        self.logp = np.empty((width, width, width, width), dtype=np.float64)
        for a in alphabet:
            for b in alphabet:
                for c in alphabet:
                    denominator = tri[(a, b, c)] + 0.1 * width
                    for d in alphabet:
                        self.logp[
                            self.char_to_id[a],
                            self.char_to_id[b],
                            self.char_to_id[c],
                            self.char_to_id[d],
                        ] = math.log(
                            (quad[(a, b, c, d)] + 0.1) / denominator
                        )
        self.target_ids = [
            self.char_to_id[char] for char in alphabet if char != " "
        ]
        frequencies = Counter(text)
        all_weights = np.array(
            [frequencies[char] for char in alphabet], dtype=float
        )
        self.all_probs = all_weights / all_weights.sum()
        weights = np.array([frequencies[alphabet[i]] for i in self.target_ids],
                           dtype=float)
        self.target_weights = weights / weights.sum()

    def plain_score(self, text: str) -> float:
        ids = [self.char_to_id[c] for c in text if c in self.char_to_id]
        if len(ids) < 4:
            return float("-inf")
        return float(np.mean([
            self.logp[a, b, c, d]
            for a, b, c, d in zip(ids, ids[1:], ids[2:], ids[3:])
        ]))


def encode_symbols(
    symbols: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    names = sorted(set(symbols))
    lookup = {symbol: index for index, symbol in enumerate(names)}
    return np.array([lookup[symbol] for symbol in symbols], dtype=np.int32), names


def ngram_table(encoded: np.ndarray):
    counts = Counter(
        zip(encoded[:-3], encoded[1:-2], encoded[2:-1], encoded[3:])
    )
    triples = np.array(list(counts), dtype=np.int32)
    weights = np.array(list(counts.values()), dtype=np.float64)
    affected = defaultdict(list)
    for index, triple in enumerate(triples):
        for symbol in set(map(int, triple)):
            affected[symbol].append(index)
    return triples, weights, {
        symbol: np.array(indices, dtype=np.int32)
        for symbol, indices in affected.items()
    }


def contributions(
    triples: np.ndarray,
    weights: np.ndarray,
    key: np.ndarray,
    lm: LM,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    rows = triples if indices is None else triples[indices]
    return (
        weights if indices is None else weights[indices]
    ) * lm.logp[
        key[rows[:, 0]], key[rows[:, 1]], key[rows[:, 2]], key[rows[:, 3]]
    ]


def fit(
    stream: Stream,
    lm: LM,
    rng: np.random.Generator,
    steps: int,
    restarts: int,
) -> dict:
    split = len(stream.symbols) // 2
    train_symbols = stream.symbols[:split]
    test_symbols = stream.symbols[split:]
    all_encoded, names = encode_symbols(stream.symbols)
    name_to_id = {name: i for i, name in enumerate(names)}
    train = np.array([name_to_id[s] for s in train_symbols], dtype=np.int32)
    test = np.array([name_to_id[s] for s in test_symbols], dtype=np.int32)
    triples, weights, affected = ngram_table(train)
    total = weights.sum()
    symbol_frequency = np.bincount(
        train, minlength=len(names)
    ).astype(np.float64)
    space_id = lm.char_to_id.get(" ")
    boundary_id = name_to_id.get(BOUNDARY)
    penalty_weight = 3.0

    def unigram_kl(counts: np.ndarray) -> float:
        probabilities = counts / counts.sum()
        mask = probabilities > 0
        return float(np.sum(
            probabilities[mask]
            * np.log(probabilities[mask] / lm.all_probs[mask])
        ))

    best_key = None
    best_score = float("-inf")
    for _ in range(restarts):
        key = rng.choice(
            lm.target_ids, size=len(names), p=lm.target_weights
        ).astype(np.int32)
        if boundary_id is not None:
            key[boundary_id] = space_id
        contrib = contributions(triples, weights, key, lm)
        decoded_frequency = np.bincount(
            key, weights=symbol_frequency, minlength=len(lm.id_to_char)
        ).astype(np.float64)
        current_kl = unigram_kl(decoded_frequency)
        current = contrib.sum() - penalty_weight * total * current_kl
        local_best = current
        local_key = key.copy()
        for step in range(steps):
            symbol = int(rng.integers(len(names)))
            if symbol == boundary_id or symbol not in affected:
                continue
            swap_symbol = None
            old_target = key[symbol]
            if rng.random() < 0.45:
                swap_symbol = int(rng.integers(len(names)))
                if (
                    swap_symbol == symbol
                    or swap_symbol == boundary_id
                    or swap_symbol not in affected
                ):
                    continue
                new_target = key[swap_symbol]
            else:
                new_target = int(
                    rng.choice(lm.target_ids, p=lm.target_weights)
                )
            if new_target == old_target:
                continue
            if swap_symbol is None:
                indices = affected[symbol]
            else:
                indices = np.union1d(
                    affected[symbol], affected[swap_symbol]
                )
            before = contrib[indices].sum()
            proposed_frequency = decoded_frequency.copy()
            proposed_frequency[old_target] -= symbol_frequency[symbol]
            proposed_frequency[new_target] += symbol_frequency[symbol]
            if swap_symbol is None:
                key[symbol] = new_target
            else:
                proposed_frequency[new_target] -= symbol_frequency[swap_symbol]
                proposed_frequency[old_target] += symbol_frequency[swap_symbol]
                key[symbol], key[swap_symbol] = (
                    key[swap_symbol], key[symbol]
                )
            proposed_kl = unigram_kl(proposed_frequency)
            proposed = contributions(
                triples, weights, key, lm, indices
            )
            delta = (
                proposed.sum() - before
                - penalty_weight * total * (proposed_kl - current_kl)
            )
            temperature = 25.0 * (1.0 - step / steps) + 0.05
            if delta >= 0 or rng.random() < math.exp(delta / temperature):
                contrib[indices] = proposed
                current += delta
                decoded_frequency = proposed_frequency
                current_kl = proposed_kl
                if current > local_best:
                    local_best = current
                    local_key = key.copy()
            else:
                if swap_symbol is None:
                    key[symbol] = old_target
                else:
                    key[symbol], key[swap_symbol] = (
                        key[swap_symbol], key[symbol]
                    )
        if local_best > best_score:
            best_score = local_best
            best_key = local_key

    assert best_key is not None
    test_triples, test_weights, _ = ngram_table(test)
    test_contrib = contributions(
        test_triples, test_weights, best_key, lm
    ).sum()
    test_score = test_contrib / max(test_weights.sum(), 1)
    train_contrib = contributions(triples, weights, best_key, lm).sum()
    train_score = train_contrib / max(total, 1)
    train_decoded_frequency = np.bincount(
        best_key, weights=symbol_frequency, minlength=len(lm.id_to_char)
    ).astype(np.float64)
    train_kl = unigram_kl(train_decoded_frequency)

    def render(encoded: np.ndarray) -> str:
        return "".join(lm.id_to_char[best_key[symbol]] for symbol in encoded)

    return {
        "stream": stream.name,
        "symbols": len(names),
        "length": len(stream.symbols),
        "train": train_score,
        "test": test_score,
        "train_kl": train_kl,
        "sample_train": render(train)[:240],
        "sample_test": render(test)[:240],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    lines = load_lines()
    streams = make_streams(lines)
    rng = np.random.default_rng(SEED)
    all_rows = []
    for language, path in CONTROLS.items():
        for spaces in (False, True):
            control = clean_control(path, language, spaces)
            cut = len(control) * 2 // 3
            lm = LM(control[:cut])
            ceiling = lm.plain_score(control[cut:])
            positive_plain = control[cut:cut + 40000]
            positive_symbols = tuple(
                BOUNDARY if char == " " else f"H:{char}:{rng.integers(4)}"
                for char in positive_plain
            )
            positive = fit(
                Stream(
                    "POSITIVE/homophonic",
                    positive_symbols,
                    spaces,
                ),
                lm,
                rng,
                args.steps,
                args.restarts,
            )
            positive.update(language=language, ceiling=ceiling)
            all_rows.append(positive)
            selected = [
                stream for stream in streams if stream.fixed_space == spaces
            ]
            for stream in selected:
                row = fit(stream, lm, rng, args.steps, args.restarts)
                row.update(language=language, ceiling=ceiling)
                all_rows.append(row)

    all_rows.sort(
        key=lambda row: row["test"] - row["ceiling"], reverse=True
    )
    for row in all_rows:
        print("\n" + "=" * 100)
        print(
            f"{row['language']:<7} {row['stream']:<32} "
            f"n={row['length']} symbols={row['symbols']} "
            f"train={row['train']:.3f} test={row['test']:.3f} "
            f"KL={row['train_kl']:.3f} ceiling={row['ceiling']:.3f} "
            f"gap={row['test']-row['ceiling']:+.3f}"
        )
        print("train:", row["sample_train"])
        print("test :", row["sample_test"])

    if args.output:
        args.output.write_text(json.dumps(all_rows, indent=2) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
