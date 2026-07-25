#!/usr/bin/env python3
"""
Search deterministic hierarchical reading orders for substituted plaintext.

The normal left-to-right transcription may not preserve the intended payload
order.  This bounded battery changes only page, line, word, and glyph order:
reversal, odd/even unweaving, boustrophedon rows, and ragged vertical reads.
Each resulting character stream is attacked as a monoalphabetic substitution
with a key fitted on its first half and frozen on its second half.

These tests address simple transposition plus substitution.  They do not test
arbitrary anagrams, keyed column widths, route ciphers, or variable per-page
keys.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stego_channel_search import (
    Channel,
    CONTROLS,
    SEED,
    TrigramLM,
    clean_control,
)


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "corpus.json"


def load_folios() -> list[tuple[str, list[list[str]]]]:
    data = json.loads(CORPUS.read_text())
    folios = []
    for folio, raw_lines in data["folios"].items():
        lines = []
        for raw_line in raw_lines:
            words = [
                word for word in raw_line["words"]
                if "?" not in word and word.isalpha()
            ]
            if words:
                lines.append(words)
        if lines:
            folios.append((folio, lines))
    return folios


def unweave(items: list, parity_first: int) -> list:
    return items[parity_first::2] + items[1 - parity_first::2]


def order_items(items: list, mode: str) -> list:
    if mode == "forward":
        return items
    if mode == "reverse":
        return list(reversed(items))
    if mode == "even_odd":
        return unweave(items, 0)
    if mode == "odd_even":
        return unweave(items, 1)
    raise ValueError(mode)


def render_row(words: list[str], mode: str, row_index: int) -> str:
    reverse_row = mode.startswith("boustro") and row_index % 2 == 1
    base_mode = mode.removeprefix("boustro_")
    selected = list(reversed(words)) if reverse_row else words
    if base_mode == "words":
        return "".join(selected)
    if base_mode == "words_chars":
        return "".join(word[::-1] for word in selected)
    if base_mode == "chars":
        text = "".join(words)
        return text[::-1] if reverse_row else text
    if base_mode == "forward":
        return "".join(words)
    if base_mode == "reverse_words":
        return "".join(reversed(words))
    if base_mode == "reverse_word_chars":
        return "".join(word[::-1] for word in words)
    if base_mode == "reverse_all":
        return "".join(words)[::-1]
    raise ValueError(mode)


def hierarchical_channel(
    folios: list[tuple[str, list[list[str]]]],
    folio_mode: str,
    line_mode: str,
    row_mode: str,
) -> str:
    selected_folios = order_items(folios, folio_mode)
    output = []
    for _, lines in selected_folios:
        selected_lines = order_items(lines, line_mode)
        output.extend(
            render_row(words, row_mode, index)
            for index, words in enumerate(selected_lines)
        )
    return "".join(output)


def transpose_rows(
    rows: list[str], column_mode: str, row_mode: str
) -> str:
    if row_mode == "bottom_top":
        rows = list(reversed(rows))
    width = max(map(len, rows), default=0)
    columns = range(width)
    if column_mode in {"right_left", "boustro_right"}:
        columns = reversed(range(width))
    output = []
    for column_index, column in enumerate(columns):
        available = [row[column] for row in rows if column < len(row)]
        if column_mode.startswith("boustro") and column_index % 2:
            available.reverse()
        output.extend(available)
    return "".join(output)


def vertical_glyph_channel(
    folios: list[tuple[str, list[list[str]]]],
    folio_mode: str,
    column_mode: str,
    row_mode: str,
) -> str:
    output = []
    for _, lines in order_items(folios, folio_mode):
        rows = ["".join(words) for words in lines]
        output.append(transpose_rows(rows, column_mode, row_mode))
    return "".join(output)


def vertical_word_channel(
    folios: list[tuple[str, list[list[str]]]],
    folio_mode: str,
    column_mode: str,
    row_mode: str,
    reverse_chars: bool,
) -> str:
    output = []
    for _, lines in order_items(folios, folio_mode):
        rows = list(reversed(lines)) if row_mode == "bottom_top" else lines
        width = max(map(len, rows), default=0)
        columns = list(range(width))
        if column_mode in {"right_left", "boustro_right"}:
            columns.reverse()
        for column_index, column in enumerate(columns):
            available = [row[column] for row in rows if column < len(row)]
            if column_mode.startswith("boustro") and column_index % 2:
                available.reverse()
            if reverse_chars:
                available = [word[::-1] for word in available]
            output.extend(available)
    return "".join(output)


def build_channels(
    folios: list[tuple[str, list[list[str]]]]
) -> list[Channel]:
    channels = []
    folio_modes = ("forward", "reverse", "even_odd", "odd_even")
    line_modes = ("forward", "reverse", "even_odd", "odd_even")
    row_modes = (
        "forward",
        "reverse_words",
        "reverse_word_chars",
        "reverse_all",
        "boustro_words",
        "boustro_words_chars",
        "boustro_chars",
    )
    for folio_mode in folio_modes:
        for line_mode in line_modes:
            for row_mode in row_modes:
                channels.append(Channel(
                    f"hier/{folio_mode}/{line_mode}/{row_mode}",
                    hierarchical_channel(
                        folios, folio_mode, line_mode, row_mode
                    ),
                ))

    for folio_mode in folio_modes:
        for column_mode in (
            "left_right", "right_left", "boustro_left", "boustro_right"
        ):
            for row_mode in ("top_bottom", "bottom_top"):
                channels.append(Channel(
                    f"glyph_columns/{folio_mode}/{column_mode}/{row_mode}",
                    vertical_glyph_channel(
                        folios, folio_mode, column_mode, row_mode
                    ),
                ))
                for reverse_chars in (False, True):
                    channels.append(Channel(
                        "word_columns/"
                        f"{folio_mode}/{column_mode}/{row_mode}/"
                        + ("reverse_chars" if reverse_chars else "chars"),
                        vertical_word_channel(
                            folios, folio_mode, column_mode, row_mode,
                            reverse_chars,
                        ),
                    ))
    return channels


def encoded_trigrams(
    text: str, source_lookup: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    encoded = np.array([source_lookup[char] for char in text], dtype=np.int16)
    counter = Counter(zip(encoded[:-2], encoded[1:-1], encoded[2:]))
    triples = np.array(list(counter), dtype=np.int16)
    weights = np.array(list(counter.values()), dtype=float)
    affected = defaultdict(list)
    for index, triple in enumerate(triples):
        for symbol in set(map(int, triple)):
            affected[symbol].append(index)
    return triples, weights, {
        symbol: np.array(indices, dtype=np.int32)
        for symbol, indices in affected.items()
    }


def fit_substitution_fast(
    text: str,
    lm: TrigramLM,
    logp: np.ndarray,
    rng: np.random.Generator,
    steps: int,
    restarts: int,
) -> tuple[dict[str, str], float, float]:
    split = len(text) // 2
    train_text, test_text = text[:split], text[split:]
    train_frequency = Counter(train_text)
    source = sorted(
        set(text), key=lambda char: (-train_frequency[char], char)
    )
    if len(source) > len(lm.alphabet) or len(source) < 4:
        return {}, float("-inf"), float("-inf")
    source_lookup = {char: index for index, char in enumerate(source)}
    train, train_weights, affected = encoded_trigrams(
        train_text, source_lookup
    )
    test, test_weights, _ = encoded_trigrams(test_text, source_lookup)

    empty = np.array([], dtype=np.int32)
    affected_pairs = {
        (first, second): np.union1d(
            affected.get(first, empty), affected.get(second, empty)
        )
        for first in range(len(source))
        for second in range(first + 1, len(source))
    }

    best_key = None
    best_train = float("-inf")
    for restart in range(restarts):
        key = np.arange(len(source), dtype=np.int16)
        if restart:
            rng.shuffle(key)
        contributions = train_weights * logp[
            key[train[:, 0]], key[train[:, 1]], key[train[:, 2]]
        ]
        current = float(contributions.sum())
        local_best = current
        local_key = key.copy()
        for step in range(steps):
            first = int(rng.integers(len(source)))
            second = int(rng.integers(len(source) - 1))
            if second >= first:
                second += 1
            pair = (
                (first, second) if first < second else (second, first)
            )
            indices = affected_pairs[pair]
            before = float(contributions[indices].sum())
            key[first], key[second] = key[second], key[first]
            rows = train[indices]
            proposed = train_weights[indices] * logp[
                key[rows[:, 0]], key[rows[:, 1]], key[rows[:, 2]]
            ]
            delta = float(proposed.sum() - before)
            temperature = 45.0 * (1.0 - step / steps) + 0.05
            if delta >= 0 or rng.random() < math.exp(delta / temperature):
                contributions[indices] = proposed
                current += delta
                if current > local_best:
                    local_best = current
                    local_key = key.copy()
            else:
                key[first], key[second] = key[second], key[first]
        local_score = local_best / max(train_weights.sum(), 1)
        if local_score > best_train:
            best_train = local_score
            best_key = local_key

    assert best_key is not None
    test_score = float(np.sum(
        test_weights * logp[
            best_key[test[:, 0]],
            best_key[test[:, 1]],
            best_key[test[:, 2]],
        ]
    ) / max(test_weights.sum(), 1))
    return (
        {
            source[index]: lm.alphabet[int(best_key[index])]
            for index in range(len(source))
        },
        best_train,
        test_score,
    )


def analyze_fast(
    channels: list[Channel],
    language: str,
    steps: int,
    restarts: int,
) -> list[dict]:
    control = clean_control(CONTROLS[language], language)
    lm_cut = len(control) * 2 // 3
    lm = TrigramLM(control[:lm_cut])
    ceiling = lm.score_plain(control[lm_cut:])
    rng = np.random.default_rng(SEED + (0 if language == "english" else 1))
    width = len(lm.alphabet)
    logp = np.empty((width, width, width), dtype=float)
    for a, first in enumerate(lm.alphabet):
        for b, second in enumerate(lm.alphabet):
            for c, third in enumerate(lm.alphabet):
                logp[a, b, c] = lm.logp(first, second, third)

    shuffled = list(channels[0].text)
    rng.shuffle(shuffled)
    _, _, null_test = fit_substitution_fast(
        "".join(shuffled), lm, logp, rng, steps, restarts
    )
    results = []
    for channel in channels:
        key, train_score, test_score = fit_substitution_fast(
            channel.text, lm, logp, rng, steps, restarts
        )
        decoded = "".join(key.get(char, "?") for char in channel.text)
        cut = len(decoded) // 2
        results.append({
            "language": language,
            "channel": channel.name,
            "length": len(channel.text),
            "alphabet": len(set(channel.text)),
            "train": train_score,
            "test": test_score,
            "null_test": null_test,
            "null_gain": test_score - null_test,
            "ceiling": ceiling,
            "sample_train": decoded[:160],
            "sample_test": decoded[cut:cut + 160],
        })
    return sorted(
        results, key=lambda row: (row["test"], row["train"]), reverse=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    folios = load_folios()
    channels = build_channels(folios)
    print(
        f"Loaded {len(folios):,} folios; generated {len(channels):,} "
        "deterministic reading orders."
    )
    all_results = []
    for language in CONTROLS:
        results = analyze_fast(
            channels, language, args.steps, args.restarts
        )
        all_results.extend(results)
        print("\n" + "=" * 100)
        print(f"{language.upper()} TARGET")
        print("=" * 100)
        for row in results[:args.top]:
            print(
                f"{row['channel']:<62} "
                f"test={row['test']:>7.3f} null={row['null_test']:>7.3f} "
                f"gain={row['null_gain']:+.3f} ceil={row['ceiling']:.3f}"
            )
            print(f"  train: {row['sample_train']}")
            print(f"  test : {row['sample_test']}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
