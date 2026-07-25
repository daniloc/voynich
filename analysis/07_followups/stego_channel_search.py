#!/usr/bin/env python3
"""
Search simple, historically plausible steganographic channels in Voynichese.

The visible text may be structured covertext while a payload is carried by a
fixed subsequence (for example, every other glyph of every other word).  This
script enumerates a bounded family of such channels, learns a monoalphabetic
key on the first half of each extracted stream, and scores the frozen key on
the second half under English and Latin character language models.

This is a triage tool, not a decipherment.  It also scores a frequency-matched
shuffle of every extracted stream.  A candidate must beat that null, approach
the positive-control band, and remain readable beyond the fitted span.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
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


@dataclass(frozen=True)
class Channel:
    name: str
    text: str


def clean_control(path: Path, language: str) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    start = re.search(r"\*\*\* start of.*?\*\*\*", text, re.S)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* end of", text, re.S)
    if end:
        text = text[:end.start()]
    text = re.sub(r"[^a-z]+", "", text)
    if language == "latin":
        text = text.replace("j", "i").replace("v", "u").replace("w", "u")
    return text


def load_lines() -> list[list[str]]:
    data = json.loads(CORPUS.read_text())
    lines = []
    for folio_lines in data["folios"].values():
        for line in folio_lines:
            words = [
                word for word in line["words"]
                if "?" not in word and word.isalpha()
            ]
            if words:
                lines.append(words)
    return lines


def choose_chars(word: str, mode: str) -> str:
    if mode == "all":
        return word
    if mode == "odd":
        return word[::2]
    if mode == "even":
        return word[1::2]
    if mode == "first":
        return word[:1]
    if mode == "last":
        return word[-1:]
    if mode == "middle":
        return word[(len(word) - 1) // 2:(len(word) - 1) // 2 + 1]
    raise ValueError(mode)


def build_channels(lines: list[list[str]]) -> list[Channel]:
    channels = []
    words = [word for line in lines for word in line]

    # Trithemius-like channels: every kth global word, then a fixed glyph
    # subsequence within each selected word.
    for stride in range(2, 7):
        for offset in range(stride):
            selected = words[offset::stride]
            for mode in ("all", "odd", "even", "first", "last", "middle"):
                text = "".join(choose_chars(word, mode) for word in selected)
                channels.append(Channel(
                    f"global_word/{stride}:{offset}/{mode}", text
                ))

    # Reset the word count on every physical line.  Historical covertext
    # systems often treat the line as the functional unit.
    for stride in range(2, 7):
        for offset in range(stride):
            selected = [
                word
                for line in lines
                for word in line[offset::stride]
            ]
            for mode in ("all", "odd", "even", "first", "last", "middle"):
                text = "".join(choose_chars(word, mode) for word in selected)
                channels.append(Channel(
                    f"line_word/{stride}:{offset}/{mode}", text
                ))

    # Fixed glyph stride over the uninterrupted cover stream.
    glyphs = "".join(words)
    for stride in range(2, 13):
        for offset in range(stride):
            channels.append(Channel(
                f"global_glyph/{stride}:{offset}", glyphs[offset::stride]
            ))

    # One glyph from a fixed edge word on each line.
    for edge_name, word_pick in (
        ("firstword", lambda line: line[0]),
        ("lastword", lambda line: line[-1]),
    ):
        for mode in ("all", "odd", "even", "first", "last", "middle"):
            channels.append(Channel(
                f"line/{edge_name}/{mode}",
                "".join(choose_chars(word_pick(line), mode) for line in lines),
            ))

    # The simplest slot-choice streams.
    for mode in ("first", "last", "middle"):
        channels.append(Channel(
            f"all_words/{mode}",
            "".join(choose_chars(word, mode) for word in words),
        ))
    return channels


class TrigramLM:
    def __init__(self, text: str, alpha: float = 0.1):
        self.alphabet = [
            char for char, _ in Counter(text).most_common()
        ]
        self.tri = Counter(zip(text, text[1:], text[2:]))
        self.bi = Counter(zip(text, text[1:]))
        self.width = len(self.alphabet)
        self.alpha = alpha

    def logp(self, a: str, b: str, c: str) -> float:
        return math.log(
            (self.tri[(a, b, c)] + self.alpha)
            / (self.bi[(a, b)] + self.alpha * self.width)
        )

    def score_plain(self, text: str) -> float:
        counts, total = trigram_counts(text)
        return sum(
            count * self.logp(a, b, c)
            for (a, b, c), count in counts.items()
        ) / max(total, 1)


def trigram_counts(text: str) -> tuple[Counter, int]:
    counts = Counter(zip(text, text[1:], text[2:]))
    return counts, sum(counts.values())


def mapped_score(
    counts: Counter,
    total: int,
    source: list[str],
    target_assignment: list[str],
    lm: TrigramLM,
) -> float:
    key = dict(zip(source, target_assignment))
    return sum(
        count * lm.logp(key[a], key[b], key[c])
        for (a, b, c), count in counts.items()
    ) / max(total, 1)


def fit_substitution(
    text: str,
    lm: TrigramLM,
    rng: np.random.Generator,
    steps: int,
    restarts: int,
) -> tuple[dict[str, str], float, float]:
    split = len(text) // 2
    train, test = text[:split], text[split:]
    train_counts, train_total = trigram_counts(train)
    test_counts, test_total = trigram_counts(test)
    train_frequency = Counter(train)
    source = sorted(
        set(text), key=lambda char: (-train_frequency[char], char)
    )
    if len(source) > len(lm.alphabet) or len(source) < 4:
        return {}, float("-inf"), float("-inf")

    frequency_seed = lm.alphabet[:len(source)]
    best_assignment = None
    best_train = float("-inf")
    for restart in range(restarts):
        assignment = frequency_seed.copy()
        if restart:
            rng.shuffle(assignment)
        current = mapped_score(
            train_counts, train_total, source, assignment, lm
        )
        temperature = 0.20
        for step in range(steps):
            i, j = rng.integers(len(source), size=2)
            if i == j:
                continue
            assignment[i], assignment[j] = assignment[j], assignment[i]
            proposed = mapped_score(
                train_counts, train_total, source, assignment, lm
            )
            delta = proposed - current
            cool = temperature * (1.0 - step / steps) + 0.002
            if delta >= 0 or rng.random() < math.exp(delta / cool):
                current = proposed
            else:
                assignment[i], assignment[j] = assignment[j], assignment[i]
        if current > best_train:
            best_train = current
            best_assignment = assignment.copy()

    assert best_assignment is not None
    test_score = mapped_score(
        test_counts, test_total, source, best_assignment, lm
    )
    return dict(zip(source, best_assignment)), best_train, test_score


def decode(text: str, key: dict[str, str]) -> str:
    return "".join(key.get(char, "?") for char in text)


def analyze(
    channels: list[Channel],
    language: str,
    steps: int,
    restarts: int,
    min_length: int,
) -> list[dict]:
    control = clean_control(CONTROLS[language], language)
    lm_cut = len(control) * 2 // 3
    lm = TrigramLM(control[:lm_cut])
    ceiling = lm.score_plain(control[lm_cut:])
    rng = np.random.default_rng(SEED + (0 if language == "english" else 1))
    results = []

    for index, channel in enumerate(channels):
        if len(channel.text) < min_length:
            continue
        key, train_score, test_score = fit_substitution(
            channel.text, lm, rng, steps, restarts
        )
        shuffled = list(channel.text)
        null_rng = np.random.default_rng(SEED + index * 17)
        null_rng.shuffle(shuffled)
        _, _, null_test = fit_substitution(
            "".join(shuffled), lm, null_rng, max(steps // 3, 100),
            max(restarts // 2, 1),
        )
        decoded = decode(channel.text, key)
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
            "sample_test": decoded[len(decoded) // 2:len(decoded) // 2 + 160],
        })
    return sorted(
        results, key=lambda row: (row["null_gain"], row["test"]), reverse=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--min-length", type=int, default=800)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    lines = load_lines()
    channels = build_channels(lines)
    print(
        f"Loaded {len(lines):,} lines; generated {len(channels):,} "
        "pre-registered extraction channels."
    )

    all_results = []
    for language in CONTROLS:
        results = analyze(
            channels, language, args.steps, args.restarts, args.min_length
        )
        all_results.extend(results)
        print("\n" + "=" * 100)
        print(f"{language.upper()} TARGET: top held-out gains over shuffled channel")
        print("=" * 100)
        for row in results[:args.top]:
            print(
                f"{row['channel']:<34} n={row['length']:>6} "
                f"a={row['alphabet']:>2} train={row['train']:>7.3f} "
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
