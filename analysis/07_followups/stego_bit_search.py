#!/usr/bin/env python3
"""
Test simple binary payloads carried by Voynich word-form choices.

Each registered feature converts a word (or a restricted subset of words) to
one bit.  The resulting streams are decoded under Bacon-5 and 7/8-bit ASCII
with all phase, polarity, and bit-order variants.  Rankings use character
trigram likelihood and are compared with shuffled-bit controls.
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
class BitStream:
    name: str
    bits: tuple[int, ...]


def clean_control(path: Path, latin: bool = False) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    start = re.search(r"\*\*\* start of.*?\*\*\*", text, re.S)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* end of", text, re.S)
    if end:
        text = text[:end.start()]
    text = re.sub(r"[^a-z]+", "", text)
    if latin:
        text = text.replace("j", "i").replace("v", "u").replace("w", "u")
    return text


class LM:
    def __init__(self, text: str):
        self.tri = Counter(zip(text, text[1:], text[2:]))
        self.bi = Counter(zip(text, text[1:]))
        self.width = len(set(text))

    def score(self, text: str) -> float:
        text = re.sub("[^a-z]", "", text.lower())
        if len(text) < 3:
            return float("-inf")
        total = 0.0
        for a, b, c in zip(text, text[1:], text[2:]):
            total += math.log(
                (self.tri[(a, b, c)] + 0.1)
                / (self.bi[(a, b)] + 0.1 * self.width)
            )
        return total / (len(text) - 2)


def load_words() -> tuple[list[str], list[list[str]]]:
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
    return [word for line in lines for word in line], lines


def xor_choice(
    words: list[str], name: str, left, right
) -> BitStream:
    bits = []
    for word in words:
        a, b = left(word), right(word)
        if a != b:
            bits.append(int(b))
    return BitStream(name, tuple(bits))


def make_streams(words: list[str]) -> list[BitStream]:
    frequency = Counter(words)
    rank = {
        word: index for index, (word, _) in enumerate(
            sorted(frequency.items(), key=lambda item: (-item[1], item[0]))
        )
    }
    streams = [
        BitStream("length_parity", tuple(len(word) % 2 for word in words)),
        BitStream("o_count_parity", tuple(word.count("o") % 2 for word in words)),
        BitStream("e_count_parity", tuple(word.count("e") % 2 for word in words)),
        BitStream("i_count_parity", tuple(word.count("i") % 2 for word in words)),
        BitStream("type_frequency_rank_parity",
                  tuple(rank[word] % 2 for word in words)),
        BitStream("starts_q", tuple(word.startswith("q") for word in words)),
        BitStream("starts_o", tuple(word.startswith("o") for word in words)),
        BitStream("ends_y", tuple(word.endswith("y") for word in words)),
        BitStream("has_gallows", tuple(any(c in word for c in "ktpf")
                                      for word in words)),
        xor_choice(words, "ch_vs_sh",
                   lambda w: "ch" in w, lambda w: "sh" in w),
        xor_choice(words, "k_vs_t",
                   lambda w: "k" in w, lambda w: "t" in w),
        xor_choice(words, "p_vs_f",
                   lambda w: "p" in w, lambda w: "f" in w),
        xor_choice(words, "final_y_vs_l",
                   lambda w: w.endswith("y"), lambda w: w.endswith("l")),
        xor_choice(words, "final_r_vs_m",
                   lambda w: w.endswith("r"), lambda w: w.endswith("m")),
        xor_choice(words, "initial_ch_vs_sh",
                   lambda w: w.startswith("ch"), lambda w: w.startswith("sh")),
        xor_choice(words, "initial_k_vs_t",
                   lambda w: w.startswith("k"), lambda w: w.startswith("t")),
    ]
    return [
        BitStream(stream.name, tuple(map(int, stream.bits)))
        for stream in streams
        if len(stream.bits) >= 200
    ]


def chunks(bits: tuple[int, ...], width: int, offset: int, reverse: bool,
           invert: bool) -> list[int]:
    selected = bits[offset:]
    values = []
    for start in range(0, len(selected) - width + 1, width):
        group = list(selected[start:start + width])
        if invert:
            group = [1 - bit for bit in group]
        if reverse:
            group.reverse()
        value = 0
        for bit in group:
            value = value * 2 + bit
        values.append(value)
    return values


def render(values: list[int], codec: str) -> str:
    if codec == "bacon5":
        return "".join(
            chr(97 + value) if value < 26 else "?"
            for value in values
        )
    if codec in {"ascii7", "ascii8"}:
        return "".join(
            chr(value) if 32 <= value <= 126 else "?"
            for value in values
        )
    raise ValueError(codec)


def printable_fraction(text: str) -> float:
    return sum(char != "?" for char in text) / max(len(text), 1)


def variants(stream: BitStream, lm: LM) -> list[dict]:
    out = []
    for codec, width in (("bacon5", 5), ("ascii7", 7), ("ascii8", 8)):
        for offset in range(width):
            for reverse in (False, True):
                for invert in (False, True):
                    text = render(
                        chunks(stream.bits, width, offset, reverse, invert),
                        codec,
                    )
                    out.append({
                        "stream": stream.name,
                        "bits": len(stream.bits),
                        "ones": sum(stream.bits) / len(stream.bits),
                        "codec": codec,
                        "offset": offset,
                        "reverse": reverse,
                        "invert": invert,
                        "score": lm.score(text),
                        "printable": printable_fraction(text),
                        "sample": text[:500],
                    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    words, _ = load_words()
    streams = make_streams(words)
    rng = np.random.default_rng(SEED)
    all_results = []
    for language, path in CONTROLS.items():
        control = clean_control(path, latin=language == "latin")
        lm = LM(control[:len(control) * 2 // 3])
        real = []
        null_best = {}
        for stream in streams:
            rows = variants(stream, lm)
            real.extend(rows)
            shuffled = list(stream.bits)
            rng.shuffle(shuffled)
            null_rows = variants(
                BitStream(stream.name, tuple(shuffled)), lm
            )
            null_best[stream.name] = max(row["score"] for row in null_rows)
        for row in real:
            row["language"] = language
            row["null_best"] = null_best[row["stream"]]
            row["gain"] = row["score"] - row["null_best"]
        real.sort(key=lambda row: (row["gain"], row["score"]), reverse=True)
        all_results.extend(real)

        print("\n" + "=" * 100)
        print(f"{language.upper()}: top binary channel variants")
        print("=" * 100)
        for row in real[:args.top]:
            print(
                f"{row['stream']:<28} {row['codec']:<7} "
                f"off={row['offset']} rev={int(row['reverse'])} "
                f"inv={int(row['invert'])} bits={row['bits']:>6} "
                f"p1={row['ones']:.3f} score={row['score']:.3f} "
                f"nullbest={row['null_best']:.3f} gain={row['gain']:+.3f} "
                f"print={row['printable']:.2f}"
            )
            print(" ", row["sample"][:180].replace("\n", "\\n"))

    if args.output:
        args.output.write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
