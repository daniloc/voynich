#!/usr/bin/env python3
"""
Selection-adjusted search for cross-quire parallel Voynich prose lines.

The earlier exploratory search ranked lines sharing stripped cores but did not
calibrate the best result after searching the full manuscript.  This gate:

* searches only cross-quire lines from the same section and Currier dialect;
* proposes pairs through uncommon shared cores;
* scores order-preserving matching blocks with inverse-document-frequency
  weights;
* repeats the complete maximum-over-pairs search after strict
  quire/section/Currier/position-preserving word shuffles.

The output is an alignment worklist and a gate on whether the best apparent
parallel exceeds corpus-wide search luck.  It does not claim plaintext.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "corpus.json"
SEED = 20260723

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
class ProseLine:
    folio: str
    line: int
    quire: str
    section: str
    currier: str
    hand: str
    words: tuple[str, ...]


@dataclass
class PairResult:
    left: str
    right: str
    left_quire: str
    right_quire: str
    left_words: int
    right_words: int
    matched_words: int
    matching_blocks: int
    longest_block: int
    weighted_score: float
    normalized_score: float
    aligned_left: str
    aligned_right: str


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
    core = residual[: -len(suffix)] if suffix else residual
    return prefix, core, suffix


def load_lines() -> list[ProseLine]:
    source = json.loads(CORPUS.read_text(encoding="utf-8"))
    lines = []
    for folio, records in source["folios"].items():
        meta = source["meta"].get(folio, {})
        for record in records:
            if locus_type(record["locus"]) != "P":
                continue
            words = tuple(
                word
                for word in record["words"]
                if "?" not in word and word.isalpha() and len(word) >= 2
            )
            if len(words) < 5:
                continue
            lines.append(
                ProseLine(
                    folio=folio,
                    line=int(record["line"]),
                    quire=str(meta.get("Q", "?")),
                    section=str(meta.get("I", "?")),
                    currier=str(meta.get("L", "?")),
                    hand=str(meta.get("H", "?")),
                    words=words,
                )
            )
    return lines


def line_id(line: ProseLine) -> str:
    return f"{line.folio}.{line.line}"


def strict_shuffle(
    lines: Sequence[ProseLine], rng: random.Random
) -> list[ProseLine]:
    slots: dict[tuple[str, str, str, str], list[tuple[int, int]]] = defaultdict(
        list
    )
    values: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    mutable = [list(line.words) for line in lines]
    for line_index, line in enumerate(lines):
        for index, word in enumerate(line.words):
            key = (
                line.quire,
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
        ProseLine(
            line.folio,
            line.line,
            line.quire,
            line.section,
            line.currier,
            line.hand,
            tuple(words),
        )
        for line, words in zip(lines, mutable)
    ]


def candidate_pairs(
    lines: Sequence[ProseLine],
    core_sequences: Sequence[Sequence[str]],
    document_frequency: Counter[str],
    maximum_df: int,
) -> set[tuple[int, int]]:
    postings: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, (line, cores) in enumerate(zip(lines, core_sequences)):
        for core in set(cores):
            if 2 <= document_frequency[core] <= maximum_df:
                postings[(line.section, line.currier, core)].append(index)

    pairs: set[tuple[int, int]] = set()
    for indices in postings.values():
        for left_offset, left in enumerate(indices):
            for right in indices[left_offset + 1 :]:
                if lines[left].quire != lines[right].quire:
                    pairs.add((left, right))
    return pairs


def score_pair(
    left: ProseLine,
    right: ProseLine,
    left_cores: Sequence[str],
    right_cores: Sequence[str],
    idf: dict[str, float],
) -> PairResult:
    matcher = difflib.SequenceMatcher(
        None, left_cores, right_cores, autojunk=False
    )
    blocks = [
        block for block in matcher.get_matching_blocks() if block.size > 0
    ]
    matched = sum(block.size for block in blocks)
    weighted = sum(
        idf.get(left_cores[block.a + offset], 0.0)
        for block in blocks
        for offset in range(block.size)
    )
    normalizer = math.sqrt(
        sum(idf.get(core, 0.0) ** 2 for core in left_cores)
        * sum(idf.get(core, 0.0) ** 2 for core in right_cores)
    )
    normalized = weighted / normalizer if normalizer else 0.0

    aligned_left = []
    aligned_right = []
    for block in blocks:
        aligned_left.append(" ".join(left.words[block.a : block.a + block.size]))
        aligned_right.append(
            " ".join(right.words[block.b : block.b + block.size])
        )
    return PairResult(
        left=line_id(left),
        right=line_id(right),
        left_quire=left.quire,
        right_quire=right.quire,
        left_words=len(left.words),
        right_words=len(right.words),
        matched_words=matched,
        matching_blocks=len(blocks),
        longest_block=max((block.size for block in blocks), default=0),
        weighted_score=weighted,
        normalized_score=normalized,
        aligned_left=" | ".join(aligned_left),
        aligned_right=" | ".join(aligned_right),
    )


def full_search(
    lines: Sequence[ProseLine],
    maximum_df: int,
    keep: int,
) -> tuple[list[PairResult], int]:
    cores = [tuple(decompose(word)[1] for word in line.words) for line in lines]
    document_frequency: Counter[str] = Counter()
    for sequence in cores:
        document_frequency.update(set(sequence))
    count = len(lines)
    idf = {
        core: math.log((count + 1) / (frequency + 1))
        for core, frequency in document_frequency.items()
    }
    pairs = candidate_pairs(lines, cores, document_frequency, maximum_df)
    results = [
        score_pair(lines[left], lines[right], cores[left], cores[right], idf)
        for left, right in pairs
    ]
    results.sort(
        key=lambda item: (
            item.normalized_score,
            item.longest_block,
            item.matched_words,
            item.weighted_score,
        ),
        reverse=True,
    )
    return results[:keep], len(pairs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nulls", type=int, default=32)
    parser.add_argument("--maximum-df", type=int, default=60)
    parser.add_argument("--keep", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data"
        / "intermediate"
        / "followups_parallel_passage_gate.json",
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nulls < 1 or args.keep < 1 or args.maximum_df < 2:
        raise ValueError("invalid search limits")
    lines = load_lines()
    observed, candidate_count = full_search(
        lines, args.maximum_df, args.keep
    )
    observed_max = observed[0].normalized_score if observed else 0.0

    rng = random.Random(SEED)
    null_maxima = []
    null_candidate_counts = []
    for replicate in range(args.nulls):
        shuffled = strict_shuffle(lines, rng)
        null_results, null_count = full_search(
            shuffled, args.maximum_df, 1
        )
        null_maxima.append(
            null_results[0].normalized_score if null_results else 0.0
        )
        null_candidate_counts.append(null_count)
        if args.progress:
            print(
                f"null {replicate + 1:02d}/{args.nulls}: "
                f"max={null_maxima[-1]:.6f} pairs={null_count}",
                flush=True,
            )

    exceedances = sum(value >= observed_max for value in null_maxima)
    p_value = (exceedances + 1) / (len(null_maxima) + 1)
    summary = {
        "lines": len(lines),
        "candidate_pairs": candidate_count,
        "observed_max": observed_max,
        "null_maximum": max(null_maxima, default=0.0),
        "null_mean_maximum": mean(null_maxima) if null_maxima else 0.0,
        "familywise_p": p_value,
        "observed_exceeds_strict_null_maximum": (
            observed_max > max(null_maxima, default=float("inf"))
        ),
        "credible_parallel_gate": (
            bool(observed)
            and observed[0].longest_block >= 4
            and observed_max > max(null_maxima, default=float("inf"))
        ),
    }
    report = {
        "method": (
            "cross-quire rare-core order-preserving alignment with complete "
            "maximum-over-pairs strict shuffles"
        ),
        "seed": SEED,
        "nulls": args.nulls,
        "maximum_document_frequency": args.maximum_df,
        "summary": summary,
        "top_pairs": [asdict(item) for item in observed],
        "null_maxima": null_maxima,
        "null_candidate_counts": null_candidate_counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print("=" * 88)
    print("PARALLEL-PASSAGE GATE")
    print("=" * 88)
    print(json.dumps(summary, indent=2))
    for rank, item in enumerate(observed[:10], 1):
        print(
            f"{rank:2d}. {item.left} Q{item.left_quire} / "
            f"{item.right} Q{item.right_quire}: "
            f"score={item.normalized_score:.6f} "
            f"matched={item.matched_words} longest={item.longest_block}"
        )
        print(f"    {item.aligned_left}")
        print(f"    {item.aligned_right}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
