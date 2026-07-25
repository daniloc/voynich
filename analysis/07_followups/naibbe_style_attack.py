#!/usr/bin/env python3
"""
Attack Voynich as a Naibbe-style one/two-letter homophonic cipher.

Naibbe mixes atomic ciphertext words, which emit one plaintext letter, with
productive prefix/suffix compounds, which emit two.  A forced two-part split
therefore is not a valid positive control.  This version learns a structural
segmenter from the first half of a known Naibbe ciphertext and freezes it:

* the code table supplies atomic/compound labels and compound boundaries;
* plaintext letter values from the table are never loaded;
* features measure only substring reuse, type support, frequency, and length;
* calibration accuracy is evaluated on the second half of Naibbe;
* the frozen ranker and classifier are then applied to Voynich.

The resulting homophonic key is learned on the first half of each emitted
symbol stream and evaluated on the second with the four-gram solver in
stego_slot_cipher.py.  Codebook inventories are selected from training only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stego_slot_cipher import CONTROLS, LM, Stream, clean_control, fit


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "corpus.json"
SEED = 20260723


def load_voynich_lines() -> list[list[str]]:
    data = json.loads(CORPUS.read_text())
    return [
        [
            word for word in line["words"]
            if "?" not in word and word.isalpha() and len(word) >= 1
        ]
        for folio_lines in data["folios"].values()
        for line in folio_lines
        if line["words"]
    ]


def load_plain_lines(path: Path) -> list[list[str]]:
    return [
        re.findall(r"[a-z]+", line.lower())
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if re.search(r"[a-z]", line.lower())
    ]


@dataclass
class StructuralCounts:
    word: Counter
    left_tokens: Counter
    right_tokens: Counter
    left_types: Counter
    right_types: Counter


def structural_counts(words: list[str]) -> StructuralCounts:
    word_counts = Counter(words)
    left_tokens, right_tokens = Counter(), Counter()
    left_types, right_types = Counter(), Counter()
    for word, count in word_counts.items():
        for split in range(1, len(word)):
            left, right = word[:split], word[split:]
            left_tokens[left] += count
            right_tokens[right] += count
            left_types[left] += 1
            right_types[right] += 1
    return StructuralCounts(
        word_counts, left_tokens, right_tokens, left_types, right_types
    )


def split_features(
    word: str, split: int, counts: StructuralCounts
) -> np.ndarray:
    left, right = word[:split], word[split:]
    own = max(1, counts.word[word])
    return np.array(
        [
            math.log1p(max(0, counts.left_tokens[left] - own)),
            math.log1p(max(0, counts.right_tokens[right] - own)),
            math.log1p(max(0, counts.left_types[left] - 1)),
            math.log1p(max(0, counts.right_types[right] - 1)),
            math.log1p(counts.word[left]),
            math.log1p(counts.word[right]),
            len(left) / len(word),
            len(right) / len(word),
            abs(len(left) - len(right)) / len(word),
            math.log1p(len(word)),
            1.0,
        ],
        dtype=float,
    )


@dataclass
class SegmentModel:
    split_weights: np.ndarray
    split_mean: np.ndarray
    split_scale: np.ndarray
    class_weights: np.ndarray
    class_mean: np.ndarray
    class_scale: np.ndarray

    def rank_split(
        self, word: str, counts: StructuralCounts
    ) -> tuple[tuple[str, str] | None, float, float]:
        if len(word) < 2:
            return None, -20.0, 20.0
        features = np.stack(
            [split_features(word, i, counts) for i in range(1, len(word))]
        )
        scores = (
            (features - self.split_mean) / self.split_scale
        ) @ self.split_weights
        best = int(scores.argmax()) + 1
        ordered = np.sort(scores)
        margin = float(ordered[-1] - ordered[-2]) if len(scores) > 1 else 0.0
        return (word[:best], word[best:]), float(scores.max()), margin

    def class_features(
        self, word: str, counts: StructuralCounts
    ) -> np.ndarray:
        split, score, margin = self.rank_split(word, counts)
        if split is None:
            return np.array(
                [
                    math.log1p(counts.word[word]), math.log1p(len(word)),
                    -20.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
                dtype=float,
            )
        left, right = split
        own = max(1, counts.word[word])
        return np.array(
            [
                math.log1p(counts.word[word]),
                math.log1p(len(word)),
                score,
                margin,
                math.log1p(max(0, counts.left_tokens[left] - own)),
                math.log1p(max(0, counts.right_tokens[right] - own)),
                math.log1p(max(0, counts.left_types[left] - 1)),
                math.log1p(max(0, counts.right_types[right] - 1)),
                math.log1p(counts.word[left]),
                math.log1p(counts.word[right]),
                len(left) / len(word),
                1.0,
            ],
            dtype=float,
        )

    def segment(
        self, word: str, counts: StructuralCounts
    ) -> tuple[str, ...]:
        split, _, _ = self.rank_split(word, counts)
        if split is None:
            return ("U:" + word,)
        features = (
            self.class_features(word, counts) - self.class_mean
        ) / self.class_scale
        logit = float(np.clip(features @ self.class_weights, -30.0, 30.0))
        compound_probability = 1.0 / (1.0 + math.exp(-logit))
        if compound_probability < 0.5:
            return ("U:" + word,)
        left, right = split
        return ("L:" + left, "R:" + right)


def table_structure(path: Path) -> tuple[set[str], set[str], set[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    atomic = {
        row["glyphs"] for row in rows if row["code"].startswith("unigram_")
    }
    left = {
        row["glyphs"] for row in rows if row["code"].startswith("prefix_")
    }
    right = {
        row["glyphs"] for row in rows if row["code"].startswith("suffix_")
    }
    return atomic, left, right


def structural_truth(
    word: str, atomic: set[str], left: set[str], right: set[str]
) -> tuple[str, tuple[str, str] | None]:
    if word in atomic:
        return "U", None
    candidates = [
        (word[:split], word[split:])
        for split in range(1, len(word))
        if word[:split] in left and word[split:] in right
    ]
    if len(candidates) == 1:
        return "B", candidates[0]
    return "?", None


def normalize_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0) + 1e-6
    mean[-1] = 0.0
    scale[-1] = 1.0
    return mean, scale


def train_split_ranker(
    words: list[str],
    counts: StructuralCounts,
    atomic: set[str],
    left: set[str],
    right: set[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    examples = []
    for word, frequency in Counter(words).items():
        label, boundary = structural_truth(word, atomic, left, right)
        if label != "B" or boundary is None:
            continue
        features = np.stack(
            [split_features(word, i, counts) for i in range(1, len(word))]
        )
        examples.append(
            (features, len(boundary[0]) - 1, min(frequency, 10))
        )
    pooled = np.concatenate([features for features, _, _ in examples])
    mean, scale = normalize_features(pooled)
    normalized = [
        ((features - mean) / scale, target, weight)
        for features, target, weight in examples
    ]

    weights = np.zeros(pooled.shape[1], dtype=float)
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    batch_size = min(128, len(normalized))
    for _ in range(4000):
        gradient = np.zeros_like(weights)
        for index in rng.choice(
            len(normalized), batch_size, replace=False
        ):
            features, target, example_weight = normalized[index]
            logits = features @ weights
            logits -= logits.max()
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            gradient += example_weight * (
                features[target] - probabilities @ features
            )
        gradient = np.clip(
            gradient / batch_size - 0.002 * weights, -20.0, 20.0
        )
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        weights += 0.025 * first / (np.sqrt(second) + 1e-8)
    return weights, mean, scale


def train_classifier(
    base: SegmentModel,
    words: list[str],
    counts: StructuralCounts,
    atomic: set[str],
    left: set[str],
    right: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    examples = []
    for word, frequency in Counter(words).items():
        label, _ = structural_truth(word, atomic, left, right)
        if label not in {"U", "B"}:
            continue
        examples.append(
            (
                base.class_features(word, counts),
                1.0 if label == "B" else 0.0,
                min(frequency, 10),
            )
        )
    features = np.stack([row[0] for row in examples])
    labels = np.array([row[1] for row in examples])
    example_weights = np.array([row[2] for row in examples])
    mean, scale = normalize_features(features)
    normalized = (features - mean) / scale

    weights = np.zeros(features.shape[1], dtype=float)
    first = np.zeros_like(weights)
    second = np.zeros_like(weights)
    for _ in range(5000):
        logits = np.clip(
            np.sum(normalized * weights, axis=1), -30.0, 30.0
        )
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = (
            np.sum(
                normalized
                * (example_weights * (labels - probabilities))[:, None],
                axis=0,
            ) / example_weights.sum()
            - 0.002 * weights
        )
        gradient = np.clip(gradient, -20.0, 20.0)
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        weights += 0.02 * first / (np.sqrt(second) + 1e-8)
    return weights, mean, scale


def calibrate(
    lines: list[list[str]], table_path: Path
) -> tuple[SegmentModel, dict]:
    words = [word for line in lines for word in line]
    cut = len(words) // 2
    training = words[:cut]
    counts = structural_counts(training)
    atomic, left, right = table_structure(table_path)
    rng = np.random.default_rng(SEED)
    split_weights, split_mean, split_scale = train_split_ranker(
        training, counts, atomic, left, right, rng
    )
    partial = SegmentModel(
        split_weights, split_mean, split_scale,
        np.zeros(12), np.zeros(12), np.ones(12),
    )
    class_weights, class_mean, class_scale = train_classifier(
        partial, training, counts, atomic, left, right
    )
    model = SegmentModel(
        split_weights, split_mean, split_scale,
        class_weights, class_mean, class_scale,
    )

    stats = {}
    for name, sample in (("train", training), ("test", words[cut:])):
        classified = joint = bigram_boundary = known = bigrams = 0
        confusion = Counter()
        for word in sample:
            label, boundary = structural_truth(
                word, atomic, left, right
            )
            if label not in {"U", "B"}:
                continue
            predicted = model.segment(word, counts)
            predicted_label = "U" if len(predicted) == 1 else "B"
            confusion[label + predicted_label] += 1
            known += 1
            classified += predicted_label == label
            correct_boundary = (
                label == "U"
                or (
                    predicted_label == "B"
                    and predicted == (
                        "L:" + boundary[0], "R:" + boundary[1]
                    )
                )
            )
            joint += correct_boundary
            if label == "B":
                bigrams += 1
                bigram_boundary += correct_boundary
        stats[name] = {
            "known_tokens": known,
            "classification_accuracy": classified / known,
            "joint_emission_accuracy": joint / known,
            "bigram_boundary_accuracy": bigram_boundary / bigrams,
            "confusion": dict(confusion),
        }
    stats["table_inventory"] = {
        "atomic": len(atomic), "left": len(left), "right": len(right)
    }
    return model, stats


def build_stream(
    lines: list[list[str]],
    model: SegmentModel,
    codebook_size: int,
    name: str,
) -> tuple[Stream, dict]:
    words = [word for line in lines for word in line]
    training = words[:len(words) // 2]
    counts = structural_counts(training)
    segmented = [model.segment(word, counts) for word in words]
    training_segments = segmented[:len(training)]
    role_counts = {"U": Counter(), "L": Counter(), "R": Counter()}
    for emission in training_segments:
        for symbol in emission:
            role_counts[symbol[0]][symbol] += 1
    inventory = set()
    for role in ("U", "L", "R"):
        inventory.update(
            symbol
            for symbol, _ in role_counts[role].most_common(codebook_size)
        )

    symbols = []
    matched = matched_train = 0
    compound = atomic = 0
    for index, emission in enumerate(segmented):
        if not all(symbol in inventory for symbol in emission):
            continue
        symbols.extend(emission)
        matched += 1
        matched_train += index < len(training)
        compound += len(emission) == 2
        atomic += len(emission) == 1
    stats = {
        "tokens": len(words),
        "matched": matched,
        "coverage": matched / len(words),
        "train_coverage": matched_train / len(training),
        "atomic_emissions": atomic,
        "compound_emissions": compound,
        "inventory": len(inventory),
    }
    return Stream(
        f"{name}/calibrated_k{codebook_size}", tuple(symbols), False
    ), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codebook-sizes", default="150,250",
        help="comma-separated inventory limit per U/L/R role",
    )
    parser.add_argument("--steps", type=int, default=90000)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--naibbe-ciphertext", type=Path, required=True)
    parser.add_argument("--naibbe-table", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sizes = [int(value) for value in args.codebook_sizes.split(",")]
    naibbe_lines = load_plain_lines(args.naibbe_ciphertext)
    model, calibration = calibrate(naibbe_lines, args.naibbe_table)
    print(json.dumps({"calibration": calibration}, indent=2))

    corpora = {
        "naibbe_control": naibbe_lines,
        "voynich": load_voynich_lines(),
    }
    rng = np.random.default_rng(SEED)
    rows = []
    for corpus_name, lines in corpora.items():
        for size in sizes:
            stream, stats = build_stream(lines, model, size, corpus_name)
            for language in ("latin", "english"):
                control = clean_control(CONTROLS[language], language, False)
                cut = len(control) * 2 // 3
                lm = LM(control[:cut])
                row = fit(
                    stream, lm, rng, args.steps, args.restarts
                )
                row.update(
                    corpus=corpus_name,
                    language=language,
                    ceiling=lm.plain_score(control[cut:]),
                    codebook=size,
                    **stats,
                )
                rows.append(row)

    rows.sort(key=lambda row: row["test"] - row["ceiling"], reverse=True)
    for row in rows:
        print("\n" + "=" * 100)
        print(
            f"{row['corpus']:<15} -> {row['language']:<7} "
            f"K={row['codebook']:<3} coverage={row['coverage']:.3f} "
            f"symbols={row['symbols']:<4} train={row['train']:.3f} "
            f"test={row['test']:.3f} KL={row['train_kl']:.3f} "
            f"ceiling={row['ceiling']:.3f} "
            f"gap={row['test']-row['ceiling']:+.3f}"
        )
        print("train:", row["sample_train"])
        print("test :", row["sample_test"])

    if args.output:
        payload = {"calibration": calibration, "results": rows}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
