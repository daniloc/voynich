#!/usr/bin/env python3
"""
Minimal recurrence-encoded decoder inspired by Kambhatla et al. (EACL 2023).

The paper canonicalizes a cipher sequence by numbering symbols in order of
first appearance, then trains a large causal Transformer on millions of
randomized homophonic ciphers.  This local prototype preserves the
canonicalization and randomized-key training, but replaces the unavailable
Transformer stack with a smoothed categorical Naive Bayes model over
recurrence-only features.  It is an applicability probe, not a reproduction.

The synthetic generator matches a distinctive complication of Naibbe: one
cipher word emits either one or two plaintext characters, and a two-character
word is assembled from independently sampled prefix/suffix homophones.

The official Naibbe plaintext is loaded only after fitting and hyperparameter
selection.  It is used for scoring and for an explicitly labelled oracle
ceiling, never for training or model selection.

Reference:
    Kambhatla, Born, and Sarkar. "Decipherment as Regression: Solving
    Historical Substitution Ciphers by Learning Symbol Recurrence Relations."
    Findings of EACL 2023. https://aclanthology.org/2023.findings-eacl.160/
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LATIN = ROOT / "data" / "controls" / "latin.txt"
DEFAULT_CIPHERTEXT = Path(
    "/tmp/naibbe-cipher/encrypted/nathist_output_ciphertext.txt"
)
DEFAULT_PLAINTEXT = Path(
    "/tmp/naibbe-cipher/respaced_plaintext/"
    "nathist_pre_encryption_respaced_plaintext.txt"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "intermediate" / "followups_recurrence_decoder.json"
)
PAPER_URL = "https://aclanthology.org/2023.findings-eacl.160/"
FEATURE_WEIGHTS = (0.05, 0.1, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class UnitCodec:
    alphabet: tuple[str, ...]
    units: tuple[str, ...]
    to_id: dict[str, int]

    @classmethod
    def latin_ascii(cls) -> "UnitCodec":
        alphabet = tuple("abcdefghijklmnopqrstuvwxyz")
        units = alphabet + tuple(a + b for a in alphabet for b in alphabet)
        return cls(alphabet, units, {unit: i for i, unit in enumerate(units)})


@dataclass
class RecurrenceNB:
    feature_names: tuple[str, ...]
    cardinalities: tuple[int, ...]
    log_prior: np.ndarray
    log_likelihoods: list[np.ndarray]

    def predict(self, features: np.ndarray, weight: float) -> np.ndarray:
        scores = np.broadcast_to(
            self.log_prior, (features.shape[0], self.log_prior.size)
        ).copy()
        for column, likelihood in enumerate(self.log_likelihoods):
            scores += weight * likelihood[:, features[:, column]].T
        return np.argmax(scores, axis=1)


FEATURE_NAMES = (
    "symbol_frequency",
    "first_occurrence_rank",
    "first_position",
    "last_position",
    "prior_occurrences",
    "future_occurrences",
    "previous_gap",
    "next_gap",
    "sequence_position",
    "left1_frequency",
    "right1_frequency",
    "left2_frequency",
    "right2_frequency",
    "left1_is_same",
    "right1_is_same",
    "left2_is_same",
    "right2_is_same",
)
CARDINALITIES = (8, 10, 10, 10, 7, 7, 10, 10, 10, 9, 9, 9, 9, 3, 3, 3, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latin", type=Path, default=LATIN)
    parser.add_argument("--ciphertext", type=Path, default=DEFAULT_CIPHERTEXT)
    parser.add_argument("--plaintext", type=Path, default=DEFAULT_PLAINTEXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--sequence-length", type=int, default=400)
    parser.add_argument("--train-sequences", type=int, default=4000)
    parser.add_argument("--validation-sequences", type=int, default=400)
    parser.add_argument("--test-sequences", type=int, default=400)
    parser.add_argument("--min-homophones", type=int, default=2)
    parser.add_argument("--max-homophones", type=int, default=8)
    parser.add_argument(
        "--bigram-probability",
        type=float,
        default=0.5,
        help="Synthetic two-character emission probability (default: 0.5).",
    )
    parser.add_argument("--smoothing", type=float, default=0.25)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def gutenberg_body(text: str) -> str:
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = text.find(start_marker)
    if start >= 0:
        start = text.find("\n", start)
    else:
        start = 0
    end = text.find(end_marker, max(start, 0))
    if end < 0:
        end = len(text)
    return text[max(start, 0) : end]


def normalize_latin(text: str) -> str:
    return "".join(re.findall(r"[a-z]", gutenberg_body(text).lower()))


def recurrence_encode(tokens: Sequence[object]) -> np.ndarray:
    """Number each new token by first appearance, exactly as in the paper."""
    lookup: dict[object, int] = {}
    encoded = np.empty(len(tokens), dtype=np.int32)
    for index, token in enumerate(tokens):
        if token not in lookup:
            lookup[token] = len(lookup)
        encoded[index] = lookup[token]
    return encoded


def _frequency_bin(value: int) -> int:
    if value <= 1:
        return 0
    return min(7, int(math.log2(value)) + 1)


def _gap_bin(value: int) -> int:
    if value < 0:
        return 0
    if value <= 2:
        return value
    if value <= 4:
        return 3
    if value <= 8:
        return 4
    if value <= 16:
        return 5
    if value <= 32:
        return 6
    if value <= 64:
        return 7
    if value <= 128:
        return 8
    return 9


def _decile(value: int, denominator: int) -> int:
    return min(9, (10 * value) // max(denominator, 1))


def extract_features(encoded: np.ndarray) -> np.ndarray:
    """Return categorical features derived only from recurrence identities."""
    length = int(encoded.size)
    unique = int(encoded.max()) + 1 if length else 0
    frequencies = np.bincount(encoded, minlength=unique)
    first = np.full(unique, length, dtype=np.int32)
    last = np.full(unique, -1, dtype=np.int32)
    np.minimum.at(first, encoded, np.arange(length, dtype=np.int32))
    np.maximum.at(last, encoded, np.arange(length, dtype=np.int32))

    prior = np.zeros(length, dtype=np.int32)
    previous_gap = np.full(length, -1, dtype=np.int32)
    seen_count = np.zeros(unique, dtype=np.int32)
    seen_at = np.full(unique, -1, dtype=np.int32)
    for position, symbol in enumerate(encoded):
        prior[position] = seen_count[symbol]
        if seen_at[symbol] >= 0:
            previous_gap[position] = position - seen_at[symbol]
        seen_count[symbol] += 1
        seen_at[symbol] = position

    future = np.zeros(length, dtype=np.int32)
    next_gap = np.full(length, -1, dtype=np.int32)
    seen_count.fill(0)
    seen_at.fill(-1)
    for position in range(length - 1, -1, -1):
        symbol = encoded[position]
        future[position] = seen_count[symbol]
        if seen_at[symbol] >= 0:
            next_gap[position] = seen_at[symbol] - position
        seen_count[symbol] += 1
        seen_at[symbol] = position

    features = np.empty((length, len(FEATURE_NAMES)), dtype=np.int16)
    for position, symbol in enumerate(encoded):
        values = [
            _frequency_bin(int(frequencies[symbol])),
            _decile(int(symbol), unique),
            _decile(int(first[symbol]), length),
            _decile(int(last[symbol]), length),
            min(6, int(prior[position])),
            min(6, int(future[position])),
            _gap_bin(int(previous_gap[position])),
            _gap_bin(int(next_gap[position])),
            _decile(position, length),
        ]
        for offset in (-1, 1, -2, 2):
            neighbor = position + offset
            if 0 <= neighbor < length:
                values.append(_frequency_bin(int(frequencies[encoded[neighbor]])))
            else:
                values.append(8)
        for offset in (-1, 1, -2, 2):
            neighbor = position + offset
            if 0 <= neighbor < length:
                values.append(int(encoded[neighbor] == symbol))
            else:
                values.append(2)
        features[position] = values
    return features


def partition_units(
    text: str,
    start: int,
    length: int,
    bigram_probability: float,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    units: list[str] = []
    cursor = start
    for _ in range(length):
        step = 2 if rng.random() < bigram_probability else 1
        units.append(text[cursor : cursor + step])
        cursor += step
    return tuple(units)


def randomized_naibbe_like_cipher(
    units: Sequence[str],
    alphabet: Sequence[str],
    homophones: int,
    rng: np.random.Generator,
) -> tuple[object, ...]:
    """
    Build arbitrary word symbols from role-specific homophone components.

    Raw component labels are randomly permuted and then discarded by
    recurrence_encode.  Only exact word recurrence survives.
    """
    roles = ("unigram", "prefix", "suffix")
    raw = np.arange(len(roles) * len(alphabet) * homophones, dtype=np.int32)
    raw = rng.permutation(raw)
    pools: dict[tuple[str, str], np.ndarray] = {}
    cursor = 0
    for role in roles:
        for character in rng.permutation(np.asarray(alphabet)):
            pools[(role, str(character))] = raw[cursor : cursor + homophones]
            cursor += homophones

    tokens: list[object] = []
    for unit in units:
        if len(unit) == 1:
            pool = pools[("unigram", unit)]
            tokens.append(("u", int(pool[rng.integers(pool.size)])))
        else:
            left = pools[("prefix", unit[0])]
            right = pools[("suffix", unit[1])]
            tokens.append(
                (
                    "b",
                    int(left[rng.integers(left.size)]),
                    int(right[rng.integers(right.size)]),
                )
            )
    return tuple(tokens)


def synthetic_sample(
    text: str,
    length: int,
    bigram_probability: float,
    homophone_range: tuple[int, int],
    codec: UnitCodec,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum_start = len(text) - 2 * length - 1
    if maximum_start <= 0:
        raise ValueError("Latin control is too short for requested sequence length")
    start = int(rng.integers(maximum_start))
    units = partition_units(text, start, length, bigram_probability, rng)
    homophones = int(rng.integers(homophone_range[0], homophone_range[1] + 1))
    tokens = randomized_naibbe_like_cipher(
        units, codec.alphabet, homophones, rng
    )
    encoded = recurrence_encode(tokens)
    features = extract_features(encoded)
    targets = np.fromiter(
        (codec.to_id[unit] for unit in units), dtype=np.int32, count=length
    )
    return encoded, features, targets


def train_model(
    text: str,
    sequences: int,
    length: int,
    bigram_probability: float,
    homophone_range: tuple[int, int],
    codec: UnitCodec,
    smoothing: float,
    rng: np.random.Generator,
    progress: bool,
) -> tuple[RecurrenceNB, dict[str, float]]:
    class_count = len(codec.units)
    priors = np.zeros(class_count, dtype=np.int64)
    counts = [
        np.zeros((class_count, cardinality), dtype=np.int64)
        for cardinality in CARDINALITIES
    ]
    unique_fractions: list[float] = []
    for sequence in range(sequences):
        encoded, features, targets = synthetic_sample(
            text,
            length,
            bigram_probability,
            homophone_range,
            codec,
            rng,
        )
        unique_fractions.append(float(np.unique(encoded).size / encoded.size))
        np.add.at(priors, targets, 1)
        for column, table in enumerate(counts):
            np.add.at(table, (targets, features[:, column]), 1)
        if progress and (sequence + 1) % max(1, sequences // 10) == 0:
            print(
                f"training {sequence + 1}/{sequences}",
                file=sys.stderr,
                flush=True,
            )

    total = int(priors.sum())
    log_prior = np.log(
        (priors + smoothing) / (total + smoothing * class_count)
    )
    log_likelihoods: list[np.ndarray] = []
    for table, cardinality in zip(counts, CARDINALITIES):
        denominator = priors[:, None] + smoothing * cardinality
        log_likelihoods.append(np.log((table + smoothing) / denominator))
    model = RecurrenceNB(
        FEATURE_NAMES, CARDINALITIES, log_prior, log_likelihoods
    )
    diagnostics = {
        "tokens": total,
        "mean_unique_fraction": float(np.mean(unique_fractions)),
        "median_unique_fraction": float(np.median(unique_fractions)),
        "observed_target_classes": int(np.count_nonzero(priors)),
        "mode_target_id": int(np.argmax(priors)),
    }
    return model, diagnostics


def unit_metrics(
    predictions: np.ndarray, targets: np.ndarray, codec: UnitCodec
) -> dict[str, float]:
    exact = predictions == targets
    true_units = [codec.units[int(target)] for target in targets]
    predicted_units = [codec.units[int(prediction)] for prediction in predictions]
    true_characters = 0
    correct_characters = 0
    correct_lengths = 0
    for predicted, truth in zip(predicted_units, true_units):
        true_characters += len(truth)
        correct_lengths += len(predicted) == len(truth)
        if len(predicted) == len(truth):
            correct_characters += sum(a == b for a, b in zip(predicted, truth))
    return {
        "unit_accuracy": float(np.mean(exact)),
        "emission_length_accuracy": float(correct_lengths / len(true_units)),
        "aligned_character_accuracy": float(
            correct_characters / max(true_characters, 1)
        ),
    }


def make_synthetic_dataset(
    text: str,
    sequences: int,
    length: int,
    bigram_probability: float,
    homophone_range: tuple[int, int],
    codec: UnitCodec,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [
        synthetic_sample(
            text,
            length,
            bigram_probability,
            homophone_range,
            codec,
            rng,
        )
        for _ in range(sequences)
    ]


def score_synthetic(
    model: RecurrenceNB,
    dataset: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
    codec: UnitCodec,
    weight: float,
    destroy_recurrence: bool = False,
) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    unique_fractions: list[float] = []
    for encoded, features, truth in dataset:
        if destroy_recurrence:
            features = extract_features(np.arange(encoded.size, dtype=np.int32))
        predictions.append(model.predict(features, weight))
        targets.append(truth)
        unique_fractions.append(float(np.unique(encoded).size / encoded.size))
    metrics = unit_metrics(
        np.concatenate(predictions), np.concatenate(targets), codec
    )
    metrics["mean_unique_fraction"] = float(np.mean(unique_fractions))
    return metrics


def load_official_units(
    ciphertext_path: Path, plaintext_path: Path
) -> tuple[list[str], list[str]]:
    cipher = ciphertext_path.read_text(encoding="utf-8").split()
    plain = plaintext_path.read_text(encoding="utf-8").split()
    if len(cipher) != len(plain):
        raise ValueError(
            "Official token counts do not align. Use the unrespaced ciphertext: "
            f"{len(cipher)} cipher tokens versus {len(plain)} plaintext units."
        )
    invalid = [
        unit
        for unit in plain
        if len(unit) not in (1, 2) or not re.fullmatch(r"[a-z]+", unit)
    ]
    if invalid:
        raise ValueError(f"Unexpected plaintext units: {invalid[:5]}")
    return cipher, plain


def official_oracle(cipher: Sequence[str], plain: Sequence[str]) -> dict[str, float]:
    mappings: dict[str, Counter[str]] = defaultdict(Counter)
    for symbol, unit in zip(cipher, plain):
        mappings[symbol][unit] += 1
    correct = sum(max(counter.values()) for counter in mappings.values())
    return {
        "majority_word_to_unit_accuracy": float(correct / len(cipher)),
        "ambiguous_word_types": int(
            sum(len(counter) > 1 for counter in mappings.values())
        ),
        "cipher_word_types": int(len(mappings)),
    }


def score_official(
    model: RecurrenceNB,
    cipher: Sequence[str],
    plain: Sequence[str],
    codec: UnitCodec,
    length: int,
    weight: float,
    destroy_recurrence: bool = False,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    unique_fractions: list[float] = []
    examples: list[dict[str, object]] = []
    usable = (len(cipher) // length) * length
    for start in range(0, usable, length):
        cipher_window = cipher[start : start + length]
        plain_window = plain[start : start + length]
        encoded = recurrence_encode(cipher_window)
        unique_fractions.append(float(np.unique(encoded).size / encoded.size))
        if destroy_recurrence:
            encoded = np.arange(length, dtype=np.int32)
        features = extract_features(encoded)
        predicted = model.predict(features, weight)
        truth = np.fromiter(
            (codec.to_id[unit] for unit in plain_window),
            dtype=np.int32,
            count=length,
        )
        predictions.append(predicted)
        targets.append(truth)
        if len(examples) < 2:
            examples.append(
                {
                    "ciphertext": list(cipher_window[:24]),
                    "truth": list(plain_window[:24]),
                    "prediction": [
                        codec.units[int(value)] for value in predicted[:24]
                    ],
                }
            )
    joined_predictions = np.concatenate(predictions)
    joined_targets = np.concatenate(targets)
    metrics = unit_metrics(joined_predictions, joined_targets, codec)
    metrics.update(
        {
            "scored_units": int(usable),
            "dropped_tail_units": int(len(cipher) - usable),
            "windows": int(usable // length),
            "mean_unique_fraction": float(np.mean(unique_fractions)),
            "whole_document_unique_fraction": float(
                len(set(cipher)) / len(cipher)
            ),
        }
    )
    return metrics, examples


def main() -> None:
    args = parse_args()
    if args.sequence_length < 16:
        raise ValueError("--sequence-length must be at least 16")
    if not (1 <= args.min_homophones <= args.max_homophones):
        raise ValueError("Invalid homophone range")

    started = time.time()
    codec = UnitCodec.latin_ascii()
    latin = normalize_latin(args.latin.read_text(encoding="utf-8"))
    split = int(0.8 * len(latin))
    train_text = latin[:split]
    heldout_text = latin[split:]

    bigram_probability = args.bigram_probability
    if not 0.0 <= bigram_probability <= 1.0:
        raise ValueError("--bigram-probability must be between zero and one")

    seed_sequence = np.random.SeedSequence(args.seed)
    train_seed, validation_seed, test_seed = seed_sequence.spawn(3)
    homophone_range = (args.min_homophones, args.max_homophones)
    model, training = train_model(
        train_text,
        args.train_sequences,
        args.sequence_length,
        bigram_probability,
        homophone_range,
        codec,
        args.smoothing,
        np.random.default_rng(train_seed),
        args.progress,
    )
    validation = make_synthetic_dataset(
        heldout_text,
        args.validation_sequences,
        args.sequence_length,
        bigram_probability,
        homophone_range,
        codec,
        np.random.default_rng(validation_seed),
    )
    validation_by_weight = {
        str(weight): score_synthetic(model, validation, codec, weight)
        for weight in FEATURE_WEIGHTS
    }
    selected_weight = max(
        FEATURE_WEIGHTS,
        key=lambda value: validation_by_weight[str(value)]["unit_accuracy"],
    )
    test = make_synthetic_dataset(
        heldout_text,
        args.test_sequences,
        args.sequence_length,
        bigram_probability,
        homophone_range,
        codec,
        np.random.default_rng(test_seed),
    )
    synthetic_test = score_synthetic(
        model, test, codec, selected_weight
    )
    synthetic_ablation = score_synthetic(
        model, test, codec, selected_weight, destroy_recurrence=True
    )
    synthetic_prior = score_synthetic(model, test, codec, 0.0)

    # The official answer file is first touched after every fitted choice is frozen.
    cipher, plain = load_official_units(args.ciphertext, args.plaintext)
    observed_bigram_probability = float(
        np.mean([len(unit) == 2 for unit in plain])
    )
    official_metrics, examples = score_official(
        model,
        cipher,
        plain,
        codec,
        args.sequence_length,
        selected_weight,
    )
    official_ablation, _ = score_official(
        model,
        cipher,
        plain,
        codec,
        args.sequence_length,
        selected_weight,
        destroy_recurrence=True,
    )
    official_prior, _ = score_official(
        model, cipher, plain, codec, args.sequence_length, 0.0
    )

    report = {
        "experiment": "minimal_recurrence_encoded_decoder",
        "paper": {
            "citation": (
                "Kambhatla, Born, and Sarkar (2023), Decipherment as "
                "Regression, Findings of EACL 2023"
            ),
            "url": PAPER_URL,
            "faithfulness": (
                "Uses first-occurrence recurrence encoding and randomized "
                "homophonic training. Replaces the paper's 12-layer causal "
                "Transformer and 2M examples with recurrence-feature Naive "
                "Bayes because no local neural backend is available."
            ),
        },
        "leakage_policy": (
            "Latin control split 80/20 by position. Synthetic keys are newly "
            "randomized per sequence. The generator's 50/50 default emission "
            "length mix is fixed without consulting Naibbe. Official Naibbe "
            "plaintext is first loaded only after fit and validation, for "
            "scoring and the labelled oracle."
        ),
        "configuration": {
            "seed": args.seed,
            "sequence_length": args.sequence_length,
            "train_sequences": args.train_sequences,
            "validation_sequences": args.validation_sequences,
            "test_sequences": args.test_sequences,
            "homophone_range": list(homophone_range),
            "bigram_probability": bigram_probability,
            "observed_naibbe_bigram_probability": observed_bigram_probability,
            "smoothing": args.smoothing,
            "feature_names": list(FEATURE_NAMES),
            "feature_weights": list(FEATURE_WEIGHTS),
            "selected_feature_weight": selected_weight,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "neural_backend_available": False,
        },
        "training": training,
        "synthetic_validation_by_weight": validation_by_weight,
        "synthetic_test": synthetic_test,
        "synthetic_test_recurrence_destroyed": synthetic_ablation,
        "synthetic_test_prior_only": synthetic_prior,
        "official_naibbe": official_metrics,
        "official_naibbe_recurrence_destroyed": official_ablation,
        "official_naibbe_prior_only": official_prior,
        "official_naibbe_evaluation_only_oracle": official_oracle(cipher, plain),
        "official_naibbe_examples": examples,
        "interpretation_guardrail": (
            "A useful EACL-style signal requires the learned decoder to beat "
            "both prior-only and recurrence-destroyed controls on synthetic "
            "holdout and transfer above those controls to untouched Naibbe. "
            "The oracle is a ceiling, not a blind-decipherment result."
        ),
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
