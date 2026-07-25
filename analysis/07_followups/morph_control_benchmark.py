#!/usr/bin/env python3
"""
Matched held-out morphotactic benchmark for Voynich and meaningful controls.

The benchmark applies one corpus-agnostic representation and scoring pipeline to:

* VOYNICH: prose lines in the repository corpus;
* NAIBBE_MEANINGFUL: the published respaced Naibbe encryption of Pliny Book 16;
* LATIN_PLAIN: ordinary words from the published Pliny corpus, starting at the
  Book 16 incipit encrypted by Naibbe;
* LATIN_SIMPLE_SUB: a bijective character substitution of LATIN_PLAIN.

All controls are reflowed through the exact Voynich prose line-length and quire
template. For each quire, counts and representation vocabularies are fitted on
the other fifteen blocks. The primary score on held-out transitions is:

    log2 P(y | predecessor, target-position)
      - log2 P(y | target-position)

The directionality score is the held-out bigram log-likelihood of each forward
line minus the likelihood after reversing only its deep interior. First, second,
penultimate, and last tokens therefore remain in their original slots.

The boundary representations are deliberately language-independent: "prefix"
and "suffix" mean the first/last two characters, not hand-selected EVA affixes.
These choices make the computation comparable, but they limit any linguistic
interpretation of the result.

A separately labelled sensitivity panel mechanically applies the exact
Voynich-selected EVA affix inventory from sequence_parallel.py and
morph_voynich_transfer.py to every corpus. It directly benchmarks the feature
map that motivated this follow-up, but it is not a language-universal or
substitution-invariant comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from random import Random
from typing import Callable, Hashable, Iterable, Optional


SEED = 20260722
ALPHA = 0.5
BOUNDARY_WIDTH = 2
N_BOOTSTRAP = 20_000
NAIBBE_GITHUB = "https://github.com/greshko/naibbe-cipher"
NAIBBE_DOI = "https://doi.org/10.1080/01611194.2025.2566408"
ROOT = Path(__file__).resolve().parents[2]

Representation = Callable[[str], Hashable]
UNK = ("<UNK>",)


def asset_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class TemplateLine:
    block: str
    words: tuple[str, ...]


@dataclass(frozen=True)
class BreakTemplateLine:
    block: str
    words: tuple[Optional[str], ...]


@dataclass(frozen=True)
class BlockScore:
    gain_sum: float
    gain_n: int
    direction_sum: float
    direction_n: int

    @property
    def gain(self) -> float:
        return self.gain_sum / self.gain_n if self.gain_n else float("nan")

    @property
    def direction(self) -> float:
        return (
            self.direction_sum / self.direction_n
            if self.direction_n
            else float("nan")
        )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--naibbe-dir",
        type=Path,
        default=Path(os.environ.get("NAIBBE_REPO", "/tmp/naibbe-cipher")),
        help="checkout of the official greshko/naibbe-cipher repository",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root
        / "data"
        / "intermediate"
        / "followups_morph_controls.txt",
        help="text report path",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="paired/equal-block bootstrap replicates",
    )
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


def locus_type(locus: str) -> str:
    match = re.search(r"[A-Za-z]", locus)
    return match.group(0).upper() if match else "?"


def load_voynich_template(corpus_path: Path) -> list[TemplateLine]:
    corpus = json.loads(corpus_path.read_text())
    lines: list[TemplateLine] = []
    for folio, folio_lines in corpus["folios"].items():
        block = str(corpus["meta"].get(folio, {}).get("Q", "?"))
        for line in folio_lines:
            if locus_type(line["locus"]) != "P":
                continue
            words = tuple(
                word
                for word in line["words"]
                if "?" not in word and len(word) >= 2
            )
            if len(words) >= 4:
                lines.append(TemplateLine(block, words))
    return lines


def load_voynich_break_template(
    corpus_path: Path,
) -> tuple[list[BreakTemplateLine], Counter[str]]:
    """Load the transfer-test protocol, preserving in-line adjacency breaks."""
    corpus = json.loads(corpus_path.read_text())
    lines = []
    audit: Counter[str] = Counter()
    for folio, folio_lines in corpus["folios"].items():
        block = str(corpus["meta"].get(folio, {}).get("Q", "?"))
        for line in folio_lines:
            if locus_type(line["locus"]) != "P":
                continue
            audit["prose_lines"] += 1
            words: list[Optional[str]] = []
            for word in line["words"]:
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
            has_transition = any(
                left is not None and right is not None
                for left, right in zip(words, words[1:])
            )
            if has_transition:
                lines.append(BreakTemplateLine(block, tuple(words)))
                audit["retained_lines"] += 1
            else:
                audit["excluded_lines"] += 1
    return lines, audit


def load_alpha_tokens(path: Path) -> list[str]:
    return re.findall(
        r"[a-z]+",
        path.read_text(encoding="utf-8", errors="ignore").lower(),
    )


def from_book16_incipit(tokens: list[str]) -> tuple[list[str], int]:
    incipit = ("pomiferae", "arbores", "quaeque", "mitioribus", "sucis")
    width = len(incipit)
    matches = [
        index
        for index in range(len(tokens) - width + 1)
        if tuple(tokens[index : index + width]) == incipit
    ]
    if not matches:
        raise ValueError("Book 16 incipit not found in Latin asset")
    # The published aggregate has Book 16 twice. Starting at the last copy
    # prevents an exact duplicate from crossing the held-out pseudo-blocks.
    offset = matches[-1]
    return tokens[offset:], offset


def reflow_to_template(
    tokens: list[str], template: list[TemplateLine]
) -> list[TemplateLine]:
    needed = sum(len(line.words) for line in template)
    if len(tokens) < needed:
        raise ValueError(f"need {needed} tokens, found {len(tokens)}")
    offset = 0
    result = []
    for line in template:
        length = len(line.words)
        result.append(
            TemplateLine(line.block, tuple(tokens[offset : offset + length]))
        )
        offset += length
    return result


def reflow_to_break_template(
    tokens: list[str], template: list[BreakTemplateLine]
) -> list[BreakTemplateLine]:
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
        result.append(BreakTemplateLine(line.block, tuple(words)))
    return result


def simple_substitution(tokens: Iterable[str]) -> list[str]:
    """Apply a fixed bijective monoalphabetic substitution (Atbash)."""
    source = "abcdefghijklmnopqrstuvwxyz"
    target = source[::-1]
    table = str.maketrans(source, target)
    return [token.translate(table) for token in tokens]


def prefix(word: str) -> str:
    return word[:BOUNDARY_WIDTH]


def suffix(word: str) -> str:
    return word[-BOUNDARY_WIDTH:]


def affix_pair(word: str) -> tuple[str, str]:
    return prefix(word), suffix(word)


def shape(word: str) -> tuple[int, bool, bool, bool]:
    """A substitution-invariant, low-cardinality orthographic shape."""
    return (
        min(len(word), 8),
        word[0] == word[-1],
        any(left == right for left, right in zip(word, word[1:])),
        len(set(word)) < len(word),
    )


REPRESENTATIONS: dict[str, Representation] = {
    "prefix2": prefix,
    "suffix2": suffix,
    "affix_pair2": affix_pair,
    "shape": shape,
}

EVA_PREFIXES = [
    "qok", "qot", "qo", "ok", "ot", "o", "y", "ch", "sh", "d",
    "cth", "ckh", "cph", "cfh",
]
EVA_SUFFIXES = [
    "eedy", "eody", "edy", "aiin", "aiir", "ain", "iin", "dy", "ol",
    "or", "ar", "al", "am", "dam", "ey", "eey", "y",
]


@lru_cache(maxsize=None)
def eva_decompose(word: str) -> tuple[str, str, str]:
    """Exact decomposition used by the repository's Voynich transfer tests."""
    selected_prefix = next(
        (
            value
            for value in sorted(EVA_PREFIXES, key=len, reverse=True)
            if word.startswith(value)
        ),
        "",
    )
    residual = word[len(selected_prefix) :]
    selected_suffix = next(
        (
            value
            for value in sorted(EVA_SUFFIXES, key=len, reverse=True)
            if residual.endswith(value) and len(residual) > len(value)
        ),
        "",
    )
    core = (
        residual[: -len(selected_suffix)]
        if selected_suffix
        else residual
    )
    return selected_prefix, core, selected_suffix


EVA_SELECTED_REPRESENTATIONS: dict[str, Representation] = {
    "eva_prefix": lambda word: eva_decompose(word)[0] or "none",
    "eva_suffix": lambda word: eva_decompose(word)[2] or "none",
    "eva_affix_pair": lambda word: (
        eva_decompose(word)[0] or "none",
        eva_decompose(word)[2] or "none",
    ),
    "eva_shape": lambda word: (
        bool(eva_decompose(word)[0]),
        bool(eva_decompose(word)[2]),
        min(len(word), 8),
        any(char in word for char in "ktpf"),
    ),
}


def position_bucket(index: int, length: int) -> str:
    if index == 0:
        return "first"
    if index == length - 1:
        return "last"
    if index == 1:
        return "second"
    if index == length - 2:
        return "penult"
    return "interior"


def train_model(
    lines: list[TemplateLine],
    held_out: str,
    representation: Representation,
) -> tuple[
    set[Hashable],
    Counter[tuple[str, Hashable]],
    Counter[str],
    Counter[tuple[str, Hashable, Hashable]],
    Counter[tuple[str, Hashable]],
]:
    vocabulary: set[Hashable] = set()
    for line in lines:
        if line.block != held_out:
            vocabulary.update(map(representation, line.words))

    unigram: Counter[tuple[str, Hashable]] = Counter()
    position_context: Counter[str] = Counter()
    bigram: Counter[tuple[str, Hashable, Hashable]] = Counter()
    predecessor_context: Counter[tuple[str, Hashable]] = Counter()
    for line in lines:
        if line.block == held_out:
            continue
        values = [representation(word) for word in line.words]
        for target_index in range(1, len(values)):
            position = position_bucket(target_index, len(values))
            previous, target = values[target_index - 1 : target_index + 1]
            unigram[(position, target)] += 1
            position_context[position] += 1
            bigram[(position, previous, target)] += 1
            predecessor_context[(position, previous)] += 1
    return (
        vocabulary,
        unigram,
        position_context,
        bigram,
        predecessor_context,
    )


def heldout_block_score(
    lines: list[TemplateLine],
    held_out: str,
    representation: Representation,
) -> BlockScore:
    (
        vocabulary,
        unigram,
        position_context,
        bigram,
        predecessor_context,
    ) = train_model(lines, held_out, representation)
    vocabulary_size = len(vocabulary) + 1

    def known(value: Hashable) -> Hashable:
        return value if value in vocabulary else UNK

    def unigram_logp(target: Hashable, position: str) -> float:
        return math.log2(
            (unigram[(position, target)] + ALPHA)
            / (position_context[position] + ALPHA * vocabulary_size)
        )

    def bigram_logp(
        previous: Hashable, target: Hashable, position: str
    ) -> float:
        return math.log2(
            (bigram[(position, previous, target)] + ALPHA)
            / (
                predecessor_context[(position, previous)]
                + ALPHA * vocabulary_size
            )
        )

    gain_sum = 0.0
    gain_n = 0
    direction_sum = 0.0
    direction_n = 0
    for line in lines:
        if line.block != held_out:
            continue
        values = [known(representation(word)) for word in line.words]
        for target_index in range(1, len(values)):
            position = position_bucket(target_index, len(values))
            previous, target = values[target_index - 1 : target_index + 1]
            gain_sum += bigram_logp(previous, target, position)
            gain_sum -= unigram_logp(target, position)
            gain_n += 1

        if len(values) < 7:
            continue
        reversed_values = (
            values[:2] + list(reversed(values[2:-2])) + values[-2:]
        )
        for target_index in range(1, len(values)):
            position = position_bucket(target_index, len(values))
            forward_previous, forward_target = values[
                target_index - 1 : target_index + 1
            ]
            reverse_previous, reverse_target = reversed_values[
                target_index - 1 : target_index + 1
            ]
            direction_sum += bigram_logp(
                forward_previous, forward_target, position
            )
            direction_sum -= bigram_logp(
                reverse_previous, reverse_target, position
            )
            direction_n += 1

    return BlockScore(gain_sum, gain_n, direction_sum, direction_n)


def score_corpus(
    lines: list[TemplateLine], representation: Representation
) -> dict[str, BlockScore]:
    blocks = sorted({line.block for line in lines})
    return {
        block: heldout_block_score(lines, block, representation)
        for block in blocks
    }


def iter_break_transitions(
    words: tuple[Optional[str], ...],
) -> Iterable[tuple[str, str, str]]:
    length = len(words)
    for target_index in range(1, length):
        previous = words[target_index - 1]
        target = words[target_index]
        if previous is None or target is None:
            continue
        yield previous, target, position_bucket(target_index, length)


def heldout_break_block_score(
    lines: list[BreakTemplateLine],
    held_out: str,
    representation: Representation,
) -> BlockScore:
    left_vocabulary: set[Hashable] = set()
    right_vocabulary: set[Hashable] = set()
    unigram: Counter[tuple[str, Hashable]] = Counter()
    position_context: Counter[str] = Counter()
    bigram: Counter[tuple[str, Hashable, Hashable]] = Counter()
    predecessor_context: Counter[tuple[str, Hashable]] = Counter()
    for line in lines:
        if line.block == held_out:
            continue
        for previous_word, target_word, position in iter_break_transitions(
            line.words
        ):
            previous = representation(previous_word)
            target = representation(target_word)
            left_vocabulary.add(previous)
            right_vocabulary.add(target)
            unigram[(position, target)] += 1
            position_context[position] += 1
            bigram[(position, previous, target)] += 1
            predecessor_context[(position, previous)] += 1
    vocabulary_size = len(right_vocabulary) + 1

    def bigram_logp(
        previous_word: str, target_word: str, position: str
    ) -> float:
        raw_previous = representation(previous_word)
        raw_target = representation(target_word)
        previous = (
            raw_previous if raw_previous in left_vocabulary else UNK
        )
        target = raw_target if raw_target in right_vocabulary else UNK
        return math.log2(
            (bigram[(position, previous, target)] + ALPHA)
            / (
                predecessor_context[(position, previous)]
                + ALPHA * vocabulary_size
            )
        )

    def gain(
        previous_word: str, target_word: str, position: str
    ) -> float:
        raw_target = representation(target_word)
        target = raw_target if raw_target in right_vocabulary else UNK
        baseline = math.log2(
            (unigram[(position, target)] + ALPHA)
            / (position_context[position] + ALPHA * vocabulary_size)
        )
        return bigram_logp(previous_word, target_word, position) - baseline

    gain_sum = 0.0
    gain_n = 0
    direction_sum = 0.0
    direction_n = 0
    for line in lines:
        if line.block != held_out:
            continue
        for previous, target, position in iter_break_transitions(line.words):
            gain_sum += gain(previous, target, position)
            gain_n += 1

        if len(line.words) < 7:
            continue
        reversed_words = list(line.words)
        eligible_interior = [
            word for word in line.words[2:-2] if word is not None
        ]
        eligible_interior.reverse()
        replacement = iter(eligible_interior)
        for index in range(2, len(reversed_words) - 2):
            if reversed_words[index] is not None:
                reversed_words[index] = next(replacement)
        for target_index in range(1, len(line.words)):
            forward_previous = line.words[target_index - 1]
            forward_target = line.words[target_index]
            reverse_previous = reversed_words[target_index - 1]
            reverse_target = reversed_words[target_index]
            if forward_previous is None or forward_target is None:
                continue
            if reverse_previous is None or reverse_target is None:
                raise AssertionError("break positions changed during reversal")
            position = position_bucket(target_index, len(line.words))
            direction_sum += bigram_logp(
                forward_previous, forward_target, position
            )
            direction_sum -= bigram_logp(
                reverse_previous, reverse_target, position
            )
            direction_n += 1
    return BlockScore(gain_sum, gain_n, direction_sum, direction_n)


def score_break_corpus(
    lines: list[BreakTemplateLine], representation: Representation
) -> dict[str, BlockScore]:
    blocks = sorted({line.block for line in lines})
    return {
        block: heldout_break_block_score(lines, block, representation)
        for block in blocks
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_mean_ci(
    values: list[float], replicates: int, seed_key: str
) -> tuple[float, float]:
    rng = Random(
        SEED
        ^ int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big")
    )
    samples = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(replicates)
    ]
    return percentile(samples, 0.025), percentile(samples, 0.975)


def exact_sign_flip_p(values: list[float]) -> float:
    """Two-sided exact paired test over the sixteen matched blocks."""
    observed = abs(sum(values))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(sum(value * sign for value, sign in zip(values, signs)))
        exceed += permuted >= observed - 1e-12
        total += 1
    return exceed / total


def metric_values(
    scores: dict[str, BlockScore], metric: str
) -> tuple[list[str], list[float]]:
    blocks = sorted(scores)
    return blocks, [getattr(scores[block], metric) for block in blocks]


def micro_score(scores: dict[str, BlockScore], metric: str) -> float:
    numerator = sum(getattr(score, f"{metric}_sum") for score in scores.values())
    denominator = sum(getattr(score, f"{metric}_n") for score in scores.values())
    return numerator / denominator


def format_summary(
    scores: dict[str, BlockScore],
    metric: str,
    bootstrap: int,
    seed_key: str,
) -> str:
    _, values = metric_values(scores, metric)
    mean = sum(values) / len(values)
    lower, upper = bootstrap_mean_ci(values, bootstrap, seed_key)
    positive = sum(value > 0 for value in values)
    return (
        f"{micro_score(scores, metric):+9.5f} "
        f"{mean:+9.5f} [{lower:+9.5f}, {upper:+9.5f}] "
        f"{positive:2d}/{len(values):2d}"
    )


def paired_contrast(
    reference: dict[str, BlockScore],
    control: dict[str, BlockScore],
    metric: str,
    bootstrap: int,
    seed_key: str,
) -> tuple[float, float, float, float]:
    blocks = sorted(reference)
    if blocks != sorted(control):
        raise ValueError("paired blocks do not match")
    differences = [
        getattr(reference[block], metric) - getattr(control[block], metric)
        for block in blocks
    ]
    mean = sum(differences) / len(differences)
    lower, upper = bootstrap_mean_ci(differences, bootstrap, seed_key)
    return mean, lower, upper, exact_sign_flip_p(differences)


def validate_inputs(
    repo_root: Path, naibbe_dir: Path
) -> tuple[Path, Path, Path]:
    corpus_path = repo_root / "data" / "corpus" / "corpus.json"
    cipher_path = (
        naibbe_dir
        / "encrypted"
        / "nathist_output_ciphertext_respaced.txt"
    )
    plain_path = (
        naibbe_dir
        / "figure_utils"
        / "gaskell_bowern_2022"
        / "data"
        / "meaningful"
        / "texts"
        / "Historical - Latin - Technical - Pliny's Natural History.txt"
    )
    for path in (corpus_path, cipher_path, plain_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return corpus_path, cipher_path, plain_path


def build_report(args: argparse.Namespace) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    corpus_path, cipher_path, plain_path = validate_inputs(
        repo_root, args.naibbe_dir
    )
    template = load_voynich_template(corpus_path)
    break_template, break_audit = load_voynich_break_template(corpus_path)
    cipher_tokens = load_alpha_tokens(cipher_path)
    all_plain_tokens = load_alpha_tokens(plain_path)
    plain_tokens, plain_offset = from_book16_incipit(all_plain_tokens)
    token_count = sum(len(line.words) for line in template)
    break_token_count = sum(
        word is not None
        for line in break_template
        for word in line.words
    )
    datasets = {
        "VOYNICH": template,
        "NAIBBE_MEANINGFUL": reflow_to_template(cipher_tokens, template),
        "LATIN_PLAIN": reflow_to_template(plain_tokens, template),
        "LATIN_SIMPLE_SUB": reflow_to_template(
            simple_substitution(plain_tokens), template
        ),
    }
    break_datasets = {
        "VOYNICH": break_template,
        "NAIBBE_MEANINGFUL": reflow_to_break_template(
            cipher_tokens, break_template
        ),
        "LATIN_PLAIN": reflow_to_break_template(
            plain_tokens, break_template
        ),
        "LATIN_SIMPLE_SUB": reflow_to_break_template(
            simple_substitution(plain_tokens), break_template
        ),
    }
    blocks = sorted({line.block for line in template})

    output = []

    def emit(text: str = "") -> None:
        output.append(text)

    emit("=" * 88)
    emit("MATCHED HELD-OUT MORPHOTACTIC CONTROL BENCHMARK")
    emit("=" * 88)
    emit(f"seed={SEED} alpha={ALPHA} boundary_width={BOUNDARY_WIDTH}")
    emit(
        f"template={len(template)} prose lines, {token_count} tokens, "
        f"{len(blocks)} quire blocks ({','.join(blocks)})"
    )
    emit(
        f"EVA sensitivity template={len(break_template)} lines, "
        f"{break_token_count} eligible tokens, "
        f"{break_audit['uncertain_breaks']} uncertain and "
        f"{break_audit['one_character_breaks']} one-character breaks"
    )
    emit("folds=leave one matched quire block out; all model counts/vocab are train-only")
    emit(
        "gain=mean held-out [log2 P(y|predecessor,target-position) "
        "- log2 P(y|target-position)]"
    )
    emit(
        "direction=mean held-out forward minus deep-interior-reversed bigram "
        "log2 probability"
    )
    emit(
        "positions=first/second/interior/penult/last; reversal fixes the four edge slots"
    )
    emit("uncertainty=equal-block bootstrap 95% CI; positive=count of blocks > 0")
    emit()
    emit("PROVENANCE")
    emit(f"  Naibbe repository: {NAIBBE_GITHUB}")
    emit(f"  paper:             {NAIBBE_DOI}")
    emit(f"  checkout revision: {git_revision(args.naibbe_dir)}")
    emit(f"  cipher asset:      {cipher_path}")
    emit(f"  cipher SHA256:     {sha256(cipher_path)}")
    emit(f"  cipher tokens:     {len(cipher_tokens)}")
    emit(f"  plaintext asset:   {plain_path}")
    emit(f"  plaintext SHA256:  {sha256(plain_path)}")
    emit(f"  plaintext tokens:  {len(all_plain_tokens)} total")
    emit(
        f"  Book 16 start:     token offset {plain_offset}; "
        f"{len(plain_tokens)} tokens remain"
    )
    emit("  aggregate note:    last of two duplicated Book 16 incipits selected")
    emit(f"  Voynich corpus:    {asset_name(corpus_path)}")
    emit(f"  corpus SHA256:     {sha256(corpus_path)}")

    all_scores: dict[
        str, dict[str, dict[str, BlockScore]]
    ] = {}
    for representation_name, representation in REPRESENTATIONS.items():
        all_scores[representation_name] = {}
        for corpus_name, lines in datasets.items():
            all_scores[representation_name][corpus_name] = score_corpus(
                lines, representation
            )
    for representation_name in REPRESENTATIONS:
        plain = all_scores[representation_name]["LATIN_PLAIN"]
        substituted = all_scores[representation_name]["LATIN_SIMPLE_SUB"]
        if plain != substituted:
            raise AssertionError(
                f"substitution invariance failed for {representation_name}"
            )

    eva_scores: dict[str, dict[str, dict[str, BlockScore]]] = {}
    for representation_name, representation in (
        EVA_SELECTED_REPRESENTATIONS.items()
    ):
        eva_scores[representation_name] = {}
        for corpus_name, lines in break_datasets.items():
            eva_scores[representation_name][corpus_name] = score_break_corpus(
                lines, representation
            )

    emit()
    emit("REPRESENTATION CARDINALITY IN THE MATCHED SAMPLE")
    emit(f"  {'representation':14s} " + " ".join(
        f"{name:>18s}" for name in datasets
    ))
    for representation_name, representation in REPRESENTATIONS.items():
        cardinalities = [
            len(
                {
                    representation(word)
                    for line in lines
                    for word in line.words
                }
            )
            for lines in datasets.values()
        ]
        emit(
            f"  {representation_name:14s} "
            + " ".join(f"{value:18d}" for value in cardinalities)
        )

    for metric, title in (
        ("gain", "HELD-OUT BIGRAM GAIN OVER POSITION-CONDITIONED UNIGRAM"),
        ("direction", "FORWARD VS DEEP-INTERIOR-REVERSED DIRECTIONALITY"),
    ):
        emit()
        emit(title)
        emit(
            f"  {'representation':14s} {'corpus':18s} "
            f"{'micro':>9s} {'block mean [95% CI]':>31s} {'positive':>8s}"
        )
        for representation_name in REPRESENTATIONS:
            for corpus_name in datasets:
                scores = all_scores[representation_name][corpus_name]
                summary = format_summary(
                    scores,
                    metric,
                    args.bootstrap,
                    f"{metric}:{representation_name}:{corpus_name}",
                )
                emit(
                    f"  {representation_name:14s} {corpus_name:18s} {summary}"
                )

        emit()
        emit("  PAIRED CONTRASTS: VOYNICH minus control, equal-block units")
        emit("  LATIN_SIMPLE_SUB is an exact invariant check, not a separate hypothesis")
        emit(
            f"  {'representation':14s} {'control':18s} "
            f"{'mean difference [95% CI]':>35s} {'raw p':>9s} {'Bonf p':>9s}"
        )
        comparisons = []
        for representation_name in REPRESENTATIONS:
            reference = all_scores[representation_name]["VOYNICH"]
            for corpus_name in (
                "NAIBBE_MEANINGFUL",
                "LATIN_PLAIN",
            ):
                contrast = paired_contrast(
                    reference,
                    all_scores[representation_name][corpus_name],
                    metric,
                    args.bootstrap,
                    f"contrast:{metric}:{representation_name}:{corpus_name}",
                )
                comparisons.append((representation_name, corpus_name, contrast))
        correction = len(comparisons)
        for representation_name, corpus_name, contrast in comparisons:
            mean, lower, upper, p_value = contrast
            emit(
                f"  {representation_name:14s} {corpus_name:18s} "
                f"{mean:+9.5f} [{lower:+9.5f}, {upper:+9.5f}] "
                f"{p_value:9.5f} {min(1.0, p_value * correction):9.5f}"
            )
        emit(f"  Bonferroni family={correction} contrasts within this metric")

    emit()
    emit("=" * 88)
    emit("VOYNICH-SELECTED EVA AFFIX SENSITIVITY")
    emit("=" * 88)
    emit(
        "These classes were hand-specified for EVA and selected on Voynich; "
        "they are not a fair language-universal feature map."
    )
    emit(
        "Unlike the generic primary arm, uncertain and one-character Voynich "
        "tokens remain fixed adjacency breaks, and controls use the same break slots."
    )
    emit(
        "The fixed EVA inventory is not substitution-invariant; "
        "LATIN_SIMPLE_SUB is descriptive here, not an invariance check."
    )
    emit(
        f"protocol={len(break_template)} lines, {break_token_count} eligible "
        "tokens; left/right vocabularies fitted separately from train transitions"
    )
    emit()
    emit("EVA-SELECTED REPRESENTATION CARDINALITY")
    emit(f"  {'representation':18s} " + " ".join(
        f"{name:>18s}" for name in break_datasets
    ))
    for representation_name, representation in (
        EVA_SELECTED_REPRESENTATIONS.items()
    ):
        cardinalities = [
            len(
                {
                    representation(word)
                    for line in lines
                    for word in line.words
                    if word is not None
                }
            )
            for lines in break_datasets.values()
        ]
        emit(
            f"  {representation_name:18s} "
            + " ".join(f"{value:18d}" for value in cardinalities)
        )

    for metric, title in (
        (
            "gain",
            "EVA-SELECTED HELD-OUT BIGRAM GAIN OVER POSITION UNIGRAM",
        ),
        (
            "direction",
            "EVA-SELECTED FORWARD VS INTERIOR-REVERSED DIRECTIONALITY",
        ),
    ):
        emit()
        emit(title)
        emit(
            f"  {'representation':18s} {'corpus':18s} "
            f"{'micro':>9s} {'block mean [95% CI]':>31s} {'positive':>8s}"
        )
        for representation_name in EVA_SELECTED_REPRESENTATIONS:
            for corpus_name in break_datasets:
                scores = eva_scores[representation_name][corpus_name]
                summary = format_summary(
                    scores,
                    metric,
                    args.bootstrap,
                    f"eva:{metric}:{representation_name}:{corpus_name}",
                )
                emit(
                    f"  {representation_name:18s} "
                    f"{corpus_name:18s} {summary}"
                )

        emit()
        emit("  PAIRED CONTRASTS: VOYNICH minus substantive control")
        emit(
            f"  {'representation':18s} {'control':18s} "
            f"{'mean difference [95% CI]':>35s} {'raw p':>9s} {'Bonf p':>9s}"
        )
        comparisons = []
        for representation_name in EVA_SELECTED_REPRESENTATIONS:
            reference = eva_scores[representation_name]["VOYNICH"]
            for corpus_name in ("NAIBBE_MEANINGFUL", "LATIN_PLAIN"):
                contrast = paired_contrast(
                    reference,
                    eva_scores[representation_name][corpus_name],
                    metric,
                    args.bootstrap,
                    (
                        f"eva-contrast:{metric}:{representation_name}:"
                        f"{corpus_name}"
                    ),
                )
                comparisons.append((representation_name, corpus_name, contrast))
        correction = len(comparisons)
        for representation_name, corpus_name, contrast in comparisons:
            mean, lower, upper, p_value = contrast
            emit(
                f"  {representation_name:18s} {corpus_name:18s} "
                f"{mean:+9.5f} [{lower:+9.5f}, {upper:+9.5f}] "
                f"{p_value:9.5f} {min(1.0, p_value * correction):9.5f}"
            )
        emit(
            f"  Bonferroni family={correction} selected-feature contrasts "
            "within this metric"
        )

    emit()
    emit("BLOCK-LEVEL SCORES")
    emit(
        "  corpus representation block gain(bits/transition) "
        "direction(bits/transition) gain_n direction_n"
    )
    for corpus_name in datasets:
        for representation_name in REPRESENTATIONS:
            scores = all_scores[representation_name][corpus_name]
            for block in blocks:
                score = scores[block]
                emit(
                    f"  {corpus_name} {representation_name} {block} "
                    f"{score.gain:+.7f} {score.direction:+.7f} "
                    f"{score.gain_n} {score.direction_n}"
                )
    for corpus_name in break_datasets:
        for representation_name in EVA_SELECTED_REPRESENTATIONS:
            scores = eva_scores[representation_name][corpus_name]
            for block in blocks:
                score = scores[block]
                emit(
                    f"  {corpus_name} {representation_name} {block} "
                    f"{score.gain:+.7f} {score.direction:+.7f} "
                    f"{score.gain_n} {score.direction_n}"
                )

    emit()
    emit("DESIGN LIMITATIONS")
    emit(
        "  1. Prefix/suffix are generic two-character boundaries, not learned "
        "morphemes or the EVA-specific inventory used by sequence_parallel.py."
    )
    emit(
        "  2. The exact Book 16 input contains only about 9,200 ordinary words. "
        "LATIN_PLAIN therefore starts at the last copy of its incipit in the "
        "larger published Pliny aggregate and continues into later material."
    )
    emit(
        "  3. Control line and quire boundaries are artificial reflows through "
        "the Voynich template; they do not preserve source sentence/page boundaries."
    )
    emit(
        "  4. Naibbe's 3% space-removal output is not token-aligned one-for-one "
        "with the pre-encryption stream, although both derive from Pliny Book 16."
    )
    emit(
        "  5. The sixteen quire blocks are heterogeneous and unequal in size; "
        "the block mean/CI is primary for uncertainty, while micro scores are descriptive."
    )
    emit(
        "  6. A positive score establishes reusable local order, not semantics: "
        "cipher rules, copying, formulae, or language can all generate it."
    )
    emit(
        "  7. LATIN_SIMPLE_SUB is an implementation control: a bijective "
        "substitution must preserve every generic representation score exactly. "
        "The Voynich-selected EVA sensitivity is intentionally not invariant."
    )
    emit(
        "  8. The generic primary arm deletes invalid tokens before scoring and "
        "can bridge their former positions. The EVA-selected sensitivity follows "
        "morph_voynich_transfer.py by retaining those positions as adjacency breaks."
    )
    return "\n".join(output) + "\n"


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report, end="")
    print(f"report written: {args.output}")


if __name__ == "__main__":
    main()
