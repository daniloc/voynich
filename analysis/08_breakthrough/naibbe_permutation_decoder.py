#!/usr/bin/env python3
"""
Blind known-family attack on Naibbe's table-constrained homophonic cipher.

The published glyph inventory identifies six tables and three emission roles.
Within each role/table block, however, this attack treats the 23 glyph values as
an unknown permutation.  No published glyph-to-letter value is loaded during
fitting.

Development is deliberately synthetic.  Random permutations are applied to the
published glyph surfaces, and a held-out Latin passage is encrypted under those
keys.  ``--mode dev`` cannot open the official ciphertext.  ``--mode gate``
runs the same frozen synthetic checks first, then fits the official untouched
(unrespaced) Naibbe control and reads its plaintext only after the key is frozen.

The search is independent of the repository's unconstrained homophonic
hillclimber:

1. rank-frequency initialization inside each 23-glyph permutation block;
2. a context-rank assignment initializer solved with the Hungarian algorithm;
3. deterministic whole-block assignment and exact permutation-swap refinement
   under a held-out Latin tetragram model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
LATIN_CONTROL = ROOT / "data" / "controls" / "latin.txt"
NAIBBE_ROOT = Path("/tmp/naibbe-cipher")
TABLE_CSV = NAIBBE_ROOT / "references" / "naibbe_tables.csv"
OFFICIAL_CIPHER = (
    NAIBBE_ROOT / "encrypted" / "nathist_output_ciphertext.txt"
)
OFFICIAL_PLAIN = (
    NAIBBE_ROOT
    / "respaced_plaintext"
    / "nathist_pre_encryption_respaced_plaintext.txt"
)

ALPHABET = "abcdefghilmnopqrstuvxyz"
TABLES = ("alpha", "beta1", "beta2", "beta3", "gamma1", "gamma2")
ROLES = ("unigram", "prefix", "suffix")
TABLE_WEIGHTS = dict(zip(TABLES, (20, 8, 8, 8, 4, 4)))
DEV_SEEDS = (314159, 271828, 161803)
LM_TRAIN_CHARS = 52_000
DEV_PLAIN_CHARS = 22_000
CONTEXT_WEIGHT = 0.55
BLOCK_PASSES = 5
SWAP_PASSES = 4
SMOOTHING = 0.08


@dataclass(frozen=True)
class Inventory:
    """Structural inventory only: role/table labels and glyph surfaces."""

    blocks: tuple[tuple[str, str], ...]
    glyphs: tuple[tuple[str, ...], ...]
    block_index: dict[tuple[str, str], int]
    glyph_locations: dict[tuple[str, str], tuple[tuple[int, int], ...]]
    unigram_surfaces: frozenset[str]
    prefix_surfaces: frozenset[str]
    suffix_surfaces: frozenset[str]


@dataclass
class CipherStream:
    block_at: list[int]
    symbol_at: list[int]
    token_count: int
    parsed_token_count: int
    ambiguous_bigram_count: int


@dataclass
class FitResult:
    key: list[list[int]]
    decoded: list[int]
    init_score: float
    final_score: float
    block_updates: int
    swap_updates: int


class TetragramLM:
    def __init__(self, text: str, alpha: float = SMOOTHING):
        self.width = len(ALPHABET)
        self.char_id = {char: i for i, char in enumerate(ALPHABET)}
        ids = [self.char_id[c] for c in text]
        self.unigrams = Counter(ids)
        self.bigrams = Counter(zip(ids, ids[1:]))
        self.trigrams = Counter(zip(ids, ids[1:], ids[2:]))
        self.quads = Counter(zip(ids, ids[1:], ids[2:], ids[3:]))
        self.alpha = alpha
        total = sum(self.unigrams.values())
        self.letter_prob = [
            (self.unigrams[i] + alpha) / (total + alpha * self.width)
            for i in range(self.width)
        ]
        self.frequency_order = sorted(
            range(self.width), key=lambda i: (-self.unigrams[i], i)
        )
        self.frequency_rank = {
            letter: rank for rank, letter in enumerate(self.frequency_order)
        }
        self.rank_left_logp, self.rank_right_logp = self._rank_contexts(ids)

    def _rank_contexts(
        self, ids: Sequence[int]
    ) -> tuple[list[list[float]], list[list[float]]]:
        left = [[0 for _ in ALPHABET] for _ in ALPHABET]
        right = [[0 for _ in ALPHABET] for _ in ALPHABET]
        for a, b in zip(ids, ids[1:]):
            right[a][self.frequency_rank[b]] += 1
            left[b][self.frequency_rank[a]] += 1

        def normalize(rows: list[list[int]]) -> list[list[float]]:
            result = []
            for row in rows:
                denominator = sum(row) + self.alpha * self.width
                result.append(
                    [
                        math.log((count + self.alpha) / denominator)
                        for count in row
                    ]
                )
            return result

        return normalize(left), normalize(right)

    def logp4(self, a: int, b: int, c: int, d: int) -> float:
        denominator = self.trigrams[(a, b, c)] + self.alpha * self.width
        return math.log((self.quads[(a, b, c, d)] + self.alpha) / denominator)

    def score(self, decoded: Sequence[int]) -> float:
        if len(decoded) < 4:
            return float("-inf")
        return sum(
            self.logp4(a, b, c, d)
            for a, b, c, d in zip(
                decoded, decoded[1:], decoded[2:], decoded[3:]
            )
        )

    def mean_score(self, decoded: Sequence[int]) -> float:
        return self.score(decoded) / max(1, len(decoded) - 3)


def clean_latin(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    normalized = "".join(
        char
        for char in normalized
        if unicodedata.category(char) != "Mn"
    )
    normalized = (
        normalized.replace("æ", "ae")
        .replace("œ", "oe")
        .replace("ð", "d")
        .replace("þ", "th")
        .replace("ł", "l")
        .replace("ß", "ss")
        .replace("ø", "o")
        .replace("j", "i")
        .replace("k", "c")
        .replace("w", "uu")
    )
    return "".join(char for char in normalized if char in ALPHABET)


def load_control_segments() -> tuple[str, str]:
    text = LATIN_CONTROL.read_text(encoding="utf-8", errors="ignore")
    start = re.search(r"\*\*\* start of.*?\*\*\*", text, re.I | re.S)
    if start:
        text = text[start.end() :]
    end = re.search(r"\*\*\* end of", text, re.I | re.S)
    if end:
        text = text[: end.start()]
    cleaned = clean_latin(text)
    if len(cleaned) < LM_TRAIN_CHARS + DEV_PLAIN_CHARS:
        raise RuntimeError(
            f"Latin control has only {len(cleaned)} normalized characters"
        )
    return (
        cleaned[:LM_TRAIN_CHARS],
        cleaned[-DEV_PLAIN_CHARS:],
    )


def load_structural_inventory(path: Path = TABLE_CSV) -> Inventory:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            # Intentionally discard the final field: it is the secret value.
            role, table, _discarded_value = row["code"].split("_")
            if role not in ROLES or table not in TABLES:
                raise ValueError(f"Unexpected structural code {row['code']!r}")
            grouped[(role, table)].append(row["glyphs"])

    blocks = tuple((role, table) for role in ROLES for table in TABLES)
    # CSV rows are ordered by their plaintext value.  Surface sorting severs
    # that otherwise complete side channel before symbol IDs are assigned.
    glyphs = tuple(tuple(sorted(grouped[block])) for block in blocks)
    if any(len(set(row)) != len(ALPHABET) for row in glyphs):
        raise RuntimeError("Each role/table block must contain 23 unique glyphs")
    block_index = {block: i for i, block in enumerate(blocks)}
    locations: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for block_id, ((role, _table), surfaces) in enumerate(zip(blocks, glyphs)):
        for symbol_id, surface in enumerate(surfaces):
            locations[(role, surface)].append((block_id, symbol_id))
    return Inventory(
        blocks=blocks,
        glyphs=glyphs,
        block_index=block_index,
        glyph_locations={
            key: tuple(value) for key, value in locations.items()
        },
        unigram_surfaces=frozenset(
            surface
            for (role, surface), _ in locations.items()
            if role == "unigram"
        ),
        prefix_surfaces=frozenset(
            surface
            for (role, surface), _ in locations.items()
            if role == "prefix"
        ),
        suffix_surfaces=frozenset(
            surface
            for (role, surface), _ in locations.items()
            if role == "suffix"
        ),
    )


def make_random_key(
    inventory: Inventory, rng: random.Random
) -> tuple[list[dict[str, str]], list[list[int]]]:
    encryption_key: list[dict[str, str]] = []
    attack_truth: list[list[int]] = []
    letter_id = {letter: i for i, letter in enumerate(ALPHABET)}
    for surfaces in inventory.glyphs:
        shuffled = list(surfaces)
        rng.shuffle(shuffled)
        encryption_key.append(dict(zip(ALPHABET, shuffled)))
        surface_value = {
            surface: letter_id[letter]
            for letter, surface in zip(ALPHABET, shuffled)
        }
        attack_truth.append([surface_value[surface] for surface in surfaces])
    return encryption_key, attack_truth


def shuffled_deck(rng: random.Random) -> list[str]:
    deck = [
        table
        for table in TABLES
        for _ in range(TABLE_WEIGHTS[table])
    ]
    rng.shuffle(deck)
    return deck


def encrypt_synthetic(
    plaintext: str,
    inventory: Inventory,
    encryption_key: list[dict[str, str]],
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    tokens: list[str] = []
    true_units: list[str] = []
    deck = shuffled_deck(rng)
    deck_index = 0

    def draw_table() -> str:
        nonlocal deck, deck_index
        if deck_index == len(deck):
            deck = shuffled_deck(rng)
            deck_index = 0
        result = deck[deck_index]
        deck_index += 1
        return result

    index = 0
    while index < len(plaintext):
        if index == len(plaintext) - 1 or rng.random() < 17 / 36:
            table = draw_table()
            block = inventory.block_index[("unigram", table)]
            tokens.append(encryption_key[block][plaintext[index]])
            true_units.append(plaintext[index])
            index += 1
            continue

        while True:
            prefix_table = draw_table()
            suffix_table = draw_table()
            prefix_block = inventory.block_index[("prefix", prefix_table)]
            suffix_block = inventory.block_index[("suffix", suffix_table)]
            prefix = encryption_key[prefix_block][plaintext[index]]
            suffix = encryption_key[suffix_block][plaintext[index + 1]]
            if prefix + suffix not in inventory.unigram_surfaces:
                tokens.append(prefix + suffix)
                true_units.append(plaintext[index : index + 2])
                index += 2
                break
    return tokens, true_units


def choose_location(
    inventory: Inventory, role: str, glyph: str
) -> tuple[int, int] | None:
    locations = inventory.glyph_locations.get((role, glyph), ())
    if not locations:
        return None
    # Only one unigram surface is duplicated.  Keeping this deterministic
    # exposes that irreducible surface collision instead of using secret values.
    return locations[0]


def parse_tokens(
    tokens: Iterable[str],
    inventory: Inventory,
    true_units: Sequence[str] | None = None,
) -> tuple[CipherStream, dict[str, float | None]]:
    block_at: list[int] = []
    symbol_at: list[int] = []
    parsed = 0
    ambiguous = 0
    structural_correct = 0
    structural_known = 0

    for token_index, token in enumerate(tokens):
        emissions: list[tuple[int, int]] = []
        if token in inventory.unigram_surfaces:
            location = choose_location(inventory, "unigram", token)
            if location:
                emissions = [location]
        else:
            candidates: list[tuple[str, str]] = []
            for split in range(1, len(token)):
                prefix, suffix = token[:split], token[split:]
                if (
                    prefix in inventory.prefix_surfaces
                    and suffix in inventory.suffix_surfaces
                ):
                    candidates.append((prefix, suffix))
            if candidates:
                if len(candidates) > 1:
                    ambiguous += 1
                # Synthetic calibration puts this above 99.5%; it is fixed
                # before the official gate.
                prefix, suffix = max(candidates, key=lambda pair: len(pair[0]))
                left = choose_location(inventory, "prefix", prefix)
                right = choose_location(inventory, "suffix", suffix)
                if left and right:
                    emissions = [left, right]
        if emissions:
            parsed += 1
            block_at.extend(block for block, _symbol in emissions)
            symbol_at.extend(symbol for _block, symbol in emissions)

        if true_units is not None:
            structural_known += 1
            expected = len(true_units[token_index])
            structural_correct += int(len(emissions) == expected)

    calibration = {
        "token_coverage": parsed / max(1, token_index + 1),
        "structural_length_accuracy": (
            structural_correct / structural_known if structural_known else None
        ),
    }
    return (
        CipherStream(
            block_at=block_at,
            symbol_at=symbol_at,
            token_count=token_index + 1 if "token_index" in locals() else 0,
            parsed_token_count=parsed,
            ambiguous_bigram_count=ambiguous,
        ),
        calibration,
    )


def hungarian_max(weights: Sequence[Sequence[float]]) -> list[int]:
    """Return the maximum-weight column assigned to each row."""

    n = len(weights)
    maximum = max(max(row) for row in weights)
    costs = [[maximum - value for value in row] for row in weights]
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for row in range(1, n + 1):
        p[0] = row
        column0 = 0
        minimum = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, n + 1):
                if used[column]:
                    continue
                current = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(n + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [0] * n
    for column in range(1, n + 1):
        assignment[p[column] - 1] = column - 1
    return assignment


def symbol_counts(stream: CipherStream, block_count: int) -> list[list[int]]:
    counts = [[0 for _ in ALPHABET] for _ in range(block_count)]
    for block, symbol in zip(stream.block_at, stream.symbol_at):
        counts[block][symbol] += 1
    return counts


def rank_symbols(counts: Sequence[Sequence[int]]) -> list[list[int]]:
    ranks = []
    for row in counts:
        order = sorted(range(len(ALPHABET)), key=lambda i: (-row[i], i))
        rank = [0] * len(ALPHABET)
        for position, symbol in enumerate(order):
            rank[symbol] = position
        ranks.append(rank)
    return ranks


def context_initializer(
    stream: CipherStream,
    lm: TetragramLM,
    block_count: int,
) -> list[list[int]]:
    counts = symbol_counts(stream, block_count)
    ranks = rank_symbols(counts)
    left_rank_counts = [
        [[0 for _ in ALPHABET] for _ in ALPHABET]
        for _ in range(block_count)
    ]
    right_rank_counts = [
        [[0 for _ in ALPHABET] for _ in ALPHABET]
        for _ in range(block_count)
    ]
    for position, (block, symbol) in enumerate(
        zip(stream.block_at, stream.symbol_at)
    ):
        if position:
            neighbor_block = stream.block_at[position - 1]
            neighbor_symbol = stream.symbol_at[position - 1]
            left_rank_counts[block][symbol][
                ranks[neighbor_block][neighbor_symbol]
            ] += 1
        if position + 1 < len(stream.block_at):
            neighbor_block = stream.block_at[position + 1]
            neighbor_symbol = stream.symbol_at[position + 1]
            right_rank_counts[block][symbol][
                ranks[neighbor_block][neighbor_symbol]
            ] += 1

    key = []
    for block in range(block_count):
        weights = []
        for symbol in range(len(ALPHABET)):
            row = []
            for letter in range(len(ALPHABET)):
                frequency_score = counts[block][symbol] * math.log(
                    lm.letter_prob[letter]
                )
                context_score = 0.0
                for rank in range(len(ALPHABET)):
                    context_score += (
                        left_rank_counts[block][symbol][rank]
                        * lm.rank_left_logp[letter][rank]
                    )
                    context_score += (
                        right_rank_counts[block][symbol][rank]
                        * lm.rank_right_logp[letter][rank]
                    )
                row.append(
                    frequency_score + CONTEXT_WEIGHT * context_score
                )
            weights.append(row)
        key.append(hungarian_max(weights))
    return key


def decode(stream: CipherStream, key: Sequence[Sequence[int]]) -> list[int]:
    return [
        key[block][symbol]
        for block, symbol in zip(stream.block_at, stream.symbol_at)
    ]


def slice_stream(
    stream: CipherStream, start: int, stop: int
) -> CipherStream:
    return CipherStream(
        block_at=stream.block_at[start:stop],
        symbol_at=stream.symbol_at[start:stop],
        token_count=stream.token_count,
        parsed_token_count=stream.parsed_token_count,
        ambiguous_bigram_count=stream.ambiguous_bigram_count,
    )


def affected_starts(
    stream: CipherStream, block_count: int
) -> list[list[tuple[int, ...]]]:
    affected = [
        [set() for _ in ALPHABET]
        for _ in range(block_count)
    ]
    last_start = len(stream.block_at) - 4
    for position, (block, symbol) in enumerate(
        zip(stream.block_at, stream.symbol_at)
    ):
        for start in range(max(0, position - 3), min(position, last_start) + 1):
            affected[block][symbol].add(start)
    return [
        [tuple(sorted(starts)) for starts in row]
        for row in affected
    ]


def symbol_positions(
    stream: CipherStream, block_count: int
) -> list[list[tuple[int, ...]]]:
    positions = [
        [[] for _ in ALPHABET]
        for _ in range(block_count)
    ]
    for position, (block, symbol) in enumerate(
        zip(stream.block_at, stream.symbol_at)
    ):
        positions[block][symbol].append(position)
    return [
        [tuple(symbol_row) for symbol_row in block_row]
        for block_row in positions
    ]


def rows_score(
    decoded: Sequence[int],
    starts: Iterable[int],
    lm: TetragramLM,
) -> float:
    return sum(
        lm.logp4(
            decoded[start],
            decoded[start + 1],
            decoded[start + 2],
            decoded[start + 3],
        )
        for start in starts
    )


def replace_block(
    stream: CipherStream,
    decoded: list[int],
    block: int,
    new_row: Sequence[int],
) -> None:
    for position, (position_block, symbol) in enumerate(
        zip(stream.block_at, stream.symbol_at)
    ):
        if position_block == block:
            decoded[position] = new_row[symbol]


def block_assignment_proposal(
    stream: CipherStream,
    key: Sequence[Sequence[int]],
    decoded: list[int],
    affected: Sequence[Sequence[Sequence[int]]],
    positions: Sequence[Sequence[Sequence[int]]],
    block: int,
    lm: TetragramLM,
) -> list[int]:
    weights = []
    current_row = key[block]
    for symbol in range(len(ALPHABET)):
        starts = affected[block][symbol]
        baseline = rows_score(decoded, starts, lm)
        row = []
        symbol_positions = positions[block][symbol]
        original = current_row[symbol]
        for letter in range(len(ALPHABET)):
            if letter == original:
                row.append(0.0)
                continue
            for position in symbol_positions:
                decoded[position] = letter
            row.append(rows_score(decoded, starts, lm) - baseline)
            for position in symbol_positions:
                decoded[position] = original
        weights.append(row)
    return hungarian_max(weights)


def refine_key(
    stream: CipherStream,
    lm: TetragramLM,
    initial_key: list[list[int]],
) -> FitResult:
    key = [row[:] for row in initial_key]
    decoded = decode(stream, key)
    affected = affected_starts(stream, len(key))
    positions = symbol_positions(stream, len(key))
    score = lm.score(decoded)
    init_score = score
    block_updates = 0
    swap_updates = 0

    for _pass in range(BLOCK_PASSES):
        changed = False
        for block in range(len(key)):
            proposal = block_assignment_proposal(
                stream, key, decoded, affected, positions, block, lm
            )
            if proposal == key[block]:
                continue
            old_row = key[block][:]
            old_score = score
            replace_block(stream, decoded, block, proposal)
            candidate_score = lm.score(decoded)
            if candidate_score > score + 1e-8:
                key[block] = proposal
                score = candidate_score
                block_updates += 1
                changed = True
            else:
                replace_block(stream, decoded, block, old_row)
                score = old_score

        for _swap_pass in range(SWAP_PASSES):
            swap_changed = False
            for block in range(len(key)):
                best_delta = 1e-8
                best_pair: tuple[int, int] | None = None
                best_starts: tuple[int, ...] = ()
                for first in range(len(ALPHABET) - 1):
                    for second in range(first + 1, len(ALPHABET)):
                        starts = tuple(
                            sorted(
                                set(affected[block][first])
                                | set(affected[block][second])
                            )
                        )
                        baseline = rows_score(decoded, starts, lm)
                        first_positions = positions[block][first]
                        second_positions = positions[block][second]
                        first_value, second_value = (
                            key[block][first],
                            key[block][second],
                        )
                        for position in first_positions:
                            decoded[position] = second_value
                        for position in second_positions:
                            decoded[position] = first_value
                        delta = rows_score(decoded, starts, lm) - baseline
                        for position in first_positions:
                            decoded[position] = first_value
                        for position in second_positions:
                            decoded[position] = second_value
                        if delta > best_delta:
                            best_delta = delta
                            best_pair = (first, second)
                            best_starts = starts
                if best_pair:
                    first, second = best_pair
                    before = rows_score(decoded, best_starts, lm)
                    key[block][first], key[block][second] = (
                        key[block][second],
                        key[block][first],
                    )
                    replace_block(stream, decoded, block, key[block])
                    after = rows_score(decoded, best_starts, lm)
                    score += after - before
                    swap_updates += 1
                    swap_changed = True
                    changed = True
            if not swap_changed:
                break
        if not changed:
            break

    # Recompute to prevent incremental roundoff from entering reported metrics.
    score = lm.score(decoded)
    return FitResult(
        key=key,
        decoded=decoded,
        init_score=init_score,
        final_score=score,
        block_updates=block_updates,
        swap_updates=swap_updates,
    )


def ids_to_text(ids: Sequence[int]) -> str:
    return "".join(ALPHABET[index] for index in ids)


def key_as_mapping(
    inventory: Inventory, key: Sequence[Sequence[int]]
) -> dict[str, dict[str, str]]:
    result = {}
    for (role, table), surfaces, values in zip(
        inventory.blocks, inventory.glyphs, key
    ):
        result[f"{role}:{table}"] = {
            surface: ALPHABET[value]
            for surface, value in zip(surfaces, values)
        }
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accuracy(decoded: Sequence[int], truth: str) -> float:
    truth_ids = [ALPHABET.index(char) for char in truth]
    limit = min(len(decoded), len(truth_ids))
    return (
        sum(decoded[i] == truth_ids[i] for i in range(limit))
        / max(len(decoded), len(truth_ids), 1)
    )


def weighted_key_accuracy(
    fit_key: Sequence[Sequence[int]],
    true_key: Sequence[Sequence[int]],
    stream: CipherStream,
) -> float:
    correct = 0
    for block, symbol in zip(stream.block_at, stream.symbol_at):
        correct += fit_key[block][symbol] == true_key[block][symbol]
    return correct / max(1, len(stream.block_at))


def run_synthetic_case(
    seed: int,
    plaintext: str,
    inventory: Inventory,
    lm: TetragramLM,
) -> dict[str, object]:
    rng = random.Random(seed)
    encryption_key, true_key = make_random_key(inventory, rng)
    tokens, units = encrypt_synthetic(
        plaintext, inventory, encryption_key, rng
    )
    stream, parsing = parse_tokens(tokens, inventory, units)
    initial_key = context_initializer(stream, lm, len(inventory.blocks))
    initial_decoded = decode(stream, initial_key)
    result = refine_key(stream, lm, initial_key)
    return {
        "seed": seed,
        "tokens": len(tokens),
        "emissions": len(stream.block_at),
        "parse": parsing,
        "ambiguous_bigram_tokens": stream.ambiguous_bigram_count,
        "initial_char_accuracy": accuracy(initial_decoded, plaintext),
        "final_char_accuracy": accuracy(result.decoded, plaintext),
        "weighted_key_accuracy": weighted_key_accuracy(
            result.key, true_key, stream
        ),
        "initial_lm": result.init_score / max(1, len(result.decoded) - 3),
        "final_lm": result.final_score / max(1, len(result.decoded) - 3),
        "truth_lm": lm.mean_score(
            [ALPHABET.index(char) for char in plaintext]
        ),
        "block_updates": result.block_updates,
        "swap_updates": result.swap_updates,
        "sample": ids_to_text(result.decoded[:240]),
    }


def load_official_tokens() -> list[str]:
    # This function is called only from the explicit gate path.
    return OFFICIAL_CIPHER.read_text(encoding="utf-8").split()


def load_official_truth() -> str:
    # Deliberately delayed until after the official key has been frozen.
    return clean_latin(OFFICIAL_PLAIN.read_text(encoding="utf-8"))


def run_official_gate(
    inventory: Inventory, lm: TetragramLM
) -> dict[str, object]:
    tokens = load_official_tokens()
    stream, parsing = parse_tokens(tokens, inventory)
    initial_key = context_initializer(stream, lm, len(inventory.blocks))
    initial_decoded = decode(stream, initial_key)
    result = refine_key(stream, lm, initial_key)

    # Freeze two directionally independent train-half keys before evaluation.
    split = len(stream.block_at) // 2
    first = slice_stream(stream, 0, split)
    second = slice_stream(stream, split, len(stream.block_at))
    first_initial = context_initializer(first, lm, len(inventory.blocks))
    second_initial = context_initializer(second, lm, len(inventory.blocks))
    first_result = refine_key(first, lm, first_initial)
    second_result = refine_key(second, lm, second_initial)

    # Every fitted key and decoded stream is frozen.  Only evaluation follows.
    truth = load_official_truth()
    identity_key = [list(range(len(ALPHABET))) for _ in inventory.blocks]
    first_truth = truth[:split]
    second_truth = truth[split:]
    return {
        "asset": str(OFFICIAL_CIPHER),
        "asset_sha256": sha256(OFFICIAL_CIPHER),
        "table_sha256": sha256(TABLE_CSV),
        "plaintext_sha256": sha256(OFFICIAL_PLAIN),
        "tokens": len(tokens),
        "emissions": len(stream.block_at),
        "parse": parsing,
        "ambiguous_bigram_tokens": stream.ambiguous_bigram_count,
        "surface_order": "lexicographic within role/table; CSV row order discarded",
        "identity_row_accuracy": accuracy(decode(stream, identity_key), truth),
        "initial_char_accuracy": accuracy(initial_decoded, truth),
        "final_char_accuracy": accuracy(result.decoded, truth),
        "heldout_halves": {
            "fit_first_test_second": {
                "train_accuracy": accuracy(first_result.decoded, first_truth),
                "test_accuracy": accuracy(
                    decode(second, first_result.key), second_truth
                ),
            },
            "fit_second_test_first": {
                "train_accuracy": accuracy(second_result.decoded, second_truth),
                "test_accuracy": accuracy(
                    decode(first, second_result.key), first_truth
                ),
            },
        },
        "initial_lm": result.init_score / max(1, len(result.decoded) - 3),
        "final_lm": result.final_score / max(1, len(result.decoded) - 3),
        "truth_lm": lm.mean_score(
            [ALPHABET.index(char) for char in truth]
        ),
        "block_updates": result.block_updates,
        "swap_updates": result.swap_updates,
        "sample": ids_to_text(result.decoded[:400]),
        "truth_length": len(truth),
        "recovered_key": key_as_mapping(inventory, result.key),
        "gate_policy": (
            "official plaintext loaded only after full and half-stream keys "
            "were frozen"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("dev", "gate"),
        default="dev",
        help="dev is synthetic-only; gate opens the official asset once",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path",
    )
    args = parser.parse_args()

    lm_text, dev_plaintext = load_control_segments()
    lm = TetragramLM(lm_text)
    inventory = load_structural_inventory()
    report: dict[str, object] = {
        "method": "table-constrained permutation decoder",
        "alphabet": ALPHABET,
        "lm_train_chars": len(lm_text),
        "dev_plain_chars": len(dev_plaintext),
        "fixed_parameters": {
            "context_weight": CONTEXT_WEIGHT,
            "block_passes": BLOCK_PASSES,
            "swap_passes": SWAP_PASSES,
            "smoothing": SMOOTHING,
        },
        "synthetic": [
            run_synthetic_case(seed, dev_plaintext, inventory, lm)
            for seed in DEV_SEEDS
        ],
    }
    if args.mode == "gate":
        report["official_gate"] = run_official_gate(inventory, lm)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
