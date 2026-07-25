#!/usr/bin/env python3
"""
Synthetic-trained recurrence decoder for Naibbe-style homophonic ciphers.

This is an intentionally gated experiment:

1. Train a causal Transformer only on Latin encrypted with fresh random keys.
2. Recover a persistent key for a reserved synthetic stream.
3. Recover the untouched official Naibbe key from ciphertext alone.
4. Only if both gates pass, freeze the model and score Voynich.

Cipher symbols are replaced by first-occurrence integers independently in each
window.  Raw glyph identity and the plaintext-bearing suffixes in the official
Naibbe table never enter the model.  The only Naibbe-specific side information
is the observable emission role: unigram, compound-left, or compound-right.

The architecture follows the central setup in Kambhatla, Born, and Sarkar
(EACL 2023): a causal language model first reproduces the recurrence-encoded
ciphertext, then generates its plaintext.  This local version is deliberately
smaller and adds persistent-key voting so long ciphers can be decoded from
overlapping fixed-length windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised by CLI error path
    raise SystemExit(
        "PyTorch is required. Run with .venv/bin/python after installing "
        "the repo-local ML environment."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
FOLLOWUPS = ROOT / "analysis" / "07_followups"
sys.path.insert(0, str(FOLLOWUPS))

from naibbe_style_attack import (  # noqa: E402
    SegmentModel,
    calibrate,
    structural_counts,
)
from residual_cipher_solver import (  # noqa: E402
    EmissionLine,
    LanguageModel,
    RawLine,
    block_folds,
    load_naibbe_raw,
    load_voynich,
    normalize_latin,
    position_bucket,
    table_maps,
    valid_runs,
)


SEED = 20260723
ALPHABET = tuple("abcdefghiklmnopqrstuxyz")
CHAR_TO_ID = {character: index for index, character in enumerate(ALPHABET)}
TABLES = ("alpha", "beta1", "beta2", "beta3", "gamma1", "gamma2")
TABLE_WEIGHTS = np.array((28, 14, 11, 11, 7, 7), dtype=np.float64)
TABLE_WEIGHTS /= TABLE_WEIGHTS.sum()
ROLES = {"U": 1, "L": 2, "R": 3}
ROLE_NAMES = {value: key for key, value in ROLES.items()}


@dataclass(frozen=True)
class CipherStream:
    symbols: tuple[str, ...]
    roles: tuple[int, ...]
    truth: tuple[Optional[str], ...]
    blocks: tuple[str, ...]
    slots: tuple[str, ...]


@dataclass
class GateResult:
    name: str
    passed: bool
    threshold: float
    accuracy: float
    covered_accuracy: float
    coverage: float
    known: int
    mapped: int
    key_symbols: int
    sample: str


@dataclass
class ResidualResult:
    fold: int
    fit_blocks: int
    validation_blocks: int
    test_blocks: int
    lambda_: float
    validation_gain: float
    test_gain: float
    mapped_coverage: float
    sample: str


@dataclass(frozen=True)
class TokenLayout:
    max_recurrences: int

    @property
    def pad(self) -> int:
        return 0

    @property
    def bos(self) -> int:
        return 1

    @property
    def rec_base(self) -> int:
        return 2

    @property
    def sep(self) -> int:
        return self.rec_base + self.max_recurrences

    @property
    def char_base(self) -> int:
        return self.sep + 1

    @property
    def eos(self) -> int:
        return self.char_base + len(ALPHABET)

    @property
    def vocab_size(self) -> int:
        return self.eos + 1

    def recurrence_token(self, value: int) -> int:
        if not 0 <= value < self.max_recurrences:
            raise ValueError(
                f"recurrence id {value} exceeds maximum "
                f"{self.max_recurrences}"
            )
        return self.rec_base + value

    def character_token(self, character: str) -> int:
        return self.char_base + CHAR_TO_ID[character]

    def token_character(self, token: int) -> str:
        index = token - self.char_base
        if not 0 <= index < len(ALPHABET):
            raise ValueError(f"token {token} is not a character token")
        return ALPHABET[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    return device


def recurrence_encode(symbols: Sequence[str]) -> list[int]:
    lookup: dict[str, int] = {}
    encoded = []
    for symbol in symbols:
        if symbol not in lookup:
            lookup[symbol] = len(lookup)
        encoded.append(lookup[symbol])
    return encoded


class Deck:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.cards: list[int] = []
        self.index = 0

    def draw(self) -> int:
        if self.index >= len(self.cards):
            self.cards = [
                table
                for table, count in enumerate((28, 14, 11, 11, 7, 7))
                for _ in range(count)
            ]
            self.rng.shuffle(self.cards)
            self.index = 0
        result = self.cards[self.index]
        self.index += 1
        return result


def synthetic_stream(
    plaintext: str,
    rng: np.random.Generator,
    key: Optional[np.ndarray] = None,
    block_size: int = 1024,
) -> tuple[CipherStream, np.ndarray]:
    """Encrypt normalized text with 18 independently permuted channels."""
    if key is None:
        key = np.stack(
            [rng.permutation(len(ALPHABET)) for _ in range(18)]
        )
    deck = Deck(rng)
    symbols: list[str] = []
    roles: list[int] = []
    truth: list[str] = []
    blocks: list[str] = []
    slots: list[str] = []
    index = 0
    while index < len(plaintext):
        if index == len(plaintext) - 1 or rng.random() < 17 / 36:
            role_sequence = ("U",)
        else:
            role_sequence = ("L", "R")
        for role_name in role_sequence:
            character = plaintext[index]
            letter = CHAR_TO_ID[character]
            role_offset = ROLES[role_name] - 1
            table = deck.draw()
            channel = role_offset * len(TABLES) + table
            slot = int(key[channel, letter])
            symbols.append(f"{role_name}:{table}:{slot}")
            roles.append(ROLES[role_name])
            truth.append(character)
            blocks.append(f"S{index // block_size:03d}")
            slots.append(f"{role_name}@stream")
            index += 1
    return (
        CipherStream(
            tuple(symbols),
            tuple(roles),
            tuple(truth),
            tuple(blocks),
            tuple(slots),
        ),
        key,
    )


def random_training_batch(
    corpus: str,
    batch_size: int,
    sequence_length: int,
    layout: TokenLayout,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = []
    role_sequences = []
    for _ in range(batch_size):
        start = int(rng.integers(0, len(corpus) - sequence_length))
        plaintext = corpus[start : start + sequence_length]
        stream, _ = synthetic_stream(plaintext, rng)
        recurrence = recurrence_encode(stream.symbols)
        source = [layout.recurrence_token(value) for value in recurrence]
        target = [layout.character_token(char) for char in plaintext]
        tokens = [layout.bos, *source, layout.sep, *target, layout.eos]
        roles = [0, *stream.roles, 0, *([0] * (len(target) + 1))]
        sequences.append(tokens)
        role_sequences.append(roles)
    token_tensor = torch.tensor(sequences, dtype=torch.long)
    role_tensor = torch.tensor(role_sequences, dtype=torch.long)
    return token_tensor[:, :-1], role_tensor[:, :-1], token_tensor[:, 1:]


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        if width % heads:
            raise ValueError("model width must be divisible by attention heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, width * 3)
        self.output = nn.Linear(width, width)
        self.dropout = dropout

    def project(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = values.shape
        qkv = self.qkv(values).view(
            batch, length, 3, self.heads, self.head_width
        )
        query, key, value = qkv.unbind(dim=2)
        return (
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        query, key, value = self.project(values)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        batch, _, length, _ = attended.shape
        joined = attended.transpose(1, 2).reshape(batch, length, self.width)
        return self.output(joined)

    def prefill(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query, key, value = self.project(values)
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )
        batch, _, length, _ = attended.shape
        joined = attended.transpose(1, 2).reshape(batch, length, self.width)
        return self.output(joined), (key, value)

    def step(
        self,
        values: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query, key, value = self.project(values)
        old_key, old_value = cache
        key = torch.cat((old_key, key), dim=2)
        value = torch.cat((old_value, value), dim=2)
        attended = F.scaled_dot_product_attention(query, key, value)
        batch = attended.shape[0]
        joined = attended.transpose(1, 2).reshape(batch, 1, self.width)
        return self.output(joined), (key, value)


class TransformerBlock(nn.Module):
    def __init__(
        self, width: int, heads: int, expansion: int, dropout: float
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(width, heads, dropout)
        self.feed_norm = nn.LayerNorm(width)
        self.feed = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.GELU(),
            nn.Linear(expansion * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.attention(self.attention_norm(values))
        return values + self.feed(self.feed_norm(values))

    def prefill(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, cache = self.attention.prefill(
            self.attention_norm(values)
        )
        values = values + attended
        return values + self.feed(self.feed_norm(values)), cache

    def step(
        self,
        values: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attended, cache = self.attention.step(
            self.attention_norm(values), cache
        )
        values = values + attended
        return values + self.feed(self.feed_norm(values)), cache


class RecurrenceLM(nn.Module):
    def __init__(
        self,
        layout: TokenLayout,
        sequence_length: int,
        width: int,
        layers: int,
        heads: int,
        expansion: int,
        dropout: float,
    ):
        super().__init__()
        self.layout = layout
        self.max_length = 2 * sequence_length + 2
        self.token_embedding = nn.Embedding(layout.vocab_size, width)
        self.position_embedding = nn.Embedding(self.max_length, width)
        self.role_embedding = nn.Embedding(4, width)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(width, heads, expansion, dropout)
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, layout.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def embed(
        self, tokens: torch.Tensor, roles: torch.Tensor, offset: int = 0
    ) -> torch.Tensor:
        positions = torch.arange(
            offset,
            offset + tokens.shape[1],
            device=tokens.device,
        )
        return self.dropout(
            self.token_embedding(tokens)
            + self.position_embedding(positions)[None, :, :]
            + self.role_embedding(roles)
        )

    def forward(
        self, tokens: torch.Tensor, roles: torch.Tensor
    ) -> torch.Tensor:
        values = self.embed(tokens, roles)
        for block in self.blocks:
            values = block(values)
        return self.output(self.norm(values))

    def prefill(
        self, tokens: torch.Tensor, roles: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
        int,
    ]:
        values = self.embed(tokens, roles)
        caches = []
        for block in self.blocks:
            values, cache = block.prefill(values)
            caches.append(cache)
        return self.output(self.norm(values[:, -1:])), caches, tokens.shape[1]

    def step(
        self,
        token: torch.Tensor,
        role: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        position: int,
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
        int,
    ]:
        values = self.embed(token, role, offset=position)
        new_caches = []
        for block, cache in zip(self.blocks, caches):
            values, cache = block.step(values, cache)
            new_caches.append(cache)
        return (
            self.output(self.norm(values)),
            new_caches,
            position + 1,
        )


def target_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    layout: TokenLayout,
    source_weight: float,
) -> tuple[torch.Tensor, float, float]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    losses = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    target_mask = (
        (flat_targets >= layout.char_base) & (flat_targets <= layout.eos)
    )
    source_mask = ~target_mask
    target = losses[target_mask].mean()
    source = losses[source_mask].mean()
    return (
        target + source_weight * source,
        float(target.detach()),
        float(source.detach()),
    )


@torch.inference_mode()
def generate_plaintext(
    model: RecurrenceLM,
    symbols: Sequence[str],
    roles: Sequence[int],
    layout: TokenLayout,
    device: torch.device,
) -> str:
    recurrence = recurrence_encode(symbols)
    source = [layout.recurrence_token(value) for value in recurrence]
    prefix = torch.tensor(
        [[layout.bos, *source, layout.sep]],
        dtype=torch.long,
        device=device,
    )
    role_tensor = torch.tensor(
        [[0, *roles, 0]], dtype=torch.long, device=device
    )
    logits, caches, position = model.prefill(prefix, role_tensor)
    output = []
    for _ in symbols:
        character_logits = logits[0, -1, layout.char_base : layout.eos]
        token = int(character_logits.argmax()) + layout.char_base
        output.append(layout.token_character(token))
        next_token = torch.tensor([[token]], device=device)
        next_role = torch.zeros((1, 1), dtype=torch.long, device=device)
        logits, caches, position = model.step(
            next_token, next_role, caches, position
        )
    return "".join(output)


def window_starts(length: int, size: int, count: int) -> list[int]:
    if length < size:
        return []
    possible = length - size
    if count <= 1 or possible == 0:
        return [0]
    return sorted(
        {
            int(round(index * possible / (count - 1)))
            for index in range(count)
        }
    )


def stream_subset(stream: CipherStream, blocks: set[str]) -> CipherStream:
    selected = [
        index
        for index, block in enumerate(stream.blocks)
        if block in blocks
    ]
    return CipherStream(
        tuple(stream.symbols[index] for index in selected),
        tuple(stream.roles[index] for index in selected),
        tuple(stream.truth[index] for index in selected),
        tuple(stream.blocks[index] for index in selected),
        tuple(stream.slots[index] for index in selected),
    )


@torch.inference_mode()
def infer_key(
    model: RecurrenceLM,
    stream: CipherStream,
    sequence_length: int,
    windows: int,
    layout: TokenLayout,
    device: torch.device,
) -> tuple[dict[str, str], dict[str, Counter]]:
    votes: dict[str, Counter] = defaultdict(Counter)
    model.eval()
    for start in window_starts(
        len(stream.symbols), sequence_length, windows
    ):
        stop = start + sequence_length
        prediction = generate_plaintext(
            model,
            stream.symbols[start:stop],
            stream.roles[start:stop],
            layout,
            device,
        )
        for symbol, character in zip(
            stream.symbols[start:stop], prediction
        ):
            votes[symbol][character] += 1
    key = {
        symbol: counts.most_common(1)[0][0]
        for symbol, counts in votes.items()
    }
    return key, votes


def evaluate_key(
    name: str,
    key: dict[str, str],
    stream: CipherStream,
    threshold: float,
    sample_length: int = 180,
) -> GateResult:
    correct = covered_correct = known = mapped = 0
    rendered = []
    for symbol, truth in zip(stream.symbols, stream.truth):
        prediction = key.get(symbol)
        if truth is None:
            continue
        known += 1
        if prediction is not None:
            mapped += 1
            covered_correct += int(prediction == truth)
            correct += int(prediction == truth)
            if len(rendered) < sample_length:
                rendered.append(prediction)
        elif len(rendered) < sample_length:
            rendered.append("?")
    accuracy = correct / max(known, 1)
    covered_accuracy = covered_correct / max(mapped, 1)
    coverage = mapped / max(known, 1)
    return GateResult(
        name=name,
        passed=accuracy >= threshold,
        threshold=threshold,
        accuracy=accuracy,
        covered_accuracy=covered_accuracy,
        coverage=coverage,
        known=known,
        mapped=mapped,
        key_symbols=len(key),
        sample="".join(rendered),
    )


def parse_table_structure(
    path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    unigram = {
        row["glyphs"]: row["code"]
        for row in rows
        if row["code"].startswith("unigram_")
    }
    prefix = {
        row["glyphs"]: row["code"]
        for row in rows
        if row["code"].startswith("prefix_")
    }
    suffix = {
        row["glyphs"]: row["code"]
        for row in rows
        if row["code"].startswith("suffix_")
    }
    return unigram, prefix, suffix


def code_truth(code: str) -> Optional[str]:
    letter = normalize_latin(code.rsplit("_", 1)[-1])
    if len(letter) != 1 or letter not in CHAR_TO_ID:
        return None
    return letter


def load_official_naibbe(
    ciphertext: Path,
    table: Path,
    n_blocks: int = 16,
) -> tuple[CipherStream, list[EmissionLine], float]:
    """
    Structurally parse official Naibbe.

    Plaintext values are attached only for evaluation.  Ambiguous compound
    boundaries are resolved structurally but their values remain unknown.
    """
    raw, _ = load_naibbe_raw(ciphertext, n_blocks=n_blocks)
    unigram_code, prefix_code, suffix_code = parse_table_structure(table)
    truth_unigram, truth_prefix, truth_suffix = table_maps(table)
    symbols: list[str] = []
    roles: list[int] = []
    truth: list[Optional[str]] = []
    blocks: list[str] = []
    slots: list[str] = []
    lines: list[EmissionLine] = []
    known = 0
    total = 0
    for line in raw:
        line_symbols: list[str] = []
        line_slots: list[str] = []
        line_truth: list[Optional[str]] = []
        length = len(line.words)
        for index, word in enumerate(line.words):
            if word is None:
                continue
            position = position_bucket(index, length)
            if word in unigram_code:
                emitted = (("U", word, truth_unigram.get(word)),)
            else:
                candidates = [
                    (word[:split], word[split:])
                    for split in range(1, len(word))
                    if word[:split] in prefix_code
                    and word[split:] in suffix_code
                ]
                if not candidates:
                    continue
                left, right = candidates[0]
                readings = {
                    (
                        truth_prefix.get(candidate_left),
                        truth_suffix.get(candidate_right),
                    )
                    for candidate_left, candidate_right in candidates
                }
                unambiguous = len(readings) == 1
                emitted = (
                    (
                        "L",
                        left,
                        truth_prefix.get(left) if unambiguous else None,
                    ),
                    (
                        "R",
                        right,
                        truth_suffix.get(right) if unambiguous else None,
                    ),
                )
            for role_name, glyph, letter in emitted:
                symbol = f"{role_name}:{glyph}"
                slot = f"{role_name}@{position}"
                symbols.append(symbol)
                roles.append(ROLES[role_name])
                truth.append(letter)
                blocks.append(line.block)
                slots.append(slot)
                line_symbols.append(symbol)
                line_slots.append(slot)
                line_truth.append(letter)
                total += 1
                known += int(letter is not None)
        lines.append(
            EmissionLine(
                line.block,
                tuple(line_symbols),
                tuple(line_slots),
                tuple(line_truth),
            )
        )
    return (
        CipherStream(
            tuple(symbols),
            tuple(roles),
            tuple(truth),
            tuple(blocks),
            tuple(slots),
        ),
        lines,
        known / max(total, 1),
    )


def emit_voynich_stream(
    raw: Sequence[RawLine],
    model: SegmentModel,
    fit_blocks: set[str],
) -> tuple[CipherStream, list[EmissionLine]]:
    training_words = [
        word
        for line in raw
        if line.block in fit_blocks
        for word in line.words
        if word is not None
    ]
    counts = structural_counts(training_words)
    symbols: list[str] = []
    roles: list[int] = []
    blocks: list[str] = []
    slots: list[str] = []
    lines: list[EmissionLine] = []
    for line in raw:
        length = len(line.words)
        for run in valid_runs(line):
            line_symbols = []
            line_slots = []
            for index, word in run:
                for symbol in model.segment(word, counts):
                    role_name = symbol.split(":", 1)[0]
                    slot = f"{role_name}@{position_bucket(index, length)}"
                    symbols.append(symbol)
                    roles.append(ROLES[role_name])
                    blocks.append(line.block)
                    slots.append(slot)
                    line_symbols.append(symbol)
                    line_slots.append(slot)
            lines.append(
                EmissionLine(
                    line.block,
                    tuple(line_symbols),
                    tuple(line_slots),
                    tuple(None for _ in line_symbols),
                )
            )
    return (
        CipherStream(
            tuple(symbols),
            tuple(roles),
            tuple(None for _ in symbols),
            tuple(blocks),
            tuple(slots),
        ),
        lines,
    )


def train_position_baseline(
    lines: Sequence[EmissionLine],
    blocks: set[str],
    key: dict[str, str],
) -> tuple[Counter, Counter]:
    counts: Counter = Counter()
    totals: Counter = Counter()
    for line in lines:
        if line.block not in blocks:
            continue
        for symbol, slot in zip(line.symbols, line.slots):
            character = key.get(symbol)
            if character is None:
                continue
            counts[(slot, character)] += 1
            totals[slot] += 1
    return counts, totals


def score_residual(
    lines: Sequence[EmissionLine],
    score_blocks: set[str],
    baseline_blocks: set[str],
    key: dict[str, str],
    lm: LanguageModel,
    lambda_: float,
    alpha: float = 0.5,
) -> tuple[float, float, int]:
    baseline, totals = train_position_baseline(lines, baseline_blocks, key)
    residual = lm_score = 0.0
    observations = 0
    width = len(lm.alphabet)
    for line in lines:
        if line.block not in score_blocks:
            continue
        run: list[tuple[str, str]] = []

        def flush() -> None:
            nonlocal residual, lm_score, observations
            if len(run) < 3:
                run.clear()
                return
            for index in range(2, len(run)):
                first, _ = run[index - 2]
                second, _ = run[index - 1]
                character, slot = run[index]
                base = (
                    baseline[(slot, character)] + alpha
                ) / (totals[slot] + alpha * width)
                language = math.exp(
                    lm.logp[
                        lm.char_to_id[first],
                        lm.char_to_id[second],
                        lm.char_to_id[character],
                    ]
                )
                mixture = (1 - lambda_) * base + lambda_ * language
                residual += math.log2(mixture / base)
                lm_score += math.log2(language)
                observations += 1
            run.clear()

        for symbol, slot in zip(line.symbols, line.slots):
            character = key.get(symbol)
            if character is None or character not in lm.char_to_id:
                flush()
            else:
                run.append((character, slot))
        flush()
    if not observations:
        return 0.0, float("-inf"), 0
    return (
        residual / observations,
        lm_score / observations,
        observations,
    )


def render_key(
    lines: Sequence[EmissionLine],
    blocks: set[str],
    key: dict[str, str],
    limit: int = 180,
) -> str:
    pieces = []
    for line in lines:
        if line.block not in blocks:
            continue
        rendered = "".join(key.get(symbol, "?") for symbol in line.symbols)
        if rendered:
            pieces.append(rendered)
        if sum(map(len, pieces)) >= limit:
            break
    return " / ".join(pieces)[:limit]


def cross_validated_residual(
    model: RecurrenceLM,
    raw: Sequence[RawLine],
    blocks: Sequence[str],
    static_stream: Optional[CipherStream],
    static_lines: Optional[Sequence[EmissionLine]],
    segmenter: Optional[SegmentModel],
    sequence_length: int,
    windows: int,
    layout: TokenLayout,
    device: torch.device,
    lm: LanguageModel,
) -> list[ResidualResult]:
    folds = block_folds(blocks)
    results = []
    for outer in range(4):
        test = {block for block in blocks if folds[block] == outer}
        validation = {
            block for block in blocks if folds[block] == (outer + 1) % 4
        }
        fit = set(blocks) - validation - test
        if static_stream is None:
            if segmenter is None:
                raise ValueError("a static stream or segmenter is required")
            stream, lines = emit_voynich_stream(raw, segmenter, fit)
        else:
            if static_lines is None:
                raise ValueError("static lines are required with static stream")
            stream, lines = static_stream, list(static_lines)
        fit_stream = stream_subset(stream, fit)
        key, _ = infer_key(
            model,
            fit_stream,
            sequence_length,
            windows,
            layout,
            device,
        )
        validation_scores = [
            (
                lambda_,
                *score_residual(
                    lines, validation, fit, key, lm, lambda_
                )[:2],
            )
            for lambda_ in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        lambda_, validation_gain, _ = max(
            validation_scores, key=lambda row: row[1]
        )
        test_gain, _, _ = score_residual(
            lines, test, fit, key, lm, lambda_
        )
        test_symbols = [
            symbol
            for line in lines
            if line.block in test
            for symbol in line.symbols
        ]
        coverage = sum(symbol in key for symbol in test_symbols) / max(
            len(test_symbols), 1
        )
        results.append(
            ResidualResult(
                fold=outer,
                fit_blocks=len(fit),
                validation_blocks=len(validation),
                test_blocks=len(test),
                lambda_=lambda_,
                validation_gain=validation_gain,
                test_gain=test_gain,
                mapped_coverage=coverage,
                sample=render_key(lines, test, key),
            )
        )
    return results


def save_checkpoint(
    path: Path,
    model: RecurrenceLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
        },
        temporary,
    )
    os.replace(temporary, path)


def train_model(
    model: RecurrenceLM,
    optimizer: torch.optim.Optimizer,
    corpus: str,
    args: argparse.Namespace,
    layout: TokenLayout,
    device: torch.device,
    start_step: int = 0,
) -> list[dict]:
    rng = np.random.default_rng(args.seed + 101)
    history = []
    model.train()
    started = time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        tokens, roles, targets = random_training_batch(
            corpus,
            args.batch_size,
            args.sequence_length,
            layout,
            rng,
        )
        tokens = tokens.to(device)
        roles = roles.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, roles)
        loss, target, source = target_loss(
            logits, targets, layout, args.source_weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.report_every == 0:
            elapsed = time.monotonic() - started
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "target_loss": target,
                "source_loss": source,
                "steps_per_second": (step - start_step) / max(elapsed, 1e-6),
            }
            history.append(row)
            print(
                f"step={step:5d} loss={row['loss']:.4f} "
                f"target={target:.4f} source={source:.4f} "
                f"steps/s={row['steps_per_second']:.2f}",
                flush=True,
            )
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            save_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                step,
                model_config(args, layout),
            )
    return history


def model_config(
    args: argparse.Namespace, layout: TokenLayout
) -> dict[str, int | float]:
    return {
        "max_recurrences": layout.max_recurrences,
        "sequence_length": args.sequence_length,
        "width": args.width,
        "layers": args.layers,
        "heads": args.heads,
        "expansion": args.expansion,
        "dropout": args.dropout,
    }


def build_model(config: dict) -> tuple[RecurrenceLM, TokenLayout]:
    layout = TokenLayout(int(config["max_recurrences"]))
    model = RecurrenceLM(
        layout=layout,
        sequence_length=int(config["sequence_length"]),
        width=int(config["width"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        expansion=int(config["expansion"]),
        dropout=float(config["dropout"]),
    )
    return model, layout


def print_gate(result: GateResult) -> None:
    state = "PASS" if result.passed else "FAIL"
    print(
        f"{result.name}: {state} accuracy={result.accuracy:.4f} "
        f"threshold={result.threshold:.4f} "
        f"covered_accuracy={result.covered_accuracy:.4f} "
        f"coverage={result.coverage:.4f} "
        f"mapped={result.mapped}/{result.known}"
    )
    print(f"  sample: {result.sample}")


def parse_args() -> argparse.Namespace:
    default_naibbe = Path(os.environ.get("NAIBBE_REPO", "/tmp/naibbe-cipher"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--naibbe-dir", type=Path, default=default_naibbe)
    parser.add_argument("--sequence-length", type=int, default=192)
    parser.add_argument("--max-recurrences", type=int, default=414)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--source-weight", type=float, default=1.0)
    parser.add_argument("--decode-windows", type=int, default=32)
    parser.add_argument("--synthetic-gate", type=float, default=0.60)
    parser.add_argument("--naibbe-gate", type=float, default=0.50)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data/intermediate/recurrence_decoder.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data/intermediate/followups_recurrence_decoder.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="load --checkpoint and run the gates without additional training",
    )
    parser.add_argument(
        "--force-voynich",
        action="store_true",
        help="diagnostic only: score Voynich even when a positive gate fails",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    print(
        f"device={device} torch={torch.__version__} "
        f"threads={torch.get_num_threads()}"
    )

    latin = normalize_latin(
        (ROOT / "data/controls/latin.txt").read_text(
            encoding="utf-8", errors="ignore"
        )
    )
    split = int(len(latin) * 0.78)
    train_latin = latin[:split]
    validation_latin = latin[split:]
    config = model_config(args, TokenLayout(args.max_recurrences))
    model, layout = build_model(config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    start_step = 0
    if args.resume or args.evaluate_only:
        if not args.checkpoint.exists():
            raise SystemExit(f"missing checkpoint: {args.checkpoint}")
        payload = torch.load(args.checkpoint, map_location=device)
        saved_config = payload["config"]
        if saved_config != config:
            raise SystemExit(
                "checkpoint configuration mismatch:\n"
                f"saved={saved_config}\nrequested={config}"
            )
        model.load_state_dict(payload["model"])
        if args.resume and not args.evaluate_only:
            optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        print(f"loaded checkpoint step={start_step}")

    history = []
    if not args.evaluate_only and start_step < args.steps:
        history = train_model(
            model,
            optimizer,
            train_latin,
            args,
            layout,
            device,
            start_step,
        )
        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            args.steps,
            config,
        )

    # Gate 1: one persistent random key over a reserved, unseen Latin span.
    validation_rng = np.random.default_rng(args.seed + 20_003)
    synthetic, _ = synthetic_stream(validation_latin, validation_rng)
    synthetic_blocks = sorted(set(synthetic.blocks))
    synthetic_cut = max(1, len(synthetic_blocks) * 2 // 3)
    synthetic_fit_blocks = set(synthetic_blocks[:synthetic_cut])
    synthetic_test_blocks = set(synthetic_blocks[synthetic_cut:])
    synthetic_key, _ = infer_key(
        model,
        stream_subset(synthetic, synthetic_fit_blocks),
        args.sequence_length,
        args.decode_windows,
        layout,
        device,
    )
    synthetic_result = evaluate_key(
        "SYNTHETIC_UNSEEN_KEY",
        synthetic_key,
        stream_subset(synthetic, synthetic_test_blocks),
        args.synthetic_gate,
    )
    print_gate(synthetic_result)

    # Gate 2: official ciphertext is read only after synthetic tuning is over.
    official_ciphertext = (
        args.naibbe_dir / "encrypted/nathist_output_ciphertext.txt"
    )
    official_table = args.naibbe_dir / "references/naibbe_tables.csv"
    for path in (official_ciphertext, official_table):
        if not path.exists():
            raise SystemExit(f"missing official Naibbe input: {path}")
    official, official_lines, official_truth_coverage = load_official_naibbe(
        official_ciphertext, official_table
    )
    official_blocks = sorted(set(official.blocks))
    official_cut = max(1, len(official_blocks) * 3 // 4)
    official_fit = set(official_blocks[:official_cut])
    official_test = set(official_blocks[official_cut:])
    official_key, _ = infer_key(
        model,
        stream_subset(official, official_fit),
        args.sequence_length,
        args.decode_windows,
        layout,
        device,
    )
    official_result = evaluate_key(
        "OFFICIAL_NAIBBE",
        official_key,
        stream_subset(official, official_test),
        args.naibbe_gate,
    )
    print_gate(official_result)

    report: dict = {
        "method": "causal recurrence decoder with persistent-key voting",
        "seed": args.seed,
        "device": str(device),
        "torch": torch.__version__,
        "config": config,
        "training": history,
        "synthetic_gate": asdict(synthetic_result),
        "official_naibbe_gate": asdict(official_result),
        "official_truth_coverage": official_truth_coverage,
        "voynich_unlocked": bool(
            synthetic_result.passed and official_result.passed
        ),
        "voynich": None,
    }

    unlocked = synthetic_result.passed and official_result.passed
    if unlocked or args.force_voynich:
        calibration_raw, _ = load_naibbe_raw(official_ciphertext)
        segmenter, calibration = calibrate(
            [
                [word for word in line.words if word is not None]
                for line in calibration_raw
            ],
            official_table,
        )
        voynich_raw, voynich_blocks = load_voynich()
        lm = LanguageModel.from_text(train_latin)
        results = cross_validated_residual(
            model=model,
            raw=voynich_raw,
            blocks=voynich_blocks,
            static_stream=None,
            static_lines=None,
            segmenter=segmenter,
            sequence_length=args.sequence_length,
            windows=args.decode_windows,
            layout=layout,
            device=device,
            lm=lm,
        )
        report["voynich"] = {
            "forced_despite_failed_gate": not unlocked,
            "structural_calibration": calibration,
            "folds": [asdict(result) for result in results],
            "mean_test_gain": float(
                np.mean([result.test_gain for result in results])
            ),
        }
        print("VOYNICH HELD-OUT RESIDUAL")
        for result in results:
            print(
                f"fold={result.fold} lambda={result.lambda_:.2f} "
                f"validation={result.validation_gain:+.4f} "
                f"test={result.test_gain:+.4f} "
                f"coverage={result.mapped_coverage:.3f}"
            )
            print(f"  sample: {result.sample}")
    else:
        print(
            "VOYNICH LOCKED: at least one positive-control gate failed. "
            "No Voynich score was computed."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
