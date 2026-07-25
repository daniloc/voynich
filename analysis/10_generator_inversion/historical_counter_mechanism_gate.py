#!/usr/bin/env python3
"""
Test a workshop-executable table/counter mechanism for Voynich production.

The extra mechanism is intentionally limited to operations a fifteenth-century
scribe could execute with a short table, tally, or rotating alphabet:

* advance once per word, glyph, or line;
* reset at a visible line, paragraph, or page boundary;
* use a cycle of 2, 3, 4, 5, 7, or 12 positions;
* at each position, bias the next glyph choice in the existing character
  grammar.

Complete repository-quire folds fit the table, select its reset/cycle and
smoothing on validation quires, and score untouched test quires.  The null
repeats the entire selection after independently rotating the phase origin in
every reset unit.  This preserves local positions and cycle sizes while
destroying a shared alignment to the visible boundary.

Controls are ordinary Latin reflowed into the Voynich layout, text sampled
from the baseline source, and text with a planted four-position line-reset
table.  A phase-offset audit asks whether anomalous test lines look like
persistent counter slips.  The experiment tests a production mechanism, not
plaintext or semantic content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

import production_algorithm_gate as source


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "corpus.json"
LATIN = ROOT / "data" / "controls" / "latin.txt"
OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_historical_counter_gate.json"
)

SEED = 20260725
MODULI = (2, 3, 4, 5, 7, 12)
ALPHAS = (256.0, 2048.0)
SCOPES = (
    "word_in_line",
    "word_in_paragraph",
    "glyph_in_line",
    "line_in_page",
    "boustrophedon_word_in_line",
)
PAGE_BACKOFF = source.ADAPT_BACKOFF
DEFAULT_NULLS = 32
CONTROL_NULLS = 8
PLANTED_MODULUS = 4
PLANTED_STRENGTH = 64.0
TABLE_ENTRY_DESCRIPTION_BITS = 1.0


@dataclass(frozen=True)
class PositionedEvent:
    event: source.Event
    line_unit: str
    paragraph_unit: str
    page_unit: str
    line_index: int
    word_in_line: int
    word_in_paragraph: int


@dataclass(frozen=True)
class Candidate:
    scope: str
    modulus: int
    alpha: float

    @property
    def name(self) -> str:
        return (
            f"{self.scope}_mod{self.modulus}_alpha{int(self.alpha)}"
        )


CANDIDATES = tuple(
    Candidate(scope, modulus, alpha)
    for scope in SCOPES
    for modulus in MODULI
    for alpha in ALPHAS
)


@dataclass
class EncodedSplit:
    symbol: np.ndarray
    group: np.ndarray
    base_probability: np.ndarray
    page_count: np.ndarray
    page_total: np.ndarray
    coordinates: dict[str, np.ndarray]
    units: dict[str, np.ndarray]
    unit_counts: dict[str, int]
    line: np.ndarray
    line_labels: tuple[str, ...]
    words: int

    @property
    def symbols(self) -> int:
        return int(self.symbol.size)


@dataclass
class PreparedFold:
    fold: int
    alphabet: tuple[str, ...]
    base_vectors: np.ndarray
    train: EncodedSplit
    validation: EncodedSplit
    test: EncodedSplit


@dataclass
class PhaseTable:
    observed: np.ndarray
    expected: np.ndarray
    multipliers: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_word(word: str) -> bool:
    return "?" not in word and word.isalpha() and len(word) >= 2


def load_positioned(
    path: Path,
) -> tuple[list[PositionedEvent], Counter[str]]:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    audit: Counter[str] = Counter()
    histories: dict[str, list[str]] = defaultdict(list)
    segments: Counter[str] = Counter()
    events: list[PositionedEvent] = []
    sequence = 0

    for folio in sorted(corpus["folios"], key=source.folio_key):
        meta = corpus["meta"].get(folio, {})
        block = str(meta.get("Q", "?"))
        paragraph = -1
        paragraph_run = 0
        paragraph_word = 0
        line_index = -1
        for raw in corpus["folios"][folio]:
            if source.locus_type(str(raw["locus"])) != "P":
                continue
            audit["prose_lines"] += 1
            line_index += 1
            locus = str(raw["locus"]).strip()
            if locus.startswith("@") or paragraph < 0:
                paragraph += 1
                paragraph_run = 0
                paragraph_word = 0

            raw_words = raw["words"]
            line_run = 0
            line_word = 0
            for index, word in enumerate(raw_words):
                audit["source_tokens"] += 1
                if not valid_word(word):
                    audit["hard_breaks"] += 1
                    histories[block].clear()
                    segments[block] += 1
                    line_run += 1
                    paragraph_run += 1
                    line_word = 0
                    paragraph_word = 0
                    continue

                history = tuple(
                    reversed(histories[block][-source.HISTORY:])
                )
                event = source.Event(
                    sequence=sequence,
                    segment=segments[block],
                    block=block,
                    folio=folio,
                    line=str(raw["line"]),
                    position=source.position_bucket(index, len(raw_words)),
                    section=str(meta.get("I", "?")),
                    currier=str(meta.get("L", "?")),
                    hand=str(meta.get("H", "?")),
                    word=word,
                    history=history,
                )
                events.append(PositionedEvent(
                    event=event,
                    line_unit=(
                        f"{folio}:{raw['line']}:run{line_run}"
                    ),
                    paragraph_unit=(
                        f"{folio}:paragraph{paragraph}:run{paragraph_run}"
                    ),
                    page_unit=folio,
                    line_index=line_index,
                    word_in_line=line_word,
                    word_in_paragraph=paragraph_word,
                ))
                histories[block].append(word)
                sequence += 1
                line_word += 1
                paragraph_word += 1
                audit["eligible_words"] += 1

    reference, reference_audit = source.load_events(path)
    if [row.event for row in events] != reference:
        raise AssertionError("positioned loader diverges from source loader")
    if audit["eligible_words"] != reference_audit["eligible_words"]:
        raise AssertionError("positioned loader audit mismatch")
    return events, audit


def replace_events(
    template: Sequence[PositionedEvent],
    events: Sequence[source.Event],
) -> list[PositionedEvent]:
    if len(template) != len(events):
        raise ValueError("replacement event count differs from template")
    return [
        replace(positioned, event=event)
        for positioned, event in zip(template, events)
    ]


def split_positioned(
    events: Sequence[PositionedEvent],
    assignment: dict[str, int],
    fold: int,
) -> tuple[
    list[PositionedEvent],
    list[PositionedEvent],
    list[PositionedEvent],
]:
    test = [
        row for row in events
        if assignment[row.event.block] == fold
    ]
    validation = [
        row for row in events
        if assignment[row.event.block] == (fold + 1) % source.N_FOLDS
    ]
    train = [
        row for row in events
        if assignment[row.event.block]
        not in {fold, (fold + 1) % source.N_FOLDS}
    ]
    return train, validation, test


def _unit_id(
    mappings: dict[str, dict[str, int]],
    scope: str,
    label: str,
) -> int:
    mapping = mappings[scope]
    if label not in mapping:
        mapping[label] = len(mapping)
    return mapping[label]


def encode_split(
    model: source.CharacterModel,
    rows: Sequence[PositionedEvent],
    group_map: dict[tuple[object, ...], int],
    base_vectors: list[list[float]],
) -> EncodedSplit:
    symbols: list[int] = []
    groups: list[int] = []
    base_probabilities: list[float] = []
    page_counts_out: list[int] = []
    page_totals_out: list[int] = []
    coordinates: dict[str, list[int]] = {
        scope: [] for scope in SCOPES
    }
    units: dict[str, list[int]] = {scope: [] for scope in SCOPES}
    unit_mappings: dict[str, dict[str, int]] = {
        scope: {} for scope in SCOPES
    }
    line_mapping: dict[str, int] = {}
    line_ids: list[int] = []
    alphabet_map = {
        symbol: index for index, symbol in enumerate(model.alphabet)
    }
    adaptive_counts: Counter[tuple[object, ...]] = Counter()
    adaptive_totals: Counter[tuple[object, ...]] = Counter()
    glyph_offsets: Counter[str] = Counter()

    for positioned in rows:
        event = positioned.event
        line_id = line_mapping.setdefault(
            positioned.line_unit, len(line_mapping)
        )
        history = ["^"] * model.order
        emitted: list[tuple[tuple[str, ...], str]] = []
        word_symbols = model.symbols(event.word)
        glyph_origin = glyph_offsets[positioned.line_unit]
        for character_index, symbol in enumerate(word_symbols):
            context = (
                tuple(history[-model.order:]) if model.order else ()
            )
            group_key = (
                event.currier,
                event.section,
                event.position,
                context,
            )
            group = group_map.get(group_key)
            if group is None:
                group = len(group_map)
                group_map[group_key] = group
                base_vectors.append([
                    model.probability(event, history, candidate_symbol)
                    for candidate_symbol in model.alphabet
                ])
            page_key = (event.folio, context)

            symbols.append(alphabet_map[symbol])
            groups.append(group)
            base_probabilities.append(base_vectors[group][alphabet_map[symbol]])
            page_counts_out.append(
                adaptive_counts[(page_key, symbol)]
            )
            page_totals_out.append(adaptive_totals[page_key])
            line_ids.append(line_id)

            scope_coordinates = {
                "word_in_line": positioned.word_in_line,
                "word_in_paragraph": positioned.word_in_paragraph,
                "glyph_in_line": glyph_origin + character_index,
                "line_in_page": positioned.line_index,
                "boustrophedon_word_in_line": (
                    positioned.word_in_line
                    if positioned.line_index % 2 == 0
                    else -positioned.word_in_line
                ),
            }
            scope_units = {
                "word_in_line": positioned.line_unit,
                "word_in_paragraph": positioned.paragraph_unit,
                "glyph_in_line": positioned.line_unit,
                "line_in_page": positioned.page_unit,
                "boustrophedon_word_in_line": positioned.line_unit,
            }
            for scope in SCOPES:
                coordinates[scope].append(scope_coordinates[scope])
                units[scope].append(_unit_id(
                    unit_mappings, scope, scope_units[scope]
                ))

            emitted.append((context, symbol))
            history.append(symbol)

        for context, symbol in emitted:
            page_key = (event.folio, context)
            adaptive_counts[(page_key, symbol)] += 1
            adaptive_totals[page_key] += 1
        glyph_offsets[positioned.line_unit] += len(word_symbols)

    return EncodedSplit(
        symbol=np.asarray(symbols, dtype=np.int32),
        group=np.asarray(groups, dtype=np.int32),
        base_probability=np.asarray(base_probabilities, dtype=np.float64),
        page_count=np.asarray(page_counts_out, dtype=np.float64),
        page_total=np.asarray(page_totals_out, dtype=np.float64),
        coordinates={
            scope: np.asarray(values, dtype=np.int64)
            for scope, values in coordinates.items()
        },
        units={
            scope: np.asarray(values, dtype=np.int32)
            for scope, values in units.items()
        },
        unit_counts={
            scope: len(mapping)
            for scope, mapping in unit_mappings.items()
        },
        line=np.asarray(line_ids, dtype=np.int32),
        line_labels=tuple(
            label
            for label, _index in sorted(
                line_mapping.items(), key=lambda item: item[1]
            )
        ),
        words=len(rows),
    )


def prepare_folds(
    events: Sequence[PositionedEvent],
) -> tuple[list[PreparedFold], dict[str, int], list[int]]:
    plain = [row.event for row in events]
    assignment, loads = source.block_folds(plain)
    prepared = []
    for fold in range(source.N_FOLDS):
        train_rows, validation_rows, test_rows = split_positioned(
            events, assignment, fold
        )
        model = source.CharacterModel(
            [row.event for row in train_rows],
            order=2,
            cell_mode="register",
        )
        group_map: dict[tuple[object, ...], int] = {}
        base_vectors: list[list[float]] = []
        train = encode_split(
            model, train_rows, group_map, base_vectors
        )
        validation = encode_split(
            model, validation_rows, group_map, base_vectors
        )
        test = encode_split(
            model, test_rows, group_map, base_vectors
        )
        prepared.append(PreparedFold(
            fold=fold,
            alphabet=model.alphabet,
            base_vectors=np.asarray(base_vectors, dtype=np.float64),
            train=train,
            validation=validation,
            test=test,
        ))
    return prepared, assignment, loads


def baseline_log2(split: EncodedSplit) -> float:
    probability = (
        split.page_count + PAGE_BACKOFF * split.base_probability
    ) / (split.page_total + PAGE_BACKOFF)
    return float(np.log2(np.maximum(probability, 1e-300)).sum())


def phase_values(
    split: EncodedSplit,
    scope: str,
    modulus: int,
    rng: np.random.Generator | None,
) -> np.ndarray:
    phase = split.coordinates[scope]
    if rng is not None:
        offsets = rng.integers(
            0,
            modulus,
            size=split.unit_counts[scope],
            dtype=np.int64,
        )
        phase = phase + offsets[split.units[scope]]
    return np.mod(phase, modulus).astype(np.int32)


def phase_seed(
    seed: int,
    fold: int,
    scope: str,
    modulus: int,
    split_name: str,
) -> int:
    payload = (
        f"{seed}:{fold}:{scope}:{modulus}:{split_name}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def fit_phase_table(
    prepared: PreparedFold,
    candidate: Candidate,
    phases: np.ndarray,
) -> PhaseTable:
    alphabet_size = len(prepared.alphabet)
    flat = phases * alphabet_size + prepared.train.symbol
    observed = np.bincount(
        flat,
        minlength=candidate.modulus * alphabet_size,
    ).reshape(candidate.modulus, alphabet_size).astype(np.float64)
    expected = np.zeros_like(observed)
    for phase in range(candidate.modulus):
        indices = phases == phase
        if np.any(indices):
            expected[phase] = prepared.base_vectors[
                prepared.train.group[indices]
            ].sum(axis=0)
    expected_totals = expected.sum(axis=1, keepdims=True)
    expected_share = np.divide(
        expected,
        expected_totals,
        out=np.full_like(expected, 1.0 / alphabet_size),
        where=expected_totals > 0,
    )
    # A small physical table is represented by one multiplier per
    # phase/glyph.  The expected baseline count is the identity table;
    # alpha shrinks every multiplier back toward one.
    multipliers = (
        observed + candidate.alpha * expected_share
    ) / np.maximum(
        expected + candidate.alpha * expected_share,
        1e-12,
    )
    return PhaseTable(
        observed=observed,
        expected=expected,
        multipliers=multipliers,
    )


def phase_log_probabilities(
    prepared: PreparedFold,
    split: EncodedSplit,
    candidate: Candidate,
    phases: np.ndarray,
    table: PhaseTable,
) -> np.ndarray:
    denominator = np.empty(split.symbols, dtype=np.float64)
    for phase in range(candidate.modulus):
        indices = phases == phase
        if np.any(indices):
            denominator[indices] = np.sum(
                prepared.base_vectors[split.group[indices]]
                * table.multipliers[phase],
                axis=1,
            )
    phase_probability = (
        split.base_probability
        * table.multipliers[phases, split.symbol]
        / np.maximum(denominator, 1e-300)
    )
    probability = (
        split.page_count + PAGE_BACKOFF * phase_probability
    ) / (split.page_total + PAGE_BACKOFF)
    return np.log2(np.maximum(probability, 1e-300))


def _phases_for_candidate(
    prepared: PreparedFold,
    candidate: Candidate,
    random_origin_seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = []
    for name, split in (
        ("train", prepared.train),
        ("validation", prepared.validation),
        ("test", prepared.test),
    ):
        rng = None
        if random_origin_seed is not None:
            rng = np.random.default_rng(phase_seed(
                random_origin_seed,
                prepared.fold,
                candidate.scope,
                candidate.modulus,
                name,
            ))
        values.append(phase_values(
            split, candidate.scope, candidate.modulus, rng
        ))
    return values[0], values[1], values[2]


def score_fold(
    prepared: PreparedFold,
    random_origin_seed: int | None = None,
    keep_model: bool = False,
) -> tuple[
    dict[str, object],
    tuple[Candidate, PhaseTable] | None,
]:
    baseline_validation = baseline_log2(prepared.validation)
    baseline_test = baseline_log2(prepared.test)
    scored: list[
        tuple[float, float, Candidate, float, PhaseTable, np.ndarray]
    ] = []

    for scope in SCOPES:
        for modulus in MODULI:
            probe = Candidate(scope, modulus, ALPHAS[0])
            train_phase, validation_phase, test_phase = (
                _phases_for_candidate(
                    prepared, probe, random_origin_seed
                )
            )
            for alpha in ALPHAS:
                candidate = Candidate(scope, modulus, alpha)
                table = fit_phase_table(
                    prepared, candidate, train_phase
                )
                validation_log2 = float(phase_log_probabilities(
                    prepared,
                    prepared.validation,
                    candidate,
                    validation_phase,
                    table,
                ).sum())
                test_log2 = float(phase_log_probabilities(
                    prepared,
                    prepared.test,
                    candidate,
                    test_phase,
                    table,
                ).sum())
                description_bits = (
                    candidate.modulus
                    * len(prepared.alphabet)
                    * TABLE_ENTRY_DESCRIPTION_BITS
                )
                scored.append((
                    validation_log2 - description_bits,
                    validation_log2,
                    candidate,
                    test_log2,
                    table,
                    test_phase,
                ))

    best_table = max(
        scored,
        key=lambda row: (
            row[0],
            -SCOPES.index(row[2].scope),
            -MODULI.index(row[2].modulus),
            -ALPHAS.index(row[2].alpha),
        ),
    )
    table_selected = best_table[0] > baseline_validation
    (
        adjusted_validation_log2,
        validation_log2,
        candidate,
        test_log2,
        table,
        _test_phase,
    ) = best_table
    if not table_selected:
        validation_log2 = baseline_validation
        test_log2 = baseline_test
    ranking = sorted(scored, key=lambda row: row[0], reverse=True)[:5]
    test_gain = test_log2 - baseline_test
    description_bits = (
        candidate.modulus
        * len(prepared.alphabet)
        * TABLE_ENTRY_DESCRIPTION_BITS
        if table_selected else 0.0
    )
    result = {
        "fold": prepared.fold,
        "selected": asdict(candidate) if table_selected else None,
        "selected_name": candidate.name if table_selected else "baseline",
        "minimum_table_description_bits": description_bits,
        "validation": {
            "symbols": prepared.validation.symbols,
            "words": prepared.validation.words,
            "baseline_bits_per_symbol": (
                -baseline_validation / prepared.validation.symbols
            ),
            "candidate_bits_per_symbol": (
                -validation_log2 / prepared.validation.symbols
            ),
            "gain_bits_per_symbol": (
                (validation_log2 - baseline_validation)
                / prepared.validation.symbols
            ),
            "net_gain_after_table_description_bits": (
                adjusted_validation_log2 - baseline_validation
                if table_selected else 0.0
            ),
        },
        "test": {
            "symbols": prepared.test.symbols,
            "words": prepared.test.words,
            "baseline_log2_probability": baseline_test,
            "candidate_log2_probability": test_log2,
            "baseline_bits_per_symbol": (
                -baseline_test / prepared.test.symbols
            ),
            "candidate_bits_per_symbol": (
                -test_log2 / prepared.test.symbols
            ),
            "gain_bits_per_symbol": (
                test_gain / prepared.test.symbols
            ),
            "gain_bits_per_word": (
                test_gain / prepared.test.words
            ),
            "net_gain_after_table_description_bits": (
                test_gain - description_bits
            ),
        },
        "validation_top_five": [
            {
                "candidate": row[2].name,
                "raw_gain_bits_per_symbol": (
                    (row[1] - baseline_validation)
                    / prepared.validation.symbols
                ),
                "net_gain_after_table_description_bits": (
                    row[0] - baseline_validation
                ),
            }
            for row in ranking
        ],
    }
    retained = (
        (candidate, table)
        if keep_model and table_selected
        else None
    )
    return result, retained


def phase_action_table(
    prepared: PreparedFold,
    candidate: Candidate,
    table: PhaseTable,
) -> list[dict[str, object]]:
    result = []
    for phase in range(candidate.modulus):
        total = int(table.observed[phase].sum())
        ranked = sorted(
            range(len(prepared.alphabet)),
            key=lambda symbol: table.multipliers[phase, symbol],
        )

        def entry(symbol: int) -> dict[str, object]:
            return {
                "symbol": prepared.alphabet[symbol],
                "multiplier": float(
                    table.multipliers[phase, symbol]
                ),
                "observed": float(table.observed[phase, symbol]),
                "baseline_expected": float(
                    table.expected[phase, symbol]
                ),
            }

        result.append({
            "phase": phase,
            "observations": total,
            "strongest_promotions": [
                entry(symbol) for symbol in reversed(ranked[-5:])
            ],
            "strongest_suppressions": [
                entry(symbol) for symbol in ranked[:5]
            ],
        })
    return result


def offset_audit(
    prepared: PreparedFold,
    candidate: Candidate,
    table: PhaseTable,
) -> dict[str, object]:
    split = prepared.test
    nominal = phase_values(
        split, candidate.scope, candidate.modulus, None
    )
    best_offsets: Counter[int] = Counter()
    lines = 0
    half_lines = 0
    half_agree = 0
    best_gain = 0.0

    for line_id in range(len(split.line_labels)):
        indices = np.flatnonzero(split.line == line_id)
        if indices.size < 8:
            continue
        scores = []
        for offset in range(candidate.modulus):
            phases = np.mod(
                nominal[indices] + offset,
                candidate.modulus,
            )
            scores.append(float(phase_log_probabilities(
                prepared,
                _slice_split(split, indices),
                candidate,
                phases,
                table,
            ).sum()))
        best = max(range(candidate.modulus), key=scores.__getitem__)
        best_offsets[best] += 1
        best_gain += scores[best] - scores[0]
        lines += 1

        midpoint = indices.size // 2
        if midpoint >= 4 and indices.size - midpoint >= 4:
            halves = (indices[:midpoint], indices[midpoint:])
            half_best = []
            for half in halves:
                half_scores = []
                for offset in range(candidate.modulus):
                    phases = np.mod(
                        nominal[half] + offset,
                        candidate.modulus,
                    )
                    half_scores.append(float(phase_log_probabilities(
                        prepared,
                        _slice_split(split, half),
                        candidate,
                        phases,
                        table,
                    ).sum()))
                half_best.append(max(
                    range(candidate.modulus),
                    key=half_scores.__getitem__,
                ))
            half_lines += 1
            half_agree += int(half_best[0] == half_best[1])

    return {
        "interpretation": (
            "A stable boundary-reset table predicts offset 0 on unseen "
            "lines. A copied or skipped counter position predicts a "
            "persistent nonzero offset and agreement between line halves."
        ),
        "eligible_test_lines": lines,
        "best_offset_counts": {
            str(offset): best_offsets[offset]
            for offset in range(candidate.modulus)
        },
        "declared_origin_best_fraction": (
            best_offsets[0] / lines if lines else None
        ),
        "mean_oracle_offset_gain_bits_per_symbol": (
            best_gain
            / sum(
                int(np.count_nonzero(split.line == line_id))
                for line_id in range(len(split.line_labels))
                if np.count_nonzero(split.line == line_id) >= 8
            )
            if lines else None
        ),
        "line_half_offset_agreement": (
            half_agree / half_lines if half_lines else None
        ),
        "line_half_comparisons": half_lines,
    }


def _slice_split(
    split: EncodedSplit,
    indices: np.ndarray,
) -> EncodedSplit:
    return EncodedSplit(
        symbol=split.symbol[indices],
        group=split.group[indices],
        base_probability=split.base_probability[indices],
        page_count=split.page_count[indices],
        page_total=split.page_total[indices],
        coordinates={
            scope: values[indices]
            for scope, values in split.coordinates.items()
        },
        units={
            scope: values[indices]
            for scope, values in split.units.items()
        },
        unit_counts=split.unit_counts,
        line=split.line[indices],
        line_labels=split.line_labels,
        words=0,
    )


def summarize_dataset(
    name: str,
    events: Sequence[PositionedEvent],
    nulls: int,
    progress: bool,
) -> dict[str, object]:
    prepared, assignment, loads = prepare_folds(events)
    fold_results = []
    retained = []
    for fold in prepared:
        result, fitted = score_fold(fold, keep_model=True)
        if fitted is not None:
            candidate, table = fitted
            result["phase_action_table"] = phase_action_table(
                fold, candidate, table
            )
            result["phase_offset_audit"] = offset_audit(
                fold, candidate, table
            )
        else:
            result["phase_action_table"] = None
            result["phase_offset_audit"] = None
        fold_results.append(result)
        retained.append(fitted)
        if progress:
            print(
                f"{name} fold {fold.fold}: {result['selected_name']} "
                f"{result['test']['gain_bits_per_symbol']:+.5f} bits/symbol",
                flush=True,
            )

    observed_log_gain = sum(
        float(row["test"]["candidate_log2_probability"])
        - float(row["test"]["baseline_log2_probability"])
        for row in fold_results
    )
    observed_symbols = sum(
        int(row["test"]["symbols"]) for row in fold_results
    )
    observed_words = sum(
        int(row["test"]["words"]) for row in fold_results
    )
    null_gains = []
    for replicate in range(nulls):
        random_origin_seed = SEED + 100_000 + replicate
        total = 0.0
        for fold in prepared:
            result, _fitted = score_fold(
                fold,
                random_origin_seed=random_origin_seed,
            )
            total += (
                float(result["test"]["candidate_log2_probability"])
                - float(result["test"]["baseline_log2_probability"])
            )
        null_gains.append(total / observed_symbols)
        if progress and (
            replicate == 0
            or (replicate + 1) % 8 == 0
            or replicate + 1 == nulls
        ):
            print(
                f"{name} null {replicate + 1}/{nulls}",
                flush=True,
            )

    observed_gain = observed_log_gain / observed_symbols
    null_array = np.asarray(null_gains, dtype=np.float64)
    p_value = (
        (1 + int(np.count_nonzero(null_array >= observed_gain)))
        / (nulls + 1)
        if nulls
        else None
    )
    selected = Counter(row["selected_name"] for row in fold_results)
    selected_scopes = Counter(
        str(row["selected"]["scope"])
        for row in fold_results
        if row["selected"] is not None
    )
    selected_moduli = Counter(
        int(row["selected"]["modulus"])
        for row in fold_results
        if row["selected"] is not None
    )
    positive_folds = sum(
        float(row["test"]["gain_bits_per_symbol"]) > 0
        for row in fold_results
    )
    summary = {
        "pooled_test_symbols": observed_symbols,
        "pooled_test_words": observed_words,
        "gain_bits_per_symbol": observed_gain,
        "gain_bits_per_word": observed_log_gain / observed_words,
        "positive_test_folds": positive_folds,
        "selected_candidate_counts": dict(selected),
        "table_selected_folds": sum(
            row["selected"] is not None for row in fold_results
        ),
        "selected_scope_counts": dict(selected_scopes),
        "selected_modulus_counts": {
            str(key): value for key, value in selected_moduli.items()
        },
        "phase_origin_null_p_one_sided": p_value,
        "null_mean_gain_bits_per_symbol": (
            float(null_array.mean()) if nulls else None
        ),
        "null_95th_percentile_gain_bits_per_symbol": (
            float(np.quantile(null_array, 0.95)) if nulls else None
        ),
        "null_replicates": nulls,
    }
    return {
        "name": name,
        "fold_assignment": assignment,
        "fold_event_loads": loads,
        "folds": fold_results,
        "null_gain_bits_per_symbol": null_gains,
        "summary": summary,
    }


def stable_preferred_symbol(
    alphabet: Sequence[str],
    previous: str,
    phase: int,
) -> str:
    payload = f"{previous}:{phase}".encode("ascii")
    index = int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "big"
    ) % len(alphabet)
    return alphabet[index]


def sample_planted_word(
    model: source.CharacterModel,
    event: source.Event,
    phase: int,
    rng: random.Random,
) -> str:
    choices = tuple(sorted(model.alphabet_set)) + (model.eos,)
    for _attempt in range(50):
        history = ["^"] * model.order
        result: list[str] = []
        for _position in range(source.MAX_WORD_LENGTH):
            preferred = stable_preferred_symbol(
                choices, history[-1], phase
            )
            weights = []
            for symbol in choices:
                probability = model.probability(
                    event, history, symbol
                )
                if symbol == preferred:
                    probability *= PLANTED_STRENGTH
                if symbol == model.eos and len(result) < 2:
                    probability = 0.0
                weights.append(probability)
            symbol = rng.choices(choices, weights=weights, k=1)[0]
            if symbol == model.eos:
                return "".join(result)
            result.append(symbol)
            history.append(symbol)
        if len(result) >= 2:
            return "".join(result)
    return "ol"


def planted_events(
    template: Sequence[PositionedEvent],
    model: source.CharacterModel,
    seed: int,
) -> list[PositionedEvent]:
    rng = random.Random(seed)
    words = [
        sample_planted_word(
            model,
            row.event,
            row.word_in_line % PLANTED_MODULUS,
            rng,
        )
        for row in template
    ]
    rebuilt = source.rebuild_histories(
        [row.event for row in template], words
    )
    return replace_events(template, rebuilt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--latin", type=Path, default=LATIN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--nulls", type=int, default=DEFAULT_NULLS)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positioned, audit = load_positioned(args.corpus)
    real = summarize_dataset(
        "VOYNICH", positioned, args.nulls, args.progress
    )
    panels = [real]
    controls: dict[str, object] = {}

    if not args.skip_controls:
        plain = [row.event for row in positioned]
        full_model = source.CharacterModel(
            plain, order=2, cell_mode="register"
        )
        channel = source.EditChannel(plain)
        base_only, _truth = source.generated_events(
            plain,
            full_model,
            channel,
            {"base": 1.0, "copy": 0.0, "edit": 0.0},
            SEED + 20_000,
        )
        control_nulls = min(args.nulls, CONTROL_NULLS)
        base_panel = summarize_dataset(
            "SYNTHETIC_BASE_ONLY",
            replace_events(positioned, base_only),
            control_nulls,
            args.progress,
        )
        latin_panel = summarize_dataset(
            "LATIN_REFLOW",
            replace_events(
                positioned, source.latin_events(plain, args.latin)
            ),
            control_nulls,
            args.progress,
        )
        planted_panel = summarize_dataset(
            "SYNTHETIC_PLANTED_WORD_LINE_MOD4",
            planted_events(positioned, full_model, SEED + 30_000),
            args.nulls,
            args.progress,
        )
        panels.extend((base_panel, latin_panel, planted_panel))

        planted_summary = planted_panel["summary"]
        planted_scope_recovery = int(
            planted_summary["selected_scope_counts"].get(
                "word_in_line", 0
            )
        )
        planted_modulus_recovery = int(
            planted_summary["selected_modulus_counts"].get("4", 0)
        )
        controls = {
            "planted": {
                "scope": "word_in_line",
                "modulus": PLANTED_MODULUS,
                "preference_multiplier": PLANTED_STRENGTH,
            },
            "planted_scope_recovery_folds": planted_scope_recovery,
            "planted_modulus_recovery_folds": planted_modulus_recovery,
            "planted_gain_pass": (
                float(planted_summary["gain_bits_per_symbol"]) > 0.02
            ),
            "planted_selection_pass": (
                planted_scope_recovery >= 3
                and planted_modulus_recovery >= 3
            ),
            "planted_null_pass": (
                planted_summary["phase_origin_null_p_one_sided"]
                is not None
                and float(
                    planted_summary["phase_origin_null_p_one_sided"]
                ) <= 0.05
            ),
            "base_only_false_positive_pass": (
                (
                    base_panel["summary"][
                        "null_95th_percentile_gain_bits_per_symbol"
                    ] is None
                )
                or (
                    float(base_panel["summary"]["gain_bits_per_symbol"])
                    <= float(
                        base_panel["summary"][
                            "null_95th_percentile_gain_bits_per_symbol"
                        ]
                    )
                )
            ),
        }

    real_summary = real["summary"]
    controls_pass = (
        all(
            bool(value)
            for key, value in controls.items()
            if key.endswith("_pass")
        )
        if controls else None
    )
    primary_pass = (
        float(real_summary["gain_bits_per_symbol"]) > 0
        and int(real_summary["positive_test_folds"]) >= 3
        and real_summary["phase_origin_null_p_one_sided"] is not None
        and float(real_summary["phase_origin_null_p_one_sided"]) <= 0.05
    )
    selected_candidate_counts = real_summary[
        "selected_candidate_counts"
    ]
    single_rule_replicates = (
        max(selected_candidate_counts.values(), default=0) >= 3
    )
    result = {
        "experiment": "historical_table_counter_mechanism_gate",
        "seed": SEED,
        "claim_boundary": (
            "A pass supports a compact boundary-reset production table. "
            "It does not identify plaintext, language, or semantics."
        ),
        "historical_model": {
            "operation": (
                "advance a small counter and consult a phase-conditioned "
                "glyph-choice table"
            ),
            "physical_memory": (
                "one tally, short row of marks, small wheel, or memorized "
                "cycle plus the ordinary glyph grammar"
            ),
            "visible_resets_only": [
                "line",
                "paragraph",
                "page",
            ],
            "searched_scopes": list(SCOPES),
            "searched_moduli": list(MODULI),
            "not_searched": [
                "modern ciphers",
                "arbitrary bit extraction",
                "computer-sized state spaces",
                "semantic language models",
                "one plaintext character per Voynich word",
            ],
        },
        "parameters": {
            "folds": source.N_FOLDS,
            "split": (
                "test=fold f; validation=fold f+1; fit=other two folds; "
                "complete repository quire blocks"
            ),
            "baseline": (
                "train-only register/position character trigram plus "
                "causal within-page adaptation"
            ),
            "candidate_count": len(CANDIDATES),
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
            "minimum_table_description_bits_per_entry": (
                TABLE_ENTRY_DESCRIPTION_BITS
            ),
            "selection_rule": (
                "validation log probability minus one lower-bound "
                "description bit per phase/glyph table entry; the baseline "
                "with no table is an explicit candidate"
            ),
            "page_adaptation_backoff": PAGE_BACKOFF,
            "phase_origin_nulls": args.nulls,
            "null": (
                "repeat full validation selection after independently "
                "rotating the phase origin within every reset unit"
            ),
        },
        "assets": {
            source.asset_name(args.corpus): sha256(args.corpus),
            source.asset_name(args.latin): sha256(args.latin),
        },
        "corpus_audit": dict(audit),
        "controls": controls,
        "panels": panels,
        "summary": {
            "voynich_gain_bits_per_symbol": real_summary[
                "gain_bits_per_symbol"
            ],
            "voynich_gain_bits_per_word": real_summary[
                "gain_bits_per_word"
            ],
            "voynich_positive_test_folds": real_summary[
                "positive_test_folds"
            ],
            "voynich_selected_scope_counts": real_summary[
                "selected_scope_counts"
            ],
            "voynich_selected_modulus_counts": real_summary[
                "selected_modulus_counts"
            ],
            "voynich_phase_origin_null_p_one_sided": real_summary[
                "phase_origin_null_p_one_sided"
            ],
            "controls_pass": controls_pass,
            "historical_counter_family_screen_pass": (
                primary_pass
                and (controls_pass is not False)
            ),
            "single_historical_counter_algorithm_pass": (
                primary_pass
                and single_rule_replicates
                and (controls_pass is not False)
            ),
        },
        "caveats": [
            (
                "The Latin control is a comparator, not a required null: "
                "ordinary language can have line-position conventions."
            ),
            (
                "A phase table is statistical. Even a pass would require "
                "an interpretable glyph table and independent manuscript "
                "replication before being called an algorithm."
            ),
            (
                "Hard-break transcription tokens reset counter coordinates "
                "because their exact glyph count and phase are unknown."
            ),
            (
                "The phase-offset audit is diagnostic and follows model "
                "selection; it is not a second confirmatory significance test."
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
