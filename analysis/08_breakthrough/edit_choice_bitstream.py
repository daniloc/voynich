#!/usr/bin/env python3
"""
Test bit payloads in conditional ranks of Voynich edit choices.

``edit_operation_channel.py`` asks whether an edit event is a homophonic
plaintext symbol.  This companion experiment tests a different carrier: for a
fixed local source or edit context, the selected alternative may encode a bit
through its rank among the alternatives available to the writer.

Rank dictionaries are learned on complete fit quires.  Codec, phase, polarity,
bit order, channel, and language are selected on disjoint validation quires,
then frozen for complete test quires.  The family includes Bacon-5 and 7/8-bit
ASCII.  Nulls shuffle extracted bits within each quire and repeat the complete
selection.  A Bacon payload embedded into the observed per-quire capacities is
the end-to-end positive control.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from edit_operation_channel import (  # noqa: E402
    CONTROLS,
    CORPUS,
    ROOT,
    SEED,
    EditEvent,
    TetragramLM,
    block_folds,
    changes,
    clean_control,
    extract_events,
    load_lines,
    position_bin,
    position_bucket,
    sha256,
)


N_FOLDS = 4
MIN_CONTEXT_OBSERVATIONS = 8
MIN_BITS = 250
LETTER_FRACTION_WEIGHT = 4.0
UNIGRAM_KL_WEIGHT = 3.0
MIN_ONE_FRACTION = 0.15
MAX_ONE_FRACTION = 0.85
NO_QUAD_LM = -20.0

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
class Codec:
    name: str
    width: int
    offset: int
    reverse: bool
    invert: bool


@dataclass(frozen=True)
class RankSpec:
    name: str
    feature: Callable[[EditEvent], tuple[str, str] | None]


@dataclass
class DecodeScore:
    score: float
    lm: float
    unigram_kl: float
    letters: int
    characters: int
    letter_fraction: float
    sample: str


@dataclass
class CandidateFold:
    candidate: str
    language: str
    fold: int
    codec: Codec
    fit_bits: int
    validation_bits: int
    test_bits: int
    validation_score: float
    validation_lm: float
    validation_letter_fraction: float
    test_score: float
    test_lm: float
    test_letter_fraction: float
    ceiling: float
    test_gap_to_ceiling: float
    sample: str


@dataclass
class FoldSelection:
    fold: int
    candidate: str
    language: str
    codec: Codec
    validation_score: float
    test_score: float
    test_gap_to_ceiling: float
    test_bits: int
    test_letter_fraction: float
    sample: str


def core(word: str) -> str:
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
    value = residual[: -len(suffix)] if suffix else residual
    return value or "<empty>"


def operation(event: EditEvent) -> str:
    edited = changes(event)
    return "".join(item.kind for item in edited) or "="


def operation_position(event: EditEvent) -> str:
    edited = changes(event)
    if not edited:
        return "=@="
    return operation(event) + "@" + ".".join(
        position_bin(item, event) for item in edited
    )


def target_delta(event: EditEvent) -> str:
    edited = changes(event)
    if not edited:
        return "="
    return "".join(item.target_char or "-" for item in edited)


def source_delta(event: EditEvent) -> str:
    edited = changes(event)
    if not edited:
        return "="
    return "".join(item.source_char or "-" for item in edited)


def full_delta(event: EditEvent) -> str:
    edited = changes(event)
    if not edited:
        return "="
    return ".".join(
        item.kind
        + ":"
        + position_bin(item, event)
        + ":"
        + (item.source_char or "-")
        + ">"
        + (item.target_char or "-")
        for item in edited
    )


def single_substitution(event: EditEvent) -> tuple[str, str] | None:
    edited = changes(event)
    if len(edited) != 1 or edited[0].kind != "S":
        return None
    item = edited[0]
    return item.source_char + "@" + position_bin(item, event), item.target_char


def single_insertion(event: EditEvent) -> tuple[str, str] | None:
    edited = changes(event)
    if len(edited) != 1 or edited[0].kind != "I":
        return None
    item = edited[0]
    return "I@" + position_bin(item, event), item.target_char


def single_deletion(event: EditEvent) -> tuple[str, str] | None:
    edited = changes(event)
    if len(edited) != 1 or edited[0].kind != "D":
        return None
    item = edited[0]
    return "D@" + position_bin(item, event), item.source_char


RANK_SPECS = (
    RankSpec("source_to_target", lambda event: (event.source, event.target)),
    RankSpec(
        "source_to_operation",
        lambda event: (event.source, operation(event)),
    ),
    RankSpec(
        "source_core_to_target_core",
        lambda event: (core(event.source), core(event.target)),
    ),
    RankSpec(
        "operation_position_to_target_delta",
        lambda event: (operation_position(event), target_delta(event)),
    ),
    RankSpec(
        "operation_position_to_full_delta",
        lambda event: (operation_position(event), full_delta(event)),
    ),
    RankSpec(
        "lag_position_to_operation",
        lambda event: (
            f"L{event.lag}|"
            + position_bucket(event.target_index, event.line_length),
            operation(event),
        ),
    ),
    RankSpec(
        "lag_position_to_target_delta",
        lambda event: (
            f"L{event.lag}|"
            + position_bucket(event.target_index, event.line_length),
            target_delta(event),
        ),
    ),
    RankSpec("substitution_glyph", single_substitution),
    RankSpec("insertion_glyph", single_insertion),
    RankSpec("deletion_glyph", single_deletion),
)


def flatten_events(
    event_runs: Sequence[Sequence[EditEvent]],
) -> dict[str, list[EditEvent]]:
    result: dict[str, list[EditEvent]] = defaultdict(list)
    for run in event_runs:
        for event in run:
            result[event.block].append(event)
    return dict(result)


def fit_rank_key(
    events: dict[str, list[EditEvent]],
    fit_blocks: set[str],
    spec: RankSpec,
    rank_method: str,
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for block in fit_blocks:
        for event in events.get(block, ()):
            featured = spec.feature(event)
            if featured is not None:
                context, choice = featured
                counts[context][choice] += 1
    result = {}
    for context, choices in counts.items():
        if sum(choices.values()) < MIN_CONTEXT_OBSERVATIONS or len(choices) < 2:
            continue
        if rank_method == "frequency":
            ordered = sorted(
                choices, key=lambda choice: (-choices[choice], choice)
            )
        elif rank_method == "lexical":
            ordered = sorted(choices)
        else:
            raise ValueError(rank_method)
        result[context] = {
            choice: rank % 2 for rank, choice in enumerate(ordered)
        }
    return result


def ranked_bits(
    events: dict[str, list[EditEvent]],
    blocks: set[str],
    spec: RankSpec,
    key: dict[str, dict[str, int]],
) -> dict[str, list[int]]:
    result = {}
    for block in sorted(blocks):
        bits = []
        for event in events.get(block, ()):
            featured = spec.feature(event)
            if featured is None:
                continue
            context, choice = featured
            context_key = key.get(context)
            if context_key is not None and choice in context_key:
                bits.append(context_key[choice])
        result[block] = bits
    return result


def direct_bits(
    events: dict[str, list[EditEvent]],
    blocks: set[str],
    feature: str,
) -> dict[str, list[int]]:
    result = {}
    eva_order = {
        character: rank
        for rank, character in enumerate("acdefghiklmnopqrstyxz")
    }
    for block in sorted(blocks):
        bits = []
        for event in events.get(block, ()):
            edited = changes(event)
            if feature == "lag_parity":
                value = event.lag % 2
            elif feature == "distance_parity":
                value = event.distance % 2
            elif feature == "edit_count_parity":
                value = len(edited) % 2
            elif feature == "first_position_parity":
                if not edited:
                    continue
                value = int(position_bin(edited[0], event)) % 2
            elif feature == "changed_glyph_parity":
                if not edited:
                    continue
                glyph = edited[0].target_char or edited[0].source_char
                if glyph not in eva_order:
                    continue
                value = eva_order[glyph] % 2
            else:
                raise ValueError(feature)
            bits.append(int(value))
        result[block] = bits
    return result


DIRECT_FEATURES = (
    "lag_parity",
    "distance_parity",
    "edit_count_parity",
    "first_position_parity",
    "changed_glyph_parity",
)


def codecs() -> list[Codec]:
    result = []
    for name, width in (("bacon5", 5), ("ascii7", 7), ("ascii8", 8)):
        for offset in range(width):
            for reverse in (False, True):
                for invert in (False, True):
                    result.append(Codec(name, width, offset, reverse, invert))
    return result


CODECS = codecs()


def decode_bits(bits: Sequence[int], codec: Codec) -> list[str | None]:
    result: list[str | None] = []
    selected = bits[codec.offset :]
    for start in range(0, len(selected) - codec.width + 1, codec.width):
        group = list(selected[start : start + codec.width])
        if codec.invert:
            group = [1 - value for value in group]
        if codec.reverse:
            group.reverse()
        value = 0
        for bit in group:
            value = value * 2 + bit
        if codec.name == "bacon5":
            result.append(chr(97 + value) if value < 26 else None)
        elif value == 32:
            result.append(None)
        elif 65 <= value <= 90:
            result.append(chr(value + 32))
        elif 97 <= value <= 122:
            result.append(chr(value))
        else:
            result.append(None)
    return result


def score_decoded(
    decoded_blocks: Sequence[Sequence[str | None]],
    lm: TetragramLM,
) -> DecodeScore:
    score_sum = 0.0
    quads = 0
    letters = 0
    characters = 0
    sample_parts = []
    letter_counts = np.zeros(lm.width, dtype=float)
    remaining = 360
    for decoded in decoded_blocks:
        characters += len(decoded)
        letters += sum(character is not None for character in decoded)
        run: list[int] = []
        rendered = []
        for character in decoded:
            if character is None or character not in lm.char_to_id:
                if len(run) >= 4:
                    for a, b, c, d in zip(
                        run, run[1:], run[2:], run[3:]
                    ):
                        score_sum += float(lm.logp[a, b, c, d])
                        quads += 1
                run = []
                rendered.append("?")
            else:
                character_id = lm.char_to_id[character]
                run.append(character_id)
                letter_counts[character_id] += 1
                rendered.append(character)
        if len(run) >= 4:
            for a, b, c, d in zip(run, run[1:], run[2:], run[3:]):
                score_sum += float(lm.logp[a, b, c, d])
                quads += 1
        if remaining > 0:
            text = "".join(rendered)[:remaining]
            sample_parts.append(text)
            remaining -= len(text)
    lm_score = score_sum / quads if quads else NO_QUAD_LM
    fraction = letters / max(1, characters)
    probabilities = letter_counts / max(1.0, letter_counts.sum())
    mask = probabilities > 0
    divergence = float(
        np.sum(
            probabilities[mask]
            * np.log(
                probabilities[mask] / lm.all_probabilities[mask]
            )
        )
    )
    combined = (
        lm_score
        + LETTER_FRACTION_WEIGHT * math.log(max(fraction, 1e-9))
        - UNIGRAM_KL_WEIGHT * divergence
    )
    return DecodeScore(
        score=combined,
        lm=lm_score,
        unigram_kl=divergence,
        letters=letters,
        characters=characters,
        letter_fraction=fraction,
        sample="/".join(sample_parts),
    )


def score_codec(
    bits_by_block: dict[str, list[int]],
    blocks: set[str],
    codec: Codec,
    lm: TetragramLM,
) -> DecodeScore:
    return score_decoded(
        [
            decode_bits(bits_by_block.get(block, ()), codec)
            for block in sorted(blocks)
        ],
        lm,
    )


def total_bits(
    bits_by_block: dict[str, list[int]],
    blocks: set[str],
) -> int:
    return sum(len(bits_by_block.get(block, ())) for block in blocks)


def best_codec(
    bits_by_block: dict[str, list[int]],
    blocks: set[str],
    lm: TetragramLM,
) -> tuple[Codec, DecodeScore]:
    scored = [
        (codec, score_codec(bits_by_block, blocks, codec, lm))
        for codec in CODECS
    ]
    return max(
        scored,
        key=lambda item: (
            item[1].score,
            item[1].letter_fraction,
            -item[0].width,
            -item[0].offset,
        ),
    )


def candidate_bits(
    events: dict[str, list[EditEvent]],
    all_blocks: set[str],
    fit_blocks: set[str],
) -> dict[str, dict[str, list[int]]]:
    result = {}
    for feature in DIRECT_FEATURES:
        result[f"direct/{feature}"] = direct_bits(
            events, all_blocks, feature
        )
    for spec in RANK_SPECS:
        for rank_method in ("frequency", "lexical"):
            key = fit_rank_key(events, fit_blocks, spec, rank_method)
            result[f"rank/{spec.name}/{rank_method}"] = ranked_bits(
                events, all_blocks, spec, key
            )
    return result


def run_fold(
    event_sources: dict[str, dict[str, list[EditEvent]]],
    blocks: Sequence[str],
    folds: dict[str, int],
    fold: int,
    lms: dict[str, TetragramLM],
    ceilings: dict[str, float],
    bit_override: dict[str, dict[str, list[int]]] | None = None,
) -> tuple[list[CandidateFold], FoldSelection]:
    all_blocks = set(blocks)
    test_blocks = {block for block in blocks if folds[block] == fold}
    validation_blocks = {
        block for block in blocks if folds[block] == (fold + 1) % N_FOLDS
    }
    fit_blocks = all_blocks - test_blocks - validation_blocks
    panels = {}
    if bit_override is not None:
        panels.update(bit_override)
    else:
        for source_mode, events in event_sources.items():
            for name, bits in candidate_bits(
                events, all_blocks, fit_blocks
            ).items():
                panels[f"{source_mode}/{name}"] = bits

    rows = []
    for candidate, bits in sorted(panels.items()):
        fit_count = total_bits(bits, fit_blocks)
        validation_count = total_bits(bits, validation_blocks)
        test_count = total_bits(bits, test_blocks)
        if min(fit_count, validation_count, test_count) < MIN_BITS:
            continue
        validation_values = [
            bit
            for block in validation_blocks
            for bit in bits.get(block, ())
        ]
        test_values = [
            bit for block in test_blocks for bit in bits.get(block, ())
        ]
        validation_ones = mean(validation_values)
        test_ones = mean(test_values)
        if not (
            MIN_ONE_FRACTION <= validation_ones <= MAX_ONE_FRACTION
            and MIN_ONE_FRACTION <= test_ones <= MAX_ONE_FRACTION
        ):
            continue
        for language, lm in lms.items():
            codec, validation = best_codec(
                bits, validation_blocks, lm
            )
            test = score_codec(bits, test_blocks, codec, lm)
            rows.append(
                CandidateFold(
                    candidate=candidate,
                    language=language,
                    fold=fold,
                    codec=codec,
                    fit_bits=fit_count,
                    validation_bits=validation_count,
                    test_bits=test_count,
                    validation_score=validation.score,
                    validation_lm=validation.lm,
                    validation_letter_fraction=validation.letter_fraction,
                    test_score=test.score,
                    test_lm=test.lm,
                    test_letter_fraction=test.letter_fraction,
                    ceiling=ceilings[language],
                    test_gap_to_ceiling=test.score - ceilings[language],
                    sample=test.sample,
                )
            )
    selected = max(
        rows,
        key=lambda row: (
            row.validation_score,
            row.validation_letter_fraction,
        ),
    )
    return rows, FoldSelection(
        fold=fold,
        candidate=selected.candidate,
        language=selected.language,
        codec=selected.codec,
        validation_score=selected.validation_score,
        test_score=selected.test_score,
        test_gap_to_ceiling=selected.test_gap_to_ceiling,
        test_bits=selected.test_bits,
        test_letter_fraction=selected.test_letter_fraction,
        sample=selected.sample,
    )


def run_panel(
    event_sources: dict[str, dict[str, list[EditEvent]]],
    blocks: Sequence[str],
    folds: dict[str, int],
    lms: dict[str, TetragramLM],
    ceilings: dict[str, float],
    bit_override: dict[str, dict[str, list[int]]] | None = None,
) -> tuple[list[CandidateFold], list[FoldSelection]]:
    rows = []
    selections = []
    for fold in range(N_FOLDS):
        fold_rows, selection = run_fold(
            event_sources,
            blocks,
            folds,
            fold,
            lms,
            ceilings,
            bit_override,
        )
        rows.extend(fold_rows)
        selections.append(selection)
    return rows, selections


def shuffle_events(
    event_sources: dict[str, dict[str, list[EditEvent]]],
    rng: np.random.Generator,
) -> dict[str, dict[str, list[EditEvent]]]:
    result = {}
    for source_mode, events_by_block in event_sources.items():
        shuffled_by_block: dict[str, list[EditEvent]] = {}
        for block, events in events_by_block.items():
            values = events.copy()
            rng.shuffle(values)
            shuffled_by_block[block] = values
        result[source_mode] = shuffled_by_block
    return result


def bits_for_all_candidates(
    event_sources: dict[str, dict[str, list[EditEvent]]],
    blocks: Sequence[str],
) -> dict[str, dict[str, list[int]]]:
    # Full-corpus rank tables are used only for null generation.  Real
    # evaluation refits ranks inside every fold.
    all_blocks = set(blocks)
    result = {}
    for source_mode, events in event_sources.items():
        for name, bits in candidate_bits(
            events, all_blocks, all_blocks
        ).items():
            result[f"{source_mode}/{name}"] = bits
    return result


def bacon_bits(text: str) -> list[int]:
    bits = []
    for character in text:
        value = ord(character) - 97
        bits.extend((value >> shift) & 1 for shift in range(4, -1, -1))
    return bits


def positive_panels(
    template: dict[str, list[int]],
    plaintext: str,
) -> dict[str, dict[str, list[int]]]:
    source_bits = bacon_bits(plaintext)
    cursor = 0
    result = {}
    for block in sorted(template):
        capacity = len(template[block])
        usable = capacity - capacity % 5
        values = [
            source_bits[(cursor + index) % len(source_bits)]
            for index in range(usable)
        ]
        cursor = (cursor + usable) % len(source_bits)
        result[block] = values
    return {"positive/bacon5": result}


def print_selections(
    title: str, selections: Sequence[FoldSelection]
) -> None:
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)
    for row in selections:
        codec = row.codec
        print(
            f"fold={row.fold} {row.language:<7} "
            f"{row.candidate:<52} "
            f"{codec.name}:off{codec.offset}:"
            f"rev{int(codec.reverse)}:inv{int(codec.invert)} "
            f"val={row.validation_score:.4f} "
            f"test={row.test_score:.4f} "
            f"gap={row.test_gap_to_ceiling:+.4f} "
            f"bits={row.test_bits} letters={row.test_letter_fraction:.3f}"
        )
        print(" ", row.sample[:220])
    print(
        "mean test gap to ceiling="
        f"{mean(row.test_gap_to_ceiling for row in selections):+.4f}"
    )


def make_lms() -> tuple[dict[str, TetragramLM], dict[str, str], dict[str, float]]:
    lms = {}
    holdouts = {}
    ceilings = {}
    for language, path in CONTROLS.items():
        control = clean_control(path, language)
        cut = len(control) * 2 // 3
        lms[language] = TetragramLM(control[:cut])
        holdouts[language] = control[cut:]
        ceilings[language] = lms[language].mean_score(holdouts[language])
    return lms, holdouts, ceilings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nulls", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    raw_lines, blocks, corpus_audit = load_lines()
    folds = block_folds(blocks)
    event_sources = {
        mode: flatten_events(extract_events(raw_lines, mode))
        for mode in ("adjacent", "nearest8")
    }
    lms, holdouts, ceilings = make_lms()

    real_rows, real_selections = run_panel(
        event_sources, blocks, folds, lms, ceilings
    )
    print(
        f"corpus lines={corpus_audit['prose_lines']} "
        f"words={corpus_audit['eligible_words']} blocks={len(blocks)} "
        f"tested fold rows={len(real_rows)} codecs={len(CODECS)}"
    )
    print_selections("REAL CONDITIONAL-RANK BIT CHANNEL", real_selections)

    # Use the largest direct stream as the capacity template.
    full_panels = bits_for_all_candidates(event_sources, blocks)
    template_name = max(
        full_panels,
        key=lambda name: sum(
            len(bits) for bits in full_panels[name].values()
        ),
    )
    positive_rows = []
    positive_selections = []
    for language in sorted(lms):
        panel = positive_panels(
            full_panels[template_name], holdouts[language]
        )
        rows, selections = run_panel(
            event_sources,
            blocks,
            folds,
            {language: lms[language]},
            {language: ceilings[language]},
            panel,
        )
        positive_rows.extend(rows)
        positive_selections.extend(selections)
    print_selections(
        "EMBEDDED BACON-5 POSITIVE CONTROL", positive_selections
    )

    # Event order is shuffled within quire.  Context/choice counts and direct
    # feature imbalance are therefore exact, while rank dictionaries are still
    # refit inside every null fold by the same leakage-free procedure.
    null_rows = []
    rng = np.random.default_rng(SEED + 500_000)
    for replicate in range(args.nulls):
        shuffled = shuffle_events(event_sources, rng)
        _rows, selections = run_panel(
            shuffled,
            blocks,
            folds,
            lms,
            ceilings,
        )
        row = {
            "replicate": replicate,
            "mean_test_gap_to_ceiling": mean(
                item.test_gap_to_ceiling for item in selections
            ),
            "max_test_score": max(item.test_score for item in selections),
            "selections": [asdict(item) for item in selections],
        }
        null_rows.append(row)
        if args.progress and (
            replicate < 4 or (replicate + 1) % 10 == 0
        ):
            print(
                f"null {replicate + 1}/{args.nulls}: "
                f"mean_gap={row['mean_test_gap_to_ceiling']:+.4f} "
                f"max_test={row['max_test_score']:.4f}"
            )

    real_gap = mean(
        item.test_gap_to_ceiling for item in real_selections
    )
    null_gaps = [
        float(row["mean_test_gap_to_ceiling"]) for row in null_rows
    ]
    positive_gap = mean(
        item.test_gap_to_ceiling for item in positive_selections
    )
    summary = {
        "real_mean_test_gap_to_ceiling": real_gap,
        "null_mean_gap_mean": mean(null_gaps),
        "null_mean_gap_max": max(null_gaps),
        "null_empirical_p": (
            1 + sum(value >= real_gap for value in null_gaps)
        )
        / (1 + len(null_gaps)),
        "positive_mean_gap_to_ceiling": positive_gap,
        "positive_pass": positive_gap > -0.05,
        "real_pass": real_gap > -0.50 and real_gap > max(null_gaps),
    }
    print("\n" + "=" * 112)
    print("GATE SUMMARY")
    print("=" * 112)
    for key, value in summary.items():
        print(f"{key}: {value}")

    payload = {
        "experiment": "edit_choice_bitstream",
        "seed": SEED,
        "parameters": {
            "nulls": args.nulls,
            "minimum_context_observations": MIN_CONTEXT_OBSERVATIONS,
            "minimum_bits_per_split": MIN_BITS,
            "letter_fraction_weight": LETTER_FRACTION_WEIGHT,
            "unigram_kl_weight": UNIGRAM_KL_WEIGHT,
            "minimum_one_fraction": MIN_ONE_FRACTION,
            "maximum_one_fraction": MAX_ONE_FRACTION,
            "rank_specs": [spec.name for spec in RANK_SPECS],
            "direct_features": list(DIRECT_FEATURES),
            "codecs": [asdict(codec) for codec in CODECS],
        },
        "assets": {
            str(CORPUS.relative_to(ROOT)): sha256(CORPUS),
            **{
                str(path.relative_to(ROOT)): sha256(path)
                for path in CONTROLS.values()
            },
        },
        "corpus_audit": dict(corpus_audit),
        "blocks": blocks,
        "folds": folds,
        "ceilings": ceilings,
        "template": template_name,
        "real_rows": [asdict(row) for row in real_rows],
        "real_selections": [asdict(row) for row in real_selections],
        "positive_rows": [asdict(row) for row in positive_rows],
        "positive_selections": [
            asdict(row) for row in positive_selections
        ],
        "nulls": null_rows,
        "summary": summary,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
