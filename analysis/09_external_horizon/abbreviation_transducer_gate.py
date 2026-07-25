#!/usr/bin/env python3
"""
Historically constrained variable-length abbreviation-transducer gate.

This is not a translation engine.  It asks a narrower question:

    Does a small, declared set of medieval-Latin abbreviation expansions make
    a Latin character model fit unseen Voynich production blocks better than
    the same fitted key restricted to one plaintext character per EVA unit?

The comparison is deliberately nested.  Both models:

* tokenize EVA bench/gallows composites as units;
* fit the same 22-way base substitution plus five possible homophones;
* preserve every observed word, core, local copy, line position, and block;
* fit on complete quires, select restarts on separate complete quires, and
  score untouched complete quires.

The variable model starts from the fitted one-character key and can change only
five frozen, position-conditioned slots.  Their expansion inventory is checked
against a compact derivative of Cappelli's XIV/XV Latin records.  Cappelli is
used to license chunks, not to assert that an EVA glyph is the same sign:

    EVA y, final       -> us | um
    EVA y, non-final   -> con | com | cum
    EVA q, initial     -> que | qui
    EVA ch, any        -> et | er
    EVA sh, any        -> per | pro | pre

Every slot also has the null choice (its base character).  The y/figure-9 pool
is the strongest palaeographic candidate.  The q and bench pools are bounded
structural probes and should not be read as sign identifications.

Controls:

* SYNTHETIC_LATIN is real held-out Latin encrypted by this exact transducer
  family under a planted key and reflowed through the Voynich line/block
  template.
* SURFACE_EXCHANGE shuffles whole words only inside
  block/section/Currier/hand/line-position cells, retaining exact words, cores,
  and the production layout.
* PROCEDURAL_COPY replicates use the repository's prefix/core/suffix sampling
  and local-copy/mutation schema, with the observed metadata and line lengths.
* If data/external/historical_abbreviation_controls.json is present, a
  no-fitting, book-separated Nuremberg Letterbooks inventory-coverage panel is
  reported as an external calibration.

The primary statistic is the pooled held-out joint-code gain per common source
observation.  Every output character pays its full Latin trigram code length,
and the variable arm additionally pays a finite channel-map description cost.
Optimization never sees a test block.  A positive pilot result would identify
a mechanism worth testing against a diplomatic glyph transcription; it would
not by itself identify Latin or recover plaintext.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus" / "corpus.json"
LATIN = ROOT / "data" / "controls" / "latin.txt"
HISTORICAL = (
    ROOT / "data" / "external" / "historical_abbreviation_controls.json"
)
OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "external_horizon_abbreviation_transducer_gate.json"
)


def asset_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


SEED = 20260724
N_FOLDS = 4
ALPHABET = tuple("abcdefghilmnopqrstuxyz")
SPACE_STATE = ("<SPACE>", "space")
MULTI_UNITS = ("cfh", "ckh", "cph", "cth", "ch", "sh")
# Four units carry the declared variable slots.  The fifth (`z`) is an
# ultra-rare homophone slot so the remaining 22 ordinary units can retain a
# strict bijection to the normalized Latin alphabet in both nested models.
SPECIAL_UNITS = ("y", "q", "ch", "sh", "z")

# This inventory is frozen before corpus fitting.  The null option is inserted
# separately and means "emit the fitted one-character base value".
ABBREVIATION_OPTIONS: dict[str, tuple[str, ...]] = {
    "y:final": ("us", "um"),
    "y:nonfinal": ("con", "com", "cum"),
    "q:initial": ("que", "qui"),
    "ch:any": ("et", "er"),
    "sh:any": ("per", "pro", "pre"),
}
VARIABLE_MODEL_BITS = sum(
    math.log2(len(options) + 1)
    for options in ABBREVIATION_OPTIONS.values()
)

GEN_PREFIXES = tuple(sorted(
    (
        "qok", "qot", "qo", "ok", "ot", "o", "y", "ch", "sh", "d",
        "cth", "ckh", "cph",
    ),
    key=len,
    reverse=True,
))
GEN_SUFFIXES = tuple(sorted(
    (
        "eedy", "eody", "edy", "aiin", "ain", "iin", "dy", "ol",
        "or", "ar", "al", "am", "y",
    ),
    key=len,
    reverse=True,
))

State = tuple[str, str]
Context = tuple[State, State, State]


@dataclass(frozen=True)
class UnitLine:
    block: str
    section: str
    currier: str
    hand: str
    words: tuple[tuple[str, ...], ...]


@dataclass
class Key:
    base: dict[str, str]
    overrides: dict[str, Optional[str]]


@dataclass(frozen=True)
class SplitScore:
    language_log2_sum: float
    language_code_bits: float
    output_characters: int
    source_observations: int
    expansion_factor: float
    model_description_bits: float
    joint_mdl_log2_sum: float
    joint_log2_per_source: float
    language_log2_per_output_character: float


@dataclass(frozen=True)
class FitResult:
    model: str
    restart: int
    train: SplitScore
    validation: SplitScore
    test: SplitScore
    key: dict[str, object]
    sample: str


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_blocks: tuple[str, ...]
    validation_blocks: tuple[str, ...]
    test_blocks: tuple[str, ...]
    train_events: int
    validation_events: int
    test_events: int
    mono: FitResult
    variable: FitResult
    heldout_joint_gain_bits_per_source: float
    planted_emission_accuracy: Optional[float]
    planted_abbreviation_event_accuracy: Optional[float]
    planted_abbreviation_correct_events: Optional[int]
    planted_abbreviation_events: Optional[int]
    planted_override_map_accuracy: Optional[float]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_latin(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    marker = "*** start of the project gutenberg ebook"
    marker_index = text.find(marker)
    if marker_index >= 0:
        text = text[text.find("\n", marker_index) + 1 :]
    text = (
        text.replace("j", "i")
        .replace("v", "u")
        .replace("w", "u")
        .replace("k", "c")
    )
    text = re.sub(f"[^{''.join(ALPHABET)}]+", " ", text)
    return re.sub(r" +", " ", text).strip()


class TrigramLM:
    def __init__(self, text: str, smoothing: float = 0.2) -> None:
        symbols = (" ",) + ALPHABET
        self.symbols = symbols
        self.smoothing = smoothing
        bigrams: Counter[tuple[str, str]] = Counter()
        trigrams: Counter[tuple[str, str, str]] = Counter()
        padded = "  " + text + " "
        for left, middle, right in zip(padded, padded[1:], padded[2:]):
            bigrams[(left, middle)] += 1
            trigrams[(left, middle, right)] += 1
        width = len(symbols)
        self.logp = {
            (left, middle, right): math.log2(
                (trigrams[(left, middle, right)] + smoothing)
                / (bigrams[(left, middle)] + smoothing * width)
            )
            for left in symbols
            for middle in symbols
            for right in symbols
        }
        unigrams = Counter(text)
        self.frequency_order = tuple(
            character
            for character, _count in sorted(
                ((character, unigrams[character]) for character in ALPHABET),
                key=lambda item: (-item[1], item[0]),
            )
        )

    def score_text(self, text: str) -> float:
        padded = "  " + text
        values = [
            self.logp[(left, middle, right)]
            for left, middle, right in zip(
                padded, padded[1:], padded[2:]
            )
        ]
        return sum(values) / len(values) if values else float("-inf")


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


def tokenize_eva(word: str) -> tuple[str, ...]:
    result = []
    index = 0
    while index < len(word):
        unit = next(
            (
                candidate
                for candidate in MULTI_UNITS
                if word.startswith(candidate, index)
            ),
            None,
        )
        if unit is None:
            unit = word[index]
        result.append(unit)
        index += len(unit)
    return tuple(result)


def load_voynich(path: Path) -> tuple[list[UnitLine], Counter[str]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    audit: Counter[str] = Counter()
    for folio, raw_lines in source["folios"].items():
        meta = source["meta"].get(folio, {})
        for raw in raw_lines:
            if locus_type(raw["locus"]) != "P":
                continue
            audit["prose_lines"] += 1
            words = []
            for word in raw["words"]:
                audit["source_words"] += 1
                if (
                    "?" in word
                    or len(word) < 2
                    or not word.isalpha()
                ):
                    audit["excluded_words"] += 1
                    continue
                units = tokenize_eva(word)
                if units:
                    words.append(units)
                    audit["eligible_words"] += 1
                    audit["units"] += len(units)
            if words:
                lines.append(UnitLine(
                    block=str(meta.get("Q", "?")),
                    section=str(meta.get("I", "?")),
                    currier=str(meta.get("L", "?")),
                    hand=str(meta.get("H", "?")),
                    words=tuple(words),
                ))
    return lines, audit


def annotate_word(units: Sequence[str]) -> list[State]:
    if len(units) == 1:
        return [(units[0], "single")]
    result = []
    for index, unit in enumerate(units):
        position = (
            "initial"
            if index == 0
            else "final"
            if index == len(units) - 1
            else "medial"
        )
        result.append((unit, position))
    return result


def slot_for_state(state: State) -> Optional[str]:
    unit, position = state
    if unit == "y":
        return "y:final" if position in {"final", "single"} else "y:nonfinal"
    if unit == "q" and position in {"initial", "single"}:
        return "q:initial"
    if unit == "ch":
        return "ch:any"
    if unit == "sh":
        return "sh:any"
    return None


def context_counts(
    lines: Sequence[UnitLine],
) -> tuple[dict[str, Counter[Context]], Counter[str]]:
    by_block: dict[str, Counter[Context]] = defaultdict(Counter)
    units: Counter[str] = Counter()
    for line in lines:
        sequence: list[State] = [SPACE_STATE, SPACE_STATE]
        for word in line.words:
            states = annotate_word(word)
            sequence.extend(states)
            sequence.append(SPACE_STATE)
            units.update(word)
        for index in range(2, len(sequence)):
            by_block[line.block][tuple(sequence[index - 2 : index + 1])] += 1
    return dict(by_block), units


def merge_counts(
    counts_by_block: dict[str, Counter[Context]],
    blocks: Iterable[str],
) -> Counter[Context]:
    result: Counter[Context] = Counter()
    for block in sorted(blocks):
        result.update(counts_by_block.get(block, {}))
    return result


def emission(state: State, key: Key) -> str:
    if state == SPACE_STATE:
        return " "
    slot = slot_for_state(state)
    if slot is not None and key.overrides.get(slot) is not None:
        value = key.overrides[slot]
        assert value is not None
        return value
    return key.base[state[0]]


def context_contribution(
    context: Context,
    count: int,
    key: Key,
    lm: TrigramLM,
) -> tuple[float, int]:
    left, middle, current = (emission(state, key) for state in context)
    combined = left + middle + current
    start = len(left) + len(middle)
    bits = 0.0
    for index in range(start, len(combined)):
        bits += lm.logp[tuple(combined[index - 2 : index + 1])]
    return bits * count, len(current) * count


class IncrementalScorer:
    def __init__(
        self,
        contexts: Counter[Context],
        key: Key,
        lm: TrigramLM,
    ) -> None:
        self.contexts = sorted(contexts.items())
        self.key = key
        self.lm = lm
        self.by_unit: dict[str, set[int]] = defaultdict(set)
        for index, (context, _count) in enumerate(self.contexts):
            for state in context:
                if state != SPACE_STATE:
                    self.by_unit[state[0]].add(index)
        self.values = [
            context_contribution(context, count, key, lm)
            for context, count in self.contexts
        ]
        self.bits = sum(value[0] for value in self.values)
        self.characters = sum(value[1] for value in self.values)
        self.source_observations = sum(
            count for _context, count in self.contexts
        )

    @property
    def score(self) -> float:
        return self.bits / self.source_observations

    def proposal(
        self, affected_units: Iterable[str]
    ) -> tuple[float, float, int, dict[int, tuple[float, int]]]:
        affected: set[int] = set()
        for unit in affected_units:
            affected.update(self.by_unit.get(unit, ()))
        replacements = {}
        old_bits = old_characters = 0
        new_bits = new_characters = 0
        for index in sorted(affected):
            old = self.values[index]
            context, count = self.contexts[index]
            new = context_contribution(context, count, self.key, self.lm)
            replacements[index] = new
            old_bits += old[0]
            old_characters += old[1]
            new_bits += new[0]
            new_characters += new[1]
        bits = self.bits - old_bits + new_bits
        characters = self.characters - old_characters + new_characters
        return (
            bits / self.source_observations,
            bits,
            characters,
            replacements,
        )

    def commit(
        self,
        bits: float,
        characters: int,
        replacements: dict[int, tuple[float, int]],
    ) -> None:
        self.bits = bits
        self.characters = characters
        for index, value in replacements.items():
            self.values[index] = value


def pooled_values(
    contexts: Counter[Context],
    key: Key,
    lm: TrigramLM,
) -> tuple[float, int, int]:
    """Raw LM log2 sum, decoded characters, and common source observations."""
    return pooled_values_from_items(sorted(contexts.items()), key, lm)


def pooled_values_from_items(
    items: Sequence[tuple[Context, int]],
    key: Key,
    lm: TrigramLM,
) -> tuple[float, int, int]:
    bits = 0.0
    characters = 0
    source_observations = 0
    for context, count in items:
        value_bits, value_characters = context_contribution(
            context, count, key, lm
        )
        bits += value_bits
        characters += value_characters
        source_observations += count
    return bits, characters, source_observations


def split_score(
    contexts: Counter[Context],
    key: Key,
    lm: TrigramLM,
    model_description_bits: float,
) -> SplitScore:
    language_log2_sum, output_characters, source_observations = pooled_values(
        contexts, key, lm
    )
    joint = language_log2_sum - model_description_bits
    return SplitScore(
        language_log2_sum=language_log2_sum,
        language_code_bits=-language_log2_sum,
        output_characters=output_characters,
        source_observations=source_observations,
        expansion_factor=output_characters / source_observations,
        model_description_bits=model_description_bits,
        joint_mdl_log2_sum=joint,
        joint_log2_per_source=joint / source_observations,
        language_log2_per_output_character=(
            language_log2_sum / output_characters
        ),
    )


def source_unit_counts(contexts: Counter[Context]) -> Counter[str]:
    """Counts each source state once, using fit contexts only."""
    result: Counter[str] = Counter()
    for context, count in contexts.items():
        state = context[-1]
        if state != SPACE_STATE:
            result[state[0]] += count
    return result


def serialize_key(key: Key) -> dict[str, object]:
    return {
        "base": dict(sorted(key.base.items())),
        "overrides": dict(sorted(key.overrides.items())),
    }


def deserialize_key(payload: dict[str, object]) -> Key:
    return Key(
        base=dict(payload["base"]),  # type: ignore[arg-type]
        overrides=dict(payload["overrides"]),  # type: ignore[arg-type]
    )


def initial_key(
    units: Sequence[str],
    unit_counts: Counter[str],
    lm: TrigramLM,
    rng: random.Random,
    randomized: bool,
) -> Key:
    primary = sorted(set(units) - set(SPECIAL_UNITS))
    if len(primary) != len(ALPHABET):
        raise ValueError(
            "expected exactly 22 ordinary EVA units after the five declared "
            f"special units, found {len(primary)}: {primary}"
        )
    if randomized:
        targets = list(ALPHABET)
        rng.shuffle(targets)
        ordered_units = primary
    else:
        ordered_units = sorted(
            primary, key=lambda unit: (-unit_counts[unit], unit)
        )
        targets = list(lm.frequency_order)
    base = dict(zip(ordered_units, targets))
    for index, unit in enumerate(SPECIAL_UNITS):
        if randomized:
            base[unit] = rng.choice(ALPHABET)
        else:
            base[unit] = lm.frequency_order[
                min(index, len(lm.frequency_order) - 1)
            ]
    return Key(
        base=base,
        overrides={slot: None for slot in ABBREVIATION_OPTIONS},
    )


def key_units(key: Key) -> tuple[list[str], list[str]]:
    primary = sorted(set(key.base) - set(SPECIAL_UNITS))
    special = sorted(set(key.base) & set(SPECIAL_UNITS))
    return primary, special


def optimize_mono_key(
    train: Counter[Context],
    validation: Counter[Context],
    test: Counter[Context],
    lm: TrigramLM,
    units: Sequence[str],
    train_unit_counts: Counter[str],
    steps: int,
    restarts: int,
    seed: int,
) -> FitResult:
    candidates = []
    for restart in range(restarts):
        rng = random.Random(seed + restart * 104729)
        key = initial_key(
            units,
            train_unit_counts,
            lm,
            rng,
            randomized=restart > 0,
        )
        key.overrides = {
            slot: None for slot in ABBREVIATION_OPTIONS
        }
        scorer = IncrementalScorer(train, key, lm)
        primary, special = key_units(key)
        best_train_score = scorer.score
        best_train_key = deepcopy(key)
        for step in range(steps):
            progress = step / max(1, steps - 1)
            temperature = 0.025 * (0.0002 / 0.025) ** progress
            move_draw = rng.random()
            if move_draw < 0.88:
                left, right = rng.sample(primary, 2)
                key.base[left], key.base[right] = (
                    key.base[right],
                    key.base[left],
                )
                changed = (left, right)
                undo = ("swap", left, right)
            else:
                unit = rng.choice(special)
                old = key.base[unit]
                key.base[unit] = rng.choice(ALPHABET)
                changed = (unit,)
                undo = ("base", unit, old)
            proposed, bits, characters, replacements = scorer.proposal(
                changed
            )
            delta = proposed - scorer.score
            accept = (
                delta >= 0
                or rng.random() < math.exp(delta / max(temperature, 1e-12))
            )
            if accept:
                scorer.commit(bits, characters, replacements)
                if scorer.score > best_train_score:
                    best_train_score = scorer.score
                    best_train_key = deepcopy(key)
            else:
                if undo[0] == "swap":
                    _kind, left, right = undo
                    key.base[left], key.base[right] = (
                        key.base[right],
                        key.base[left],
                    )
                elif undo[0] == "base":
                    _kind, unit, old = undo
                    key.base[unit] = old
        validation_score = IncrementalScorer(
            validation, best_train_key, lm
        ).score
        candidates.append((
            validation_score,
            restart,
            best_train_score,
            best_train_key,
        ))
    validation_score, restart, train_score, winner = max(
        candidates,
        key=lambda row: (row[0], row[2], -row[1]),
    )
    return FitResult(
        model="mono",
        restart=restart,
        train=split_score(train, winner, lm, 0.0),
        validation=split_score(validation, winner, lm, 0.0),
        test=split_score(test, winner, lm, 0.0),
        key=serialize_key(winner),
        sample="",
    )


def fit_abbreviation_overrides(
    train: Counter[Context],
    validation: Counter[Context],
    test: Counter[Context],
    lm: TrigramLM,
    mono: FitResult,
) -> FitResult:
    """
    Exhaust the declared override family while holding the mono base key fixed.

    Freezing the base is essential: the variable arm must not receive an extra
    substitution-key search budget and call the resulting gain "abbreviation".
    Override values are fitted on train only; validation and test are scored
    after selection.
    """
    mono_key = deserialize_key(mono.key)
    slots = tuple(sorted(ABBREVIATION_OPTIONS))
    choices = [
        (None,) + ABBREVIATION_OPTIONS[slot] for slot in slots
    ]
    train_items = sorted(train.items())
    winner: Optional[tuple[float, Key]] = None
    for values in itertools.product(*choices):
        key = Key(
            base=mono_key.base.copy(),
            overrides=dict(zip(slots, values)),
        )
        language_log2_sum, _characters, _source = pooled_values_from_items(
            train_items, key, lm
        )
        if winner is None or language_log2_sum > winner[0]:
            winner = (language_log2_sum, key)
    assert winner is not None
    _train_language_log2_sum, key = winner
    return FitResult(
        model="variable",
        restart=mono.restart,
        train=split_score(
            train, key, lm, VARIABLE_MODEL_BITS
        ),
        validation=split_score(
            validation, key, lm, VARIABLE_MODEL_BITS
        ),
        test=split_score(
            test, key, lm, VARIABLE_MODEL_BITS
        ),
        key=serialize_key(key),
        sample="",
    )


def block_folds(
    counts_by_block: dict[str, Counter[Context]],
) -> tuple[dict[str, int], list[int]]:
    """Greedy size balance; block identity and content are not inspected."""
    loads = [0] * N_FOLDS
    assignment = {}
    sizes = {
        block: sum(counts.values())
        for block, counts in counts_by_block.items()
    }
    for block in sorted(sizes, key=lambda value: (-sizes[value], value)):
        fold = min(range(N_FOLDS), key=lambda value: (loads[value], value))
        assignment[block] = fold
        loads[fold] += sizes[block]
    return assignment, loads


def render_sample(
    lines: Sequence[UnitLine],
    blocks: set[str],
    key: Key,
    maximum: int = 360,
) -> str:
    result = []
    remaining = maximum
    for line in lines:
        if line.block not in blocks:
            continue
        words = []
        for word in line.words:
            text = "".join(emission(state, key) for state in annotate_word(word))
            words.append(text)
        part = " ".join(words)
        result.append(part[:remaining])
        remaining -= len(part)
        if remaining <= 0:
            break
    return " / ".join(result)[:maximum]


def emission_accuracy(
    counts: Counter[Context],
    fitted: Key,
    planted: Key,
) -> float:
    correct = total = 0
    for context, count in counts.items():
        state = context[-1]
        if state == SPACE_STATE:
            continue
        total += count
        correct += count * int(
            emission(state, fitted) == emission(state, planted)
        )
    return correct / total if total else float("nan")


def abbreviation_accuracy(
    counts: Counter[Context],
    fitted: Key,
    planted: Key,
) -> tuple[float, int, int, float]:
    """Event-weighted active-slot accuracy and unweighted exact slot recovery."""
    correct = total = 0
    for context, count in counts.items():
        state = context[-1]
        slot = slot_for_state(state)
        if slot is None or planted.overrides.get(slot) is None:
            continue
        total += count
        correct += count * int(
            emission(state, fitted) == emission(state, planted)
        )
    slots = [
        slot
        for slot, value in planted.overrides.items()
        if value is not None
    ]
    exact = sum(
        fitted.overrides.get(slot) == planted.overrides[slot]
        for slot in slots
    )
    return (
        correct / total if total else float("nan"),
        correct,
        total,
        exact / len(slots) if slots else float("nan"),
    )


def run_panel(
    name: str,
    lines: Sequence[UnitLine],
    lm: TrigramLM,
    steps: int,
    restarts: int,
    seed: int,
    planted: Optional[Key] = None,
) -> dict[str, object]:
    counts_by_block, full_unit_counts = context_counts(lines)
    folds, loads = block_folds(counts_by_block)
    units = sorted(full_unit_counts)
    results = []
    for fold in range(N_FOLDS):
        test_blocks = {block for block, value in folds.items() if value == fold}
        validation_blocks = {
            block
            for block, value in folds.items()
            if value == (fold + 1) % N_FOLDS
        }
        train_blocks = set(folds) - test_blocks - validation_blocks
        train = merge_counts(counts_by_block, train_blocks)
        validation = merge_counts(counts_by_block, validation_blocks)
        test = merge_counts(counts_by_block, test_blocks)
        train_unit_counts = source_unit_counts(train)
        mono = optimize_mono_key(
            train,
            validation,
            test,
            lm,
            units,
            train_unit_counts,
            steps,
            restarts,
            seed + fold * 1_000_003,
        )
        mono_key = deserialize_key(mono.key)
        variable = fit_abbreviation_overrides(
            train,
            validation,
            test,
            lm,
            mono,
        )
        variable_key = deserialize_key(variable.key)
        mono = replace(
            mono,
            sample=render_sample(lines, test_blocks, mono_key),
        )
        variable = replace(
            variable,
            sample=render_sample(lines, test_blocks, variable_key),
        )
        abbreviation_event_accuracy = None
        abbreviation_correct_events = None
        abbreviation_events = None
        override_map_accuracy = None
        if planted is not None:
            (
                abbreviation_event_accuracy,
                abbreviation_correct_events,
                abbreviation_events,
                override_map_accuracy,
            ) = abbreviation_accuracy(test, variable_key, planted)
        results.append(FoldResult(
            fold=fold,
            train_blocks=tuple(sorted(train_blocks)),
            validation_blocks=tuple(sorted(validation_blocks)),
            test_blocks=tuple(sorted(test_blocks)),
            train_events=sum(train.values()),
            validation_events=sum(validation.values()),
            test_events=sum(test.values()),
            mono=mono,
            variable=variable,
            heldout_joint_gain_bits_per_source=(
                variable.test.joint_log2_per_source
                - mono.test.joint_log2_per_source
            ),
            planted_emission_accuracy=(
                emission_accuracy(test, variable_key, planted)
                if planted is not None
                else None
            ),
            planted_abbreviation_event_accuracy=(
                abbreviation_event_accuracy
            ),
            planted_abbreviation_correct_events=(
                abbreviation_correct_events
            ),
            planted_abbreviation_events=abbreviation_events,
            planted_override_map_accuracy=override_map_accuracy,
        ))
    gains = [result.heldout_joint_gain_bits_per_source for result in results]
    mono_language = sum(
        result.mono.test.language_log2_sum for result in results
    )
    variable_language = sum(
        result.variable.test.language_log2_sum for result in results
    )
    mono_characters = sum(
        result.mono.test.output_characters for result in results
    )
    variable_characters = sum(
        result.variable.test.output_characters for result in results
    )
    mono_model_bits = sum(
        result.mono.test.model_description_bits for result in results
    )
    variable_model_bits = sum(
        result.variable.test.model_description_bits for result in results
    )
    source_observations = sum(
        result.mono.test.source_observations for result in results
    )
    if source_observations != sum(
        result.variable.test.source_observations for result in results
    ):
        raise AssertionError("nested arms do not share source observations")
    pooled_mono_joint = mono_language - mono_model_bits
    pooled_variable_joint = variable_language - variable_model_bits
    pooled_gain = (
        pooled_variable_joint - pooled_mono_joint
    ) / source_observations
    return {
        "name": name,
        "fold_assignment": folds,
        "fold_event_loads": loads,
        "folds": [asdict(result) for result in results],
        "mean_fold_joint_gain_bits_per_source": sum(gains) / len(gains),
        "pooled_heldout_joint_gain_bits_per_source": pooled_gain,
        "pooled_test": {
            "source_observations": source_observations,
            "mono": {
                "language_log2_sum": mono_language,
                "language_code_bits": -mono_language,
                "output_characters": mono_characters,
                "expansion_factor": mono_characters / source_observations,
                "model_description_bits": mono_model_bits,
                "joint_mdl_log2_sum": pooled_mono_joint,
            },
            "variable": {
                "language_log2_sum": variable_language,
                "language_code_bits": -variable_language,
                "output_characters": variable_characters,
                "expansion_factor": (
                    variable_characters / source_observations
                ),
                "model_description_bits": variable_model_bits,
                "joint_mdl_log2_sum": pooled_variable_joint,
            },
        },
        "positive_folds": sum(gain > 0 for gain in gains),
    }


def encode_word(
    word: str,
    inverse_base: dict[str, str],
    planted: Key,
) -> tuple[tuple[str, ...], Counter[str]]:
    options: list[tuple[str, str, str]] = []
    for slot, expansion in planted.overrides.items():
        if expansion is not None:
            options.append((slot.split(":", 1)[0], slot, expansion))
    memo: dict[int, tuple[tuple[str, ...], Counter[str]]] = {}

    def solve(index: int) -> tuple[tuple[str, ...], Counter[str]]:
        if index == len(word):
            return (), Counter()
        if index in memo:
            return memo[index]
        candidates = []
        base_unit = inverse_base[word[index]]
        tail, used = solve(index + 1)
        candidates.append(((base_unit,) + tail, used.copy()))
        for unit, slot, expansion in options:
            if not word.startswith(expansion, index):
                continue
            ending = index + len(expansion)
            position = slot.split(":", 1)[1]
            if position == "initial" and index != 0:
                continue
            if position == "final" and ending != len(word):
                continue
            if position == "nonfinal" and ending == len(word):
                continue
            tail, used = solve(ending)
            counts = used.copy()
            counts[slot] += 1
            candidates.append(((unit,) + tail, counts))
        # Prefer shorter source spellings, then more planted abbreviations.
        winner = min(
            candidates,
            key=lambda item: (
                len(item[0]),
                -sum(item[1].values()),
                item[0],
            ),
        )
        memo[index] = winner
        return winner

    return solve(0)


def synthetic_positive(
    template: Sequence[UnitLine],
    latin_tokens: Sequence[str],
    seed: int,
) -> tuple[list[UnitLine], Key, Counter[str]]:
    observed_units = sorted({
        unit for line in template for word in line.words for unit in word
    })
    primary = sorted(set(observed_units) - set(SPECIAL_UNITS))
    if len(primary) != len(ALPHABET):
        raise ValueError("synthetic key requires 22 ordinary source units")
    rng = random.Random(seed)
    targets = list(ALPHABET)
    rng.shuffle(targets)
    planted = Key(
        base=dict(zip(primary, targets)),
        overrides={
            "y:final": "us",
            "y:nonfinal": "con",
            "q:initial": "qui",
            "ch:any": "et",
            "sh:any": "per",
        },
    )
    for index, unit in enumerate(SPECIAL_UNITS):
        planted.base[unit] = targets[index]
    inverse = {target: source for source, target in planted.base.items()
               if source not in SPECIAL_UNITS}
    eligible = [
        token
        for token in latin_tokens
        if token and all(character in inverse for character in token)
    ]
    if not eligible:
        raise ValueError("no encodable Latin tokens")
    cursor = 0
    usage: Counter[str] = Counter()
    result = []
    for line in template:
        words = []
        for _source_word in line.words:
            word = eligible[cursor % len(eligible)]
            cursor += 1
            encoded, used = encode_word(word, inverse, planted)
            words.append(encoded)
            usage.update(used)
        result.append(UnitLine(
            line.block,
            line.section,
            line.currier,
            line.hand,
            tuple(words),
        ))
    return result, planted, usage


def exchange_words(
    lines: Sequence[UnitLine], rng: random.Random
) -> list[UnitLine]:
    cells: dict[tuple[str, ...], list[tuple[int, int]]] = defaultdict(list)
    mutable = [list(line.words) for line in lines]
    for line_index, line in enumerate(lines):
        for word_index, _word in enumerate(line.words):
            cells[(
                line.block,
                line.section,
                line.currier,
                line.hand,
                position_bucket(word_index, len(line.words)),
            )].append((line_index, word_index))
    for locations in cells.values():
        values = [lines[i].words[j] for i, j in locations]
        rng.shuffle(values)
        for (line_index, word_index), value in zip(locations, values):
            mutable[line_index][word_index] = value
    return [
        UnitLine(
            line.block,
            line.section,
            line.currier,
            line.hand,
            tuple(words),
        )
        for line, words in zip(lines, mutable)
    ]


def decompose(word: str) -> tuple[str, str, str]:
    prefix = next((value for value in GEN_PREFIXES if word.startswith(value)), "")
    residual = word[len(prefix) :]
    suffix = next(
        (
            value
            for value in GEN_SUFFIXES
            if residual.endswith(value) and len(residual) > len(value)
        ),
        "",
    )
    core = residual[: -len(suffix)] if suffix else residual
    return prefix, core, suffix


class SchemaCopyGenerator:
    def __init__(self, words: Sequence[str]) -> None:
        core_counts: Counter[str] = Counter()
        prefixes: dict[str, Counter[str]] = defaultdict(Counter)
        suffixes: dict[str, Counter[str]] = defaultdict(Counter)
        for word in words:
            prefix, core, suffix = decompose(word)
            if not core:
                continue
            core_counts[core] += 1
            prefixes[core][prefix] += 1
            suffixes[core][suffix] += 1
        self.cores = tuple(core_counts)
        self.core_weights = tuple(core_counts.values())
        self.prefixes = {
            core: (tuple(values), tuple(values.values()))
            for core, values in prefixes.items()
        }
        self.suffixes = {
            core: (tuple(values), tuple(values.values()))
            for core, values in suffixes.items()
        }
        self.glyphs = tuple(sorted(set("".join(words))))

    def sample(self, rng: random.Random) -> str:
        core = rng.choices(self.cores, weights=self.core_weights, k=1)[0]
        prefixes, prefix_weights = self.prefixes[core]
        suffixes, suffix_weights = self.suffixes[core]
        return (
            rng.choices(prefixes, weights=prefix_weights, k=1)[0]
            + core
            + rng.choices(suffixes, weights=suffix_weights, k=1)[0]
        )


def procedural_copy(
    lines: Sequence[UnitLine], rng: random.Random
) -> list[UnitLine]:
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    all_words = []
    for line in lines:
        key = (line.section, line.currier, line.hand)
        for units in line.words:
            word = "".join(units)
            grouped[key].append(word)
            all_words.append(word)
    global_generator = SchemaCopyGenerator(all_words)
    generators = {
        key: SchemaCopyGenerator(words)
        for key, words in grouped.items()
        if len(words) >= 50 and len({decompose(word)[1] for word in words}) >= 4
    }
    history: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    result = []
    for line in lines:
        key = (line.section, line.currier, line.hand)
        generator = generators.get(key, global_generator)
        words = []
        for _ in line.words:
            old = history[key]
            if old and rng.random() < 0.12:
                word = old[-rng.randint(1, min(40, len(old)))]
                if len(word) > 1 and rng.random() < 0.4:
                    index = rng.randrange(len(word))
                    word = (
                        word[:index]
                        + rng.choice(generator.glyphs)
                        + word[index + 1 :]
                    )
            else:
                word = generator.sample(rng)
            history[key].append(word)
            words.append(tokenize_eva(word))
        result.append(UnitLine(
            line.block,
            line.section,
            line.currier,
            line.hand,
            tuple(words),
        ))
    return result


def normalized_letters(value: object) -> str:
    text = str(value or "").lower()
    text = (
        text.replace("j", "i")
        .replace("v", "u")
        .replace("w", "u")
        .replace("k", "c")
    )
    return re.sub("[^a-z]", "", text)


def historical_calibration(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"available": False, "path": str(path.relative_to(ROOT))}
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunks = sorted({
        chunk for options in ABBREVIATION_OPTIONS.values() for chunk in options
    })
    result: dict[str, object] = {
        "available": True,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "declared_chunks": chunks,
    }
    cappelli = payload.get("cappelli_latin_xiv_xv", {})
    records = cappelli.get("records", []) if isinstance(cappelli, dict) else []
    support = Counter()
    for record in records:
        if not isinstance(record, dict):
            continue
        count = int(record.get("count", 1))
        visible = normalized_letters(record.get("visible"))
        transcription = normalized_letters(record.get("transcription"))
        marked = normalized_letters(record.get("marked_components"))
        for chunk in chunks:
            if chunk in marked or (
                chunk in transcription and chunk not in visible
            ):
                support[chunk] += count
    result["cappelli"] = {
        "record_rows": len(records),
        "support_by_chunk": {chunk: support[chunk] for chunk in chunks},
        "all_chunks_supported": all(support[chunk] > 0 for chunk in chunks),
    }
    nuremberg = payload.get("nuremberg_letterbooks", {})
    groups = nuremberg.get("groups", []) if isinstance(nuremberg, dict) else []
    by_book: dict[str, Counter[str]] = defaultdict(Counter)
    for group in groups:
        if not isinstance(group, dict):
            continue
        book = str(group.get("book", "?"))
        for pair in group.get("pairs", []):
            if not isinstance(pair, dict):
                continue
            count = int(pair.get("count", 1))
            omitted = normalized_letters(pair.get("omitted"))
            by_book[book]["events"] += count
            if omitted in chunks:
                by_book[book]["exact_covered"] += count
            if any(chunk in omitted for chunk in chunks):
                by_book[book]["substring_covered"] += count
    result["nuremberg_book_heldout_coverage"] = {
        book: {
            **dict(values),
            "exact_fraction": (
                values["exact_covered"] / values["events"]
                if values["events"]
                else None
            ),
            "substring_fraction": (
                values["substring_covered"] / values["events"]
                if values["events"]
                else None
            ),
        }
        for book, values in sorted(by_book.items())
    }
    return result


def panel_summary(panel: dict[str, object]) -> str:
    return (
        f"{panel['name']}: mean_delta="
        f"{panel['mean_fold_joint_gain_bits_per_source']:+.5f} "
        f"pooled_delta="
        f"{panel['pooled_heldout_joint_gain_bits_per_source']:+.5f} "
        f"positive_folds={panel['positive_folds']}/{N_FOLDS}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--latin", type=Path, default=LATIN)
    parser.add_argument("--historical", type=Path, default=HISTORICAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--steps", type=int, default=1400)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--procedural-nulls", type=int, default=19)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines, corpus_audit = load_voynich(args.corpus)
    latin = clean_latin(args.latin)
    split = len(latin) * 2 // 3
    lm_text = latin[:split]
    latin_holdout = latin[split:]
    lm = TrigramLM(lm_text)
    ceiling = lm.score_text(latin_holdout)
    latin_tokens = re.findall(
        f"[{''.join(ALPHABET)}]+", latin_holdout
    )

    panels = []
    real = run_panel(
        "VOYNICH",
        lines,
        lm,
        args.steps,
        args.restarts,
        SEED,
    )
    panels.append(real)
    if args.progress:
        print(panel_summary(real), flush=True)

    positive_lines, planted, usage = synthetic_positive(
        lines, latin_tokens, SEED + 10_000
    )
    positive = run_panel(
        "SYNTHETIC_LATIN",
        positive_lines,
        lm,
        args.steps,
        args.restarts,
        SEED + 20_000,
        planted=planted,
    )
    panels.append(positive)
    if args.progress:
        print(panel_summary(positive), flush=True)

    exchanged = exchange_words(lines, random.Random(SEED + 30_000))
    exchange = run_panel(
        "SURFACE_EXCHANGE",
        exchanged,
        lm,
        args.steps,
        args.restarts,
        SEED + 40_000,
    )
    panels.append(exchange)
    if args.progress:
        print(panel_summary(exchange), flush=True)

    procedural_panels = []
    for replicate in range(args.procedural_nulls):
        generated = procedural_copy(
            lines, random.Random(SEED + 50_000 + replicate)
        )
        panel = run_panel(
            f"PROCEDURAL_COPY_{replicate:02d}",
            generated,
            lm,
            args.steps,
            args.restarts,
            SEED + 60_000 + replicate * 10_000,
        )
        procedural_panels.append(panel)
        panels.append(panel)
        if args.progress:
            print(panel_summary(panel), flush=True)

    real_gain = float(
        real["pooled_heldout_joint_gain_bits_per_source"]
    )
    positive_gain = float(
        positive["pooled_heldout_joint_gain_bits_per_source"]
    )
    exchange_gain = float(
        exchange["pooled_heldout_joint_gain_bits_per_source"]
    )
    procedural_gains = [
        float(panel["pooled_heldout_joint_gain_bits_per_source"])
        for panel in procedural_panels
    ]
    all_null_gains = [exchange_gain] + procedural_gains
    positive_accuracy = [
        fold["planted_emission_accuracy"] for fold in positive["folds"]
    ]
    positive_abbreviation_accuracy = [
        fold["planted_abbreviation_event_accuracy"]
        for fold in positive["folds"]
    ]
    positive_override_accuracy = [
        fold["planted_override_map_accuracy"]
        for fold in positive["folds"]
    ]
    positive_abbreviation_global_accuracy = (
        sum(
            fold["planted_abbreviation_correct_events"]
            for fold in positive["folds"]
        )
        / sum(
            fold["planted_abbreviation_events"]
            for fold in positive["folds"]
        )
    )
    positive_abbreviation_min_accuracy = min(
        value
        for value in positive_abbreviation_accuracy
        if value is not None
    )
    positive_control_pass = (
        positive["positive_folds"] == N_FOLDS
        and positive_abbreviation_min_accuracy >= 0.90
    )
    gates = {
        "synthetic_positive_delta_pass": positive_gain > 0,
        "synthetic_positive_all_fold_delta_pass": (
            positive["positive_folds"] == N_FOLDS
        ),
        "synthetic_positive_abbreviation_recovery_pass": (
            positive_control_pass
        ),
        "voynich_positive_delta": real_gain > 0,
        "voynich_beats_all_matched_nulls": (
            real_gain > max(all_null_gains) if all_null_gains else False
        ),
        "bounded_mechanism_screen_pass": (
            positive_gain > 0
            and positive_control_pass
            and real_gain > 0
            and real_gain > max(all_null_gains)
        ),
    }
    procedural_rank = 1 + sum(
        value >= real_gain for value in procedural_gains
    )
    payload = {
        "experiment": "historically_constrained_abbreviation_transducer_gate",
        "seed": SEED,
        "claim_boundary": (
            "A pass nominates a variable-length mechanism for diplomatic-"
            "transcription follow-up; it does not identify Latin or plaintext."
        ),
        "parameters": {
            "folds": N_FOLDS,
            "split": (
                "test=fold f; validation=fold f+1; fit=other two folds; "
                "folds greedily event-balanced without reading content"
            ),
            "steps": args.steps,
            "restarts": args.restarts,
            "procedural_nulls": args.procedural_nulls,
            "variable_override_configurations": math.prod(
                len(values) + 1
                for values in ABBREVIATION_OPTIONS.values()
            ),
            "variable_base_key": (
                "frozen fitted mono base; override map exhaustively selected "
                "on fit blocks only"
            ),
            "alphabet": "".join(ALPHABET),
            "multi_units": list(MULTI_UNITS),
            "special_units": list(SPECIAL_UNITS),
            "abbreviation_options": {
                slot: list(values)
                for slot, values in ABBREVIATION_OPTIONS.items()
            },
            "primary_statistic": (
                "pooled held-out joint MDL log2 gain per common source "
                "observation: full Latin trigram log probability minus "
                "channel-map description bits, variable minus nested mono"
            ),
            "length_and_channel_prior": (
                "Every decoded character pays its full LM code length; scores "
                "are never normalized by decoded length. The shared frozen "
                "base-key code cancels. The variable map pays a uniform "
                "sum(log2(slot choices including null)) code."
            ),
            "variable_model_description_bits_per_fold": (
                VARIABLE_MODEL_BITS
            ),
            "initialization_counts": "fit blocks only in every fold",
        },
        "assets": {
            asset_name(args.corpus): sha256(args.corpus),
            asset_name(args.latin): sha256(args.latin),
        },
        "corpus_audit": dict(corpus_audit),
        "latin": {
            "language_model_characters": len(lm_text),
            "reserved_holdout_characters": len(latin_holdout),
            "reserved_ceiling_bits_per_character": ceiling,
        },
        "synthetic_positive": {
            "planted_key": serialize_key(planted),
            "planted_abbreviation_usage": dict(usage),
        },
        "historical_calibration": historical_calibration(args.historical),
        "panels": panels,
        "summary": {
            "status": "pilot_unresolved",
            "voynich_pooled_joint_gain_bits_per_source": real_gain,
            "synthetic_positive_pooled_joint_gain_bits_per_source": (
                positive_gain
            ),
            "surface_exchange_pooled_joint_gain_bits_per_source": (
                exchange_gain
            ),
            "procedural_null_pooled_joint_gains_bits_per_source": (
                procedural_gains
            ),
            "procedural_null_rank": (
                f"{procedural_rank}/{len(procedural_gains) + 1}"
            ),
            "procedural_null_exceedances": procedural_rank - 1,
            "procedural_null_replicates": len(procedural_gains),
            "matched_null_max": max(all_null_gains),
            "synthetic_positive_mean_emission_accuracy": (
                sum(value for value in positive_accuracy if value is not None)
                / len(positive_accuracy)
            ),
            "synthetic_positive_min_abbreviation_event_accuracy": (
                positive_abbreviation_min_accuracy
            ),
            "synthetic_positive_mean_abbreviation_event_accuracy": (
                sum(
                    value
                    for value in positive_abbreviation_accuracy
                    if value is not None
                )
                / len(positive_abbreviation_accuracy)
            ),
            "synthetic_positive_global_abbreviation_event_accuracy": (
                positive_abbreviation_global_accuracy
            ),
            "synthetic_positive_mean_override_map_accuracy": (
                sum(
                    value
                    for value in positive_override_accuracy
                    if value is not None
                )
                / len(positive_override_accuracy)
            ),
            **gates,
        },
        "caveats": [
            "EVA is not a diplomatic stroke transcription; its unit choices may "
            "erase the allographs and abbreviation marks this gate needs.",
            "Cappelli licenses expansion strings only.  No textual Cappelli "
            "record proves that an EVA-labelled unit is the same visual sign.",
            "The q and bench slot assignments are bounded structural probes, "
            "weaker palaeographically than the y/figure-9 candidate.",
            "A character language-model improvement can reflect Latin-like word "
            "shape without yielding stable lexical readings.",
            "The score is a source-observation-normalized joint MDL proxy: "
            "full plaintext LM code plus a deterministic channel-map code. It "
            "does not model a stochastic palaeographic abbreviation process.",
            "Null ranks are a pilot screen, not calibrated significance. Even "
            "19 procedural replicates only resolve an extreme rank to 1/20.",
            "The checked-in Latin control is classical oratory, not a matched "
            "fifteenth-century technical/diplomatic corpus.",
            "Four event-balanced folds are atomic by repository quire code, not "
            "Lisa Fagin Davis's unpublished/true codicological collation.",
        ],
    }
    if args.historical.exists():
        payload["assets"][asset_name(args.historical)] = sha256(
            args.historical
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\n".join(panel_summary(panel) for panel in panels))
    print("GATE", json.dumps(payload["summary"], sort_keys=True))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
