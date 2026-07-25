"""
Does cross-word morphology survive word-Markov and local-copy controls?

The outcome is the EVA prefix of word N+1.  Two source arrows are tested:

  * prefix(N) -> prefix(N+1), the held-out arrow found in the follow-up work;
  * suffix(N) -> prefix(N+1), the qualitative arrow reported in the manuscript.

All scores are leave-one-complete-quire-out.  The nested predictors are:

  position
  position + source morphology
  position + exact previous word
  exact previous word with source morphology as a hierarchical backoff
  position + adaptive surface context
  adaptive surface context + source morphology

The adaptive surface context uses an exact previous word only when it occurred
at least five times as a training predecessor.  Rarer words are pooled by their
stripped core.  Both cases include a causal lag state describing whether the
previous word itself was copied within the preceding 40 eligible tokens.  The
five-count threshold, add-0.5 outcome smoothing, and backoff mass of eight are
fixed before testing.

The primary residual is

  log2 P(next prefix | adaptive surface, source morphology, target position)
    - log2 P(next prefix | adaptive surface, target position).

For frequent exact-word contexts, source morphology is deterministically
contained in the surface word and the two probabilities are identical.  The
residual therefore measures whether morphological pooling helps on rare or
unseen surface contexts, not whether a parser can rediscover a prefix already
visible in a known word.

Uncertain and one-character tokens remain adjacency breaks.  Counts,
vocabularies, rare/frequent decisions, and probability estimates exclude the
test quire.  A separately fitted deep-interior analysis excludes every
transition touching the first, second, penultimate, or last position.

Finally, a fitted word-level Markov generator supplies a calibration panel.
It preserves quire/line layouts and adjacency breaks but contains no explicit
cross-word morphological rule.  This calibration is parametric and descriptive:
its generator is fitted once to the full real corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Hashable, Iterable, Optional, Sequence

import numpy as np


SEED = 20260723
ALPHA = 0.5
BACKOFF_MASS = 8.0
FREQUENT_PREDECESSOR = 5
N_BOOTSTRAP = 20000
DEFAULT_SURROGATES = 24
UNKNOWN = "<TRAIN-ONLY-UNK>"

PREFIXES = tuple(sorted(
    (
        "qok", "qot", "qo", "ok", "ot", "o", "y", "ch", "sh", "d",
        "cth", "ckh", "cph", "cfh",
    ),
    key=len,
    reverse=True,
))
SUFFIXES = tuple(sorted(
    (
        "eedy", "eody", "edy", "aiin", "aiir", "ain", "iin", "dy",
        "ol", "or", "ar", "al", "am", "dam", "ey", "eey", "y",
    ),
    key=len,
    reverse=True,
))

Value = Hashable


@dataclass(frozen=True)
class Line:
    folio: str
    line: str
    quire: str
    section: str
    currier: str
    hand: str
    words: tuple[Optional[str], ...]
    copy_states: tuple[Optional[str], ...] = ()


@dataclass
class BlockScore:
    label: str
    transitions: int = 0
    sums: Counter = field(default_factory=Counter)
    residual_cells: dict = field(
        default_factory=lambda: defaultdict(lambda: [0.0, 0])
    )

    def mean(self, metric: str) -> float:
        return (
            self.sums[metric] / self.transitions
            if self.transitions
            else float("nan")
        )


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def say(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def folio_key(folio: str) -> tuple[int, int, int]:
    match = re.match(r"f(\d+)([rv])(\d*)", folio)
    if not match:
        return (10**9, 0, 0)
    return (
        int(match.group(1)),
        0 if match.group(2) == "r" else 1,
        int(match.group(3) or 0),
    )


def locus_type(locus: str) -> str:
    match = re.search(r"[A-Za-z]", locus)
    return match.group(0).upper() if match else "?"


@lru_cache(maxsize=None)
def decompose(word: str) -> tuple[str, str, str]:
    prefix = next((item for item in PREFIXES if word.startswith(item)), "")
    residual = word[len(prefix):]
    suffix = next(
        (
            item for item in SUFFIXES
            if residual.endswith(item) and len(residual) > len(item)
        ),
        "",
    )
    core = residual[:-len(suffix)] if suffix else residual
    return prefix, core, suffix


def prefix(word: str) -> str:
    return decompose(word)[0] or "none"


def suffix(word: str) -> str:
    return decompose(word)[2] or "none"


def core(word: str) -> str:
    return decompose(word)[1] or "<EMPTY-CORE>"


SOURCE_FEATURES: dict[str, Callable[[str], str]] = {
    "previous_prefix": prefix,
    "previous_suffix": suffix,
}


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


def assign_copy_states(lines: Sequence[Line]) -> list[Line]:
    """Assign causal exact-word lag bins, resetting history at each quire."""
    clocks: Counter = Counter()
    last_seen: dict[str, dict[str, int]] = defaultdict(dict)
    result = []
    for line in lines:
        states: list[Optional[str]] = []
        quire = line.quire
        for word in line.words:
            if word is None:
                states.append(None)
                continue
            clock = clocks[quire]
            previous = last_seen[quire].get(word)
            if previous is None or clock - previous > 40:
                state = "new_or_lag>40"
            else:
                lag = clock - previous
                if lag == 1:
                    state = "lag1"
                elif lag <= 5:
                    state = "lag2-5"
                else:
                    state = "lag6-40"
            states.append(state)
            last_seen[quire][word] = clock
            clocks[quire] += 1
        result.append(replace(line, copy_states=tuple(states)))
    return result


def load_lines(path: Path) -> tuple[list[Line], Counter]:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    audit: Counter = Counter()
    lines = []
    for folio in sorted(corpus["folios"], key=folio_key):
        meta = corpus["meta"].get(folio, {})
        for raw in corpus["folios"][folio]:
            if locus_type(raw["locus"]) != "P":
                continue
            audit["prose_lines"] += 1
            words = []
            for word in raw["words"]:
                audit["source_tokens"] += 1
                if "?" in word:
                    words.append(None)
                    audit["uncertain_breaks"] += 1
                elif len(word) < 2:
                    words.append(None)
                    audit["one_char_breaks"] += 1
                else:
                    words.append(word)
                    audit["eligible_tokens"] += 1
            lines.append(Line(
                folio=folio,
                line=str(raw["line"]),
                quire=str(meta.get("Q", "?")),
                section=str(meta.get("I", "?")),
                currier=str(meta.get("L", "?")),
                hand=str(meta.get("H", "?")),
                words=tuple(words),
            ))
    return assign_copy_states(lines), audit


def transitions(
    line: Line,
    deep_only: bool,
) -> Iterable[tuple[str, str, str, str]]:
    for target_index in range(1, len(line.words)):
        left = line.words[target_index - 1]
        right = line.words[target_index]
        if left is None or right is None:
            continue
        left_position = position_bucket(target_index - 1, len(line.words))
        target_position = position_bucket(target_index, len(line.words))
        if deep_only and (
            left_position != "interior" or target_position != "interior"
        ):
            continue
        copy_state = line.copy_states[target_index - 1]
        assert copy_state is not None
        yield left, right, target_position, copy_state


@dataclass
class NestedModel:
    source_function: Callable[[str], str]
    outcome_vocabulary: set[str]
    source_vocabulary: set[str]
    word_vocabulary: set[str]
    core_vocabulary: set[str]
    predecessor_frequency: Counter
    base: Counter
    base_totals: Counter
    morph: Counter
    morph_totals: Counter
    exact: Counter
    exact_totals: Counter
    surface: Counter
    surface_totals: Counter
    surface_morph: Counter
    surface_morph_totals: Counter

    @property
    def outcome_size(self) -> int:
        return len(self.outcome_vocabulary) + 1

    def map_outcome(self, value: str) -> str:
        return value if value in self.outcome_vocabulary else UNKNOWN

    def map_source(self, value: str) -> str:
        return value if value in self.source_vocabulary else UNKNOWN

    def map_word(self, value: str) -> str:
        return value if value in self.word_vocabulary else UNKNOWN

    def map_core(self, value: str) -> str:
        return value if value in self.core_vocabulary else UNKNOWN

    def p_base(self, position: str, outcome: str) -> float:
        return (
            self.base[(position, outcome)] + ALPHA
        ) / (
            self.base_totals[position] + ALPHA * self.outcome_size
        )

    @staticmethod
    def backed_probability(
        counts: Counter,
        totals: Counter,
        context: Value,
        outcome: str,
        backoff: float,
    ) -> float:
        return (
            counts[(context, outcome)] + BACKOFF_MASS * backoff
        ) / (
            totals[context] + BACKOFF_MASS
        )

    def p_morph(
        self,
        position: str,
        source: str,
        outcome: str,
    ) -> float:
        context = (position, source)
        return self.backed_probability(
            self.morph,
            self.morph_totals,
            context,
            outcome,
            self.p_base(position, outcome),
        )

    def p_exact(
        self,
        position: str,
        word: str,
        outcome: str,
        morph_backoff: Optional[float] = None,
    ) -> float:
        context = (position, word)
        backoff = (
            self.p_base(position, outcome)
            if morph_backoff is None
            else morph_backoff
        )
        return self.backed_probability(
            self.exact,
            self.exact_totals,
            context,
            outcome,
            backoff,
        )

    def surface_key(
        self,
        word: str,
        copy_state: str,
    ) -> tuple[str, str, str]:
        if self.predecessor_frequency[word] >= FREQUENT_PREDECESSOR:
            return ("word", self.map_word(word), copy_state)
        return ("family", self.map_core(core(word)), copy_state)

    def p_surface(
        self,
        position: str,
        surface_key: Value,
        outcome: str,
    ) -> float:
        context = (position, surface_key)
        return self.backed_probability(
            self.surface,
            self.surface_totals,
            context,
            outcome,
            self.p_base(position, outcome),
        )

    def p_surface_morph(
        self,
        position: str,
        surface_key: Value,
        source: str,
        outcome: str,
    ) -> float:
        surface_probability = self.p_surface(
            position, surface_key, outcome
        )
        # A frequent exact word already determines source morphology.  Returning
        # the identical probability avoids manufacturing gain by smoothing the
        # same counts twice.
        if surface_key[0] == "word":
            return surface_probability
        context = (position, surface_key, source)
        return self.backed_probability(
            self.surface_morph,
            self.surface_morph_totals,
            context,
            outcome,
            surface_probability,
        )

    def probabilities(
        self,
        left_word: str,
        right_word: str,
        position: str,
        copy_state: str,
    ) -> tuple[dict[str, float], str, str, str]:
        outcome = self.map_outcome(prefix(right_word))
        source = self.map_source(self.source_function(left_word))
        word = self.map_word(left_word)
        surface_key = self.surface_key(left_word, copy_state)

        p0 = self.p_base(position, outcome)
        pm = self.p_morph(position, source, outcome)
        pe = self.p_exact(position, word, outcome)
        pem = self.p_exact(position, word, outcome, pm)
        ps = self.p_surface(position, surface_key, outcome)
        psm = self.p_surface_morph(
            position, surface_key, source, outcome
        )
        return {
            "base_logp": math.log2(p0),
            "morph_gain": math.log2(pm / p0),
            "exact_gain": math.log2(pe / p0),
            "exact_morph_residual": math.log2(pem / pe),
            "surface_gain": math.log2(ps / p0),
            "surface_morph_residual": math.log2(psm / ps),
        }, source, outcome, surface_key[0]


def fit_model(
    lines: Sequence[Line],
    source_function: Callable[[str], str],
    deep_only: bool,
) -> NestedModel:
    rows = [
        item
        for line in lines
        for item in transitions(line, deep_only)
    ]
    outcome_vocabulary = {prefix(right) for _, right, _, _ in rows}
    source_vocabulary = {source_function(left) for left, _, _, _ in rows}
    word_vocabulary = {left for left, _, _, _ in rows}
    core_vocabulary = {core(left) for left, _, _, _ in rows}
    predecessor_frequency = Counter(left for left, _, _, _ in rows)

    base: Counter = Counter()
    base_totals: Counter = Counter()
    morph: Counter = Counter()
    morph_totals: Counter = Counter()
    exact: Counter = Counter()
    exact_totals: Counter = Counter()
    surface: Counter = Counter()
    surface_totals: Counter = Counter()
    surface_morph: Counter = Counter()
    surface_morph_totals: Counter = Counter()

    def train_surface_key(word: str, copy_state: str):
        if predecessor_frequency[word] >= FREQUENT_PREDECESSOR:
            return ("word", word, copy_state)
        return ("family", core(word), copy_state)

    for left, right, position, copy_state in rows:
        outcome = prefix(right)
        source = source_function(left)
        surface_key = train_surface_key(left, copy_state)
        morph_context = (position, source)
        exact_context = (position, left)
        surface_context = (position, surface_key)
        combined_context = (position, surface_key, source)

        base[(position, outcome)] += 1
        base_totals[position] += 1
        morph[(morph_context, outcome)] += 1
        morph_totals[morph_context] += 1
        exact[(exact_context, outcome)] += 1
        exact_totals[exact_context] += 1
        surface[(surface_context, outcome)] += 1
        surface_totals[surface_context] += 1
        if surface_key[0] == "family":
            surface_morph[(combined_context, outcome)] += 1
            surface_morph_totals[combined_context] += 1

    return NestedModel(
        source_function=source_function,
        outcome_vocabulary=outcome_vocabulary,
        source_vocabulary=source_vocabulary,
        word_vocabulary=word_vocabulary,
        core_vocabulary=core_vocabulary,
        predecessor_frequency=predecessor_frequency,
        base=base,
        base_totals=base_totals,
        morph=morph,
        morph_totals=morph_totals,
        exact=exact,
        exact_totals=exact_totals,
        surface=surface,
        surface_totals=surface_totals,
        surface_morph=surface_morph,
        surface_morph_totals=surface_morph_totals,
    )


def score_block(
    label: str,
    lines: Sequence[Line],
    model: NestedModel,
    deep_only: bool,
    localize: bool,
) -> BlockScore:
    score = BlockScore(label)
    for line in lines:
        for left, right, position, copy_state in transitions(line, deep_only):
            values, source, outcome, surface_kind = model.probabilities(
                left, right, position, copy_state
            )
            score.transitions += 1
            score.sums.update(values)
            if localize:
                key = (position, source, outcome, surface_kind)
                score.residual_cells[key][0] += values[
                    "surface_morph_residual"
                ]
                score.residual_cells[key][1] += 1
    return score


def score_corpus(
    lines: Sequence[Line],
    source_function: Callable[[str], str],
    deep_only: bool,
    localize: bool = False,
) -> list[BlockScore]:
    quires = sorted({line.quire for line in lines if line.quire != "?"})
    by_quire: dict[str, list[Line]] = defaultdict(list)
    for line in lines:
        by_quire[line.quire].append(line)
    blocks = []
    for held_out in quires:
        train = [line for line in lines if line.quire != held_out]
        model = fit_model(train, source_function, deep_only)
        blocks.append(score_block(
            held_out,
            by_quire[held_out],
            model,
            deep_only,
            localize,
        ))
    return blocks


def exact_sign_p(values: Sequence[float]) -> float:
    nonzero = [value for value in values if abs(value) > 1e-15]
    if not nonzero:
        return 1.0
    positives = sum(value > 0 for value in nonzero)
    return sum(
        math.comb(len(nonzero), count)
        for count in range(positives, len(nonzero) + 1)
    ) / (2 ** len(nonzero))


def block_summary(
    blocks: Sequence[BlockScore],
    metric: str,
    rng: np.random.Generator,
) -> dict:
    eligible = [block for block in blocks if block.transitions]
    bits = np.asarray([block.sums[metric] for block in eligible])
    counts = np.asarray([block.transitions for block in eligible])
    gains = bits / counts
    observed = bits.sum() / counts.sum()

    sample = rng.integers(
        0, len(eligible), size=(N_BOOTSTRAP, len(eligible))
    )
    bootstrap = bits[sample].sum(axis=1) / counts[sample].sum(axis=1)
    bootstrap_interval = np.quantile(bootstrap, (0.025, 0.975))

    signed = np.array([0.0])
    for value in bits:
        signed = np.concatenate((signed + value, signed - value))
    null = signed / counts.sum()
    flip_interval = np.quantile(null, (0.025, 0.975))
    flip_p = np.mean(null >= observed - 1e-15)
    return {
        "gain": observed,
        "bootstrap_low": bootstrap_interval[0],
        "bootstrap_high": bootstrap_interval[1],
        "positive": int(np.sum(gains > 0)),
        "blocks": len(eligible),
        "sign_p": exact_sign_p(gains),
        "flip_low": flip_interval[0],
        "flip_high": flip_interval[1],
        "flip_p": flip_p,
        "transitions": int(counts.sum()),
    }


def emit_summary(
    reporter: Reporter,
    blocks: Sequence[BlockScore],
    metric: str,
    rng: np.random.Generator,
    indent: str = "    ",
) -> dict:
    result = block_summary(blocks, metric, rng)
    reporter.say(
        f"{indent}{metric}: {result['gain']:+.5f} bits/transition; "
        f"block-bootstrap 95% [{result['bootstrap_low']:+.5f},"
        f"{result['bootstrap_high']:+.5f}]"
    )
    reporter.say(
        f"{indent}positive={result['positive']}/{result['blocks']}; "
        f"sign p={result['sign_p']:.5f}; exact block-flip p="
        f"{result['flip_p']:.5f}; flip-null 95% "
        f"[{result['flip_low']:+.5f},{result['flip_high']:+.5f}]"
    )
    return result


def report_real(
    reporter: Reporter,
    results: dict[str, dict[str, list[BlockScore]]],
) -> dict[str, float]:
    reporter.say("=" * 92)
    reporter.say("REAL VOYNICH: NESTED HELD-OUT NEXT-PREFIX PREDICTION")
    reporter.say("=" * 92)
    reporter.say(
        "Primary residual = adaptive surface+morphology minus adaptive "
        "surface. Exact-morph residual changes only the word-Markov backoff."
    )
    reporter.say(
        "Deterministic-coarsening warning: a source prefix/suffix is a "
        "function of the exact source word, so it cannot add population "
        "information once exact-word identity is known. These residuals test "
        "held-out pooling/backoff generalization on sparse surface contexts."
    )
    rng = np.random.default_rng(SEED)
    aggregate_residuals = {}
    for source_name, modes in results.items():
        all_blocks = modes["all"]
        deep_blocks = modes["deep"]
        deep_by_label = {block.label: block for block in deep_blocks}
        reporter.say(f"\n{source_name}")
        reporter.say(
            f"  {'quire':>5s} {'n':>6s} {'raw morph':>10s} "
            f"{'exact resid':>12s} {'surface resid':>13s} "
            f"{'deep resid':>11s}"
        )
        for block in all_blocks:
            deep = deep_by_label[block.label]
            reporter.say(
                f"  {block.label:>5s} {block.transitions:6d} "
                f"{block.mean('morph_gain'):+10.5f} "
                f"{block.mean('exact_morph_residual'):+12.5f} "
                f"{block.mean('surface_morph_residual'):+13.5f} "
                f"{deep.mean('surface_morph_residual'):+11.5f}"
            )

        reporter.say("  All-transition nested contrasts:")
        for metric in (
            "morph_gain",
            "exact_gain",
            "exact_morph_residual",
            "surface_gain",
            "surface_morph_residual",
        ):
            summary = emit_summary(
                reporter, all_blocks, metric, rng, indent="    "
            )
            if metric == "surface_morph_residual":
                aggregate_residuals[f"{source_name}:all"] = summary["gain"]
        reporter.say("  Deep-interior primary residual:")
        summary = emit_summary(
            reporter,
            deep_blocks,
            "surface_morph_residual",
            rng,
            indent="    ",
        )
        aggregate_residuals[f"{source_name}:deep"] = summary["gain"]
    return aggregate_residuals


def stable_cells(
    blocks: Sequence[BlockScore],
) -> tuple[list[dict], list[dict]]:
    combined: dict[Value, dict[str, tuple[float, int]]] = defaultdict(dict)
    for block in blocks:
        for key, (bits, count) in block.residual_cells.items():
            combined[key][block.label] = (bits, count)
    candidates = []
    for key, by_block in combined.items():
        count = sum(value[1] for value in by_block.values())
        if count < 20 or len(by_block) < 5:
            continue
        bits = sum(value[0] for value in by_block.values())
        positive = np.mean([value[0] > 0 for value in by_block.values()])
        candidates.append({
            "key": key,
            "bits": bits,
            "count": count,
            "quires": len(by_block),
            "positive": positive,
            "mean": bits / count,
        })
    positive = sorted(
        (
            row for row in candidates
            if row["bits"] > 0 and row["positive"] >= 0.70
        ),
        key=lambda row: row["bits"],
        reverse=True,
    )[:8]
    negative = sorted(
        (
            row for row in candidates
            if row["bits"] < 0 and row["positive"] <= 0.30
        ),
        key=lambda row: row["bits"],
    )[:8]
    return positive, negative


def report_localization(
    reporter: Reporter,
    results: dict[str, dict[str, list[BlockScore]]],
) -> None:
    reporter.say("\n" + "=" * 92)
    reporter.say("STABLE PRIMARY-RESIDUAL CELLS")
    reporter.say("=" * 92)
    reporter.say(
        "All-transition adaptive residual; >=20 events in >=5 held-out "
        "quires and the same sign in >=70% of those quires. Descriptive only."
    )
    for source_name, modes in results.items():
        positive, negative = stable_cells(modes["all"])
        reporter.say(f"\n{source_name}")
        for heading, rows in (
            ("stable positive", positive),
            ("stable negative", negative),
        ):
            reporter.say(f"  {heading}:")
            if not rows:
                reporter.say("    none at the declared threshold")
            for row in rows:
                position, source, outcome, surface_kind = row["key"]
                reporter.say(
                    f"    {position} {source} -> {outcome} "
                    f"[{surface_kind}]: total={row['bits']:+.2f}, "
                    f"mean={row['mean']:+.4f}, n={row['count']}, "
                    f"quires={row['quires']}, positive={row['positive']:.2f}"
                )


class CounterSampler:
    def __init__(self, counter: Counter):
        self.values = tuple(counter)
        self.cumulative = np.cumsum(
            [counter[value] for value in self.values], dtype=float
        )
        self.total = float(self.cumulative[-1]) if self.values else 0.0

    def sample(self, rng: np.random.Generator) -> str:
        draw = rng.random() * self.total
        index = int(np.searchsorted(self.cumulative, draw, side="right"))
        return self.values[index]


class WordMarkovGenerator:
    def __init__(self, lines: Sequence[Line]):
        vocabulary = sorted({
            word for line in lines for word in line.words if word is not None
        })
        self.vocabulary = tuple(vocabulary)
        token_counts: dict[str, Counter] = defaultdict(Counter)
        transition_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for line in lines:
            for index, word in enumerate(line.words):
                if word is None:
                    continue
                position = position_bucket(index, len(line.words))
                token_counts[position][word] += 1
            for left, right, position, _ in transitions(line, False):
                transition_counts[(position, left)][right] += 1
        self.token_samplers = {
            position: CounterSampler(counts)
            for position, counts in token_counts.items()
        }
        self.transition_samplers = {
            context: CounterSampler(counts)
            for context, counts in transition_counts.items()
        }

    def baseline(self, position: str, rng: np.random.Generator) -> str:
        sampler = self.token_samplers[position]
        empirical_mass = sampler.total / (
            sampler.total + ALPHA * len(self.vocabulary)
        )
        if rng.random() < empirical_mass:
            return sampler.sample(rng)
        return self.vocabulary[int(rng.integers(0, len(self.vocabulary)))]

    def next_word(
        self,
        previous: str,
        position: str,
        rng: np.random.Generator,
    ) -> str:
        sampler = self.transition_samplers.get((position, previous))
        if sampler is not None:
            empirical_mass = sampler.total / (
                sampler.total + BACKOFF_MASS
            )
            if rng.random() < empirical_mass:
                return sampler.sample(rng)
        return self.baseline(position, rng)

    def generate(
        self,
        template: Sequence[Line],
        rng: np.random.Generator,
    ) -> list[Line]:
        generated = []
        for line in template:
            words: list[Optional[str]] = []
            previous: Optional[str] = None
            for index, original in enumerate(line.words):
                if original is None:
                    words.append(None)
                    previous = None
                    continue
                position = position_bucket(index, len(line.words))
                word = (
                    self.baseline(position, rng)
                    if previous is None
                    else self.next_word(previous, position, rng)
                )
                words.append(word)
                previous = word
            generated.append(replace(
                line,
                words=tuple(words),
                copy_states=(),
            ))
        return assign_copy_states(generated)


def aggregate_metric(blocks: Sequence[BlockScore], metric: str) -> float:
    return sum(block.sums[metric] for block in blocks) / sum(
        block.transitions for block in blocks
    )


def report_surrogates(
    reporter: Reporter,
    real_lines: Sequence[Line],
    real_aggregates: dict[str, float],
    replicates: int,
) -> None:
    reporter.say("\n" + "=" * 92)
    reporter.say("FITTED WORD-MARKOV SURROGATE CALIBRATION")
    reporter.say("=" * 92)
    reporter.say(
        "The generator preserves layout/breaks and samples words from a "
        "target-position-conditioned first-order exact-word Markov model. "
        "It has no separately specified cross-word morph rule."
    )
    if replicates <= 0:
        reporter.say("Skipped (--surrogates 0).")
        return

    generator = WordMarkovGenerator(real_lines)
    rng = np.random.default_rng(SEED + 100)
    distributions: dict[str, list[float]] = defaultdict(list)
    for replicate in range(replicates):
        generated = generator.generate(real_lines, rng)
        for source_name, source_function in SOURCE_FEATURES.items():
            for mode, deep_only in (("all", False), ("deep", True)):
                blocks = score_corpus(
                    generated, source_function, deep_only, localize=False
                )
                distributions[f"{source_name}:{mode}"].append(
                    aggregate_metric(blocks, "surface_morph_residual")
                )
        if (replicate + 1) % max(1, replicates // 4) == 0:
            reporter.say(
                f"  completed surrogate {replicate + 1}/{replicates}"
            )

    for key in sorted(distributions):
        values = np.asarray(distributions[key])
        real = real_aggregates[key]
        low, high = np.quantile(values, (0.025, 0.975))
        upper_p = (np.sum(values >= real - 1e-15) + 1) / (
            len(values) + 1
        )
        reporter.say(
            f"  {key:28s} real={real:+.5f}; surrogate mean="
            f"{values.mean():+.5f}; 95% [{low:+.5f},{high:+.5f}]; "
            f"empirical upper p={upper_p:.5f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/corpus.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/intermediate/followups_morph_markov_residual.txt"
        ),
    )
    parser.add_argument(
        "--surrogates",
        type=int,
        default=DEFAULT_SURROGATES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reporter = Reporter()
    lines, audit = load_lines(args.corpus)

    reporter.say("=" * 92)
    reporter.say("MORPHOLOGY AFTER WORD-MARKOV/LOCAL-COPY CONTROLS")
    reporter.say("=" * 92)
    reporter.say(
        f"prose lines={audit['prose_lines']}; source tokens="
        f"{audit['source_tokens']}; eligible={audit['eligible_tokens']}; "
        f"uncertain breaks={audit['uncertain_breaks']}; "
        f"one-character breaks={audit['one_char_breaks']}"
    )
    reporter.say(
        f"alpha={ALPHA}; hierarchical backoff mass={BACKOFF_MASS}; "
        f"frequent predecessor threshold={FREQUENT_PREDECESSOR}; "
        f"seed={SEED}"
    )

    results: dict[str, dict[str, list[BlockScore]]] = {}
    for source_name, source_function in SOURCE_FEATURES.items():
        results[source_name] = {
            "all": score_corpus(
                lines, source_function, False, localize=True
            ),
            "deep": score_corpus(
                lines, source_function, True, localize=False
            ),
        }
    real_aggregates = report_real(reporter, results)
    report_localization(reporter, results)
    report_surrogates(
        reporter, lines, real_aggregates, args.surrogates
    )

    reporter.say("\n" + "=" * 92)
    reporter.say("INTERPRETATION BOUNDARY")
    reporter.say("=" * 92)
    reporter.say(
        "A positive residual means source morphology pools predictive "
        "information across rare surface-word contexts. Copied formulae, "
        "scribal procedure, and a structured generator can all create the "
        "same residual."
    )
    reporter.write(args.output)


if __name__ == "__main__":
    main()
