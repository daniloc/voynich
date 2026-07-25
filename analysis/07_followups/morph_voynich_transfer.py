"""
Held-out transfer test for Voynich morphotactic ordering.

The score is deliberately narrower than a language-model score.  For each
representation and held-out production block, it compares

    P(next class | previous class, target-position)

against

    P(next class | target-position).

Thus a positive gain says that the predecessor adds reusable information beyond
the five-way line-position bucket.  It does not by itself identify syntax,
semantics, or even linguistic generation.

Primary inference is leave-one-complete-quire-out.  Secondary tests transfer
between identified hands and Currier classes, with quire portions retained as
the inference blocks.  Every vocabulary and count is fitted without the test
block.  A fixed add-0.5 estimator is used, avoiding test-informed tuning.

Uncertain and one-character tokens follow the filtering convention in
sequence_parallel.py, but they break adjacency rather than being deleted and
silently joining their neighbours.  All eligible prose lines are retained.
"""

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


SEED = 20260722
ALPHA = 0.5
N_BOOTSTRAP = 20000
N_FLIP_MONTE_CARLO = 100000
UNKNOWN = "<TRAIN-ONLY-UNK>"

PREFIXES = [
    "qok", "qot", "qo", "ok", "ot", "o", "y", "ch", "sh", "d",
    "cth", "ckh", "cph", "cfh",
]
SUFFIXES = [
    "eedy", "eody", "edy", "aiin", "aiir", "ain", "iin", "dy", "ol",
    "or", "ar", "al", "am", "dam", "ey", "eey", "y",
]


def locus_type(locus):
    match = re.search(r"[A-Za-z]", locus)
    return match.group(0).upper() if match else "?"


def decompose(word):
    prefix = next(
        (
            value
            for value in sorted(PREFIXES, key=len, reverse=True)
            if word.startswith(value)
        ),
        "",
    )
    residual = word[len(prefix):]
    suffix = next(
        (
            value
            for value in sorted(SUFFIXES, key=len, reverse=True)
            if residual.endswith(value) and len(residual) > len(value)
        ),
        "",
    )
    core = residual[:-len(suffix)] if suffix else residual
    return prefix, core, suffix


def position_bucket(index, length):
    if index == 0:
        return "first"
    if index == length - 1:
        return "last"
    if index == 1:
        return "second"
    if index == length - 2:
        return "penult"
    return "interior"


def representations():
    return {
        "prefix": lambda word: decompose(word)[0] or "none",
        "suffix": lambda word: decompose(word)[2] or "none",
        "affix_pair": lambda word: (
            decompose(word)[0] or "none",
            decompose(word)[2] or "none",
        ),
        "shape": lambda word: (
            bool(decompose(word)[0]),
            bool(decompose(word)[2]),
            min(len(word), 8),
            any(char in word for char in "ktpf"),
        ),
    }


def load_lines():
    with open("data/corpus/corpus.json", encoding="utf-8") as handle:
        corpus = json.load(handle)

    lines = []
    audit = Counter()
    for folio, folio_lines in corpus["folios"].items():
        meta = corpus["meta"].get(folio, {})
        for source in folio_lines:
            if locus_type(source["locus"]) != "P":
                continue
            audit["prose_lines"] += 1
            words = []
            for word in source["words"]:
                audit["source_tokens"] += 1
                if "?" in word:
                    words.append(None)
                    audit["uncertain_tokens_breaking_adjacency"] += 1
                elif len(word) < 2:
                    words.append(None)
                    audit["one_char_tokens_breaking_adjacency"] += 1
                else:
                    words.append(word)
                    audit["eligible_tokens"] += 1
            line = {
                "folio": folio,
                "line": source["line"],
                "locus": source["locus"],
                "section": meta.get("I", "?"),
                "currier": meta.get("L", "?"),
                "hand": meta.get("H", "?"),
                "quire": meta.get("Q", "?"),
                "words": words,
            }
            if any(True for _ in iter_raw_transitions(line, False)):
                lines.append(line)
                audit["lines_with_eligible_transition"] += 1
            else:
                audit["lines_without_eligible_transition"] += 1
    return lines, audit


def iter_raw_transitions(line, deep_only):
    words = line["words"]
    length = len(words)
    for index in range(length - 1):
        left, right = words[index], words[index + 1]
        if left is None or right is None:
            continue
        left_position = position_bucket(index, length)
        target_position = position_bucket(index + 1, length)
        if deep_only and not (
            left_position == "interior" and target_position == "interior"
        ):
            continue
        yield left, right, target_position


@dataclass
class Model:
    function: object
    left_vocabulary: set
    right_vocabulary: set
    baseline: Counter
    position_totals: Counter
    bigram: Counter
    context_totals: Counter

    @property
    def vocabulary_size(self):
        return len(self.right_vocabulary) + 1

    def map_left(self, value):
        return value if value in self.left_vocabulary else UNKNOWN

    def map_right(self, value):
        return value if value in self.right_vocabulary else UNKNOWN

    def gain(self, left_word, right_word, target_position):
        left = self.map_left(self.function(left_word))
        right = self.map_right(self.function(right_word))
        size = self.vocabulary_size
        bigram_probability = (
            self.bigram[(target_position, left, right)] + ALPHA
        ) / (
            self.context_totals[(target_position, left)] + ALPHA * size
        )
        baseline_probability = (
            self.baseline[(target_position, right)] + ALPHA
        ) / (
            self.position_totals[target_position] + ALPHA * size
        )
        return math.log2(bigram_probability / baseline_probability), left, right


def fit_model(lines, function, deep_only):
    raw = []
    left_vocabulary = set()
    right_vocabulary = set()
    for line in lines:
        for left_word, right_word, target_position in iter_raw_transitions(
            line, deep_only
        ):
            left = function(left_word)
            right = function(right_word)
            left_vocabulary.add(left)
            right_vocabulary.add(right)
            raw.append((target_position, left, right))

    baseline = Counter()
    position_totals = Counter()
    bigram = Counter()
    context_totals = Counter()
    for target_position, left, right in raw:
        baseline[(target_position, right)] += 1
        position_totals[target_position] += 1
        bigram[(target_position, left, right)] += 1
        context_totals[(target_position, left)] += 1
    return Model(
        function,
        left_vocabulary,
        right_vocabulary,
        baseline,
        position_totals,
        bigram,
        context_totals,
    )


@dataclass
class BlockScore:
    label: str
    bits: float = 0.0
    transitions: int = 0
    contributions: dict = field(default_factory=lambda: defaultdict(list))

    @property
    def gain(self):
        return self.bits / self.transitions if self.transitions else float("nan")


def score_block(label, lines, model, deep_only, keep_contributions=False):
    score = BlockScore(label)
    for line in lines:
        for left_word, right_word, target_position in iter_raw_transitions(
            line, deep_only
        ):
            gain, left, right = model.gain(
                left_word, right_word, target_position
            )
            score.bits += gain
            score.transitions += 1
            if keep_contributions:
                key = (target_position, left, right)
                values = score.contributions[key]
                if values:
                    values[0] += gain
                    values[1] += 1
                else:
                    score.contributions[key] = [gain, 1]
    return score


def exact_sign_p(gains):
    nonzero = [value for value in gains if abs(value) > 1e-15]
    if not nonzero:
        return 1.0
    positives = sum(value > 0 for value in nonzero)
    return sum(
        math.comb(len(nonzero), count)
        for count in range(positives, len(nonzero) + 1)
    ) / (2 ** len(nonzero))


def flip_distribution(bits, total_transitions, rng):
    count = len(bits)
    if count <= 20:
        signed_sums = np.array([0.0])
        for value in bits:
            signed_sums = np.concatenate(
                (signed_sums + value, signed_sums - value)
            )
        return signed_sums / total_transitions, True

    signs = rng.choice(
        (-1.0, 1.0), size=(N_FLIP_MONTE_CARLO, count)
    )
    return signs @ np.asarray(bits) / total_transitions, False


def block_summary(blocks, rng):
    blocks = [block for block in blocks if block.transitions]
    if not blocks:
        return None
    bits = np.asarray([block.bits for block in blocks], dtype=float)
    transitions = np.asarray(
        [block.transitions for block in blocks], dtype=float
    )
    gains = bits / transitions
    observed = bits.sum() / transitions.sum()

    indices = rng.integers(
        0, len(blocks), size=(N_BOOTSTRAP, len(blocks))
    )
    boot_bits = bits[indices].sum(axis=1)
    boot_transitions = transitions[indices].sum(axis=1)
    boot = boot_bits / boot_transitions
    bootstrap_interval = np.quantile(boot, (0.025, 0.975))

    null, exact = flip_distribution(bits, transitions.sum(), rng)
    flip_interval = np.quantile(null, (0.025, 0.975))
    if exact:
        flip_p = np.mean(null >= observed - 1e-15)
    else:
        flip_p = (np.sum(null >= observed - 1e-15) + 1) / (len(null) + 1)

    return {
        "blocks": len(blocks),
        "transitions": int(transitions.sum()),
        "gain": observed,
        "median": float(np.median(gains)),
        "positive": int(np.sum(gains > 0)),
        "bootstrap_low": bootstrap_interval[0],
        "bootstrap_high": bootstrap_interval[1],
        "sign_p": exact_sign_p(gains),
        "flip_low": flip_interval[0],
        "flip_high": flip_interval[1],
        "flip_p": flip_p,
        "flip_exact": exact,
    }


def print_summary(summary, indent="  "):
    if summary is None:
        print(f"{indent}no eligible held-out transitions")
        return
    method = "exact" if summary["flip_exact"] else "MC"
    print(
        f"{indent}gain={summary['gain']:+.5f} bits/transition; "
        f"block-bootstrap 95% CI "
        f"[{summary['bootstrap_low']:+.5f}, "
        f"{summary['bootstrap_high']:+.5f}]"
    )
    print(
        f"{indent}blocks positive={summary['positive']}/"
        f"{summary['blocks']}; median={summary['median']:+.5f}; "
        f"one-sided sign p={summary['sign_p']:.5f}"
    )
    print(
        f"{indent}block sign-flip null 95% "
        f"[{summary['flip_low']:+.5f}, {summary['flip_high']:+.5f}]; "
        f"{method} one-sided p={summary['flip_p']:.5f}; "
        f"transitions={summary['transitions']}"
    )


def leave_one_quire_out(lines, reps):
    quires = sorted({line["quire"] for line in lines if line["quire"] != "?"})
    results = {
        name: {"all": [], "deep": []}
        for name in reps
    }

    for name, function in reps.items():
        for quire in quires:
            train = [line for line in lines if line["quire"] != quire]
            test = [line for line in lines if line["quire"] == quire]
            for mode, deep_only in (("all", False), ("deep", True)):
                model = fit_model(train, function, deep_only)
                score = score_block(
                    quire,
                    test,
                    model,
                    deep_only,
                    keep_contributions=(mode == "all"),
                )
                results[name][mode].append(score)
    return results


def print_primary(results):
    print("\n" + "=" * 88)
    print("PRIMARY: LEAVE ONE COMPLETE QUIRE OUT")
    print("=" * 88)
    print(
        "Gain = log2 P(next | predecessor,target-position) - "
        "log2 P(next | target-position)"
    )
    print(
        "Deep sensitivity fits and scores only transitions whose two tokens "
        "are both in the interior bucket."
    )
    rng = np.random.default_rng(SEED)
    primary_ps = []
    for name, modes in results.items():
        print(f"\n{name}")
        print(
            f"  {'quire':>5s} {'all n':>7s} {'all gain':>10s} "
            f"{'deep n':>7s} {'deep gain':>10s}"
        )
        all_by_label = {score.label: score for score in modes["all"]}
        deep_by_label = {score.label: score for score in modes["deep"]}
        for label in sorted(all_by_label):
            all_score = all_by_label[label]
            deep_score = deep_by_label[label]
            deep_gain = (
                f"{deep_score.gain:+.5f}"
                if deep_score.transitions
                else "NA"
            )
            print(
                f"  {label:>5s} {all_score.transitions:7d} "
                f"{all_score.gain:+10.5f} {deep_score.transitions:7d} "
                f"{deep_gain:>10s}"
            )
        print("  All eligible transitions:")
        all_summary = block_summary(modes["all"], rng)
        print_summary(all_summary, "    ")
        print("  Deep-interior-only sensitivity:")
        print_summary(block_summary(modes["deep"], rng), "    ")
        primary_ps.append(all_summary["flip_p"])

    print("\n  Multiplicity check for the four planned all-transition tests:")
    for name, raw_p in zip(results, primary_ps):
        print(
            f"    {name:12s} raw block-flip p={raw_p:.5f}; "
            f"Bonferroni p={min(1.0, 4 * raw_p):.5f}"
        )


def evaluate_transfer_target(
    train,
    test,
    block_key,
    function,
    deep_only,
):
    model = fit_model(train, function, deep_only)
    groups = defaultdict(list)
    for line in test:
        groups[block_key(line)].append(line)
    return [
        score_block(label, groups[label], model, deep_only)
        for label in sorted(groups)
    ]


def print_cross_hand(lines, reps):
    known = sorted(
        {
            line["hand"]
            for line in lines
            if line["hand"] not in ("?", "")
        }
    )
    rng = np.random.default_rng(SEED + 1)
    print("\n" + "=" * 88)
    print("SECONDARY: LEAVE ONE IDENTIFIED HAND OUT")
    print("=" * 88)
    print(
        "Training excludes the target hand and all unidentified-hand lines; "
        "test inference blocks are complete quire portions in that hand."
    )
    print(
        "H=X/Y are retained as identifiable corpus categories but should not "
        "be assumed equivalent to numbered palaeographic hands."
    )
    for name, function in reps.items():
        print(f"\n{name}")
        print(
            f"  {'hand':>5s} {'blocks':>6s} {'all gain':>10s} "
            f"{'all CI':>22s} {'flip p':>8s} {'deep gain':>10s}"
        )
        for target in known:
            train = [
                line
                for line in lines
                if line["hand"] not in ("?", "", target)
            ]
            test = [line for line in lines if line["hand"] == target]
            all_blocks = evaluate_transfer_target(
                train, test, lambda line: line["quire"], function, False
            )
            deep_blocks = evaluate_transfer_target(
                train, test, lambda line: line["quire"], function, True
            )
            all_summary = block_summary(all_blocks, rng)
            deep_summary = block_summary(deep_blocks, rng)
            if all_summary is None:
                continue
            interval = (
                f"[{all_summary['bootstrap_low']:+.4f},"
                f"{all_summary['bootstrap_high']:+.4f}]"
            )
            deep_gain = (
                f"{deep_summary['gain']:+.5f}"
                if deep_summary is not None
                else "NA"
            )
            print(
                f"  {target:>5s} {all_summary['blocks']:6d} "
                f"{all_summary['gain']:+10.5f} {interval:>22s} "
                f"{all_summary['flip_p']:8.5f} {deep_gain:>10s}"
            )


def print_cross_currier(lines, reps):
    rng = np.random.default_rng(SEED + 2)
    print("\n" + "=" * 88)
    print("SECONDARY: DIRECT CROSS-CURRIER TRANSFER")
    print("=" * 88)
    print(
        "Only identified A/B lines enter.  Each direction trains exclusively "
        "on the source Currier class and scores target quire portions."
    )
    for name, function in reps.items():
        print(f"\n{name}")
        print(
            f"  {'direction':>9s} {'blocks':>6s} {'all gain':>10s} "
            f"{'all CI':>22s} {'flip p':>8s} {'deep gain':>10s}"
        )
        for source, target in (("A", "B"), ("B", "A")):
            train = [line for line in lines if line["currier"] == source]
            test = [line for line in lines if line["currier"] == target]
            all_blocks = evaluate_transfer_target(
                train, test, lambda line: line["quire"], function, False
            )
            deep_blocks = evaluate_transfer_target(
                train, test, lambda line: line["quire"], function, True
            )
            all_summary = block_summary(all_blocks, rng)
            deep_summary = block_summary(deep_blocks, rng)
            interval = (
                f"[{all_summary['bootstrap_low']:+.4f},"
                f"{all_summary['bootstrap_high']:+.4f}]"
            )
            deep_gain = (
                f"{deep_summary['gain']:+.5f}"
                if deep_summary is not None
                else "NA"
            )
            print(
                f"  {source + '->' + target:>9s} "
                f"{all_summary['blocks']:6d} "
                f"{all_summary['gain']:+10.5f} {interval:>22s} "
                f"{all_summary['flip_p']:8.5f} {deep_gain:>10s}"
            )


def display_value(value):
    if isinstance(value, tuple):
        return "(" + ",".join(str(item) for item in value) + ")"
    return str(value)


def stable_contributions(blocks):
    by_transition = defaultdict(dict)
    for block in blocks:
        for key, (bits, count) in block.contributions.items():
            by_transition[key][block.label] = (bits, count)

    candidates = []
    for key, block_values in by_transition.items():
        total_bits = sum(value[0] for value in block_values.values())
        total_count = sum(value[1] for value in block_values.values())
        if total_count < 20 or len(block_values) < 5:
            continue
        positive_fraction = np.mean(
            [value[0] > 0 for value in block_values.values()]
        )
        candidates.append(
            {
                "key": key,
                "bits": total_bits,
                "count": total_count,
                "blocks": len(block_values),
                "positive_fraction": positive_fraction,
                "bits_per_event": total_bits / total_count,
            }
        )
    positive = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["positive_fraction"] >= 0.70
            and candidate["bits"] > 0
        ),
        key=lambda candidate: candidate["bits"],
        reverse=True,
    )[:6]
    negative = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["positive_fraction"] <= 0.30
            and candidate["bits"] < 0
        ),
        key=lambda candidate: candidate["bits"],
    )[:6]
    return positive, negative


def print_localization(results):
    print("\n" + "=" * 88)
    print("LOCALIZATION: STABLE HELD-OUT TRANSITION CONTRIBUTIONS")
    print("=" * 88)
    print(
        "Primary all-transition scores only.  Listed cells occur >=20 times "
        "in >=5 held-out quires and have the same contribution sign in >=70% "
        "of those quires.  Ranking uses total held-out bits, so this is "
        "descriptive localization rather than a second significance test."
    )
    for name, modes in results.items():
        positive, negative = stable_contributions(modes["all"])
        print(f"\n{name}")
        for label, candidates in (
            ("stable positive", positive),
            ("stable negative", negative),
        ):
            print(f"  {label}:")
            if not candidates:
                print("    none at the declared support threshold")
                continue
            for candidate in candidates:
                position, left, right = candidate["key"]
                print(
                    f"    {position:8s} {display_value(left)} -> "
                    f"{display_value(right)}: total="
                    f"{candidate['bits']:+.2f} bits, "
                    f"mean={candidate['bits_per_event']:+.3f}, "
                    f"n={candidate['count']}, "
                    f"quires={candidate['blocks']}, "
                    f"positive={candidate['positive_fraction']:.2f}"
                )


def print_audit(lines, audit):
    print("=" * 88)
    print("VOYNICH MORPHOTACTIC TRANSFER BENCHMARK")
    print("=" * 88)
    print(
        f"prose lines={audit['prose_lines']}; retained with an eligible "
        f"transition={len(lines)}; source tokens={audit['source_tokens']}"
    )
    print(
        f"eligible tokens={audit['eligible_tokens']}; uncertain breaks="
        f"{audit['uncertain_tokens_breaking_adjacency']}; one-character "
        f"breaks={audit['one_char_tokens_breaking_adjacency']}"
    )
    print(
        "Filtering does not bridge across excluded tokens.  Fixed smoothing "
        f"alpha={ALPHA}; bootstrap={N_BOOTSTRAP}; seed={SEED}."
    )


def main():
    lines, audit = load_lines()
    reps = representations()
    print_audit(lines, audit)
    results = leave_one_quire_out(lines, reps)
    print_primary(results)
    print_cross_hand(lines, reps)
    print_cross_currier(lines, reps)
    print_localization(results)

    print("\n" + "=" * 88)
    print("INTERPRETATION BOUNDARY")
    print("=" * 88)
    print(
        "Positive held-out gain establishes reusable directional class "
        "constraints beyond target position.  Copying, scribal procedure, "
        "and algorithmic generation remain viable causes; this test does not "
        "identify syntax, semantics, or plaintext."
    )


if __name__ == "__main__":
    main()
