#!/usr/bin/env python3
"""
Held-out adversarial controls for the table-constrained permutation decoder.

The target attack in naibbe_permutation_decoder.py obtains an excellent
in-sample official Naibbe decode.  This script asks whether that improvement
transfers when the fitted table permutations are frozen.

Primary statistic
-----------------
For a fit stream and an untouched test stream with the same 18 role/table
blocks, fit the target solver only on the former.  On test report

    mean log2 P_Latin(refined frozen key)
      - mean log2 P_Latin(context-initializer frozen key).

The comparison is paired at every held-out emission and neither test symbols
nor test plaintext affect fitting.

Positive control
----------------
Fit official Naibbe first-half -> second-half and the reverse direction.
Official plaintext is loaded only after both keys are frozen.

Adversarial plaintext-free controls
-----------------------------------
SPLIT_KEY
    Apply independently randomized within-table symbol permutations to the two
    halves of the official structural stream.  Each half separately has a
    perfect Latin key, but no single key transfers.

BLOCK_EXCHANGE
    Exchange symbol identities among positions of the same role/table block,
    separately in each half.  This preserves block labels and exact blockwise
    symbol histograms while destroying sequential assignment.

TABLE_MARKOV
    Generate symbols from a stationary first-order model conditioned on the
    previous symbol/block and current role/table label.  The fixed block-label
    stream, blockwise marginals, and first-order recurrence are preserved
    without a plaintext or global substitution key.

Twenty independently seeded null attacks are pooled.  Their strict maximum is
a finite-sample one-sided threshold with resolution 1/(20+1)=0.0476 for one
predeclared attack.  A latent/group search over K candidates must repeat the
entire K-way selection inside every null replicate; this script's threshold is
not a license to search K groupings on the final test split.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from naibbe_permutation_decoder import (  # noqa: E402
    ALPHABET,
    CipherStream,
    TetragramLM,
    context_initializer,
    decode,
    load_control_segments,
    load_official_tokens,
    load_official_truth,
    load_structural_inventory,
    parse_tokens,
    refine_key,
)


SEED = 20260723
NULL_COUNTS = {
    "SPLIT_KEY": 7,
    "BLOCK_EXCHANGE": 7,
    "TABLE_MARKOV": 6,
}
LOG2 = math.log(2.0)


@dataclass
class AttackJob:
    name: str
    seed: int
    train: CipherStream
    test: CipherStream
    return_decoded: bool = False


@dataclass
class AttackResult:
    name: str
    seed: int
    train_emissions: int
    test_emissions: int
    train_initial_lm: float
    train_final_lm: float
    train_residual: float
    test_initial_lm: float
    test_final_lm: float
    test_residual: float
    block_updates: int
    swap_updates: int
    test_decoded: Optional[list[int]] = None


def slice_stream(
    stream: CipherStream,
    start: int,
    stop: int,
) -> CipherStream:
    return CipherStream(
        block_at=stream.block_at[start:stop],
        symbol_at=stream.symbol_at[start:stop],
        token_count=stop - start,
        parsed_token_count=stop - start,
        ambiguous_bigram_count=0,
    )


def stream_with_symbols(
    template: CipherStream,
    symbols: Sequence[int],
) -> CipherStream:
    if len(symbols) != len(template.block_at):
        raise ValueError("symbol/template length mismatch")
    return CipherStream(
        block_at=list(template.block_at),
        symbol_at=list(symbols),
        token_count=template.token_count,
        parsed_token_count=template.parsed_token_count,
        ambiguous_bigram_count=template.ambiguous_bigram_count,
    )


def lm_bits(lm: TetragramLM, decoded: Sequence[int]) -> float:
    return lm.mean_score(decoded) / LOG2


def run_attack(job: AttackJob) -> AttackResult:
    lm_text, _ = load_control_segments()
    lm = TetragramLM(lm_text)
    block_count = 18
    initial_key = context_initializer(job.train, lm, block_count)
    initial_train = decode(job.train, initial_key)
    fit = refine_key(job.train, lm, initial_key)

    initial_test = decode(job.test, initial_key)
    final_test = decode(job.test, fit.key)
    train_initial_lm = lm_bits(lm, initial_train)
    train_final_lm = lm_bits(lm, fit.decoded)
    test_initial_lm = lm_bits(lm, initial_test)
    test_final_lm = lm_bits(lm, final_test)
    return AttackResult(
        name=job.name,
        seed=job.seed,
        train_emissions=len(job.train.block_at),
        test_emissions=len(job.test.block_at),
        train_initial_lm=train_initial_lm,
        train_final_lm=train_final_lm,
        train_residual=train_final_lm - train_initial_lm,
        test_initial_lm=test_initial_lm,
        test_final_lm=test_final_lm,
        test_residual=test_final_lm - test_initial_lm,
        block_updates=fit.block_updates,
        swap_updates=fit.swap_updates,
        test_decoded=final_test if job.return_decoded else None,
    )


def permuted_symbols(
    stream: CipherStream,
    rng: random.Random,
) -> list[int]:
    permutations = []
    for _ in range(18):
        row = list(range(len(ALPHABET)))
        rng.shuffle(row)
        permutations.append(row)
    return [
        permutations[block][symbol]
        for block, symbol in zip(stream.block_at, stream.symbol_at)
    ]


def block_exchange_symbols(
    stream: CipherStream,
    rng: random.Random,
) -> list[int]:
    result = list(stream.symbol_at)
    positions: dict[int, list[int]] = defaultdict(list)
    for index, block in enumerate(stream.block_at):
        positions[block].append(index)
    for indices in positions.values():
        values = [result[index] for index in indices]
        rng.shuffle(values)
        for index, value in zip(indices, values):
            result[index] = value
    return result


@dataclass
class MarkovTables:
    marginal: dict[int, tuple[list[int], list[int]]]
    transition: dict[
        tuple[int, int, int],
        tuple[list[int], list[int]],
    ]


def counter_table(
    counter: Counter,
) -> tuple[list[int], list[int]]:
    values = sorted(counter)
    cumulative = []
    total = 0
    for value in values:
        total += counter[value]
        cumulative.append(total)
    return values, cumulative


def build_markov_tables(stream: CipherStream) -> MarkovTables:
    marginal: dict[int, Counter] = defaultdict(Counter)
    transition: dict[tuple[int, int, int], Counter] = defaultdict(Counter)
    for index, (block, symbol) in enumerate(
        zip(stream.block_at, stream.symbol_at)
    ):
        marginal[block][symbol] += 1
        if index:
            context = (
                stream.block_at[index - 1],
                stream.symbol_at[index - 1],
                block,
            )
            transition[context][symbol] += 1
    return MarkovTables(
        marginal={
            context: counter_table(counts)
            for context, counts in marginal.items()
        },
        transition={
            context: counter_table(counts)
            for context, counts in transition.items()
        },
    )


def weighted_draw(
    table: tuple[list[int], list[int]],
    rng: random.Random,
) -> int:
    values, cumulative = table
    draw = rng.randrange(cumulative[-1])
    low, high = 0, len(cumulative)
    while low < high:
        middle = (low + high) // 2
        if draw < cumulative[middle]:
            high = middle
        else:
            low = middle + 1
    return values[low]


def markov_symbols(
    template: CipherStream,
    tables: MarkovTables,
    rng: random.Random,
) -> list[int]:
    result = []
    for index, block in enumerate(template.block_at):
        table = None
        if result and rng.random() < 0.90:
            context = (
                template.block_at[index - 1],
                result[-1],
                block,
            )
            table = tables.transition.get(context)
        if table is None:
            table = tables.marginal[block]
        result.append(weighted_draw(table, rng))
    return result


def make_jobs(
    official: CipherStream,
) -> tuple[list[AttackJob], list[tuple[int, int]]]:
    cut = len(official.block_at) // 2
    first = slice_stream(official, 0, cut)
    second = slice_stream(official, cut, len(official.block_at))
    jobs = [
        AttackJob("NAIBBE_FIRST_TO_SECOND", SEED, first, second, True),
        AttackJob("NAIBBE_SECOND_TO_FIRST", SEED + 1, second, first, True),
    ]
    truth_slices = [
        (cut, len(official.block_at)),
        (0, cut),
    ]

    seed = SEED + 100
    for replicate in range(NULL_COUNTS["SPLIT_KEY"]):
        rng = random.Random(seed)
        jobs.append(AttackJob(
            f"SPLIT_KEY_{replicate + 1:02d}",
            seed,
            stream_with_symbols(first, permuted_symbols(first, rng)),
            stream_with_symbols(second, permuted_symbols(second, rng)),
        ))
        seed += 1

    for replicate in range(NULL_COUNTS["BLOCK_EXCHANGE"]):
        rng = random.Random(seed)
        jobs.append(AttackJob(
            f"BLOCK_EXCHANGE_{replicate + 1:02d}",
            seed,
            stream_with_symbols(
                first, block_exchange_symbols(first, rng)
            ),
            stream_with_symbols(
                second, block_exchange_symbols(second, rng)
            ),
        ))
        seed += 1

    tables = build_markov_tables(official)
    for replicate in range(NULL_COUNTS["TABLE_MARKOV"]):
        rng = random.Random(seed)
        jobs.append(AttackJob(
            f"TABLE_MARKOV_{replicate + 1:02d}",
            seed,
            stream_with_symbols(first, markov_symbols(first, tables, rng)),
            stream_with_symbols(
                second, markov_symbols(second, tables, rng)
            ),
        ))
        seed += 1
    return jobs, truth_slices


def plaintext_accuracy(
    decoded: Sequence[int],
    truth: str,
) -> float:
    truth_ids = [ALPHABET.index(character) for character in truth]
    length = min(len(decoded), len(truth_ids))
    return sum(
        decoded[index] == truth_ids[index]
        for index in range(length)
    ) / max(len(decoded), len(truth_ids), 1)


def quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_family(results: Sequence[AttackResult]) -> dict:
    residuals = [result.test_residual for result in results]
    final_lm = [result.test_final_lm for result in results]
    train = [result.train_residual for result in results]
    return {
        "n": len(results),
        "train_residual_mean": sum(train) / len(train),
        "test_residual_mean": sum(residuals) / len(residuals),
        "test_residual_min": min(residuals),
        "test_residual_max": max(residuals),
        "test_final_lm_mean": sum(final_lm) / len(final_lm),
        "test_final_lm_max": max(final_lm),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data/intermediate/followups_permutation_controls.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = load_structural_inventory()
    official, parsing = parse_tokens(
        load_official_tokens(), inventory
    )
    jobs, truth_slices = make_jobs(official)

    print("=" * 96)
    print("HELD-OUT CONTROLS FOR TABLE-CONSTRAINED PERMUTATION DECODER")
    print("=" * 96)
    print(
        f"official emissions={len(official.block_at)}; split="
        f"{len(jobs[0].train.block_at)}/{len(jobs[0].test.block_at)}; "
        f"parse coverage={parsing['token_coverage']:.4f}"
    )
    print(
        f"null attacks={sum(NULL_COUNTS.values())}; workers={args.jobs}; "
        "primary units=bits per held-out tetragram"
    )

    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(run_attack, job): job.name for job in jobs
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"  [{completed:02d}/{len(jobs)}] {result.name:26s} "
                f"train={result.train_residual:+.4f} "
                f"test={result.test_residual:+.4f} "
                f"test LM={result.test_final_lm:+.4f}"
            )
    by_name = {result.name: result for result in results}

    # Plaintext is first opened after both positive-control keys are frozen.
    truth = load_official_truth()
    positive = []
    for name, (start, stop) in zip(
        ("NAIBBE_FIRST_TO_SECOND", "NAIBBE_SECOND_TO_FIRST"),
        truth_slices,
    ):
        result = by_name[name]
        truth_part = truth[start:stop]
        result_dict = asdict(result)
        result_dict["plaintext_accuracy"] = plaintext_accuracy(
            result.test_decoded or [], truth_part
        )
        result_dict.pop("test_decoded", None)
        positive.append(result_dict)

    null_results = [
        result
        for result in results
        if not result.name.startswith("NAIBBE_")
    ]
    null_residuals = [result.test_residual for result in null_results]
    null_final_lm = [result.test_final_lm for result in null_results]
    residual_threshold = max(null_residuals)
    lm_threshold = max(null_final_lm)

    families = {}
    for family in NULL_COUNTS:
        members = [
            result for result in null_results
            if result.name.startswith(family)
        ]
        families[family] = summarize_family(members)

    print("\nPOSITIVE CONTROL, FROZEN-KEY HELD-OUT")
    for row in positive:
        print(
            f"  {row['name']:26s} residual={row['test_residual']:+.5f} "
            f"LM={row['test_final_lm']:+.5f} "
            f"accuracy={row['plaintext_accuracy']:.4f}"
        )

    print("\nNULL FAMILY SUMMARIES")
    for family, row in families.items():
        print(
            f"  {family:16s} n={row['n']:2d} "
            f"train residual={row['train_residual_mean']:+.5f}; "
            f"test mean/range={row['test_residual_mean']:+.5f} "
            f"[{row['test_residual_min']:+.5f},"
            f"{row['test_residual_max']:+.5f}]"
        )

    positives_passing = sum(
        row["test_residual"] > residual_threshold
        and row["test_final_lm"] > lm_threshold
        for row in positive
    )
    print("\nPREDECLARED ATTACK THRESHOLD")
    print(
        f"  held-out residual must be > {residual_threshold:+.5f} "
        "bits/tetragram (strict maximum of 20 null attacks)"
    )
    print(
        f"  held-out final Latin LM must be > {lm_threshold:+.5f} "
        "bits/tetragram"
    )
    print(
        f"  positive folds passing both={positives_passing}/2; "
        "null false positives=0/20 by construction; conformal resolution="
        "1/21=0.0476"
    )
    print(
        "  For a latent/group Voynich search, freeze grouping, inventory, "
        "and key before the final test. If K candidates are searched, each "
        "null replicate must repeat the same K-way selection and the threshold "
        "must use replicate-wise maxima."
    )

    report = {
        "method": "held-out adversarial controls for table permutation solver",
        "seed": SEED,
        "official_emissions": len(official.block_at),
        "split": {
            "train": len(jobs[0].train.block_at),
            "test": len(jobs[0].test.block_at),
        },
        "null_counts": NULL_COUNTS,
        "primary_metric": (
            "held-out mean log2 tetragram gain: refined frozen key minus "
            "context-initializer frozen key"
        ),
        "positive": positive,
        "null_families": families,
        "null_results": [
            {
                key: value
                for key, value in asdict(result).items()
                if key != "test_decoded"
            }
            for result in sorted(null_results, key=lambda item: item.name)
        ],
        "threshold": {
            "residual_strict_gt": residual_threshold,
            "final_lm_strict_gt": lm_threshold,
            "null_residual_q95": quantile(null_residuals, 0.95),
            "null_final_lm_q95": quantile(null_final_lm, 0.95),
            "positive_folds_passing": positives_passing,
            "null_false_positives": 0,
            "null_attacks": len(null_results),
            "one_sided_resolution": 1 / (len(null_results) + 1),
            "scope": (
                "one predeclared attack; latent/group candidate search must "
                "be nested inside every null replicate"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
