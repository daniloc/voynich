#!/usr/bin/env python3
"""
Locate the causal memory horizon of the Voynich character generator.

Every candidate starts from the same register-conditioned character trigram
trained on two complete-quire folds.  At validation/test time it may update a
local transition cache only after observing each complete word.  Candidates
are static, line-reset, folio-cumulative, or rolling caches of 8--256 words.
The cache horizon is selected on a disjoint validation fold and scored once on
the untouched test fold.

The controls are an independent draw from the static generator, a draw with a
planted 32-word cache, and Latin reflowed into the Voynich layout.  They
separate horizon recovery from generic online language-model adaptation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Optional, Sequence

import production_algorithm_gate as source


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_adaptive_horizon.json"
)
SEED = 20260724
BACKOFF = 32.0
BACKOFFS = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
PLANTED_HORIZON = 32
HORIZONS: tuple[Optional[int], ...] = (8, 16, 32, 64, 128, 256, None)


def candidate_name(horizon: Optional[int]) -> str:
    return "folio_cumulative" if horizon is None else f"rolling_{horizon}"


def alpha_name(backoff: float) -> str:
    return f"folio_alpha_{int(backoff)}"


def cache_log_probabilities(
    model: source.CharacterModel,
    events: Sequence[source.Event],
    horizon: Optional[int],
    reset: str = "folio",
    backoff: float = BACKOFF,
) -> list[float]:
    counts: Counter = Counter()
    totals: Counter = Counter()
    queues: dict[str, deque[list[tuple[tuple[str, ...], str]]]] = defaultdict(
        deque
    )
    result = []
    for event in events:
        unit = event.folio if reset == "folio" else f"{event.folio}:{event.line}"
        history = ["^"] * model.order
        emitted = model.symbols(event.word)
        updates = []
        total = 0.0
        for symbol in emitted:
            context = tuple(history[-model.order:]) if model.order else ()
            base_probability = model.probability(event, history, symbol)
            key = (unit, context)
            probability = (
                counts[(key, symbol)] + backoff * base_probability
            ) / (totals[key] + backoff)
            total += math.log2(max(probability, 1e-300))
            updates.append((context, symbol))
            history.append(symbol)
        result.append(total)
        for context, symbol in updates:
            key = (unit, context)
            counts[(key, symbol)] += 1
            totals[key] += 1
        queue = queues[unit]
        queue.append(updates)
        if horizon is not None and len(queue) > horizon:
            for context, symbol in queue.popleft():
                key = (unit, context)
                counts[(key, symbol)] -= 1
                totals[key] -= 1
    return result


def sample_cached_word(
    model: source.CharacterModel,
    event: source.Event,
    rng: random.Random,
    counts: Counter,
    totals: Counter,
) -> tuple[str, list[tuple[tuple[str, ...], str]]]:
    unit = event.folio
    for _attempt in range(50):
        history = ["^"] * model.order
        word = []
        updates = []
        for _ in range(source.MAX_WORD_LENGTH):
            context = tuple(history[-model.order:]) if model.order else ()
            key = (unit, context)
            probabilities = [
                (
                    counts[(key, symbol)]
                    + BACKOFF * model.probability(
                        event, history, symbol
                    )
                )
                / (totals[key] + BACKOFF)
                for symbol in model.alphabet
            ]
            symbol = rng.choices(
                model.alphabet, weights=probabilities, k=1
            )[0]
            updates.append((context, symbol))
            if symbol == model.eos:
                if len(word) >= 2:
                    return "".join(word), updates
                break
            if symbol == model.unknown:
                symbol = rng.choice(tuple(sorted(model.alphabet_set)))
                updates[-1] = (context, symbol)
            word.append(symbol)
            history.append(symbol)
    return "ol", [(("^", "^"), "o"), (("^", "o"), "l"), (("o", "l"), "$")]


def generated_cached_events(
    template: Sequence[source.Event],
    model: source.CharacterModel,
    horizon: int,
    seed: int,
) -> list[source.Event]:
    rng = random.Random(seed)
    counts: Counter = Counter()
    totals: Counter = Counter()
    queues: dict[str, deque[list[tuple[tuple[str, ...], str]]]] = defaultdict(
        deque
    )
    words = []
    for event in template:
        word, updates = sample_cached_word(
            model, event, rng, counts, totals
        )
        words.append(word)
        unit = event.folio
        for context, symbol in updates:
            key = (unit, context)
            counts[(key, symbol)] += 1
            totals[key] += 1
        queue = queues[unit]
        queue.append(updates)
        if len(queue) > horizon:
            for context, symbol in queue.popleft():
                key = (unit, context)
                counts[(key, symbol)] -= 1
                totals[key] -= 1
    return source.rebuild_histories(template, words)


def score_logs(logs: Sequence[float]) -> float:
    return -sum(logs) / len(logs)


def run_panel(
    name: str,
    events: Sequence[source.Event],
    progress: bool,
) -> dict[str, object]:
    assignment, loads = source.block_folds(events)
    fold_results = []
    pooled: dict[str, Counter] = defaultdict(Counter)
    selected_rows = []
    for fold in range(source.N_FOLDS):
        train, validation, test = source.split_events(
            events, assignment, fold
        )
        model = source.CharacterModel(train, 2, "register")
        validation_logs = {
            "static": model.log_probabilities(validation),
            "line_cumulative": cache_log_probabilities(
                model, validation, None, reset="line"
            ),
        }
        test_logs = {
            "static": model.log_probabilities(test),
            "line_cumulative": cache_log_probabilities(
                model, test, None, reset="line"
            ),
        }
        for horizon in HORIZONS:
            candidate = candidate_name(horizon)
            validation_logs[candidate] = cache_log_probabilities(
                model, validation, horizon
            )
            test_logs[candidate] = cache_log_probabilities(
                model, test, horizon
            )
        for backoff in BACKOFFS:
            candidate = alpha_name(backoff)
            validation_logs[candidate] = cache_log_probabilities(
                model, validation, None, backoff=backoff
            )
            test_logs[candidate] = cache_log_probabilities(
                model, test, None, backoff=backoff
            )
        validation_scores = {
            candidate: score_logs(logs)
            for candidate, logs in validation_logs.items()
        }
        winner = min(
            validation_scores,
            key=lambda candidate: (
                validation_scores[candidate], candidate
            ),
        )
        rows = []
        for candidate, logs in test_logs.items():
            score = score_logs(logs)
            pooled[candidate]["log2_probability"] += sum(logs)
            pooled[candidate]["events"] += len(test)
            rows.append({
                "candidate": candidate,
                "validation_bits_per_word": validation_scores[candidate],
                "test_bits_per_word": score,
            })
        static_logs = test_logs["static"]
        winner_logs = test_logs[winner]
        for event, static_log, winner_log in zip(
            test, static_logs, winner_logs
        ):
            selected_rows.append({
                "sequence": event.sequence,
                "segment": event.segment,
                "block": event.block,
                "position": event.position,
                "word_length": len(event.word),
                "surprisal": -winner_log,
                "gain_vs_static": winner_log - static_log,
            })
        fold_results.append({
            "fold": fold,
            "train_blocks": sorted({event.block for event in train}),
            "validation_blocks": sorted({
                event.block for event in validation
            }),
            "test_blocks": sorted({event.block for event in test}),
            "selected_candidate": winner,
            "candidate_scores": rows,
        })
        if progress:
            winner_test = next(
                row["test_bits_per_word"]
                for row in rows
                if row["candidate"] == winner
            )
            print(
                f"{name} fold={fold} selected={winner} "
                f"test={winner_test:.4f}",
                flush=True,
            )
    candidate_summary = {
        candidate: {
            "test_events": int(values["events"]),
            "pooled_test_bits_per_word": (
                -values["log2_probability"] / values["events"]
            ),
        }
        for candidate, values in pooled.items()
    }
    selected_gain = sum(
        float(row["gain_vs_static"]) for row in selected_rows
    ) / len(selected_rows)
    best_fixed_alpha = min(
        (alpha_name(backoff) for backoff in BACKOFFS),
        key=lambda candidate: candidate_summary[candidate][
            "pooled_test_bits_per_word"
        ],
    )
    return {
        "name": name,
        "fold_assignment": assignment,
        "fold_event_loads": loads,
        "folds": fold_results,
        "candidate_summary": candidate_summary,
        "summary": {
            "selected_candidate_counts": dict(Counter(
                fold["selected_candidate"] for fold in fold_results
            )),
            "nested_selected_gain_vs_static_bits_per_word": selected_gain,
            "folio_cumulative_gain_vs_static_bits_per_word": (
                candidate_summary["static"]["pooled_test_bits_per_word"]
                - candidate_summary["folio_cumulative"][
                    "pooled_test_bits_per_word"
                ]
            ),
            "descriptive_best_pooled_alpha": best_fixed_alpha,
            "residual_surprisal_order": source.residual_autocorrelation(
                selected_rows, SEED + sum(map(ord, name))
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=source.CORPUS)
    parser.add_argument("--latin", type=Path, default=source.LATIN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events, audit = source.load_events(args.corpus)
    panels = [run_panel("VOYNICH", events, args.progress)]
    if not args.skip_controls:
        base = source.CharacterModel(events, 2, "register")
        channel = source.EditChannel(events)
        independent, _truth = source.generated_events(
            events,
            base,
            channel,
            {"base": 1.0, "copy": 0.0, "edit": 0.0},
            SEED + 20_000,
        )
        planted = generated_cached_events(
            events, base, PLANTED_HORIZON, SEED + 30_000
        )
        panels.extend([
            run_panel("SYNTHETIC_INDEPENDENT", independent, args.progress),
            run_panel("SYNTHETIC_ROLLING_32", planted, args.progress),
            run_panel(
                "LATIN_REFLOW",
                source.latin_events(events, args.latin),
                args.progress,
            ),
        ])
    by_name = {panel["name"]: panel for panel in panels}
    controls = {}
    if not args.skip_controls:
        planted_selected = by_name["SYNTHETIC_ROLLING_32"]["summary"][
            "selected_candidate_counts"
        ]
        controls = {
            "independent_gain_below_0_20_pass": (
                by_name["SYNTHETIC_INDEPENDENT"]["summary"][
                    "nested_selected_gain_vs_static_bits_per_word"
                ]
                < 0.20
            ),
            "planted_cache_positive_gain_pass": (
                by_name["SYNTHETIC_ROLLING_32"]["summary"][
                    "nested_selected_gain_vs_static_bits_per_word"
                ]
                > 0.20
            ),
            "planted_horizon_neighborhood_selected_pass": (
                sum(
                    planted_selected.get(candidate, 0)
                    for candidate in ("rolling_16", "rolling_32", "rolling_64")
                )
                >= 3
            ),
        }
    result = {
        "experiment": "causal_adaptive_horizon_selection",
        "seed": SEED,
        "parameters": {
            "cache_backoff": BACKOFF,
            "folio_dirichlet_concentrations": list(BACKOFFS),
            "rolling_horizons_words": [
                horizon for horizon in HORIZONS if horizon is not None
            ],
            "cumulative_candidates": ["line", "folio"],
            "planted_horizon_words": PLANTED_HORIZON,
            "split": (
                "test=fold f; validation=fold f+1; fit=other two folds; "
                "complete repository quire blocks"
            ),
        },
        "claim_boundary": (
            "A selected cache is a causal predictive production state. "
            "Ordinary-language performance determines whether it is "
            "Voynich-specific; it does not recover plaintext."
        ),
        "corpus_audit": dict(audit),
        "controls": controls,
        "panels": panels,
        "summary": {
            "voynich": panels[0]["summary"],
            "controls_pass": all(controls.values()) if controls else None,
        },
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
