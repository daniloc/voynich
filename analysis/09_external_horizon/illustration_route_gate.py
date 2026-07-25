#!/usr/bin/env python3
"""
Test whether illustrations specify text order rather than text values.

This experiment is deliberately disjoint from the K21/K22 visual-key family:
no pixel value, pigment, graph statistic, or zodiac state changes a symbol.
The drawings are allowed to do only one thing: prescribe a permutation of
already-written text units.

Plant data-gap screen
---------------------
The committed herbal graph summaries do not trace individual branches, so a
plant traversal is not identifiable from the current data.  For pipeline
diagnostics only, we construct a coarse rooted tree from the aggregate node,
branch-point, and terminal-group counts.  Folio prose is divided into one
contiguous chunk per graph node in visual top-to-bottom order, then screened
with root-to-tip breadth-first and depth-first traversals.  This result is
explicitly excluded from the inferential verdict: a real traced plant graph
could have a materially different route.

Zodiac gate
-----------
The committed zodiac records contain complete ordered circular node sets.
Labels are read clockwise or counterclockwise.  All ring edges, including the
closing edge, are scored, making the evaluation invariant to the arbitrary
choice of the first node.

The zodiac primary statistic is the best predeclared canonical route.  Its
null repeats the same maximum over random route permutations.  The plant
diagnostic also receives random and complete-graph rematching screens.
Positive controls scramble ordinary order with a declared route and require
that route to recover the planted sequence.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BODY_SPATIAL = ROOT / "data" / "corpus" / "body_spatial.json"
VISUAL_GRAPHS = ROOT / "data" / "grounding" / "multimodal_visual_graphs.json"
ZODIAC_NODES = ROOT / "data" / "grounding" / "zodiac_all12_visual_nodes.json"
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "external_illustration_route_gate.json"
)
DEFAULT_SEED = 20260724

PREFIXES = tuple(
    sorted(
        (
            "qok",
            "qot",
            "cth",
            "ckh",
            "cph",
            "cfh",
            "qo",
            "ok",
            "ot",
            "ch",
            "sh",
            "o",
            "y",
            "d",
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
            "aiin",
            "aiir",
            "edy",
            "eey",
            "ain",
            "iin",
            "dam",
            "dy",
            "ol",
            "or",
            "ar",
            "al",
            "am",
            "ey",
            "y",
        ),
        key=len,
        reverse=True,
    )
)
SIGNATURE_CHANNELS = ("edges", "affixes", "shape")


@dataclass(frozen=True)
class CoarsePlantGraph:
    node_count: int
    children: tuple[tuple[int, ...], ...]
    kinds: tuple[str, ...]

    def bfs(self) -> list[int]:
        order: list[int] = []
        queue = deque([0])
        while queue:
            node = queue.popleft()
            order.append(node)
            queue.extend(self.children[node])
        return order

    def dfs(self) -> list[int]:
        order: list[int] = []

        def visit(node: int) -> None:
            order.append(node)
            for child in self.children[node]:
                visit(child)

        visit(0)
        return order

    def depths(self) -> list[int]:
        values = [0] * self.node_count
        queue = deque([0])
        while queue:
            node = queue.popleft()
            for child in self.children[node]:
                values[child] = values[node] + 1
                queue.append(child)
        return values

    def display_order(self) -> list[int]:
        # The plant panels place terminal organs above stem and root.  Nodes
        # without recorded left/right coordinates retain construction order.
        depths = self.depths()
        return sorted(range(self.node_count), key=lambda n: (-depths[n], n))


class TransitionModel:
    def __init__(self, sequences: Iterable[Sequence[str]], alpha: float = 0.25):
        self.alpha = alpha
        self.pairs: Counter[tuple[str, str]] = Counter()
        self.contexts: Counter[str] = Counter()
        self.unigrams: Counter[str] = Counter()
        for sequence in sequences:
            for word in sequence:
                self.unigrams[word] += 1
            for left, right in zip(sequence, sequence[1:]):
                self.pairs[(left, right)] += 1
                self.contexts[left] += 1
        self.total = sum(self.unigrams.values())
        self.vocab = max(len(self.unigrams) + 1, 2)

    def log2_probability(self, left: str, right: str) -> float:
        unigram = (self.unigrams[right] + self.alpha) / (
            self.total + self.alpha * self.vocab
        )
        conditional = (
            self.pairs[(left, right)] + self.alpha * self.vocab * unigram
        ) / (self.contexts[left] + self.alpha * self.vocab)
        return math.log2(max(conditional, 1e-300))


def clean_word(value: str) -> str:
    return "".join(char for char in value.lower() if "a" <= char <= "z")


def affix(value: str, choices: Sequence[str]) -> str:
    return next((item for item in choices if value.startswith(item)), "_")


def suffix(value: str) -> str:
    return next((item for item in SUFFIXES if value.endswith(item)), "_")


def length_bin(value: str) -> str:
    length = len(value)
    if length <= 3:
        return "s"
    if length <= 5:
        return "m"
    if length <= 7:
        return "l"
    return "x"


def signature(word: str, channel: str) -> str:
    word = clean_word(word)
    if not word:
        return "_"
    if channel == "edges":
        return f"{word[0]}:{word[-1]}:{length_bin(word)}"
    if channel == "affixes":
        return f"{affix(word, PREFIXES)}:{suffix(word)}"
    if channel == "shape":
        gallows = sum(char in "ktpf" for char in word)
        benches = word.count("ch") + word.count("sh")
        repeated = int(any(a == b for a, b in zip(word, word[1:])))
        return (
            f"{length_bin(word)}:g{min(gallows, 2)}:"
            f"b{min(benches, 2)}:r{repeated}"
        )
    raise ValueError(channel)


def load_body() -> tuple[list[dict], dict[str, dict]]:
    source = json.loads(BODY_SPATIAL.read_text(encoding="utf-8"))
    tokens = [
        row
        for row in source["tokens"]
        if clean_word(row["word"]) and not row.get("uncertain", False)
    ]
    return tokens, source["pages"]


def prose_lines(
    tokens: Sequence[dict],
    excluded_quire: str,
    excluded_lines: set[tuple[str, int, str]] | None = None,
) -> list[list[str]]:
    excluded_lines = excluded_lines or set()
    grouped: dict[tuple[str, int, str], list[tuple[int, str]]] = defaultdict(list)
    for row in tokens:
        if row["quire"] == excluded_quire or row["role"] != "paragraph":
            continue
        key = (row["folio"], int(row["line"]), row["locus"])
        if key in excluded_lines:
            continue
        grouped[key].append(
            (int(row["idx"]), clean_word(row["word"]))
        )
    return [
        [word for _, word in sorted(values)]
        for values in grouped.values()
        if len(values) >= 2
    ]


def build_models(
    tokens: Sequence[dict],
    excluded_quire: str,
    excluded_lines: set[tuple[str, int, str]] | None = None,
) -> dict[str, TransitionModel]:
    lines = prose_lines(tokens, excluded_quire, excluded_lines)
    return {
        channel: TransitionModel(
            [[signature(word, channel) for word in line] for line in lines]
        )
        for channel in SIGNATURE_CHANNELS
    }


def score_word_pairs(
    pairs: Sequence[tuple[str, str]],
    models: dict[str, TransitionModel],
) -> float:
    if not pairs:
        return float("-inf")
    channel_scores = []
    for channel, model in models.items():
        values = [
            model.log2_probability(
                signature(left, channel),
                signature(right, channel),
            )
            for left, right in pairs
        ]
        channel_scores.append(mean(values))
    return mean(channel_scores)


def split_chunks(words: Sequence[str], count: int) -> list[list[str]]:
    if count < 1 or len(words) < count:
        raise ValueError(f"cannot divide {len(words)} words into {count} chunks")
    quotient, remainder = divmod(len(words), count)
    chunks = []
    start = 0
    for index in range(count):
        size = quotient + int(index < remainder)
        chunks.append(list(words[start:start + size]))
        start += size
    return chunks


def route_boundary_pairs(
    chunks: Sequence[Sequence[str]],
    route: Sequence[int],
) -> list[tuple[str, str]]:
    ordered = [chunks[index] for index in route]
    return [
        (left[-1], right[0])
        for left, right in zip(ordered, ordered[1:])
        if left and right
    ]


def circular_pairs(words: Sequence[str]) -> list[tuple[str, str]]:
    if len(words) < 2:
        return []
    return list(zip(words, words[1:] + words[:1]))


def coarse_plant_graph(row: dict) -> CoarsePlantGraph:
    graph = row["graph"]
    node_count = max(int(graph["node_count"]), 3)
    branch_count = min(
        int(graph["branch_point_count"]),
        max(node_count - 3, 0),
    )
    terminal_count = min(
        max(int(graph["terminal_group_count"]), 1),
        node_count - 2 - branch_count,
    )
    support_count = node_count - 2 - branch_count - terminal_count

    kinds = ["root", "stem"]
    kinds.extend("branch" for _ in range(branch_count))
    kinds.extend("terminal" for _ in range(terminal_count))
    kinds.extend("organ" for _ in range(support_count))
    children: list[list[int]] = [[] for _ in range(node_count)]
    children[0].append(1)

    branch_nodes = list(range(2, 2 + branch_count))
    terminal_start = 2 + branch_count
    terminal_nodes = list(range(terminal_start, terminal_start + terminal_count))
    support_nodes = list(range(terminal_start + terminal_count, node_count))

    if branch_nodes:
        children[1].extend(branch_nodes)
        parents = branch_nodes
    else:
        parents = [1]
    for index, node in enumerate(terminal_nodes):
        children[parents[index % len(parents)]].append(node)
    for index, node in enumerate(support_nodes):
        children[parents[index % len(parents)]].append(node)

    result = CoarsePlantGraph(
        node_count=node_count,
        children=tuple(tuple(values) for values in children),
        kinds=tuple(kinds),
    )
    if len(result.bfs()) != node_count:
        raise RuntimeError(f"disconnected constructed graph for {row['folio']}")
    return result


def plant_words(tokens: Sequence[dict], folio: str) -> list[str]:
    rows = [
        row
        for row in tokens
        if row["folio"] == folio and row["role"] == "paragraph"
    ]
    rows.sort(key=lambda row: (int(row["line"]), int(row["idx"])))
    return [clean_word(row["word"]) for row in rows]


def graph_route(graph: CoarsePlantGraph, name: str) -> list[int]:
    display_index = {
        node: index for index, node in enumerate(graph.display_order())
    }
    if name == "root_dfs":
        nodes = graph.dfs()
    elif name == "root_bfs":
        nodes = graph.bfs()
    elif name == "display":
        nodes = graph.display_order()
    else:
        raise ValueError(name)
    return [display_index[node] for node in nodes]


def plant_route_scores(
    records: Sequence[dict],
    tokens: Sequence[dict],
    pages: dict[str, dict],
    models_by_quire: dict[str, dict[str, TransitionModel]],
    graph_assignment: dict[str, dict] | None = None,
    random_routes: dict[str, list[list[int]]] | None = None,
) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = defaultdict(list)
    graph_assignment = graph_assignment or {row["folio"]: row for row in records}
    for row in records:
        folio = row["folio"]
        words = plant_words(tokens, folio)
        donor = graph_assignment[folio]
        graph = coarse_plant_graph(donor)
        chunks = split_chunks(words, graph.node_count)
        quire = pages[folio]["quire"]
        models = models_by_quire[quire]
        display = list(range(graph.node_count))
        scores["display"].append(
            score_word_pairs(route_boundary_pairs(chunks, display), models)
        )
        if random_routes is None:
            for name in ("root_dfs", "root_bfs"):
                route = graph_route(graph, name)
                scores[name].append(
                    score_word_pairs(route_boundary_pairs(chunks, route), models)
                )
        else:
            for index, route in enumerate(random_routes[folio]):
                scores[f"random_{index}"].append(
                    score_word_pairs(route_boundary_pairs(chunks, route), models)
                )
    return scores


def route_gain(scores: dict[str, list[float]], name: str) -> float:
    return mean(
        candidate - baseline
        for candidate, baseline in zip(scores[name], scores["display"])
    )


def empirical_upper_p(observed: float, nulls: Sequence[float]) -> float:
    return (1 + sum(value >= observed - 1e-12 for value in nulls)) / (
        len(nulls) + 1
    )


def randomized_routes(
    records: Sequence[dict],
    rng: random.Random,
    count: int = 2,
) -> dict[str, list[list[int]]]:
    routes = {}
    for row in records:
        size = coarse_plant_graph(row).node_count
        candidates = []
        for _ in range(count):
            route = list(range(size))
            rng.shuffle(route)
            candidates.append(route)
        routes[row["folio"]] = candidates
    return routes


def rematched_graphs(
    records: Sequence[dict],
    pages: dict[str, dict],
    rng: random.Random,
) -> tuple[dict[str, dict], int]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        groups[pages[row["folio"]]["quire"]].append(row)
    assignment: dict[str, dict] = {}
    moved = 0
    for rows in groups.values():
        donors = list(rows)
        if len(rows) > 1:
            for _ in range(20):
                rng.shuffle(donors)
                if all(a["folio"] != b["folio"] for a, b in zip(rows, donors)):
                    break
        for target, donor in zip(rows, donors):
            assignment[target["folio"]] = donor
            moved += int(target["folio"] != donor["folio"])
    return assignment, moved


def plant_positive_control(
    records: Sequence[dict],
    tokens: Sequence[dict],
    pages: dict[str, dict],
    models_by_quire: dict[str, dict[str, TransitionModel]],
    rng: random.Random,
    replicates: int,
) -> dict:
    planted_scores: list[float] = []
    display_scores: list[float] = []
    random_scores: list[list[float]] = [[] for _ in range(replicates)]
    for row in records:
        folio = row["folio"]
        graph = coarse_plant_graph(row)
        natural = plant_words(tokens, folio)
        natural_chunks = split_chunks(natural, graph.node_count)
        planted_route = graph_route(graph, "root_dfs")
        displayed: list[list[str]] = [[] for _ in range(graph.node_count)]
        for natural_index, display_index in enumerate(planted_route):
            displayed[display_index] = natural_chunks[natural_index]
        models = models_by_quire[pages[folio]["quire"]]
        planted_scores.append(score_word_pairs(
            route_boundary_pairs(displayed, planted_route), models
        ))
        display_scores.append(score_word_pairs(
            route_boundary_pairs(displayed, list(range(graph.node_count))),
            models,
        ))
        for replicate in range(replicates):
            route = list(range(graph.node_count))
            rng.shuffle(route)
            random_scores[replicate].append(score_word_pairs(
                route_boundary_pairs(displayed, route), models
            ))
    planted_gain = mean(
        candidate - baseline
        for candidate, baseline in zip(planted_scores, display_scores)
    )
    null_gains = [
        mean(value - baseline for value, baseline in zip(values, display_scores))
        for values in random_scores
    ]
    p_value = empirical_upper_p(planted_gain, null_gains)
    return {
        "planted_route": "root_dfs",
        "gain_bits_per_boundary": planted_gain,
        "random_null_mean_gain": mean(null_gains),
        "random_null_p_upper": p_value,
        "passed": planted_gain > 0 and p_value <= 0.05,
    }


def run_plant_gate(
    all_records: Sequence[dict],
    tokens: Sequence[dict],
    pages: dict[str, dict],
    rng: random.Random,
    replicates: int,
) -> dict:
    records = [row for row in all_records if row["domain"] == "herbal"]
    quires = sorted({pages[row["folio"]]["quire"] for row in records})
    models = {quire: build_models(tokens, quire) for quire in quires}
    observed = plant_route_scores(records, tokens, pages, models)
    gains = {
        name: route_gain(observed, name)
        for name in ("root_dfs", "root_bfs")
    }
    observed_statistic = max(gains.values())

    random_nulls = []
    for _ in range(replicates):
        routes = randomized_routes(records, rng)
        scores = plant_route_scores(
            records,
            tokens,
            pages,
            models,
            random_routes=routes,
        )
        random_nulls.append(max(
            route_gain(scores, "random_0"),
            route_gain(scores, "random_1"),
        ))

    rematch_nulls = []
    moved_counts = []
    for _ in range(replicates):
        assignment, moved = rematched_graphs(records, pages, rng)
        scores = plant_route_scores(
            records,
            tokens,
            pages,
            models,
            graph_assignment=assignment,
        )
        rematch_nulls.append(max(
            route_gain(scores, "root_dfs"),
            route_gain(scores, "root_bfs"),
        ))
        moved_counts.append(moved)

    per_folio = []
    for index, row in enumerate(records):
        per_folio.append({
            "folio": row["folio"],
            "quire": pages[row["folio"]]["quire"],
            "node_count": coarse_plant_graph(row).node_count,
            "display_score_bits": observed["display"][index],
            "root_dfs_gain_bits_per_boundary": (
                observed["root_dfs"][index] - observed["display"][index]
            ),
            "root_bfs_gain_bits_per_boundary": (
                observed["root_bfs"][index] - observed["display"][index]
            ),
        })

    return {
        "identifiable": False,
        "inferential_status": "not_testable_from_aggregate_graph_summaries",
        "excluded_from_decision": True,
        "data_gap": (
            "The records provide aggregate node, branch-point, and terminal "
            "counts but no observed node-edge adjacency or sibling geometry. "
            "DFS/BFS orders are therefore properties of a constructed "
            "diagnostic tree, not recoverable properties of the drawings."
        ),
        "unit": "complete herbal folio; reference model excludes its quire",
        "n_folios": len(records),
        "n_quires": len(quires),
        "routes": {
            "root_dfs": (
                "preorder root-to-tip traversal of the coarse organ tree"
            ),
            "root_bfs": (
                "level-order root-to-tip traversal of the coarse organ tree"
            ),
        },
        "route_gains_bits_per_boundary": gains,
        "selected_statistic": observed_statistic,
        "selected_route": max(gains, key=gains.get),
        "random_route_null": {
            "replicates": replicates,
            "mean": mean(random_nulls),
            "observed_debug_quantile": mean(
                value <= observed_statistic for value in random_nulls
            ),
        },
        "within_quire_graph_rematch_null": {
            "replicates": replicates,
            "mean": mean(rematch_nulls),
            "observed_debug_quantile": mean(
                value <= observed_statistic for value in rematch_nulls
            ),
            "mean_moved_folios": mean(moved_counts),
            "singleton_quires_remain_fixed": True,
        },
        "null_interpretation": (
            "exploratory software diagnostics only; no plant null probability "
            "has inferential meaning without observed node-edge topology"
        ),
        "positive_control": plant_positive_control(
            records, tokens, pages, models, rng, replicates
        ),
        "per_folio": per_folio,
    }


def zodiac_sequences(source: dict) -> list[dict]:
    sequences = []
    for folio, page in source["folios"].items():
        for tier in ("outer", "inner"):
            records = sorted(
                (row for row in page["records"] if row["tier"] == tier),
                key=lambda row: int(row["cyclic_index"]),
            )
            words = [clean_word(row["label_primary"]) for row in records]
            if len(words) >= 3 and all(words):
                sequences.append({
                    "folio": folio,
                    "tier": tier,
                    "words": words,
                    "confidence": page["alignment"][tier]["confidence"],
                })
    return sequences


def zodiac_route_score(
    words: Sequence[str],
    route: Sequence[int],
    models: dict[str, TransitionModel],
) -> float:
    ordered = [words[index] for index in route]
    return score_word_pairs(circular_pairs(ordered), models)


def zodiac_positive_control(
    sequences: Sequence[dict],
    tokens: Sequence[dict],
    pages: dict[str, dict],
    rng: random.Random,
    replicates: int,
) -> dict:
    grouped: dict[
        tuple[str, int, str],
        list[tuple[int, str, str]],
    ] = defaultdict(list)
    for row in tokens:
        if row["role"] != "paragraph":
            continue
        key = (row["folio"], int(row["line"]), row["locus"])
        grouped[key].append((
            int(row["idx"]),
            clean_word(row["word"]),
            row["quire"],
        ))
    donor_lines = [
        (
            key,
            values[0][2],
            [word for _, word, _ in sorted(values)],
        )
        for key, values in grouped.items()
        if len(values) >= 3
    ]

    planted_forward = []
    planted_reverse = []
    control_rows = []
    random_forward: list[list[float]] = [[] for _ in range(replicates)]
    random_reverse: list[list[float]] = [[] for _ in range(replicates)]
    for sequence in sequences:
        quire = pages[sequence["folio"]]["quire"]
        candidates = [
            row
            for row in donor_lines
            if row[1] != quire and len(row[2]) >= len(sequence["words"])
        ]
        donor_key, donor_quire, donor_words = candidates[
            (sum(map(ord, sequence["folio"])) + len(sequence["tier"]))
            % len(candidates)
        ]
        natural = donor_words[:len(sequence["words"])]
        size = len(natural)
        ccw = list(reversed(range(size)))
        displayed = [""] * size
        for natural_index, node_index in enumerate(ccw):
            displayed[node_index] = natural[natural_index]
        # This model sees neither the complete target quire nor the exact
        # ordinary prose line from which the planted sequence was drawn.
        models = build_models(tokens, quire, {donor_key})
        planted_forward.append(zodiac_route_score(
            displayed, list(range(size)), models
        ))
        planted_reverse.append(zodiac_route_score(displayed, ccw, models))
        control_rows.append({
            "target_folio": sequence["folio"],
            "target_tier": sequence["tier"],
            "target_quire_excluded": quire,
            "donor_line_excluded": {
                "folio": donor_key[0],
                "line": donor_key[1],
                "locus": donor_key[2],
                "quire": donor_quire,
            },
        })
        for replicate in range(replicates):
            route = list(range(size))
            rng.shuffle(route)
            random_forward[replicate].append(
                zodiac_route_score(displayed, route, models)
            )
            random_reverse[replicate].append(
                zodiac_route_score(displayed, list(reversed(route)), models)
            )
    planted_means = {
        "clockwise": mean(planted_forward),
        "counterclockwise": mean(planted_reverse),
    }
    planted_score = max(planted_means.values())
    selected_route = max(planted_means, key=planted_means.get)
    null_values = [
        max(mean(forward), mean(reverse))
        for forward, reverse in zip(random_forward, random_reverse)
    ]
    p_value = empirical_upper_p(planted_score, null_values)
    return {
        "planted_route": "counterclockwise",
        "selected_route": selected_route,
        "route_scores_bits_per_transition": planted_means,
        "selected_score_bits_per_transition": planted_score,
        "random_null_mean_score": mean(null_values),
        "random_null_p_upper": p_value,
        "training_exclusion": (
            "each control model excludes the complete target quire and the "
            "exact donor prose line"
        ),
        "passed": (
            selected_route == "counterclockwise"
            and planted_score > mean(null_values)
            and p_value <= 0.05
        ),
        "control_rows": control_rows,
    }


def run_zodiac_gate(
    source: dict,
    tokens: Sequence[dict],
    pages: dict[str, dict],
    rng: random.Random,
    replicates: int,
) -> dict:
    sequences = zodiac_sequences(source)
    quires = sorted({pages[row["folio"]]["quire"] for row in sequences})
    models = {quire: build_models(tokens, quire) for quire in quires}
    route_scores: dict[str, list[float]] = defaultdict(list)
    per_ring = []
    for sequence in sequences:
        words = sequence["words"]
        size = len(words)
        models_for_ring = models[pages[sequence["folio"]]["quire"]]
        cw_score = zodiac_route_score(
            words, list(range(size)), models_for_ring
        )
        ccw_score = zodiac_route_score(
            words, list(reversed(range(size))), models_for_ring
        )
        route_scores["clockwise"].append(cw_score)
        route_scores["counterclockwise"].append(ccw_score)
        per_ring.append({
            "folio": sequence["folio"],
            "tier": sequence["tier"],
            "nodes": size,
            "alignment_confidence": sequence["confidence"],
            "clockwise_score_bits_per_transition": cw_score,
            "counterclockwise_score_bits_per_transition": ccw_score,
        })
    means = {name: mean(values) for name, values in route_scores.items()}
    observed = max(means.values())

    nulls = []
    for _ in range(replicates):
        forward_scores = []
        reverse_scores = []
        for sequence in sequences:
            route = list(range(len(sequence["words"])))
            rng.shuffle(route)
            models_for_ring = models[pages[sequence["folio"]]["quire"]]
            forward_scores.append(zodiac_route_score(
                sequence["words"], route, models_for_ring
            ))
            reverse_scores.append(zodiac_route_score(
                sequence["words"], list(reversed(route)), models_for_ring
            ))
        nulls.append(max(mean(forward_scores), mean(reverse_scores)))

    return {
        "identifiable": True,
        "evaluation": (
            "two fixed routes evaluated over three physical quires; no route "
            "is fitted, and quire exclusion applies only to the reference "
            "transition model"
        ),
        "unit": (
            "complete folio/ring tier; circular score is invariant to start "
            "anchor; reference model excludes the complete quire"
        ),
        "n_folios": len({row["folio"] for row in sequences}),
        "n_rings": len(sequences),
        "n_quires": len(quires),
        "quires": quires,
        "route_scores_bits_per_transition": means,
        "selected_statistic": observed,
        "selected_route": max(means, key=means.get),
        "random_hamiltonian_route_null": {
            "replicates": replicates,
            "mean": mean(nulls),
            "p_upper": empirical_upper_p(observed, nulls),
            "construction": (
                "one random circular order per ring, scored forward and in "
                "its exact reverse; statistic is the maximum global mean"
            ),
        },
        "positive_control": zodiac_positive_control(
            sequences, tokens, pages, rng, replicates
        ),
        "per_ring": per_ring,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates < 20:
        raise ValueError("--replicates must be at least 20")
    tokens, pages = load_body()
    visual_source = json.loads(VISUAL_GRAPHS.read_text(encoding="utf-8"))
    zodiac_source = json.loads(ZODIAC_NODES.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    plant = run_plant_gate(
        visual_source["records"],
        tokens,
        pages,
        rng,
        args.replicates,
    )
    zodiac = run_zodiac_gate(
        zodiac_source,
        tokens,
        pages,
        rng,
        args.replicates,
    )
    result = {
        "experiment": "external illustration-derived route gate",
        "seed": args.seed,
        "replicates": args.replicates,
        "predeclared_scope": {
            "allowed": (
                "illustration-derived permutation of existing text units"
            ),
            "excluded": [
                "visual value shifts",
                "pigment keys",
                "visual state-conditioned symbol values",
                "phase selection from text",
                "reverse or alternating plant routes without traced geometry",
            ],
            "signature_channels": list(SIGNATURE_CHANNELS),
            "score": (
                "mean heldout log2 transition probability across fixed "
                "morphological signature channels"
            ),
        },
        "plant_gate": plant,
        "zodiac_gate": zodiac,
        "decision": {
            "plant_route_supported": None,
            "plant_route_status": (
                "not testable until traced herbal node-edge graphs exist"
            ),
            "zodiac_route_supported": (
                zodiac["random_hamiltonian_route_null"]["p_upper"] <= 0.05
                and zodiac["positive_control"]["passed"]
            ),
        },
        "caveats": [
            (
                "The herbal annotations contain graph summaries rather than "
                "traced node coordinates or edges. Plant results are pipeline "
                "diagnostics only and provide no evidence for or against real "
                "plant traversals."
            ),
            (
                "The gate detects whether a route restores ordinary Voynich "
                "morphological transition order. It does not assume or recover "
                "plaintext."
            ),
            (
                "Zodiac alignment confidence is retained in the output, but "
                "the circular statistic removes start-node phase uncertainty."
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "plant": {
            "identifiable": plant["identifiable"],
            "status": plant["inferential_status"],
            "route": plant["selected_route"],
            "gain": plant["selected_statistic"],
            "diagnostic_only": True,
            "control": plant["positive_control"]["passed"],
        },
        "zodiac": {
            "route": zodiac["selected_route"],
            "score": zodiac["selected_statistic"],
            "random_p": zodiac["random_hamiltonian_route_null"]["p_upper"],
            "control": zodiac["positive_control"]["passed"],
        },
        "decision": result["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
