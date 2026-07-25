"""
Follow-up: look for sequence structure above exact words, then surface approximate
parallel lines that could serve as cryptanalytic anchors.

Sequence test
-------------
For several word representations, measure adjacent-class mutual information within
section x Currier strata. The null permutes whole words among equivalent physical
positions while preserving either:

  coarse: section x Currier x position
  strict: section x Currier x hand x quire x position

This retains known layout and production-block effects while destroying local
adjacency. The representations are declared before the test and all use the same
permutation stream.

Parallel search
---------------
Rank prose-line pairs from different quires, but the same section and Currier dialect,
by TF-IDF cosine over exact stripped cores. Common cores are excluded from candidate
generation. LCS/core-order similarity is reported but not used for selection.
"""

import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

RNG = np.random.default_rng(20260722)
N_PERM = 200

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
        (value for value in sorted(PREFIXES, key=len, reverse=True)
         if word.startswith(value)),
        "",
    )
    residual = word[len(prefix):]
    suffix = next(
        (value for value in sorted(SUFFIXES, key=len, reverse=True)
         if residual.endswith(value) and len(residual) > len(value)),
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


def entropy(counter):
    total = sum(counter.values())
    return -sum(
        count / total * math.log2(count / total)
        for count in counter.values()
        if count
    ) if total else 0.0


def mutual_information(pairs):
    joint = Counter(pairs)
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    total = len(pairs)
    if not total:
        return 0.0
    return sum(
        count / total
        * math.log2(
            (count / total)
            / ((left[a] / total) * (right[b] / total))
        )
        for (a, b), count in joint.items()
    )


def load_lines():
    corpus = json.load(open("data/corpus/corpus.json"))
    lines = []
    for folio, folio_lines in corpus["folios"].items():
        meta = corpus["meta"].get(folio, {})
        for line in folio_lines:
            if locus_type(line["locus"]) != "P":
                continue
            words = [
                word for word in line["words"]
                if "?" not in word and len(word) >= 2
            ]
            if len(words) < 4:
                continue
            lines.append({
                "folio": folio,
                "line": line["line"],
                "locus": line["locus"],
                "section": meta.get("I", "?"),
                "currier": meta.get("L", "?"),
                "hand": meta.get("H", "?"),
                "quire": meta.get("Q", "?"),
                "words": words,
            })
    return lines


def build_slots(lines):
    slots = []
    line_slot_ids = []
    for line_id, line in enumerate(lines):
        ids = []
        for index, word in enumerate(line["words"]):
            ids.append(len(slots))
            slots.append({
                "line_id": line_id,
                "word": word,
                "position": position_bucket(index, len(line["words"])),
                "section": line["section"],
                "currier": line["currier"],
                "hand": line["hand"],
                "quire": line["quire"],
            })
        line_slot_ids.append(ids)
    return slots, line_slot_ids


def representations(words):
    counts = Counter(words)

    def frequency_band(word):
        count = counts[word]
        if count == 1:
            return "hapax"
        if count <= 3:
            return "rare"
        if count <= 10:
            return "mid"
        if count <= 50:
            return "common"
        return "very_common"

    return {
        "exact_word": lambda word: word,
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
        "frequency_band": frequency_band,
    }


def stratified_mi(slot_values, slots, line_slot_ids):
    by_stratum = defaultdict(list)
    total_pairs = 0
    for ids in line_slot_ids:
        for left_id, right_id in zip(ids, ids[1:]):
            slot = slots[left_id]
            stratum = (slot["section"], slot["currier"])
            by_stratum[stratum].append(
                (slot_values[left_id], slot_values[right_id])
            )
            total_pairs += 1
    return sum(
        len(pairs) / total_pairs * mutual_information(pairs)
        for pairs in by_stratum.values()
    ) if total_pairs else 0.0


def permutation_groups(slots, strict):
    groups = defaultdict(list)
    for index, slot in enumerate(slots):
        key = (
            slot["section"],
            slot["currier"],
            slot["position"],
        )
        if strict:
            key += (slot["hand"], slot["quire"])
        groups[key].append(index)
    return groups


def permuted_words(words, groups):
    result = list(words)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        source = RNG.permutation(indices)
        for target, origin in zip(indices, source):
            result[target] = words[origin]
    return result


def sequence_battery(lines):
    slots, line_slot_ids = build_slots(lines)
    words = [slot["word"] for slot in slots]
    reps = representations(words)
    print("=" * 78)
    print("SEQUENCE TEST: adjacent-class MI beyond layout and production blocks")
    print("=" * 78)
    print(
        f"lines={len(lines)} tokens={len(words)} adjacent pairs="
        f"{sum(len(ids) - 1 for ids in line_slot_ids)} permutations={N_PERM}"
    )

    for strict in (False, True):
        label = "STRICT section+Currier+hand+quire+position" if strict else (
            "COARSE section+Currier+position"
        )
        groups = permutation_groups(slots, strict)
        observed = {
            name: stratified_mi(
                [function(word) for word in words], slots, line_slot_ids
            )
            for name, function in reps.items()
        }
        null = {name: [] for name in reps}
        for _ in range(N_PERM):
            shuffled = permuted_words(words, groups)
            for name, function in reps.items():
                values = [function(word) for word in shuffled]
                null[name].append(
                    stratified_mi(values, slots, line_slot_ids)
                )

        print(f"\nNull: {label} ({len(groups)} exchangeability groups)")
        print(
            f"  {'representation':18s} {'observed':>10s} {'null':>10s} "
            f"{'excess':>10s} {'z':>8s} {'p':>8s}"
        )
        raw_ps = []
        for name in reps:
            values = np.asarray(null[name])
            excess = observed[name] - values.mean()
            z_score = excess / values.std() if values.std() else float("nan")
            p_value = (
                np.sum(values >= observed[name] - 1e-12) + 1
            ) / (len(values) + 1)
            raw_ps.append(p_value)
            print(
                f"  {name:18s} {observed[name]:10.4f} "
                f"{values.mean():10.4f} {excess:+10.4f} "
                f"{z_score:8.2f} {p_value:8.4f}"
            )
        print(
            f"  Bonferroni minimum p: "
            f"{min(1.0, min(raw_ps) * len(raw_ps)):.4f}"
        )


def sign_flip_p(block_differences, n_perm=5000):
    values = np.asarray(block_differences, dtype=float)
    observed = values.sum()
    if not len(values):
        return 1.0
    exceed = 0
    for _ in range(n_perm):
        if np.sum(values * RNG.choice((-1, 1), size=len(values))) >= observed:
            exceed += 1
    return (exceed + 1) / (n_perm + 1)


def directionality_battery(lines):
    """Cross-validated forward-vs-interior-reversed scoring.

    Reversing only the deep interior keeps the specialized first/second and
    penultimate/last slots fixed. Models are trained on three of four whole-quire
    folds and evaluated on the omitted fold.
    """
    words = [word for line in lines for word in line["words"]]
    reps = representations(words)
    reps.pop("exact_word")
    quires = sorted(set(line["quire"] for line in lines))
    fold_of = {quire: index % 4 for index, quire in enumerate(quires)}
    alpha = 0.5

    print("\n" + "=" * 78)
    print("HELD-OUT DIRECTIONALITY: forward vs interior-reversed lines")
    print("=" * 78)
    print(
        "four folds by whole quire; first/second and penult/last remain fixed"
    )
    print(
        f"  {'representation':18s} {'bits/transition':>16s} "
        f"{'positive lines':>15s} {'quire-flip p':>13s}"
    )

    raw_ps = []
    for name, function in reps.items():
        line_values = [[function(word) for word in line["words"]] for line in lines]
        alphabet = set(value for values in line_values for value in values)
        unknown = ("<unknown>", name)
        alphabet.add(unknown)
        vocabulary_size = len(alphabet)
        differences = []
        quire_differences = defaultdict(float)
        transitions = 0

        for fold in range(4):
            joint = Counter()
            context = Counter()
            train_ids = [
                index for index, line in enumerate(lines)
                if fold_of[line["quire"]] != fold
            ]
            test_ids = [
                index for index, line in enumerate(lines)
                if fold_of[line["quire"]] == fold
            ]
            for index in train_ids:
                values = line_values[index]
                for left, right in zip(values, values[1:]):
                    joint[(left, right)] += 1
                    context[left] += 1

            def line_score(values):
                score = 0.0
                for left, right in zip(values, values[1:]):
                    left = left if left in alphabet else unknown
                    right = right if right in alphabet else unknown
                    score += math.log2(
                        (joint[(left, right)] + alpha)
                        / (context[left] + alpha * vocabulary_size)
                    )
                return score

            for index in test_ids:
                actual = line_values[index]
                if len(actual) < 7:
                    continue
                reversed_interior = (
                    actual[:2] + list(reversed(actual[2:-2])) + actual[-2:]
                )
                difference = line_score(actual) - line_score(reversed_interior)
                differences.append(difference)
                quire_differences[lines[index]["quire"]] += difference
                transitions += len(actual) - 1

        gain = sum(differences) / max(1, transitions)
        positive = np.mean(np.asarray(differences) > 0)
        p_value = sign_flip_p(list(quire_differences.values()))
        raw_ps.append(p_value)
        print(
            f"  {name:18s} {gain:+16.5f} {positive:15.3f} "
            f"{p_value:13.4f}"
        )
    print(
        f"  Bonferroni minimum p: "
        f"{min(1.0, min(raw_ps) * len(raw_ps)):.4f}"
    )


def lcs_length(left, right):
    previous = [0] * (len(right) + 1)
    for a in left:
        current = [0]
        for index, b in enumerate(right, 1):
            current.append(
                previous[index - 1] + 1
                if a == b
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def parallel_search(lines):
    core_sequences = [
        [decompose(word)[1] for word in line["words"]]
        for line in lines
    ]
    document_frequency = Counter()
    for sequence in core_sequences:
        document_frequency.update(set(sequence))
    n_lines = len(lines)
    vectors = []
    norms = []
    for sequence in core_sequences:
        counts = Counter(sequence)
        vector = {
            core: count * math.log((n_lines + 1) / (document_frequency[core] + 1))
            for core, count in counts.items()
        }
        vectors.append(vector)
        norms.append(math.sqrt(sum(value * value for value in vector.values())))

    postings = defaultdict(list)
    for line_id, sequence in enumerate(core_sequences):
        for core in set(sequence):
            if 2 <= document_frequency[core] <= 120:
                postings[core].append(line_id)

    candidate_pairs = set()
    for ids in postings.values():
        for left, right in combinations(ids, 2):
            a, b = lines[left], lines[right]
            if (
                a["quire"] != b["quire"]
                and a["section"] == b["section"]
                and a["currier"] == b["currier"]
            ):
                candidate_pairs.add((left, right))

    scored = []
    for left, right in candidate_pairs:
        if not norms[left] or not norms[right]:
            continue
        shared = set(vectors[left]) & set(vectors[right])
        cosine = sum(
            vectors[left][core] * vectors[right][core]
            for core in shared
        ) / (norms[left] * norms[right])
        if cosine <= 0:
            continue
        lcs = lcs_length(core_sequences[left], core_sequences[right])
        order_ratio = lcs / min(
            len(core_sequences[left]), len(core_sequences[right])
        )
        scored.append((cosine, order_ratio, left, right, shared))
    scored.sort(reverse=True)

    print("\n" + "=" * 78)
    print("PARALLEL SEARCH: cross-quire rare-core line similarity")
    print("=" * 78)
    print(
        f"candidate pairs={len(candidate_pairs)}; ranking is exploratory and "
        "selection-adjusted significance is not claimed"
    )
    for rank, (cosine, order_ratio, left, right, shared) in enumerate(
        scored[:12], 1
    ):
        a, b = lines[left], lines[right]
        rare_shared = sorted(
            shared,
            key=lambda core: (document_frequency[core], core),
        )[:8]
        print(
            f"\n{rank:2d}. cosine={cosine:.3f} core-LCS={order_ratio:.2f} "
            f"shared={rare_shared}"
        )
        print(
            f"    {a['folio']}.{a['line']} Q={a['quire']} "
            f"I={a['section']} L={a['currier']}: {' '.join(a['words'])}"
        )
        print(
            f"    {b['folio']}.{b['line']} Q={b['quire']} "
            f"I={b['section']} L={b['currier']}: {' '.join(b['words'])}"
        )


if __name__ == "__main__":
    prose_lines = load_lines()
    sequence_battery(prose_lines)
    directionality_battery(prose_lines)
    parallel_search(prose_lines)
