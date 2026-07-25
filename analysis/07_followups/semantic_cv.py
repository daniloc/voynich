#!/usr/bin/env python3
"""
Block-held-out page-level illustration coupling.

Question
--------
Can the text on a folio predict its independently supplied IVTFF illustration
class on a production block (quire) that was not seen during training?

This is deliberately a coupling test, not a decoding claim. Illustration class
is strongly confounded with quire, Currier dialect, hand, and locus/layout. The
script therefore:

* holds out complete quires;
* excludes classes that occur in too few quires from the primary analysis;
* compares lexical/core features with majority, metadata-only, and layout-only
  baselines;
* repeats useful contrasts inside one Currier value;
* uses Currier-stratified and Currier+hand-stratified label permutations;
* reports a deliberately injected positive control;
* identifies core families with independent block and hand recurrence filters.

Only numpy and the checked-in data/corpus/body.json are required.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BODY = ROOT / "data" / "corpus" / "body.json"
SEED = 20260722
N_PERM = 300

PREFIXES = [
    "qok", "qot", "cth", "ckh", "cph", "cfh", "qo", "ok", "ot",
    "ch", "sh", "o", "y", "d",
]
SUFFIXES = [
    "eedy", "eody", "aiin", "aiir", "edy", "dam", "ain", "iin",
    "dy", "ol", "or", "ar", "al", "am", "eey", "ey", "y",
]


def decompose(word):
    prefix = next((p for p in PREFIXES if word.startswith(p)), "")
    remainder = word[len(prefix):]
    suffix = next(
        (
            s for s in SUFFIXES
            if remainder.endswith(s) and len(remainder) > len(s)
        ),
        "",
    )
    core = remainder[:-len(suffix)] if suffix else remainder
    return prefix, core, suffix


def load_pages():
    body = json.loads(BODY.read_text())
    pages = {}
    for folio, meta in body["pages"].items():
        pages[folio] = {
            "folio": folio,
            "label": meta["section"],
            "quire": meta["quire"],
            "currier": meta["currier"],
            "hand": meta["hand"],
            "tokens": [],
        }
    for token in body["tokens"]:
        if token.get("uncertain") or "?" in token["word"] or len(token["word"]) < 2:
            continue
        pages[token["folio"]]["tokens"].append(token)
    return [p for p in pages.values() if p["tokens"]]


def lexical_features(page, representation, roles=None):
    features = Counter()
    for token in page["tokens"]:
        if roles is not None and token["role"] not in roles:
            continue
        word = token["word"]
        prefix, core, suffix = decompose(word)
        if representation == "word":
            features["word=" + word] += 1
        elif representation == "core":
            if core:
                features["core=" + core] += 1
        elif representation == "morph":
            if prefix:
                features["prefix=" + prefix] += 1
            if core:
                features["core=" + core] += 1
            if suffix:
                features["suffix=" + suffix] += 1
        elif representation == "char3":
            padded = "^" + word + "$"
            for i in range(len(padded) - 2):
                features["char3=" + padded[i:i + 3]] += 1
        else:
            raise ValueError(representation)
    return features


def metadata_features(page):
    return Counter({
        "currier=" + page["currier"]: 1,
        "hand=" + page["hand"]: 1,
    })


def layout_features(page, roles=None):
    features = Counter()
    for token in page["tokens"]:
        if roles is not None and token["role"] not in roles:
            continue
        features["role=" + token["role"]] += 1
        features["position=" + token["linepos"]] += 1
        length = len(token["word"])
        features["length=" + str(min(length, 10))] += 1
    return features


def entropy(values):
    counts = Counter(values)
    n = sum(counts.values())
    return -sum((v / n) * math.log2(v / n) for v in counts.values())


def mutual_information(x, y):
    joint = Counter(zip(x, y))
    cx, cy = Counter(x), Counter(y)
    n = len(x)
    return sum(
        (count / n) * math.log2((count * n) / (cx[a] * cy[b]))
        for (a, b), count in joint.items()
    )


def fit_nb(feature_rows, labels, classes, alpha=0.5):
    """Uniform-prior multinomial Naive Bayes."""
    document_frequency = Counter()
    for row in feature_rows:
        document_frequency.update(row.keys())
    vocabulary = {
        feature for feature, count in document_frequency.items() if count >= 2
    }
    if not vocabulary:
        vocabulary = set(document_frequency)

    class_counts = {label: Counter() for label in classes}
    totals = Counter()
    for row, label in zip(feature_rows, labels):
        kept = Counter({k: v for k, v in row.items() if k in vocabulary})
        class_counts[label].update(kept)
        totals[label] += sum(kept.values())

    width = max(len(vocabulary), 1)
    log_prob = {}
    for label in classes:
        denominator = totals[label] + alpha * width
        log_prob[label] = {
            feature: math.log((class_counts[label][feature] + alpha) / denominator)
            for feature in vocabulary
        }
    return vocabulary, log_prob


def predict_nb(model, rows, classes):
    vocabulary, log_prob = model
    predictions = []
    for row in rows:
        scores = {}
        for label in classes:
            scores[label] = sum(
                value * log_prob[label][feature]
                for feature, value in row.items()
                if feature in vocabulary
            )
        predictions.append(max(classes, key=lambda label: (scores[label], label)))
    return predictions


def metrics(y_true, y_pred, classes):
    confusion = {label: Counter() for label in classes}
    for truth, prediction in zip(y_true, y_pred):
        confusion[truth][prediction] += 1
    recalls, f1s = [], []
    for label in classes:
        tp = confusion[label][label]
        actual = sum(confusion[label].values())
        predicted = sum(confusion[truth][label] for truth in classes)
        recall = tp / actual if actual else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        recalls.append(recall)
        f1s.append(f1)
    accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)
    return {
        "accuracy": accuracy,
        "balanced": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "recall": dict(zip(classes, recalls)),
        "confusion": confusion,
        "n": len(y_true),
    }


def leave_one_quire_out(pages, feature_rows, labels=None):
    y = labels if labels is not None else [p["label"] for p in pages]
    classes = sorted(set(y))
    predictions, truths, majority_predictions = [], [], []
    tested_folios = []
    for quire in sorted(set(p["quire"] for p in pages)):
        train = [i for i, p in enumerate(pages) if p["quire"] != quire]
        test = [i for i, p in enumerate(pages) if p["quire"] == quire]
        if not test:
            continue
        model = fit_nb(
            [feature_rows[i] for i in train],
            [y[i] for i in train],
            classes,
        )
        fold_predictions = predict_nb(
            model, [feature_rows[i] for i in test], classes
        )
        majority = Counter(y[i] for i in train).most_common(1)[0][0]
        predictions.extend(fold_predictions)
        truths.extend(y[i] for i in test)
        majority_predictions.extend([majority] * len(test))
        tested_folios.extend(pages[i]["folio"] for i in test)
    result = metrics(truths, predictions, classes)
    majority_result = metrics(truths, majority_predictions, classes)
    result["majority"] = majority_result
    result["folios"] = tested_folios
    return result


def stratified_folio_cv(pages, feature_rows, n_splits=5):
    rng = np.random.default_rng(SEED)
    y = [p["label"] for p in pages]
    classes = sorted(set(y))
    fold_for = {}
    for label in classes:
        indices = [i for i, value in enumerate(y) if value == label]
        rng.shuffle(indices)
        for offset, index in enumerate(indices):
            fold_for[index] = offset % n_splits
    truths, predictions = [], []
    for fold in range(n_splits):
        train = [i for i in range(len(pages)) if fold_for[i] != fold]
        test = [i for i in range(len(pages)) if fold_for[i] == fold]
        model = fit_nb(
            [feature_rows[i] for i in train],
            [y[i] for i in train],
            classes,
        )
        predictions.extend(predict_nb(model, [feature_rows[i] for i in test], classes))
        truths.extend(y[i] for i in test)
    return metrics(truths, predictions, classes)


def permute_within(pages, rng, fields):
    labels = [p["label"] for p in pages]
    permuted = labels[:]
    strata = defaultdict(list)
    for i, page in enumerate(pages):
        strata[tuple(page[field] for field in fields)].append(i)
    for indices in strata.values():
        shuffled = [labels[i] for i in indices]
        rng.shuffle(shuffled)
        for index, label in zip(indices, shuffled):
            permuted[index] = label
    return permuted


def permutation_baseline(pages, feature_rows, observed, fields):
    rng = np.random.default_rng(SEED + 1)
    null = []
    for _ in range(N_PERM):
        shuffled = permute_within(pages, rng, fields)
        null.append(
            leave_one_quire_out(pages, feature_rows, shuffled)["balanced"]
        )
    null = np.array(null)
    p_value = (np.sum(null >= observed) + 1) / (len(null) + 1)
    return float(null.mean()), float(null.std()), float(np.quantile(null, 0.95)), p_value


def eligible_labels(pages, min_quires, min_pages):
    by_label = defaultdict(list)
    for page in pages:
        by_label[page["label"]].append(page)
    return sorted(
        label for label, rows in by_label.items()
        if len(rows) >= min_pages
        and len(set(row["quire"] for row in rows)) >= min_quires
    )


def select_pages(pages, labels, currier=None, min_role_tokens=0, roles=None):
    selected = []
    for page in pages:
        if page["label"] not in labels:
            continue
        if currier is not None and page["currier"] != currier:
            continue
        if roles is not None:
            count = sum(token["role"] in roles for token in page["tokens"])
            if count < min_role_tokens:
                continue
        selected.append(page)
    return selected


def format_scores(name, result):
    recalls = " ".join(
        f"{label}:{value:.2f}" for label, value in result["recall"].items()
    )
    print(
        f"  {name:16s} acc={result['accuracy']:.3f} "
        f"bal={result['balanced']:.3f} macroF1={result['macro_f1']:.3f} "
        f"recall=[{recalls}]"
    )


def run_experiment(title, pages, roles=None, do_random=True, do_permutation=True):
    labels = sorted(set(p["label"] for p in pages))
    quires = sorted(set(p["quire"] for p in pages))
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    print(
        f"  pages={len(pages)} classes={labels} quires={quires} "
        f"counts={dict(Counter(p['label'] for p in pages))}"
    )
    print(
        "  Currier x class="
        + str(dict(Counter((p["currier"], p["label"]) for p in pages)))
    )

    rows_by_representation = {
        representation: [
            lexical_features(page, representation, roles) for page in pages
        ]
        for representation in ("word", "core", "morph", "char3")
    }
    rows_by_representation["metadata"] = [metadata_features(p) for p in pages]
    rows_by_representation["layout"] = [layout_features(p, roles) for p in pages]

    results = {}
    for representation in ("word", "core", "morph", "char3", "metadata", "layout"):
        result = leave_one_quire_out(
            pages, rows_by_representation[representation]
        )
        results[representation] = result
        format_scores(representation, result)
    format_scores("majority", results["core"]["majority"])

    if do_random:
        random_result = stratified_folio_cv(pages, rows_by_representation["core"])
        print(
            f"  {'core random-CV':16s} acc={random_result['accuracy']:.3f} "
            f"bal={random_result['balanced']:.3f} "
            f"macroF1={random_result['macro_f1']:.3f}"
        )

    if do_permutation:
        for fields, label in (
            (("currier",), "Currier"),
            (("currier", "hand"), "Currier+hand"),
        ):
            null_mean, null_sd, null_95, p_value = permutation_baseline(
                pages,
                rows_by_representation["core"],
                results["core"]["balanced"],
                fields,
            )
            print(
                f"  core {label}-stratified permutations (n={N_PERM}): "
                f"mean={null_mean:.3f} sd={null_sd:.3f} q95={null_95:.3f} "
                f"p={p_value:.4f}"
            )

    positive_rows = []
    for page, row in zip(pages, rows_by_representation["core"]):
        marked = row.copy()
        marked["INJECTED_CONTROL=" + page["label"]] += 100000
        positive_rows.append(marked)
    positive = leave_one_quire_out(pages, positive_rows)
    print(
        f"  injected-label positive control: bal={positive['balanced']:.3f} "
        f"(pipeline sensitivity only; not manuscript evidence)"
    )
    return results


def stable_pharma_cores(pages):
    """
    Require within-hand enrichment in mixed quire O and recurrence in quire S.

    Quire O contains herbal and pharmaceutical pages by both hand 4 and unknown
    hand, allowing two hand-matched contrasts. Quire S supplies an independent
    pharmaceutical block. Rates are selected on all three checks, so they are
    candidate anchors rather than confirmatory p-values.
    """
    relevant = [
        p for p in pages
        if p["currier"] == "A"
        and p["label"] in {"herbal", "pharmaceutical"}
    ]
    counts = defaultdict(Counter)
    totals = Counter()
    for page in relevant:
        if page["quire"] == "O":
            prefix = "pharma" if page["label"] == "pharmaceutical" else "herbal"
            bucket = f"{prefix}_O_hand_{page['hand']}"
        elif page["label"] == "pharmaceutical" and page["quire"] == "S":
            bucket = "pharma_S"
        else:
            bucket = "herbal_all"
        for token in page["tokens"]:
            if token["role"] != "paragraph":
                continue
            core = decompose(token["word"])[1]
            if len(core) < 2:
                continue
            counts[bucket][core] += 1
            totals[bucket] += 1

    rows = []
    buckets = (
        "pharma_O_hand_4",
        "herbal_O_hand_4",
        "pharma_O_hand_?",
        "herbal_O_hand_?",
        "pharma_S",
        "herbal_all",
    )
    for core in (
        set(counts["pharma_O_hand_4"])
        & set(counts["pharma_O_hand_?"])
        & set(counts["pharma_S"])
    ):
        if counts["pharma_S"][core] < 2:
            continue
        rates = {
            bucket: 1000 * counts[bucket][core] / totals[bucket]
            for bucket in buckets
        }
        if not (
            rates["pharma_O_hand_4"] > rates["herbal_O_hand_4"]
            and rates["pharma_O_hand_?"] > rates["herbal_O_hand_?"]
            and rates["pharma_S"] > rates["herbal_all"]
        ):
            continue
        score = min(
            math.log2(
                (rates["pharma_O_hand_4"] + 0.1)
                / (rates["herbal_O_hand_4"] + 0.1)
            ),
            math.log2(
                (rates["pharma_O_hand_?"] + 0.1)
                / (rates["herbal_O_hand_?"] + 0.1)
            ),
            math.log2(
                (rates["pharma_S"] + 0.1) / (rates["herbal_all"] + 0.1)
            ),
        )
        rows.append((score, core, rates))
    rows.sort(reverse=True)
    print("\n" + "=" * 88)
    print("HAND-MATCHED, CROSS-QUIRE PHARMACEUTICAL CORE CANDIDATES (paragraph only)")
    print("=" * 88)
    print(
        "  Required: pharma > herbal separately for hand 4 and unknown hand in "
        "mixed quire O, plus pharma-S > all Currier-A herbal."
    )
    print(
        "  core       O-P4/k O-H4/k O-P?/k O-H?/k   S-P/k all-H/k  min-log2"
    )
    for score, core, rates in rows[:12]:
        print(
            f"  {core:10s} "
            f"{rates['pharma_O_hand_4']:7.2f} "
            f"{rates['herbal_O_hand_4']:7.2f} "
            f"{rates['pharma_O_hand_?']:7.2f} "
            f"{rates['herbal_O_hand_?']:7.2f} "
            f"{rates['pharma_S']:7.2f} "
            f"{rates['herbal_all']:7.2f} "
            f"{score:9.2f}"
        )
    if not rows:
        print("  none met the cross-block recurrence and direction criteria")


def pharmaceutical_block_checks(pages):
    relevant = [
        p for p in pages
        if p["currier"] == "A"
        and p["label"] in {"herbal", "pharmaceutical"}
        and sum(t["role"] == "paragraph" for t in p["tokens"]) >= 20
    ]
    feature_rows = [
        lexical_features(p, "core", {"paragraph"}) for p in relevant
    ]
    labels = [p["label"] for p in relevant]
    classes = ["herbal", "pharmaceutical"]
    print("\n" + "=" * 88)
    print("PHARMACEUTICAL PRODUCTION-BLOCK CHECKS (Currier A, paragraph cores)")
    print("=" * 88)
    for test_quire in ("O", "S"):
        train = [i for i, p in enumerate(relevant) if p["quire"] != test_quire]
        test = [i for i, p in enumerate(relevant) if p["quire"] == test_quire]
        model = fit_nb(
            [feature_rows[i] for i in train],
            [labels[i] for i in train],
            classes,
        )
        predictions = predict_nb(model, [feature_rows[i] for i in test], classes)
        truth_counts = Counter(labels[i] for i in test)
        correct = Counter()
        predicted = Counter()
        for i, prediction in zip(test, predictions):
            predicted[prediction] += 1
            if labels[i] == prediction:
                correct[labels[i]] += 1
        if len(truth_counts) == 2:
            balanced = np.mean(
                [
                    correct[label] / truth_counts[label]
                    for label in classes
                ]
            )
            balanced_text = f" balanced={balanced:.3f}"
        else:
            balanced_text = ""
        print(
            f"  hold out quire {test_quire}: truth={dict(truth_counts)} "
            f"predicted={dict(predicted)} correct-by-class={dict(correct)}"
            f"{balanced_text}"
        )
    print(
        "  Quire S contains no herbal pages, so its row checks transfer of the "
        "pharmaceutical signature only; quire O supplies the actual two-class test."
    )


def print_confounding(pages):
    labels = [p["label"] for p in pages]
    h_label = entropy(labels)
    print("=" * 88)
    print("IDENTIFIABILITY AND CONFOUNDING")
    print("=" * 88)
    print(
        f"  all pages={len(pages)} H(section)={h_label:.3f} bits; "
        "normalized MI is I(metadata;section)/H(section)"
    )
    for field in ("currier", "hand", "quire"):
        values = [p[field] for p in pages]
        mi = mutual_information(values, labels)
        print(
            f"  {field:8s}: I={mi:.3f} bits, normalized={mi / h_label:.3f}, "
            f"H(section|{field})={h_label - mi:.3f}"
        )
    groups = defaultdict(list)
    for page in pages:
        groups[page["quire"]].append(page["label"])
    pure = sum(len(set(values)) == 1 for values in groups.values())
    print(
        f"  quire purity: {pure}/{len(groups)} quires contain only one "
        "illustration class"
    )
    print("  classes by independent quire count:")
    for label in sorted(set(labels)):
        rows = [p for p in pages if p["label"] == label]
        print(
            f"    {label:16s} pages={len(rows):3d} "
            f"quires={sorted(set(p['quire'] for p in rows))} "
            f"Currier={dict(Counter(p['currier'] for p in rows))} "
            f"hands={dict(Counter(p['hand'] for p in rows))}"
        )


def main():
    pages = load_pages()
    print_confounding(pages)

    primary_labels = eligible_labels(pages, min_quires=3, min_pages=8)
    primary = select_pages(pages, primary_labels)
    run_experiment(
        "PRIMARY: classes represented on >=3 quires; all textual loci",
        primary,
    )

    expanded_labels = eligible_labels(pages, min_quires=2, min_pages=5)
    expanded = select_pages(pages, expanded_labels)
    run_experiment(
        "EXPLORATORY EXPANDED: classes represented on >=2 quires; all textual loci",
        expanded,
    )

    unknown_cz = select_pages(
        pages, {"cosmological", "zodiac"}, currier="?"
    )
    run_experiment(
        "WITHIN CURRIER ?: cosmological vs zodiac; complete-quire holdout",
        unknown_cz,
    )

    currier_a_hp = select_pages(
        pages, {"herbal", "pharmaceutical"}, currier="A"
    )
    run_experiment(
        "WITHIN CURRIER A: herbal vs pharmaceutical; all textual loci",
        currier_a_hp,
    )

    currier_a_hp_prose = select_pages(
        pages,
        {"herbal", "pharmaceutical"},
        currier="A",
        min_role_tokens=20,
        roles={"paragraph"},
    )
    run_experiment(
        "WITHIN CURRIER A: herbal vs pharmaceutical; paragraph prose only",
        currier_a_hp_prose,
        roles={"paragraph"},
    )

    pharmaceutical_block_checks(pages)
    stable_pharma_cores(pages)

    print("\n" + "=" * 88)
    print("INTERPRETATION GUARD")
    print("=" * 88)
    print(
        "  Above-chance held-out prediction establishes cross-quire coupling only. "
        "It can arise from subject vocabulary, scribal convention, page layout, "
        "or a domain-conditioned content-free generator. It does not establish "
        "translation, reference, or propositional meaning."
    )


if __name__ == "__main__":
    main()
