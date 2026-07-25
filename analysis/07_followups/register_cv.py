"""
Held-out label/prose register test.

This is a follow-up to analysis/01_structure/register.py.  That script found that
label and paragraph tokens often share decomposed cores and that a label folio's
cores overlap its own prose more than a few randomly selected pages.  The original
comparison did not hold section, Currier, hand, or production block fixed.

This script asks three narrower questions:

1. LOCALITY: do label cores overlap prose on the same folio more than prose from
   another folio in the same section x Currier x hand x quire stratum?
2. HELD-OUT CORE PREDICTION: in leave-one-quire-out evaluation, does the presence
   of a core in a test folio's prose improve prediction of its label-core types
   over a train-only label-frequency prior?
3. CANONICAL FORM: conditional on a core occurring in both registers, can the
   canonical prose affix pair predict the canonical label affix pair (and vice
   versa) on a held-out folio or quire?

All label outcomes from a test folio/quire are excluded from fitting.  Permutation
nulls preserve complete folio prose profiles and metadata/production strata for
the locality and core-ranking tests.  The form null preserves each folio's exact
output-affix frequencies.  Synthetic positive controls exercise the same held-out
pipelines.  None of these tests identifies semantics: a positive result is also
compatible with local copying or a shared positional generator.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/corpus/body.json"
SEED = 20260722
N_PERM = 5000
N_FORM_PERM = 500
MIN_LABEL_TOKENS = 4
MIN_PROSE_TOKENS = 30

PREFIXES = (
    "qok", "qot", "qo", "ok", "ot", "o", "y", "ch", "sh", "d",
    "cth", "ckh", "cph", "cfh",
)
SUFFIXES = (
    "eedy", "eody", "edy", "aiin", "aiir", "ain", "iin", "dy", "ol",
    "or", "ar", "al", "am", "dam", "ey", "eey", "y",
)
PREFIX_DOMAIN = ("",) + PREFIXES
SUFFIX_DOMAIN = ("",) + SUFFIXES
PAIR_DOMAIN = tuple((p, s) for p in PREFIX_DOMAIN for s in SUFFIX_DOMAIN)


def decomp(word: str) -> tuple[str, str, str]:
    prefix = next(
        (p for p in sorted(PREFIXES, key=len, reverse=True) if word.startswith(p)),
        "",
    )
    rem = word[len(prefix):]
    suffix = next(
        (
            s
            for s in sorted(SUFFIXES, key=len, reverse=True)
            if rem.endswith(s) and len(rem) > len(s)
        ),
        "",
    )
    core = rem[:-len(suffix)] if suffix else rem
    return prefix, core, suffix


def valid_token(token: dict) -> bool:
    word = token["word"]
    return not token["uncertain"] and "?" not in word and len(word) >= 2


def stable_key(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def canonical(counter: Counter) -> tuple[str, str, str]:
    """Most frequent decomposition, with a deterministic lexical tie break."""
    return min(counter, key=lambda item: (-counter[item], item))


def average_precision(scores: dict[str, float], targets: set[str]) -> float:
    if not targets:
        return float("nan")
    ranked = sorted(scores, key=lambda core: (-scores[core], core))
    hits = 0
    total = 0.0
    for rank, core in enumerate(ranked, 1):
        if core in targets:
            hits += 1
            total += hits / rank
    return total / len(targets)


def recall_at_k(scores: dict[str, float], targets: set[str]) -> float:
    if not targets:
        return float("nan")
    ranked = sorted(scores, key=lambda core: (-scores[core], core))
    predicted = set(ranked[:len(targets)])
    return len(predicted & targets) / len(targets)


def upper_p(null: np.ndarray, observed: float) -> float:
    return (np.count_nonzero(null >= observed) + 1) / (len(null) + 1)


@dataclass(frozen=True)
class Page:
    folio: str
    section: str
    currier: str
    hand: str
    quire: str
    prose: Counter
    labels: Counter

    @property
    def meta_stratum(self) -> tuple[str, str, str]:
        return self.section, self.currier, self.hand

    @property
    def block_stratum(self) -> tuple[str, str, str, str]:
        return self.section, self.currier, self.hand, self.quire

    @property
    def prose_cores(self) -> set[str]:
        return {core for _, core, _ in self.prose}

    @property
    def label_core_counts(self) -> Counter:
        out = Counter()
        for (_, core, _), count in self.labels.items():
            out[core] += count
        return out


def load_pages() -> tuple[list[Page], set[str]]:
    data = json.loads(CORPUS.read_text())
    prose = defaultdict(Counter)
    labels = defaultdict(Counter)
    all_cores = set()
    for token in data["tokens"]:
        if not valid_token(token):
            continue
        parts = decomp(token["word"])
        if token["role"] == "paragraph":
            all_cores.add(parts[1])
            prose[token["folio"]][parts] += 1
        elif token["role"] == "label":
            all_cores.add(parts[1])
            labels[token["folio"]][parts] += 1

    pages = []
    for folio in sorted(set(prose) & set(labels)):
        if sum(labels[folio].values()) < MIN_LABEL_TOKENS:
            continue
        if sum(prose[folio].values()) < MIN_PROSE_TOKENS:
            continue
        meta = data["pages"][folio]
        pages.append(
            Page(
                folio=folio,
                section=meta["section"],
                currier=meta["currier"],
                hand=meta["hand"],
                quire=meta["quire"],
                prose=prose[folio],
                labels=labels[folio],
            )
        )
    return pages, all_cores


def grouped(pages: Sequence[Page], key: Callable[[Page], tuple]) -> dict[tuple, list[Page]]:
    out = defaultdict(list)
    for page in pages:
        out[key(page)].append(page)
    return out


def permutation_assignments(
    pages: Sequence[Page],
    key: Callable[[Page], tuple],
    rng: np.random.Generator,
) -> dict[str, str]:
    assignment = {}
    for members in grouped(pages, key).values():
        donors = [page.folio for page in members]
        shuffled = list(rng.permutation(donors))
        assignment.update(
            {page.folio: donor for page, donor in zip(members, shuffled)}
        )
    return assignment


def eligible_for_permutation(
    pages: Sequence[Page], key: Callable[[Page], tuple]
) -> list[Page]:
    groups = grouped(pages, key)
    return [page for page in pages if len(groups[key(page)]) >= 2]


def locality_test(pages: Sequence[Page], rng: np.random.Generator) -> None:
    by_folio = {page.folio: page for page in pages}
    movable = eligible_for_permutation(pages, lambda page: page.block_stratum)

    def page_coverage(label_page: Page, prose_page: Page) -> tuple[float, float]:
        label_counts = label_page.label_core_counts
        prose_cores = prose_page.prose_cores
        type_cov = len(set(label_counts) & prose_cores) / len(label_counts)
        token_cov = (
            sum(count for core, count in label_counts.items() if core in prose_cores)
            / sum(label_counts.values())
        )
        return type_cov, token_cov

    observed = np.array([page_coverage(page, page) for page in movable])
    null_type = np.empty(N_PERM)
    null_token = np.empty(N_PERM)
    for perm_ix in range(N_PERM):
        assignment = permutation_assignments(
            movable, lambda page: page.block_stratum, rng
        )
        values = np.array(
            [
                page_coverage(page, by_folio[assignment[page.folio]])
                for page in movable
            ]
        )
        null_type[perm_ix] = values[:, 0].mean()
        null_token[perm_ix] = values[:, 1].mean()

    print("\n" + "=" * 78)
    print("A. BLOCK-MATCHED LOCALITY")
    print("=" * 78)
    print(
        f"Eligible pages: {len(pages)}; movable within "
        f"section x Currier x hand x quire: {len(movable)}"
    )
    print(
        "Complete prose folios are permuted; label/prose token counts and all "
        "block metadata remain fixed."
    )
    for name, column, null in (
        ("label-core TYPE coverage", 0, null_type),
        ("label-core TOKEN coverage", 1, null_token),
    ):
        obs = observed[:, column].mean()
        print(
            f"  {name:27s}: own={obs:.4f}  perm={null.mean():.4f} "
            f"[95% {np.quantile(null, .025):.4f}, {np.quantile(null, .975):.4f}] "
            f"delta={obs-null.mean():+.4f}  p={upper_p(null, obs):.4f}"
        )


def synthetic_targets(page: Page) -> set[str]:
    """Known local-core signal with the same target-set size as the real labels."""
    k = min(len(page.label_core_counts), len(page.prose_cores))
    ordered = sorted(
        page.prose_cores, key=lambda core: (stable_key(page.folio, core), core)
    )
    return set(ordered[:k])


def target_sets(pages: Sequence[Page], synthetic: bool) -> dict[str, set[str]]:
    if synthetic:
        return {page.folio: synthetic_targets(page) for page in pages}
    return {page.folio: set(page.label_core_counts) for page in pages}


def core_model_scores(
    test_page: Page,
    donor_page: Page,
    train_pages: Sequence[Page],
    targets: dict[str, set[str]],
    vocabulary: set[str],
) -> tuple[dict[str, float], dict[str, float], float]:
    """
    Train-only naive-Bayes page-incidence model.

    The baseline is a hierarchical label-core page-frequency prior.  The linked
    model adds one learned likelihood ratio: whether a candidate core is present
    in the target (or permuted donor) prose profile.  It has no core-specific
    page-local parameter.
    """
    global_target = Counter()
    stratum_target = defaultdict(Counter)
    n_stratum = Counter()
    n11 = n10 = n01 = n00 = 0
    for page in train_pages:
        ys = targets[page.folio]
        xs = page.prose_cores
        global_target.update(ys)
        stratum_target[page.meta_stratum].update(ys)
        n_stratum[page.meta_stratum] += 1
        both = len(ys & xs)
        n11 += both
        n10 += len(ys) - both
        n01 += len(xs - ys)
        n00 += len(vocabulary) - len(ys | xs)

    n_train = len(train_pages)
    tau = 5.0
    px_y1 = (n11 + 0.5) / (n11 + n10 + 1.0)
    px_y0 = (n01 + 0.5) / (n01 + n00 + 1.0)
    lr_present = math.log((px_y1 + 1e-15) / (px_y0 + 1e-15))
    lr_absent = math.log((1.0 - px_y1 + 1e-15) / (1.0 - px_y0 + 1e-15))

    prior_scores = {}
    linked_scores = {}
    local = donor_page.prose_cores
    stratum = test_page.meta_stratum
    ns = n_stratum[stratum]
    for core in vocabulary:
        global_prob = (global_target[core] + 0.5) / (n_train + 1.0)
        prob = (
            stratum_target[stratum][core] + tau * global_prob
        ) / (ns + tau)
        prob = min(max(prob, 1e-12), 1.0 - 1e-12)
        prior = math.log(prob / (1.0 - prob))
        prior_scores[core] = prior
        linked_scores[core] = prior + (
            lr_present if core in local else lr_absent
        )
    return prior_scores, linked_scores, lr_present - lr_absent


def core_prediction_test(
    pages: Sequence[Page],
    vocabulary: set[str],
    rng: np.random.Generator,
    synthetic: bool = False,
) -> None:
    targets = target_sets(pages, synthetic)
    by_folio = {page.folio: page for page in pages}
    movable = eligible_for_permutation(pages, lambda page: page.block_stratum)

    prior_ap = {}
    own_ap = {}
    prior_rk = {}
    own_rk = {}
    link_weight = {}
    ap_matrix = {}
    rk_matrix = {}

    for test_page in pages:
        # A quire is treated as the production block and held out in its entirety.
        train = [page for page in pages if page.quire != test_page.quire]
        donors = grouped(pages, lambda page: page.block_stratum)[
            test_page.block_stratum
        ]
        for donor in donors:
            prior, linked, weight = core_model_scores(
                test_page, donor, train, targets, vocabulary
            )
            truth = targets[test_page.folio]
            ap_matrix[(test_page.folio, donor.folio)] = average_precision(
                linked, truth
            )
            rk_matrix[(test_page.folio, donor.folio)] = recall_at_k(linked, truth)
            if donor.folio == test_page.folio:
                prior_ap[test_page.folio] = average_precision(prior, truth)
                own_ap[test_page.folio] = ap_matrix[(test_page.folio, donor.folio)]
                prior_rk[test_page.folio] = recall_at_k(prior, truth)
                own_rk[test_page.folio] = rk_matrix[(test_page.folio, donor.folio)]
                link_weight[test_page.folio] = weight

    # The inferential subset excludes strata with no possible reassignment.
    obs_ap = np.mean([own_ap[page.folio] for page in movable])
    obs_rk = np.mean([own_rk[page.folio] for page in movable])
    base_ap = np.mean([prior_ap[page.folio] for page in movable])
    base_rk = np.mean([prior_rk[page.folio] for page in movable])
    null_ap = np.empty(N_PERM)
    null_rk = np.empty(N_PERM)
    for perm_ix in range(N_PERM):
        assignment = permutation_assignments(
            movable, lambda page: page.block_stratum, rng
        )
        null_ap[perm_ix] = np.mean(
            [ap_matrix[(page.folio, assignment[page.folio])] for page in movable]
        )
        null_rk[perm_ix] = np.mean(
            [rk_matrix[(page.folio, assignment[page.folio])] for page in movable]
        )

    title = (
        "C. SYNTHETIC LOCAL-CORE POSITIVE CONTROL"
        if synthetic
        else "B. LEAVE-ONE-QUIRE-OUT LABEL-CORE PREDICTION"
    )
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(
        f"Target pages: {len(pages)} ({len(movable)} movable in the "
        "block-matched null); candidate cores: "
        f"{len(vocabulary)}"
    )
    if synthetic:
        exact_size = sum(
            len(targets[page.folio]) == len(page.label_core_counts)
            for page in pages
        )
        print(
            "Synthetic target sets are deterministic samples of each page's prose "
            f"cores; {exact_size}/{len(pages)} match its real number of distinct "
            "label cores (the rest are capped by prose-core count)."
        )
    print(
        f"  learned local-presence log-odds boost: "
        f"median={np.median(list(link_weight.values())):.3f}"
    )
    print(
        f"  macro average precision: frequency={base_ap:.4f}  "
        f"own-prose={obs_ap:.4f}  delta={obs_ap-base_ap:+.4f}"
    )
    print(
        f"                           perm-prose={null_ap.mean():.4f} "
        f"[95% {np.quantile(null_ap, .025):.4f}, "
        f"{np.quantile(null_ap, .975):.4f}]  "
        f"own-vs-perm p={upper_p(null_ap, obs_ap):.4f}"
    )
    print(
        f"  macro recall at label-K: frequency={base_rk:.4f}  "
        f"own-prose={obs_rk:.4f}  delta={obs_rk-base_rk:+.4f}"
    )
    print(
        f"                           perm-prose={null_rk.mean():.4f} "
        f"[95% {np.quantile(null_rk, .025):.4f}, "
        f"{np.quantile(null_rk, .975):.4f}]  "
        f"own-vs-perm p={upper_p(null_rk, obs_rk):.4f}"
    )


@dataclass(frozen=True)
class FormRow:
    folio: str
    section: str
    currier: str
    hand: str
    quire: str
    core: str
    prose_pair: tuple[str, str]
    label_pair: tuple[str, str]

    @property
    def meta_stratum(self) -> tuple[str, str, str]:
        return self.section, self.currier, self.hand

    @property
    def block_stratum(self) -> tuple[str, str, str, str]:
        return self.section, self.currier, self.hand, self.quire


def form_rows(pages: Sequence[Page]) -> list[FormRow]:
    rows = []
    for page in pages:
        by_prose = defaultdict(Counter)
        by_label = defaultdict(Counter)
        for parts, count in page.prose.items():
            by_prose[parts[1]][parts] += count
        for parts, count in page.labels.items():
            by_label[parts[1]][parts] += count
        for core in sorted(set(by_prose) & set(by_label)):
            prose = canonical(by_prose[core])
            label = canonical(by_label[core])
            rows.append(
                FormRow(
                    folio=page.folio,
                    section=page.section,
                    currier=page.currier,
                    hand=page.hand,
                    quire=page.quire,
                    core=core,
                    prose_pair=(prose[0], prose[2]),
                    label_pair=(label[0], label[2]),
                )
            )
    return rows


def outcome_domain(field: str) -> Sequence:
    if field == "pair":
        return PAIR_DOMAIN
    if field == "prefix":
        return PREFIX_DOMAIN
    if field == "suffix":
        return SUFFIX_DOMAIN
    raise ValueError(field)


def select_field(pair: tuple[str, str], field: str):
    if field == "pair":
        return pair
    if field == "prefix":
        return pair[0]
    if field == "suffix":
        return pair[1]
    raise ValueError(field)


def form_probabilities(
    train: Sequence[FormRow],
    test: FormRow,
    x_attr: str,
    y_attr: str,
    field: str,
) -> tuple[dict, dict]:
    """
    Dirichlet-smoothed hierarchical categorical models.

    Frequency model: P(y | section, Currier, hand).
    Operator model: additionally P(y | source affix, stratum), backed off through
    the global source-affix relation and the frequency model.
    """
    domain = outcome_domain(field)
    global_y = Counter()
    stratum_y = defaultdict(Counter)
    x_y = defaultdict(Counter)
    stratum_x_y = defaultdict(Counter)
    for row in train:
        x = getattr(row, x_attr)
        y = select_field(getattr(row, y_attr), field)
        global_y[y] += 1
        stratum_y[row.meta_stratum][y] += 1
        x_y[x][y] += 1
        stratum_x_y[(row.meta_stratum, x)][y] += 1

    alpha = 0.1
    tau_stratum = 10.0
    tau_relation = 5.0
    total = sum(global_y.values())
    stratum = test.meta_stratum
    x = getattr(test, x_attr)
    ns = sum(stratum_y[stratum].values())
    nx = sum(x_y[x].values())
    nsx = sum(stratum_x_y[(stratum, x)].values())
    freq = {}
    operator = {}
    for y in domain:
        pg = (global_y[y] + alpha) / (total + alpha * len(domain))
        ps = (stratum_y[stratum][y] + tau_stratum * pg) / (
            ns + tau_stratum
        )
        px = (x_y[x][y] + tau_relation * pg) / (nx + tau_relation)
        relation_weight = nx / (nx + tau_relation)
        backoff = relation_weight * px + (1.0 - relation_weight) * ps
        psx = (
            stratum_x_y[(stratum, x)][y] + tau_relation * backoff
        ) / (nsx + tau_relation)
        freq[y] = ps
        operator[y] = psx
    return freq, operator


def top_hit(scores: dict, truth, k: int) -> bool:
    ranked = sorted(scores, key=lambda outcome: (-scores[outcome], outcome))
    return truth in ranked[:k]


def heldout_form_metrics(
    rows: Sequence[FormRow],
    group_attr: str,
    direction: str,
) -> dict[str, dict[str, float]]:
    if direction == "forward":
        x_attr, y_attr = "prose_pair", "label_pair"
    elif direction == "reverse":
        x_attr, y_attr = "label_pair", "prose_pair"
    else:
        raise ValueError(direction)

    hits = defaultdict(Counter)
    for test in rows:
        test_group = getattr(test, group_attr)
        train = [row for row in rows if getattr(row, group_attr) != test_group]
        for field in ("pair", "prefix", "suffix"):
            truth = select_field(getattr(test, y_attr), field)
            freq, operator = form_probabilities(
                train, test, x_attr, y_attr, field
            )
            for model, scores in (("freq", freq), ("operator", operator)):
                hits[(field, model)]["top1"] += top_hit(scores, truth, 1)
                hits[(field, model)]["top3"] += top_hit(scores, truth, 3)

    n = len(rows)
    return {
        f"{field}_{model}": {
            metric: count / n for metric, count in values.items()
        }
        for (field, model), values in hits.items()
    }


def print_form_metrics(
    metrics: dict[str, dict[str, float]], label: str
) -> None:
    print(f"  {label}")
    print(
        f"    {'field':8s} {'freq@1':>8s} {'operator@1':>11s} "
        f"{'delta':>8s} {'freq@3':>8s} {'operator@3':>11s}"
    )
    for field in ("pair", "prefix", "suffix"):
        freq = metrics[f"{field}_freq"]
        operator = metrics[f"{field}_operator"]
        print(
            f"    {field:8s} {freq['top1']:8.3f} {operator['top1']:11.3f} "
            f"{operator['top1']-freq['top1']:+8.3f} "
            f"{freq['top3']:8.3f} {operator['top3']:11.3f}"
        )


def print_prefix_breakdown(rows: Sequence[FormRow]) -> None:
    """Show whether the forward prefix result is confined to one test quire."""
    by_quire = defaultdict(Counter)
    for test in rows:
        train = [row for row in rows if row.quire != test.quire]
        truth = test.label_pair[0]
        freq, operator = form_probabilities(
            train, test, "prose_pair", "label_pair", "prefix"
        )
        freq_pred = min(freq, key=lambda value: (-freq[value], value))
        operator_pred = min(operator, key=lambda value: (-operator[value], value))
        by_quire[test.quire]["n"] += 1
        by_quire[test.quire]["freq"] += freq_pred == truth
        by_quire[test.quire]["operator"] += operator_pred == truth
    print("  forward prefix @1 by held-out quire:")
    print(f"    {'quire':5s} {'n':>4s} {'freq':>7s} {'operator':>9s} {'delta':>8s}")
    for quire, counts in sorted(by_quire.items()):
        n = counts["n"]
        freq = counts["freq"] / n
        operator = counts["operator"] / n
        print(
            f"    {quire:5s} {n:4d} {freq:7.3f} {operator:9.3f} "
            f"{operator-freq:+8.3f}"
        )


def permute_form_outputs(
    rows: Sequence[FormRow], rng: np.random.Generator
) -> list[FormRow]:
    out = []
    groups = defaultdict(list)
    for ix, row in enumerate(rows):
        # This is stricter than a metadata-block shuffle: keep every folio's exact
        # label-affix histogram fixed and break only its pairing to prose affixes.
        groups[row.folio].append(ix)
    permuted_labels = {}
    for indices in groups.values():
        values = [rows[ix].label_pair for ix in indices]
        values = list(rng.permutation(np.array(values, dtype=object)))
        for ix, value in zip(indices, values):
            permuted_labels[ix] = tuple(value)
    for ix, row in enumerate(rows):
        out.append(replace(row, label_pair=permuted_labels[ix]))
    return out


def rotate_pair(pair: tuple[str, str]) -> tuple[str, str]:
    prefix = PREFIX_DOMAIN[
        (PREFIX_DOMAIN.index(pair[0]) + 1) % len(PREFIX_DOMAIN)
    ]
    suffix = SUFFIX_DOMAIN[
        (SUFFIX_DOMAIN.index(pair[1]) + 1) % len(SUFFIX_DOMAIN)
    ]
    return prefix, suffix


def canonical_form_test(
    pages: Sequence[Page], rng: np.random.Generator
) -> None:
    rows = form_rows(pages)
    folios = len({row.folio for row in rows})
    print("\n" + "=" * 78)
    print("D. HELD-OUT CANONICAL REGISTER-FORM PREDICTION")
    print("=" * 78)
    print(
        f"Rows: {len(rows)} folio-core pairs on {folios} folios. Each row is "
        "conditional on that core occurring in both prose and labels."
    )
    print(
        "Forward: prose affix -> label affix. With the shared core supplied, an "
        "exact affix-pair hit is also exact canonical-word reconstruction."
    )
    for group in ("folio", "quire"):
        metrics = heldout_form_metrics(rows, group, "forward")
        print_form_metrics(metrics, f"leave-one-{group}-out")
    print_prefix_breakdown(rows)
    print("Reverse (invertibility check): label affix -> prose affix.")
    reverse_quire = heldout_form_metrics(rows, "quire", "reverse")
    print_form_metrics(reverse_quire, "leave-one-quire-out")

    observed_forward = heldout_form_metrics(rows, "quire", "forward")
    fields = ("pair", "prefix", "suffix")
    obs_delta = {
        field: (
            observed_forward[f"{field}_operator"]["top1"]
            - observed_forward[f"{field}_freq"]["top1"]
        )
        for field in fields
    }
    null = {field: np.empty(N_FORM_PERM) for field in fields}
    for ix in range(N_FORM_PERM):
        permuted = permute_form_outputs(rows, rng)
        metrics = heldout_form_metrics(permuted, "quire", "forward")
        for field in fields:
            null[field][ix] = (
                metrics[f"{field}_operator"]["top1"]
                - metrics[f"{field}_freq"]["top1"]
            )
    print("  forward @1 advantage vs within-folio output shuffle:")
    for field in fields:
        values = null[field]
        print(
            f"    {field:8s} observed={obs_delta[field]:+.3f}, "
            f"null={values.mean():+.3f} "
            f"[95% {np.quantile(values, .025):+.3f}, "
            f"{np.quantile(values, .975):+.3f}], "
            f"raw p={upper_p(values, obs_delta[field]):.4f}"
        )
    null_max = np.max(np.column_stack([null[field] for field in fields]), axis=1)
    best_field = max(fields, key=lambda field: obs_delta[field])
    print(
        f"    max-field correction: best={best_field} "
        f"delta={obs_delta[best_field]:+.3f}, "
        f"family-wise p={upper_p(null_max, obs_delta[best_field]):.4f}"
    )

    synthetic = [replace(row, label_pair=rotate_pair(row.prose_pair)) for row in rows]
    print("Synthetic positive control: a known bijective rotation of both affixes.")
    positive_forward = heldout_form_metrics(synthetic, "quire", "forward")
    positive_reverse = heldout_form_metrics(synthetic, "quire", "reverse")
    print_form_metrics(positive_forward, "forward leave-one-quire-out")
    print_form_metrics(positive_reverse, "reverse leave-one-quire-out")


def print_inventory(pages: Sequence[Page], vocabulary: set[str]) -> None:
    strata = grouped(pages, lambda page: page.block_stratum)
    print("=" * 78)
    print("HELD-OUT LABEL/PROSE REGISTER FOLLOW-UP")
    print("=" * 78)
    print(f"Corpus: {CORPUS.relative_to(ROOT)}")
    print(
        f"Eligibility: >= {MIN_LABEL_TOKENS} clean label tokens and >= "
        f"{MIN_PROSE_TOKENS} clean paragraph tokens on a folio."
    )
    print(
        f"Pages={len(pages)}, label tokens="
        f"{sum(sum(page.labels.values()) for page in pages)}, "
        f"prose tokens={sum(sum(page.prose.values()) for page in pages)}, "
        f"fixed candidate core vocabulary={len(vocabulary)}"
    )
    print("Permutation/production strata (section, Currier, hand, quire):")
    for stratum, members in sorted(strata.items()):
        print(f"  {stratum}: {len(members)} folio(s)")


def main() -> None:
    pages, vocabulary = load_pages()
    if len(pages) < 10:
        raise RuntimeError("Too few eligible label+prose folios")
    print_inventory(pages, vocabulary)
    locality_test(pages, np.random.default_rng(SEED))
    core_prediction_test(
        pages, vocabulary, np.random.default_rng(SEED + 1), synthetic=False
    )
    core_prediction_test(
        pages, vocabulary, np.random.default_rng(SEED + 2), synthetic=True
    )
    canonical_form_test(pages, np.random.default_rng(SEED + 3))
    print("\n" + "=" * 78)
    print("INTERPRETATION LIMIT")
    print("=" * 78)
    print(
        "A positive locality/core result supports page-local register coupling, "
        "not semantics: copying and a local shared generator predict it too. Form "
        "prediction is tested only after selecting cores present in both registers. "
        "A null form result therefore does not negate core sharing; a positive form "
        "result would establish a reproducible orthographic transform only."
    )


if __name__ == "__main__":
    main()
