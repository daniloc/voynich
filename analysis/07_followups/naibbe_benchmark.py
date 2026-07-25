#!/usr/bin/env python3
"""
Benchmark a known-meaningful Naibbe ciphertext against the repository's diagnostics.

The Naibbe material is not vendored. Pass a checkout of Michael Greshko's public
reference implementation:

    python3 analysis/07_followups/naibbe_benchmark.py \
        --naibbe-dir /path/to/naibbe-cipher

Reference implementation and data:
  https://github.com/greshko/naibbe-cipher
  https://doi.org/10.1080/01611194.2025.2566408

The primary comparison is token-count and line-template matched. "LATIN_PLAIN"
uses the larger Pliny Natural History corpus bundled with the Naibbe repository;
the bundled ciphertext was generated from Book 16 of that work. The plaintext
and ciphertext are therefore from the same work, but are not token-aligned:
Naibbe turns one- and two-letter plaintext units into ciphertext words.

This script deliberately reproduces several in-repository diagnostics, including
their limitations. In particular, the slot decomposition uses EVA-specific,
hand-selected affixes, and the optional cipher attack only fits a bijective
character substitution. Neither is a general-purpose language/cipher test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


SEED = 20260722
CHECKPOINTS = (0, 10, 25, 50, 100)
NAIBBE_GITHUB = "https://github.com/greshko/naibbe-cipher"
NAIBBE_DOI = "https://doi.org/10.1080/01611194.2025.2566408"

PRE = (
    "qok", "qot", "cth", "ckh", "cph", "cfh", "qo", "ok", "ot",
    "ch", "sh", "o", "y", "d",
)
SUF = (
    "eedy", "eody", "aiin", "aiir", "edy", "dam", "eey", "iin",
    "ain", "dy", "ol", "or", "ar", "al", "am", "ey", "y",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_naibbe = os.environ.get("NAIBBE_REPO", "/tmp/naibbe-cipher")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--naibbe-dir",
        type=Path,
        default=Path(default_naibbe),
        help="checkout of greshko/naibbe-cipher (or set NAIBBE_REPO)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data/intermediate/followups_naibbe.txt",
        help="captured text report",
    )
    parser.add_argument(
        "--skip-attack",
        action="store_true",
        help="skip the slower bijective-character substitution attack",
    )
    parser.add_argument("--attack-steps", type=int, default=2500)
    parser.add_argument("--attack-restarts", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def locus_type(locus: str) -> str:
    match = re.search(r"[A-Za-z]", locus)
    return match.group(0).upper() if match else "?"


def folio_key(folio: str) -> tuple[int, int, int]:
    match = re.match(r"f(\d+)([rv])(\d*)", folio)
    if not match:
        return (10**9, 0, 0)
    return (
        int(match.group(1)),
        0 if match.group(2) == "r" else 1,
        int(match.group(3) or 0),
    )


def load_voynich_lines(repo_root: Path) -> list[list[str]]:
    corpus = json.loads((repo_root / "data/corpus/corpus.json").read_text())
    lines = []
    for folio in sorted(corpus["folios"], key=folio_key):
        for line in corpus["folios"][folio]:
            if locus_type(line["locus"]) != "P":
                continue
            words = [
                word
                for word in line["words"]
                if "?" not in word and len(word) >= 2
            ]
            if words:
                lines.append(words)
    return lines


def normalized_latin_words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    replacements = {
        "æ": "ae",
        "œ": "oe",
        "ð": "d",
        "þ": "th",
        "ł": "l",
        "ß": "ss",
        "ø": "o",
    }
    normalized = "".join(replacements.get(char, char) for char in normalized.lower())
    normalized = normalized.replace("w", "uu").replace("j", "i").replace("k", "c")
    return re.findall(r"[a-z]+", normalized)


def load_naibbe_ciphertext(path: Path) -> list[str]:
    return re.findall(r"[a-z]+", path.read_text(encoding="utf-8", errors="ignore").lower())


def decompose(word: str) -> tuple[str, str, str]:
    prefix = next((item for item in PRE if word.startswith(item)), "")
    remainder = word[len(prefix):]
    suffix = next(
        (
            item
            for item in SUF
            if remainder.endswith(item) and len(remainder) > len(item)
        ),
        "",
    )
    core = remainder[:-len(suffix)] if suffix else remainder
    return prefix, core, suffix


def weighted_choice(rng: random.Random, counts: Counter[str]) -> str:
    values = list(counts)
    return rng.choices(values, weights=[counts[value] for value in values], k=1)[0]


def schema_copy_surrogate(words: list[str], n: int) -> list[str]:
    """Reproduce residual.py's Voynich-fitted schema/copy null."""
    rng = random.Random(SEED)
    core_counts = Counter(decompose(word)[1] for word in words)
    prefixes: dict[str, Counter[str]] = defaultdict(Counter)
    suffixes: dict[str, Counter[str]] = defaultdict(Counter)
    for word in words:
        prefix, core, suffix = decompose(word)
        prefixes[core][prefix] += 1
        suffixes[core][suffix] += 1
    glyphs = sorted(set("".join(words)))

    def sample_word() -> str:
        core = weighted_choice(rng, core_counts)
        return (
            weighted_choice(rng, prefixes[core])
            + core
            + weighted_choice(rng, suffixes[core])
        )

    generated: list[str] = []
    for _ in range(n):
        if generated and rng.random() < 0.12:
            window = min(40, len(generated))
            word = generated[-rng.randint(1, window)]
            if len(word) > 1 and rng.random() < 0.4:
                index = rng.randrange(len(word))
                word = word[:index] + rng.choice(glyphs) + word[index + 1:]
            generated.append(word)
        else:
            generated.append(sample_word())
    return generated


def reflow(words: list[str], line_lengths: list[int], n: int) -> list[list[str]]:
    if len(words) < n:
        raise ValueError(f"need {n} tokens, found {len(words)}")
    lines: list[list[str]] = []
    offset = 0
    for length in line_lengths:
        if offset >= n:
            break
        take = min(length, n - offset)
        lines.append(words[offset:offset + take])
        offset += take
    if offset != n:
        raise ValueError(f"line template holds {offset} of {n} tokens")
    return lines


def entropy(counts: Counter[object]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(
        count / total * math.log2(count / total)
        for count in counts.values()
        if count
    )


def bigram_conditional_entropy(lines: list[list[str]]) -> float:
    joint: Counter[tuple[str, str]] = Counter()
    context: Counter[str] = Counter()
    for line in lines:
        for first, second in zip(line, line[1:]):
            joint[(first, second)] += 1
            context[first] += 1
    total = sum(joint.values())
    return -sum(
        count / total * math.log2(count / context[first])
        for (first, _), count in joint.items()
    )


def exact_order_metric(
    lines: list[list[str]], name: str, permutations: int = 20
) -> dict[str, float]:
    observed = bigram_conditional_entropy(lines)
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big") ^ SEED
    rng = random.Random(seed)
    shuffled = []
    for _ in range(permutations):
        sample = []
        for line in lines:
            copy = line.copy()
            rng.shuffle(copy)
            sample.append(copy)
        shuffled.append(bigram_conditional_entropy(sample))
    mean = statistics.fmean(shuffled)
    return {
        "observed": observed,
        "shuffle": mean,
        "gap": mean - observed,
        "shuffle_sd": statistics.pstdev(shuffled),
    }


def positional_entropy(words: list[str], bins: int = 5) -> tuple[float, int]:
    positions: dict[str, list[int]] = defaultdict(lambda: [0] * bins)
    frequencies: Counter[str] = Counter()
    for word in words:
        for index, char in enumerate(word):
            relative = index / (len(word) - 1) if len(word) > 1 else 0.5
            bucket = min(bins - 1, int(relative * bins))
            positions[char][bucket] += 1
            frequencies[char] += 1
    total = sum(frequencies.values())
    weighted = sum(
        frequencies[char] / total * entropy(Counter(dict(enumerate(counts))))
        for char, counts in positions.items()
    )
    return weighted, len(frequencies)


def unit_h2(words: list[list[str]]) -> float:
    context: Counter[str] = Counter()
    joint: Counter[tuple[str, str]] = Counter()
    for word in words:
        sequence = ["^", *word, "$"]
        for first, second in zip(sequence, sequence[1:]):
            context[first] += 1
            joint[(first, second)] += 1
    total = sum(joint.values())
    return -sum(
        count / total * math.log2(count / context[first])
        for (first, _), count in joint.items()
    )


def bpe_track(words: list[str]) -> dict[int, dict[str, float | int]]:
    units = [list(word) for word in words]
    track: dict[int, dict[str, float | int]] = {}
    for merge_index in range(max(CHECKPOINTS) + 1):
        if merge_index in CHECKPOINTS:
            track[merge_index] = {
                "h2": unit_h2(units),
                "units_per_word": statistics.fmean(map(len, units)),
                "alphabet": len({unit for word in units for unit in word}),
            }
        if merge_index == max(CHECKPOINTS):
            break
        pairs: Counter[tuple[str, str]] = Counter()
        for word in units:
            pairs.update(zip(word, word[1:]))
        if not pairs:
            break
        (left, right), _ = pairs.most_common(1)[0]
        merged = left + right
        for word in units:
            index = 0
            while index < len(word) - 1:
                if word[index] == left and word[index + 1] == right:
                    word[index:index + 2] = [merged]
                else:
                    index += 1
    return track


def mutual_information(xs: list[str], ys: list[str]) -> float:
    total = len(xs)
    left = Counter(xs)
    right = Counter(ys)
    joint = Counter(zip(xs, ys))
    return sum(
        count / total
        * math.log2(
            (count / total)
            / ((left[x] / total) * (right[y] / total))
        )
        for (x, y), count in joint.items()
    )


def slot_metrics(words: list[str]) -> dict[str, float | int]:
    decomposed = [decompose(word) for word in words]
    prefixes = [prefix for prefix, _, _ in decomposed]
    cores = [core for _, core, _ in decomposed]
    suffixes = [suffix for _, _, suffix in decomposed]
    return {
        "coverage": sum(bool(prefix or suffix) for prefix, _, suffix in decomposed)
        / len(decomposed),
        "prefix_types": len(set(prefixes)),
        "core_types": len(set(cores)),
        "suffix_types": len(set(suffixes)),
        "mi_pre_suf": mutual_information(prefixes, suffixes),
        "mi_pre_core": mutual_information(prefixes, cores),
        "mi_core_suf": mutual_information(cores, suffixes),
    }


def burstiness(words: list[str], minimum_count: int = 10) -> tuple[float, int]:
    positions: dict[str, list[int]] = defaultdict(list)
    for index, word in enumerate(words):
        positions[word].append(index)
    values = []
    for word_positions in positions.values():
        if len(word_positions) < minimum_count:
            continue
        gaps = [
            second - first
            for first, second in zip(word_positions, word_positions[1:])
        ]
        mean = statistics.fmean(gaps)
        sd = statistics.pstdev(gaps)
        if mean + sd:
            values.append((sd - mean) / (sd + mean))
    return (statistics.fmean(values) if values else float("nan"), len(values))


def hapax_type_fraction(words: list[str]) -> float:
    counts = Counter(words)
    return sum(count == 1 for count in counts.values()) / len(counts)


def char_stream(words: list[str]) -> str:
    return " ".join(words)


def latin_attack_text(path: Path, cap: int = 400_000) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    text = re.sub(r"[^a-z]+", " ", text)
    text = text.replace("j", "i").replace("v", "u").replace("w", "u").replace("k", "c")
    return re.sub(r"\s+", " ", text).strip()[:cap]


def build_trigram_lm(text: str, alphabet: list[str]):
    allowed = set(alphabet) | {" "}
    text = "".join(char if char in allowed else " " for char in text)
    trigrams: Counter[tuple[str, str, str]] = Counter()
    bigrams: Counter[tuple[str, str]] = Counter()
    for index in range(len(text) - 2):
        first, second, third = text[index:index + 3]
        trigrams[(first, second, third)] += 1
        bigrams[(first, second)] += 1
    vocabulary_size = len(allowed)

    def log_probability(first: str, second: str, third: str) -> float:
        return math.log(
            (trigrams[(first, second, third)] + 1)
            / (bigrams[(first, second)] + vocabulary_size)
        )

    return log_probability


def trigram_counts(text: str) -> tuple[Counter[tuple[str, str, str]], int]:
    counts = Counter(
        (text[index], text[index + 1], text[index + 2])
        for index in range(len(text) - 2)
    )
    return counts, sum(counts.values())


def score_counts(counts, total: int, key: dict[str, str], log_probability) -> float:
    def mapped(char: str) -> str:
        return " " if char == " " else key.get(char, " ")

    score = 0.0
    for (first, second, third), count in counts.items():
        score += count * log_probability(
            mapped(first), mapped(second), mapped(third)
        )
    return score / max(1, total)


def score_plain(text: str, log_probability) -> float:
    counts, total = trigram_counts(text)
    return sum(
        count * log_probability(first, second, third)
        for (first, second, third), count in counts.items()
    ) / max(1, total)


def hillclimb(
    train_counts,
    train_total: int,
    source_alphabet: list[str],
    destination_alphabet: list[str],
    log_probability,
    rng: random.Random,
    steps: int,
    restarts: int,
) -> dict[str, str]:
    source = list(source_alphabet)
    destination = list(destination_alphabet)
    size = min(len(source), len(destination))
    source = source[:size]
    best_key: dict[str, str] = {}
    best_score = -math.inf
    for _ in range(restarts):
        permutation = destination[:size]
        rng.shuffle(permutation)
        key = dict(zip(source, permutation))
        current = score_counts(train_counts, train_total, key, log_probability)
        for _ in range(steps):
            first, second = rng.sample(source, 2)
            key[first], key[second] = key[second], key[first]
            score = score_counts(train_counts, train_total, key, log_probability)
            if score >= current:
                current = score
            else:
                key[first], key[second] = key[second], key[first]
        if current > best_score:
            best_score = current
            best_key = key.copy()
    return best_key


def attack_one(
    text: str,
    latin_alphabet: list[str],
    log_probability,
    rng: random.Random,
    steps: int,
    restarts: int,
) -> tuple[float, float]:
    midpoint = len(text) // 2
    train, test = text[:midpoint], text[midpoint:]
    train_counts, train_total = trigram_counts(train)
    test_counts, test_total = trigram_counts(test)
    source_alphabet = [
        char
        for char, _ in Counter(text.replace(" ", "")).most_common()
    ]
    key = hillclimb(
        train_counts,
        train_total,
        source_alphabet,
        latin_alphabet,
        log_probability,
        rng,
        steps,
        restarts,
    )
    return (
        score_counts(train_counts, train_total, key, log_probability),
        score_counts(test_counts, test_total, key, log_probability),
    )


def cipher_attack(
    corpora: dict[str, list[str]],
    latin_control_path: Path,
    steps: int,
    restarts: int,
) -> dict[str, object]:
    rng = random.Random(SEED + 991)
    latin = latin_attack_text(latin_control_path)
    cutoff = int(len(latin) * 0.55)
    language_model_text, held_out = latin[:cutoff], latin[cutoff:]
    latin_alphabet = [
        char
        for char, _ in Counter(language_model_text.replace(" ", "")).most_common()
    ]
    log_probability = build_trigram_lm(language_model_text, latin_alphabet)
    ceiling = score_plain(held_out, log_probability)

    encryption_permutation = latin_alphabet.copy()
    rng.shuffle(encryption_permutation)
    encryption_key = dict(zip(latin_alphabet, encryption_permutation))
    encrypted_latin = "".join(
        " " if char == " " else encryption_key.get(char, " ")
        for char in held_out
    )
    floor = score_plain(encrypted_latin, log_probability)

    rows = {
        "POS_CTRL_SIMPLE_SUB": attack_one(
            encrypted_latin,
            latin_alphabet,
            log_probability,
            rng,
            steps,
            restarts,
        )
    }
    for name in ("VOYNICH", "NAIBBE_MEANINGFUL", "SCHEMA_COPY_NULL"):
        rows[name] = attack_one(
            char_stream(corpora[name]),
            latin_alphabet,
            log_probability,
            rng,
            steps,
            restarts,
        )
    return {"ceiling": ceiling, "floor": floor, "rows": rows}


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_report(
    provenance: dict[str, object],
    corpora: dict[str, list[str]],
    order: dict[str, dict[str, float]],
    positions: dict[str, tuple[float, int]],
    bpe: dict[str, dict[int, dict[str, float | int]]],
    slots: dict[str, dict[str, float | int]],
    lexical: dict[str, dict[str, float | int]],
    attack: dict[str, object] | None,
) -> str:
    names = list(corpora)
    lines = [
        "NAIBBE MEANINGFUL-CIPHERTEXT BENCHMARK",
        "=" * 80,
        f"Naibbe implementation: {provenance['github']}",
        f"Paper: {provenance['doi']}",
        f"Naibbe revision: {provenance['revision']}",
        f"Ciphertext SHA-256: {provenance['ciphertext_sha256']}",
        f"Book 16 source SHA-256: {provenance['source_sha256']}",
        f"Plaintext SHA-256: {provenance['plaintext_sha256']}",
        f"Matched token count: {provenance['tokens']}",
        (
            "Line control: every stream reflowed to the same Voynich prose-line "
            "length template"
        ),
        (
            "Plaintext relation: published Pliny Natural History comparator; "
            "ciphertext source is Book 16 of the same work, not token-aligned"
        ),
        "",
        "CORPORA",
        "-" * 80,
    ]
    for name in names:
        lines.append(
            f"{name:<20} tokens={len(corpora[name]):>6} "
            f"types={len(set(corpora[name])):>6} "
            f"glyphs={len(set(''.join(corpora[name]))):>3}"
        )

    lines.extend([
        "",
        "EXACT-TOKEN ORDER (H(next word|word); positive gap means order helps)",
        "-" * 80,
        f"{'corpus':<20}{'real H':>10}{'shuffle H':>12}{'gap':>10}{'shuffle SD':>12}",
    ])
    for name in names:
        row = order[name]
        lines.append(
            f"{name:<20}{row['observed']:>10.3f}{row['shuffle']:>12.3f}"
            f"{row['gap']:>+10.3f}{row['shuffle_sd']:>12.3f}"
        )

    lines.extend([
        "",
        "GLYPH POSITION LOCK (5 relative-position bins; lower = more locked)",
        "-" * 80,
        f"{'corpus':<20}{'H_position':>14}{'glyphs':>10}",
    ])
    for name in names:
        position_h, alphabet = positions[name]
        lines.append(f"{name:<20}{position_h:>14.3f}{alphabet:>10}")

    lines.extend([
        "",
        "BPE UNIT H2 (same greedy merge procedure)",
        "-" * 80,
        f"{'corpus':<20}" + "".join(f"{'m=' + str(m):>10}" for m in CHECKPOINTS)
        + f"{'climb':>10}",
    ])
    for name in names:
        values = [float(bpe[name][merge]["h2"]) for merge in CHECKPOINTS]
        lines.append(
            f"{name:<20}"
            + "".join(f"{value:>10.3f}" for value in values)
            + f"{values[-1] - values[0]:>+10.3f}"
        )

    lines.extend([
        "",
        "HAND-SPECIFIED EVA SLOT DECOMPOSITION",
        "-" * 80,
        (
            f"{'corpus':<20}{'coverage':>10}{'P/C/S types':>18}"
            f"{'I(P;S)':>10}{'I(P;C)':>10}{'I(C;S)':>10}"
        ),
    ])
    for name in names:
        row = slots[name]
        cardinalities = (
            f"{row['prefix_types']}/{row['core_types']}/{row['suffix_types']}"
        )
        lines.append(
            f"{name:<20}{float(row['coverage']):>10.3f}{cardinalities:>18}"
            f"{float(row['mi_pre_suf']):>10.3f}"
            f"{float(row['mi_pre_core']):>10.3f}"
            f"{float(row['mi_core_suf']):>10.3f}"
        )

    lines.extend([
        "",
        "LEXICAL TAIL / BURSTINESS",
        "-" * 80,
        f"{'corpus':<20}{'hapax/type':>14}{'burst B':>12}{'eligible':>12}",
    ])
    for name in names:
        row = lexical[name]
        lines.append(
            f"{name:<20}{float(row['hapax']):>14.3f}"
            f"{float(row['burst']):>12.3f}{int(row['burst_n']):>12}"
        )

    if attack:
        ceiling = float(attack["ceiling"])
        floor = float(attack["floor"])
        lines.extend([
            "",
            "BIJECTIVE CHARACTER-SUBSTITUTION ATTACK (repository R1 hypothesis class)",
            "-" * 80,
            f"Latin LM ceiling={ceiling:.3f}; random-key floor={floor:.3f}",
            f"{'corpus':<24}{'train':>12}{'held-out':>12}{'gap closed':>14}",
        ])
        for name, (train, test) in attack["rows"].items():
            fraction = (test - floor) / (ceiling - floor)
            lines.append(
                f"{name:<24}{train:>12.3f}{test:>12.3f}{fraction:>13.1%}"
            )

    naibbe_order_retention = (
        order["NAIBBE_MEANINGFUL"]["gap"] / order["LATIN_PLAIN"]["gap"]
    )
    lines.extend([
        "",
        "RESULT SUMMARY",
        "-" * 80,
        (
            "Known-meaningful Naibbe is more position-locked than Voynich "
            f"({positions['NAIBBE_MEANINGFUL'][0]:.3f} vs "
            f"{positions['VOYNICH'][0]:.3f}) and has a similar BPE climb "
            f"({float(bpe['NAIBBE_MEANINGFUL'][100]['h2']) - float(bpe['NAIBBE_MEANINGFUL'][0]['h2']):+.3f} "
            f"vs {float(bpe['VOYNICH'][100]['h2']) - float(bpe['VOYNICH'][0]['h2']):+.3f})."
        ),
        (
            "Its EVA slot coverage and MI profile are also close to Voynich: "
            f"coverage {float(slots['NAIBBE_MEANINGFUL']['coverage']):.3f} vs "
            f"{float(slots['VOYNICH']['coverage']):.3f}; "
            f"I(prefix;suffix) {float(slots['NAIBBE_MEANINGFUL']['mi_pre_suf']):.3f} "
            f"vs {float(slots['VOYNICH']['mi_pre_suf']):.3f}."
        ),
        (
            "Some exact-token order survives encryption: Naibbe's shuffle gap is "
            f"{order['NAIBBE_MEANINGFUL']['gap']:+.3f} bits, "
            f"{naibbe_order_retention:.0%} of the plaintext comparator's "
            f"{order['LATIN_PLAIN']['gap']:+.3f}, while Voynich is "
            f"{order['VOYNICH']['gap']:+.3f}."
        ),
        (
            "Naibbe does not match Voynich's lexical tail or clumping: "
            f"hapax/type {float(lexical['NAIBBE_MEANINGFUL']['hapax']):.3f} vs "
            f"{float(lexical['VOYNICH']['hapax']):.3f}; burst B "
            f"{float(lexical['NAIBBE_MEANINGFUL']['burst']):+.3f} vs "
            f"{float(lexical['VOYNICH']['burst']):+.3f}."
        ),
        (
            "Consequently, position lock, BPE climb, EVA slot structure, and the "
            "bijective R1 attack cannot exclude meaningful verbose ciphertext. "
            "The order and lexical differences remain usable constraints on a "
            "Voynich-specific cipher model."
        ),
    ])

    lines.extend([
        "",
        "INTERPRETATION LIMITS",
        "-" * 80,
        (
            "1. Naibbe is a constructed positive control, not evidence that MS 408 "
            "uses this cipher."
        ),
        (
            "2. The exact-token entropy estimator is the repository's in-sample "
            "diagnostic; sparse vocabularies bias it. Read only its shuffle gap."
        ),
        (
            "3. The slot inventory is hand-selected for EVA. Its Latin row is a "
            "mechanical negative control, not a Latin morphology analysis."
        ),
        (
            "4. The schema/copy null is fitted from Voynich word parts and is not "
            "an independent content-free process."
        ),
        (
            "5. The R1 attack can recover bijective character substitutions only. "
            "Failure on Naibbe is expected and measures the attack's scope."
        ),
        (
            "6. A diagnostic that assigns Naibbe and Voynich the same verdict cannot "
            "by itself distinguish meaningful verbose ciphertext from asemic text."
        ),
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    naibbe_root = args.naibbe_dir.resolve()
    ciphertext_path = naibbe_root / "encrypted/nathist_output_ciphertext.txt"
    source_path = naibbe_root / "input/examples/nathist_book16.txt"
    plaintext_path = (
        naibbe_root
        / "figure_utils/gaskell_bowern_2022/data/meaningful/texts"
        / "Historical - Latin - Technical - Pliny's Natural History.txt"
    )
    required = (ciphertext_path, source_path, plaintext_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "Missing Naibbe reference data:\n  "
            + "\n  ".join(missing)
            + f"\nClone {NAIBBE_GITHUB} and pass --naibbe-dir."
        )

    voynich_lines = load_voynich_lines(repo_root)
    voynich = [word for line in voynich_lines for word in line]
    n = len(voynich)
    line_lengths = [len(line) for line in voynich_lines]
    naibbe = load_naibbe_ciphertext(ciphertext_path)[:n]
    plaintext = normalized_latin_words(
        plaintext_path.read_text(encoding="utf-8", errors="ignore")
    )[:n]
    surrogate = schema_copy_surrogate(voynich, n)

    corpora = {
        "VOYNICH": voynich,
        "NAIBBE_MEANINGFUL": naibbe,
        "LATIN_PLAIN": plaintext,
        "SCHEMA_COPY_NULL": surrogate,
    }
    if any(len(words) != n for words in corpora.values()):
        sizes = ", ".join(f"{name}={len(words)}" for name, words in corpora.items())
        raise SystemExit(f"Could not token-match corpora: {sizes}")

    line_corpora = {
        name: reflow(words, line_lengths, n)
        for name, words in corpora.items()
    }
    order = {
        name: exact_order_metric(line_corpora[name], name)
        for name in corpora
    }
    positions = {
        name: positional_entropy(words)
        for name, words in corpora.items()
    }
    bpe = {
        name: bpe_track(words)
        for name, words in corpora.items()
    }
    slots = {
        name: slot_metrics(words)
        for name, words in corpora.items()
    }
    lexical = {}
    for name, words in corpora.items():
        burst, eligible = burstiness(words)
        lexical[name] = {
            "hapax": hapax_type_fraction(words),
            "burst": burst,
            "burst_n": eligible,
        }

    attack = None
    if not args.skip_attack:
        attack = cipher_attack(
            corpora,
            repo_root / "data/controls/latin.txt",
            args.attack_steps,
            args.attack_restarts,
        )

    provenance = {
        "github": NAIBBE_GITHUB,
        "doi": NAIBBE_DOI,
        "revision": git_revision(naibbe_root),
        "ciphertext_sha256": sha256(ciphertext_path),
        "source_sha256": sha256(source_path),
        "plaintext_sha256": sha256(plaintext_path),
        "tokens": n,
    }
    report = render_report(
        provenance,
        corpora,
        order,
        positions,
        bpe,
        slots,
        lexical,
        attack,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
