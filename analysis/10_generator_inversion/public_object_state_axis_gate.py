#!/usr/bin/env python3
"""
Run the frozen state-axis discriminator on complete-object embeddings.

The scoring implementation is imported unchanged from visual_state_axis_gate.
Only the visual artifact and experiment metadata differ from the earlier
coarse-mask/DINO run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import production_algorithm_gate as source
import visual_state_axis_gate as gate


ROOT = Path(__file__).resolve().parents[2]
OBJECTS = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_public_object_embeddings.json"
)
OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_public_object_state_axis_gate.json"
)


def run(
    corpus: Path,
    objects: Path,
    output: Path,
    progress: bool = True,
) -> dict:
    result = gate.run(
        corpus,
        objects,
        objects,
        output,
        progress=progress,
    )
    artifact = json.loads(objects.read_text(encoding="utf-8"))
    result["experiment"] = (
        "public_complete_object_prediction_of_text_state_axes"
    )
    result["visual_representation"] = {
        "artifact_experiment": artifact["experiment"],
        "view_semantics": {
            "full_rgb": (
                "original pixels inside completed object masks, white outside"
            ),
            "tight_rgb": "tight union crop of the full RGB object mask",
            "full_silhouette": (
                "foreground strokes inside completed object masks"
            ),
            "tight_silhouette": (
                "tight union crop of the foreground-stroke silhouette"
            ),
            "handcrafted_combined": (
                "completed-mask, pigment, skeleton, endpoint, junction, "
                "component, grid, and projection features"
            ),
        },
        "object_summary": artifact["summary"],
        "mapping_audit": artifact["mapping_audit"],
        "models": artifact["models"],
    }
    result["parameters"]["visual_candidate_interpretation"] = {
        "dino_full_rgb_cls": "completed objects at page coordinates",
        "dino_tight_rgb_cls": "completed objects tightly cropped",
        "dino_full_silhouette_cls": (
            "complete in-object stroke topology at page coordinates"
        ),
        "dino_tight_silhouette_cls": (
            "complete in-object stroke topology tightly cropped"
        ),
        "dino_full_rgb_both": (
            "completed-object CLS plus mean patch embedding"
        ),
        "dino_tight_rgb_both": (
            "tight completed-object CLS plus mean patch embedding"
        ),
        "dino_full_tight_rgb_cls": (
            "concatenated page-coordinate and tight object CLS"
        ),
        "handcrafted_combined": "explicit object topology descriptor",
    }
    result["claim_boundary"] = (
        "A pass establishes out-of-quire prediction of frozen text-state "
        "axes from publicly pretrained, foldout-aware complete-object "
        "representations under exact quire x Currier x section rematching. "
        "It still does not identify plaintext or image semantics. A failure "
        "closes this detector/SAM/DINO/topology route and its linear kernel "
        "map, not manually labeled iconographic correspondences."
    )
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=source.CORPUS)
    parser.add_argument("--objects", type=Path, default=OBJECTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        args.corpus,
        args.objects,
        args.output,
        progress=not args.quiet,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
