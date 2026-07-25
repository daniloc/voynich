#!/usr/bin/env python3
"""
Put herbal and zodiac diagrams into one explicit visual-graph schema.

Pixel descriptors are identical across domains.  Structural descriptors use
the same graph vocabulary, while retaining a provenance flag because herbal
nodes come from coarse organ tags and zodiac nodes come from ring geometry.
This file does not assert that a plant root and a zodiac ring mean the same
thing; it creates the shared coordinates needed to test that proposition.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import build_zodiac_visual_nodes as zodiac  # noqa: E402


PLANT_TAGS = ROOT / "data" / "grounding" / "plant_tags.json"
ZODIAC_NODES = ROOT / "data" / "grounding" / "zodiac_all12_visual_nodes.json"
OUTPUT_DIR = ROOT / "images" / "crops" / "multimodal_graphs"
DEFAULT_OUTPUT = ROOT / "data" / "grounding" / "multimodal_visual_graphs.json"

PLANT_FOLIOS = {
    "m0_a": "f4r",
    "m0_b": "f6r",
    "m0_c": "f9r",
    "m0_d": "f11r",
    "m1_a": "f15r",
    "m1_b": "f18r",
    "m1_c": "f20r",
    "m1_d": "f23r",
    "m2_a": "f31r",
    "m2_b": "f35r",
    "m2_c": "f43r",
    "m2_d": "f52r",
    "r016_f8": "f8r",
    "r026_f16": "f16r",
    "r050_f26": "f26r",
}


@dataclass(frozen=True)
class PlantImageSpec:
    source: str
    crop: Optional[tuple[int, int, int, int]]


def montage_specs(index: int) -> dict[str, PlantImageSpec]:
    source = f"images/derived/plants_montage_{index}.png"
    prefix = f"m{index}"
    return {
        f"{prefix}_a": PlantImageSpec(source, (8, 30, 568, 806)),
        f"{prefix}_b": PlantImageSpec(source, (576, 30, 1136, 806)),
        f"{prefix}_c": PlantImageSpec(source, (8, 836, 568, 1612)),
        f"{prefix}_d": PlantImageSpec(source, (576, 836, 1136, 1612)),
    }


PLANT_IMAGES = {
    **montage_specs(0),
    **montage_specs(1),
    **montage_specs(2),
    "r016_f8": PlantImageSpec("images/facsimile/raw_016.png", None),
    "r026_f16": PlantImageSpec("images/facsimile/raw_026.png", None),
    "r050_f26": PlantImageSpec("images/facsimile/raw_050.png", None),
}


def root_axes(tag: str) -> int:
    return 3 if tag.startswith("fibrous") else 1


def root_branch(tag: str) -> int:
    return int(
        "branched" in tag
        or "forked" in tag
        or tag.startswith("fibrous")
    )


def stem_axes(tag: str) -> int:
    return {
        "single_erect": 1,
        "branching": 2,
        "multi_radiating": 3,
    }[tag]


def leaf_order(tag: str) -> int:
    if tag in {"none_small", "basal_brown_mass"}:
        return 0
    if tag == "palmate_lobed":
        return 3
    if tag == "pinnate_fern":
        return 4
    if "paired" in tag:
        return 2
    return 1


def inflorescence_arity(tag: str) -> int:
    if tag.startswith("none"):
        return 0
    if "pair" in tag:
        return 2
    if "cluster" in tag or tag == "umbel":
        return 3
    return 1


def plant_graph(row: dict) -> dict[str, int]:
    roots = root_axes(row["root"])
    stems = stem_axes(row["stem"])
    leaves = leaf_order(row["leaf"])
    flowers = inflorescence_arity(row["infl"])
    branches = (
        root_branch(row["root"])
        + int(row["stem"] == "branching")
        + max(stems - 1, 0)
    )
    return {
        "node_count": 1 + roots + stems + leaves + flowers,
        "axis_count": roots + stems,
        "branch_point_count": branches,
        "terminal_group_count": leaves + flowers,
        "cycle_rank": 0,
        "hierarchy_depth": 4,
        "radial_layer_count": 0,
        "horizontal_axis": int(row["root"].startswith("rhizome")),
        "repetition_count": max(leaves, flowers),
    }


def zodiac_graph(row: dict) -> dict[str, int]:
    nodes = int(row["ring_label_count"])
    tiers = len(row["alignment"])
    return {
        "node_count": nodes,
        "axis_count": tiers,
        "branch_point_count": 0,
        "terminal_group_count": nodes,
        "cycle_rank": tiers,
        "hierarchy_depth": tiers,
        "radial_layer_count": tiers,
        "horizontal_axis": 0,
        "repetition_count": nodes,
    }


def load_plant_image(spec: PlantImageSpec) -> Image.Image:
    image = Image.open(ROOT / spec.source).convert("RGB")
    if spec.crop is not None:
        image = image.crop(spec.crop)
    return image


def build(output: Path) -> dict:
    if not ZODIAC_NODES.exists():
        zodiac.build(ZODIAC_NODES)
    zodiac_data = json.loads(ZODIAC_NODES.read_text(encoding="utf-8"))
    tags = json.loads(PLANT_TAGS.read_text(encoding="utf-8"))["plants"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for row in tags:
        item_id = row["id"]
        if item_id not in PLANT_IMAGES:
            continue
        spec = PLANT_IMAGES[item_id]
        image = load_plant_image(spec)
        panel_path = OUTPUT_DIR / f"{item_id}_panel.jpg"
        image.save(panel_path, quality=92)
        records.append({
            "domain": "herbal",
            "id": item_id,
            "folio": PLANT_FOLIOS[item_id],
            "source": spec.source,
            "source_crop": list(spec.crop) if spec.crop else None,
            "panel_image": str(panel_path.relative_to(ROOT)),
            "panel_size": list(image.size),
            "visual": zodiac.visual_features(image, zodiac.PANEL_FEATURE_SIZE),
            "graph": plant_graph(row),
            "graph_provenance": "coarse_organ_tags",
            "tags": {
                key: row[key] for key in ("root", "stem", "leaf", "infl")
            },
        })

    for folio, row in zodiac_data["folios"].items():
        records.append({
            "domain": "zodiac",
            "id": folio,
            "folio": folio,
            "source": row["source"],
            "source_crop": row["source_crop"],
            "panel_image": row["panel_image"],
            "panel_size": row["panel_size"],
            "visual": row["panel_visual"],
            "graph": zodiac_graph(row),
            "graph_provenance": "ordered_ring_geometry",
            "tags": {
                "outer_nodes": sum(
                    record["tier"] == "outer" for record in row["records"]
                ),
                "inner_nodes": sum(
                    record["tier"] == "inner" for record in row["records"]
                ),
                "outside_labels": row["outside_label_count"],
            },
        })

    result = {
        "built": "2026-07-23",
        "method": (
            "shared resolution-normalized pixel descriptors and explicit "
            "domain-specific graph construction"
        ),
        "pixel_schema": zodiac_data["feature_schema"]["pixel_features"],
        "graph_schema": [
            "node_count",
            "axis_count",
            "branch_point_count",
            "terminal_group_count",
            "cycle_rank",
            "hierarchy_depth",
            "radial_layer_count",
            "horizontal_axis",
            "repetition_count",
        ],
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.output)
    domains = {
        name: sum(row["domain"] == name for row in result["records"])
        for name in ("herbal", "zodiac")
    }
    print(f"wrote {args.output}")
    print(f"records={len(result['records'])} domains={domains}")


if __name__ == "__main__":
    main()
