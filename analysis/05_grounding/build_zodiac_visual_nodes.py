#!/usr/bin/env python3
"""
Build an auditable visual-node dataset for all twelve Voynich zodiac diagrams.

The source transcription supplies a clockwise cyclic order for each ring tier.
Four diagrams also have label readings anchored directly against the scans in
z10_bindings.json.  For the remaining diagrams, this script uses the modal
start position learned from those four calibrators and marks the phase as
inferred.  No semantic attributes are invented for the inferred records.

Every ring label is bound to:
  * a cyclic graph position and tier;
  * a documented source-image coordinate;
  * resolution-normalized pixel descriptors from a local patch; and
  * explicit alignment method/confidence fields.

The generated QC overlays are part of the protocol: they make a displaced
center, radius, direction, or phase visible before the data reach a solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LABELS = ROOT / "data" / "grounding" / "z10" / "all12_labels.json"
ANCHORS = ROOT / "data" / "grounding" / "z10_bindings.json"
IMAGE_ROOT = ROOT / "images" / "crops"
OUTPUT_DIR = IMAGE_ROOT / "zodiac_all12"
DEFAULT_OUTPUT = ROOT / "data" / "grounding" / "zodiac_all12_visual_nodes.json"

TIERS = ("outer", "inner")
FEATURE_SIZE = 96
PANEL_FEATURE_SIZE = 512


@dataclass(frozen=True)
class PanelSpec:
    source: str
    canvas_id: str
    crop: Optional[tuple[int, int, int, int]]
    center: tuple[float, float]
    outer_radius: tuple[float, float]
    inner_radius: tuple[float, float]


# Coordinates were placed against the Yale IIIF canvases and are expressed in
# the cropped panel coordinate system.  The QC overlays generated below are the
# review surface for this small, deliberately explicit geometry table.
PANEL_SPECS: dict[str, PanelSpec] = {
    "f70v2": PanelSpec(
        "zodiac_z10/f70v2_full.jpg",
        "1006200",
        None,
        (2075, 1830),
        (1300, 1260),
        (790, 805),
    ),
    "f70v1": PanelSpec(
        "zodiac_all12/f70v1_source.jpg",
        "1006201",
        None,
        (1475, 1450),
        (900, 965),
        (515, 560),
    ),
    "f71r": PanelSpec(
        "zodiac_z10/f71r_full.jpg",
        "1006202",
        None,
        (1395, 1740),
        (1050, 1080),
        (590, 600),
    ),
    "f71v": PanelSpec(
        "zodiac_all12/f71v_f72r_source.jpg",
        "1006203",
        (0, 0, 2300, 3018),
        (1200, 1500),
        (850, 850),
        (500, 485),
    ),
    "f72r1": PanelSpec(
        "zodiac_all12/f71v_f72r_source.jpg",
        "1006203",
        (2000, 0, 4400, 3018),
        (1200, 1500),
        (850, 850),
        (500, 485),
    ),
    "f72r2": PanelSpec(
        "zodiac_z10/f72r2_full.jpg",
        "1006203",
        None,
        (980, 1400),
        (790, 1050),
        (470, 590),
    ),
    "f72r3": PanelSpec(
        "zodiac_all12/f71v_f72r_source.jpg",
        "1006203",
        (6200, 0, 8865, 3018),
        (1150, 1480),
        (850, 880),
        (500, 500),
    ),
    "f72v1": PanelSpec(
        "zodiac_all12/f72v_wide_source.jpg",
        "1006204",
        (1000, 0, 3400, 3794),
        (1270, 1780),
        (850, 960),
        (510, 565),
    ),
    "f72v2": PanelSpec(
        "zodiac_all12/f72v_wide_source.jpg",
        "1006204",
        (3400, 0, 5976, 3794),
        (1180, 1850),
        (875, 940),
        (510, 565),
    ),
    "f72v3": PanelSpec(
        "zodiac_all12/f72v_part_source.jpg",
        "1006205",
        None,
        (1490, 1450),
        (980, 1040),
        (575, 610),
    ),
    "f73r": PanelSpec(
        "zodiac_all12/f73r_source.jpg",
        "1006206",
        None,
        (1410, 1500),
        (990, 1110),
        (590, 630),
    ),
    "f73v": PanelSpec(
        "zodiac_z10/f73v_full.jpg",
        "1006207",
        None,
        (1650, 1700),
        (1080, 1100),
        (620, 635),
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(spec: PanelSpec) -> tuple[Image.Image, Path]:
    source = IMAGE_ROOT / spec.source
    image = Image.open(source).convert("RGB")
    if spec.crop is not None:
        image = image.crop(spec.crop)
    return image, source


def label_tokens(raw: str) -> list[str]:
    return [
        "".join(char for char in token.lower() if "a" <= char <= "z")
        for token in raw.split(".")
        if any("a" <= char.lower() <= "z" for char in token)
    ]


def normalized_array(image: Image.Image, size: int) -> np.ndarray:
    fitted = image.copy()
    fitted.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), tuple(np.median(
        np.asarray(fitted, dtype=np.uint8).reshape(-1, 3), axis=0
    ).astype(np.uint8)))
    left = (size - fitted.width) // 2
    top = (size - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    return np.asarray(canvas, dtype=np.float32)


def visual_features(image: Image.Image, size: int) -> dict[str, float]:
    rgb = normalized_array(image, size)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    paper = float(np.percentile(gray, 85))
    ink = gray < paper - 22.0
    chroma_floor = gray < paper - 4.0

    red = (
        (rgb[..., 0] - rgb[..., 1] > 13)
        & (rgb[..., 0] - rgb[..., 2] > 8)
        & chroma_floor
    )
    green = (
        (rgb[..., 1] - rgb[..., 0] > 8)
        & (rgb[..., 1] - rgb[..., 2] > 1)
        & chroma_floor
    )
    blue = (
        (rgb[..., 2] - rgb[..., 0] > 8)
        & (rgb[..., 2] - rgb[..., 1] > 2)
        & chroma_floor
    )
    gold = (
        (rgb[..., 0] - rgb[..., 2] > 14)
        & (rgb[..., 1] - rgb[..., 2] > 8)
        & (rgb[..., 0] - rgb[..., 1] > 2)
        & chroma_floor
    )

    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_density = (float(np.mean(dx > 14)) + float(np.mean(dy > 14))) / 2

    clipped = np.clip(gray, 0, 255).astype(np.uint8)
    hist = np.bincount((clipped // 16).ravel(), minlength=16).astype(float)
    probs = hist[hist > 0] / hist.sum()
    entropy = float(-np.sum(probs * np.log2(probs)))

    denom = max(float(np.mean(np.abs(gray - paper))), 1.0)
    horizontal_symmetry = 1.0 - min(
        float(np.mean(np.abs(gray - np.fliplr(gray)))) / (2 * denom),
        1.0,
    )
    vertical_symmetry = 1.0 - min(
        float(np.mean(np.abs(gray - np.flipud(gray)))) / (2 * denom),
        1.0,
    )

    yy, xx = np.indices(gray.shape)
    if ink.any():
        weights = np.maximum(paper - gray, 0) * ink
        total = float(weights.sum())
        centroid_x = float(
            (weights * xx).sum() / total / max(gray.shape[1] - 1, 1) * 2 - 1
        )
        centroid_y = float(
            (weights * yy).sum() / total / max(gray.shape[0] - 1, 1) * 2 - 1
        )
    else:
        centroid_x = 0.0
        centroid_y = 0.0

    # Text glyphs normally form small disconnected components at this scale.
    # Retaining only larger components gives the solver a coarse silhouette
    # channel without handing it a proxy for the adjacent label's length.
    visited = np.zeros(ink.shape, dtype=bool)
    component_sizes: list[int] = []
    component_pixels: list[list[tuple[int, int]]] = []
    height, width = ink.shape
    for start_y, start_x in zip(*np.nonzero(ink)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels = []
        while stack:
            point_y, point_x = stack.pop()
            pixels.append((point_y, point_x))
            for near_y in range(max(0, point_y - 1), min(height, point_y + 2)):
                for near_x in range(max(0, point_x - 1), min(width, point_x + 2)):
                    if ink[near_y, near_x] and not visited[near_y, near_x]:
                        visited[near_y, near_x] = True
                        stack.append((near_y, near_x))
        component_sizes.append(len(pixels))
        component_pixels.append(pixels)

    minimum_large = max(12, round(size * size * 0.0025))
    large_mask = np.zeros(ink.shape, dtype=bool)
    large_sizes = []
    for pixels, component_size in zip(component_pixels, component_sizes):
        if component_size < minimum_large:
            continue
        large_sizes.append(component_size)
        for point_y, point_x in pixels:
            large_mask[point_y, point_x] = True
    if large_mask.any():
        large_y, large_x = np.nonzero(large_mask)
        large_centroid_x = float(
            large_x.mean() / max(width - 1, 1) * 2 - 1
        )
        large_centroid_y = float(
            large_y.mean() / max(height - 1, 1) * 2 - 1
        )
    else:
        large_centroid_x = 0.0
        large_centroid_y = 0.0
    large_horizontal_symmetry = 1.0 - float(
        np.mean(large_mask != np.fliplr(large_mask))
    )
    large_vertical_symmetry = 1.0 - float(
        np.mean(large_mask != np.flipud(large_mask))
    )

    values = {
        "ink_fraction": float(np.mean(ink)),
        "red_fraction": float(np.mean(red)),
        "green_fraction": float(np.mean(green)),
        "blue_fraction": float(np.mean(blue)),
        "gold_fraction": float(np.mean(gold)),
        "edge_density": edge_density,
        "contrast": float(np.std(gray) / 255.0),
        "entropy": entropy,
        "horizontal_symmetry": horizontal_symmetry,
        "vertical_symmetry": vertical_symmetry,
        "ink_centroid_x": centroid_x,
        "ink_centroid_y": centroid_y,
        "large_component_fraction": float(np.mean(large_mask)),
        "largest_component_fraction": (
            max(large_sizes, default=0) / float(size * size)
        ),
        "large_component_count": float(len(large_sizes)),
        "large_component_centroid_x": large_centroid_x,
        "large_component_centroid_y": large_centroid_y,
        "large_horizontal_symmetry": large_horizontal_symmetry,
        "large_vertical_symmetry": large_vertical_symmetry,
    }
    return {name: round(value, 8) for name, value in values.items()}


def crop_patch(
    image: Image.Image,
    x: float,
    y: float,
    radius: int,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    box = (
        max(0, int(round(x - radius))),
        max(0, int(round(y - radius))),
        min(image.width, int(round(x + radius))),
        min(image.height, int(round(y + radius))),
    )
    return image.crop(box), box


def calibrator_starts(anchors: dict) -> dict[str, int]:
    starts: dict[str, list[int]] = {tier: [] for tier in TIERS}
    for folio in anchors["folios"].values():
        for tier in TIERS:
            alignment = folio.get("alignment", {}).get(tier, {})
            if alignment.get("direction") == "cw":
                starts[tier].append(int(alignment["start_clock"]))
    return {
        tier: Counter(values).most_common(1)[0][0]
        for tier, values in starts.items()
    }


def alignment_for(
    folio: str,
    tier: str,
    anchors: dict,
    inferred_starts: dict[str, int],
) -> dict:
    anchored = anchors["folios"].get(folio)
    if anchored and tier in anchored.get("alignment", {}):
        row = anchored["alignment"][tier]
        return {
            "start_clock": int(row["start_clock"]),
            "direction": row["direction"],
            "method": "pixel_read_anchors",
            "confidence": "high",
            "anchors_matched": int(row.get("anchors_matched", 0)),
            "anchors_contradicting": int(row.get("anchors_contradicting", 0)),
            "phase_uncertainty_nodes": [0],
        }
    return {
        "start_clock": inferred_starts[tier],
        "direction": "cw",
        "method": "modal_start_from_four_anchored_diagrams",
        "confidence": "low",
        "anchors_matched": 0,
        "anchors_contradicting": 0,
        "phase_uncertainty_nodes": [-1, 0, 1],
    }


def angle_for(start_clock: float, index: int, count: int, direction: str) -> float:
    start = math.radians(start_clock * 30.0 - 90.0)
    step = 2 * math.pi * index / count
    return start + step if direction == "cw" else start - step


def draw_qc(
    image: Image.Image,
    spec: PanelSpec,
    records: list[dict],
    output: Path,
) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    cx, cy = spec.center
    colors = {"outer": (190, 40, 30), "inner": (20, 90, 190)}
    for tier, radii in (
        ("outer", spec.outer_radius),
        ("inner", spec.inner_radius),
    ):
        rx, ry = radii
        draw.ellipse(
            (cx - rx, cy - ry, cx + rx, cy + ry),
            outline=colors[tier],
            width=max(3, round(min(image.size) / 700)),
        )
    node_radius = max(9, round(min(image.size) / 180))
    font = ImageFont.load_default()
    for record in records:
        x, y = record["image_xy"]
        color = colors[record["tier"]]
        draw.ellipse(
            (x - node_radius, y - node_radius, x + node_radius, y + node_radius),
            fill=(255, 255, 255),
            outline=color,
            width=3,
        )
        text = str(record["cyclic_index"] + 1)
        draw.text(
            (x + node_radius + 2, y - node_radius),
            text,
            fill=color,
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
    overlay.thumbnail((1500, 1500), Image.Resampling.LANCZOS)
    overlay.save(output, quality=92)


def build(output: Path) -> dict:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    anchors = json.loads(ANCHORS.read_text(encoding="utf-8"))
    inferred_starts = calibrator_starts(anchors)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "built": "2026-07-23",
        "method": (
            "clockwise transcription order plus scan-calibrated ring geometry; "
            "four phases are pixel-read anchors, eight use the anchored modal start"
        ),
        "feature_schema": {
            "pixel_features": [
                "ink_fraction",
                "red_fraction",
                "green_fraction",
                "blue_fraction",
                "gold_fraction",
                "edge_density",
                "contrast",
                "entropy",
                "horizontal_symmetry",
                "vertical_symmetry",
                "ink_centroid_x",
                "ink_centroid_y",
                "large_component_fraction",
                "largest_component_fraction",
                "large_component_count",
                "large_component_centroid_x",
                "large_component_centroid_y",
                "large_horizontal_symmetry",
                "large_vertical_symmetry",
            ],
            "graph_features": [
                "cyclic_index",
                "cyclic_fraction",
                "tier_depth",
                "degree",
                "ring_size",
            ],
            "patch_normalized_px": FEATURE_SIZE,
            "panel_normalized_px": PANEL_FEATURE_SIZE,
        },
        "calibration": {
            "anchor_folios": sorted(anchors["folios"]),
            "inferred_start_clock": inferred_starts,
            "inferred_phase_uncertainty_nodes": [-1, 0, 1],
        },
        "folios": {},
    }

    for folio, rows in labels.items():
        if folio not in PANEL_SPECS:
            raise RuntimeError(f"no panel geometry for {folio}")
        spec = PANEL_SPECS[folio]
        panel, source = load_panel(spec)
        patch_radius = max(72, round(min(panel.size) * 0.045))
        context_radius = max(110, round(min(panel.size) * 0.075))
        folio_records: list[dict] = []
        alignments = {}

        for tier in TIERS:
            tier_rows = [row for row in rows if row["tier"] == tier]
            if not tier_rows:
                continue
            alignment = alignment_for(folio, tier, anchors, inferred_starts)
            alignments[tier] = alignment
            radius_x, radius_y = (
                spec.outer_radius if tier == "outer" else spec.inner_radius
            )
            count = len(tier_rows)
            for index, row in enumerate(tier_rows):
                angle = angle_for(
                    alignment["start_clock"],
                    index,
                    count,
                    alignment["direction"],
                )
                x = spec.center[0] + radius_x * math.cos(angle)
                y = spec.center[1] + radius_y * math.sin(angle)
                patch, box = crop_patch(panel, x, y, patch_radius)
                context_patch, context_box = crop_patch(
                    panel, x, y, context_radius
                )
                tokens = label_tokens(row["label"])
                folio_records.append({
                    "locus": int(row["locus"]),
                    "label_raw": row["label"],
                    "label_tokens": tokens,
                    "label_primary": tokens[0] if tokens else "",
                    "tier": tier,
                    "cyclic_index": index,
                    "cyclic_fraction": round(index / count, 8),
                    "tier_depth": 0 if tier == "outer" else 1,
                    "degree": 2,
                    "ring_size": count,
                    "clock": round((alignment["start_clock"] + 12 * index / count) % 12, 4),
                    "angle_radians": round(angle % (2 * math.pi), 8),
                    "image_xy": [round(x, 3), round(y, 3)],
                    "patch_box": list(box),
                    "patch_radius": patch_radius,
                    "visual": visual_features(patch, FEATURE_SIZE),
                    "context_patch_box": list(context_box),
                    "context_patch_radius": context_radius,
                    "context_visual": visual_features(
                        context_patch, FEATURE_SIZE
                    ),
                    "alignment_method": alignment["method"],
                    "alignment_confidence": alignment["confidence"],
                })

        panel_path = OUTPUT_DIR / f"{folio}_panel.jpg"
        review_panel = panel.copy()
        review_panel.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        review_panel.save(panel_path, quality=90)
        qc_path = OUTPUT_DIR / f"{folio}_qc.jpg"
        draw_qc(panel, spec, folio_records, qc_path)
        outside = [row for row in rows if row["tier"] not in TIERS]
        result["folios"][folio] = {
            "source": str(source.relative_to(ROOT)),
            "source_sha256": file_sha256(source),
            "yale_canvas_id": spec.canvas_id,
            "source_crop": list(spec.crop) if spec.crop else None,
            "panel_image": str(panel_path.relative_to(ROOT)),
            "panel_image_size": list(review_panel.size),
            "qc_overlay": str(qc_path.relative_to(ROOT)),
            "panel_size": list(panel.size),
            "center": list(spec.center),
            "ring_radii": {
                "outer": list(spec.outer_radius),
                "inner": list(spec.inner_radius),
            },
            "patch_radius": patch_radius,
            "context_patch_radius": context_radius,
            "panel_visual": visual_features(panel, PANEL_FEATURE_SIZE),
            "alignment": alignments,
            "ring_label_count": len(folio_records),
            "outside_label_count": len(outside),
            "outside_labels": outside,
            "records": folio_records,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.output)
    folios = len(result["folios"])
    records = sum(row["ring_label_count"] for row in result["folios"].values())
    anchored = sum(
        1
        for row in result["folios"].values()
        if row["alignment"]["outer"]["method"] == "pixel_read_anchors"
    )
    print(f"wrote {args.output}")
    print(f"folios={folios} ring_nodes={records} anchored={anchored} inferred={folios-anchored}")


if __name__ == "__main__":
    main()
