#!/usr/bin/env python3
"""
Build foldout-aware, complete-object visual embeddings for Voynich pages.

Public Surya and HORAE detectors propose figure regions.  SAM 2.1 expands
those boxes into object masks; directional pigment components provide a
domain-specific proposal fallback without using Voynich text labels.
Filling object-mask holes before reapplying the page foreground preserves
unpainted figures, while a narrow pigment support band retains thin stems.

The output uses the feature-key contract of visual_state_axis_gate.py so the
previously frozen causal discriminator can be reused without changing its
text model, splits, hyperparameter grid, or relabeling null.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Sequence

import cv2
import fitz
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from ultralytics import SAM, YOLO

import build_guarded_dinov2_embeddings as dino
import build_page_illustration_features as guarded


ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "images" / "facsimile" / "Voynich_Manuscript.pdf"
CANVASES = ROOT / "data" / "iiif" / "iiif_canvases.json"
CORPUS = ROOT / "data" / "corpus" / "corpus.json"
FOLDOUT_MAP = ROOT / "data" / "grounding" / "foldout_visual_map.json"
CACHE = ROOT / ".cache" / "public_object_attack"
SURYA_MODEL = CACHE / "surya_layout2"
HORAE_MODEL = CACHE / "horae_yolo12s" / "best.pt"
SAM_MODEL = CACHE / "sam2" / "sam2.1_t.pt"
DINO_REPOSITORY = CACHE / "dinov2" / "repository"
DINO_WEIGHTS = CACHE / "dinov2" / "dinov2_vits14_pretrain.pth"
OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_public_object_embeddings.json"
)
QC = ROOT / "images" / "crops" / "generator_inversion_public_objects.png"

MAX_IMAGE_SIDE = 1536
MAX_PROPOSALS = 5
SURYA_THRESHOLD = 0.12
HORAE_THRESHOLD = 0.10
VISUAL_LABELS = {
    "Image",
    "Figure",
    "Diagram",
    "Chemical-Block",
    "Complex-Block",
}
HORAE_CLASSES = {
    "miniature",
    "marginal medallion",
    "marginal decoration",
}
QC_FOLIOS = (
    "f23r",
    "f57v",
    "f68r2",
    "f75r",
    "f88r",
    "f90v1",
    "f103r",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def bounded_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    if max(image.size) <= MAX_IMAGE_SIDE:
        return image
    scale = MAX_IMAGE_SIDE / max(image.size)
    return image.resize(
        (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )


def crop_normalized(
    image: Image.Image,
    crop: Sequence[float],
) -> Image.Image:
    left, top, right, bottom = crop
    box = (
        max(0, round(left * image.width)),
        max(0, round(top * image.height)),
        min(image.width, round(right * image.width)),
        min(image.height, round(bottom * image.height)),
    )
    return bounded_image(image.crop(box))


def target_map(
    labels: Sequence[str],
    corpus: Path,
    mapping: Path,
) -> tuple[dict[int, list[dict]], dict]:
    corpus_folios = set(
        json.loads(corpus.read_text(encoding="utf-8"))["folios"]
    )
    foldouts = json.loads(mapping.read_text(encoding="utf-8"))
    exceptions = {
        int(page["pdf_page_index"]): page
        for page in foldouts["pages"]
    }
    result: dict[int, list[dict]] = {}
    targets = []
    for index, label in enumerate(labels):
        if index in exceptions:
            page = exceptions[index]
            if page["source_label"] != label:
                raise ValueError(
                    f"foldout label mismatch at {index}: "
                    f"{page['source_label']} != {label}"
                )
            result[index] = []
            for target in page["targets"]:
                record = {
                    "folio": str(target["folio"]),
                    "crop": [float(value) for value in target["crop"]],
                    "foldout_mapped": True,
                    "shared_canvas": bool(
                        target.get("shared_canvas", False)
                    ),
                }
                result[index].append(record)
                targets.append(record["folio"])
        elif re.fullmatch(r"\d+[rv]", label):
            folio = f"f{label}"
            if folio in corpus_folios:
                result[index] = [{
                    "folio": folio,
                    "crop": [0.0, 0.0, 1.0, 1.0],
                    "foldout_mapped": False,
                    "shared_canvas": False,
                }]
                targets.append(folio)
    counts = Counter(targets)
    duplicate_targets = sorted(
        folio for folio, count in counts.items() if count > 1
    )
    missing_targets = sorted(corpus_folios - set(targets))
    extra_targets = sorted(set(targets) - corpus_folios)
    if duplicate_targets or missing_targets or extra_targets:
        raise ValueError({
            "duplicate_targets": duplicate_targets,
            "missing_targets": missing_targets,
            "extra_targets": extra_targets,
        })
    return result, {
        "corpus_folios": len(corpus_folios),
        "mapped_visual_targets": len(targets),
        "unique_visual_targets": len(set(targets)),
        "foldout_target_sides": sum(
            row["foldout_mapped"]
            for values in result.values()
            for row in values
        ),
        "shared_canvas_target_sides": sum(
            row["shared_canvas"]
            for values in result.values()
            for row in values
        ),
        "unmapped_corpus_folios": missing_targets,
        "duplicate_target_folios": duplicate_targets,
    }


def clip_box(
    box: Sequence[float],
    width: int,
    height: int,
) -> list[float]:
    left, top, right, bottom = box
    return [
        max(0.0, min(float(width - 1), float(left))),
        max(0.0, min(float(height - 1), float(top))),
        max(1.0, min(float(width), float(right))),
        max(1.0, min(float(height), float(bottom))),
    ]


def box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return max(
        0.0,
        min(first[2], second[2]) - max(first[0], second[0]),
    ) * max(
        0.0,
        min(first[3], second[3]) - max(first[1], second[1]),
    )


def expand_box(
    box: Sequence[float],
    width: int,
    height: int,
    fraction: float = 0.02,
) -> list[float]:
    padding_x = width * fraction
    padding_y = height * fraction
    return clip_box(
        [
            box[0] - padding_x,
            box[1] - padding_y,
            box[2] + padding_x,
            box[3] + padding_y,
        ],
        width,
        height,
    )


def valid_proposal(
    box: Sequence[float],
    width: int,
    height: int,
) -> bool:
    fraction = box_area(box) / (width * height)
    box_width = (box[2] - box[0]) / width
    box_height = (box[3] - box[1]) / height
    return (
        0.002 <= fraction <= 0.92
        and box_width >= 0.025
        and box_height >= 0.025
    )


def foreground_masks(
    image: Image.Image,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), 8.0)
    local_ink = (local - gray > 13.0) & (gray < 235.0)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    channels = rgb.astype(np.int16)
    red = channels[..., 0]
    green = channels[..., 1]
    blue = channels[..., 2]
    green_pigment = (
        (green - blue > 14)
        & (green - red > -6)
        & (red - green < 18)
    )
    blue_pigment = (
        (blue - red > 5)
        & (blue - green > 3)
    )
    red_pigment = (
        (red - green > 18)
        & (np.abs(green - blue) < 18)
    )
    saturation = hsv[..., 1]
    pigment = (
        (green_pigment & (saturation > 48))
        | (blue_pigment & (saturation > 60))
        | (red_pigment & (saturation > 115))
    ) & (
        (hsv[..., 2] > 55)
        & (hsv[..., 2] < 248)
    )
    foreground = local_ink | pigment
    margin_x = max(1, round(image.width * 0.012))
    margin_y = max(1, round(image.height * 0.012))
    foreground[:margin_y, :] = False
    foreground[-margin_y:, :] = False
    foreground[:, :margin_x] = False
    foreground[:, -margin_x:] = False
    pigment[:margin_y, :] = False
    pigment[-margin_y:, :] = False
    pigment[:, :margin_x] = False
    pigment[:, -margin_x:] = False
    return foreground, pigment, rgb


def component_proposals(
    mask: np.ndarray,
    source: str,
    score: float,
    minimum_area_fraction: float,
    minimum_height_fraction: float,
) -> list[dict]:
    height, width = mask.shape
    working = mask.astype(np.uint8)
    if source == "pigment":
        working = cv2.morphologyEx(
            working,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        working = cv2.dilate(
            working,
            np.ones((3, 3), dtype=np.uint8),
        )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        working,
        connectivity=8,
    )
    result = []
    for index in range(1, count):
        left, top, box_width, box_height, area = [
            int(value) for value in stats[index]
        ]
        if area / (width * height) < minimum_area_fraction:
            continue
        if box_height / height < minimum_height_fraction:
            continue
        if source == "pigment" and (
            left < width * 0.025
            or top < height * 0.025
            or left + box_width > width * 0.975
            or top + box_height > height * 0.975
        ):
            continue
        box = expand_box(
            [left, top, left + box_width, top + box_height],
            width,
            height,
            fraction=0.04 if source == "pigment" else 0.025,
        )
        if valid_proposal(box, width, height):
            result.append({
                "source": source,
                "label": source,
                "score": score,
                "box": box,
            })
    return result


def detector_proposals(
    image: Image.Image,
    surya,
    horae: YOLO,
) -> list[dict]:
    width, height = image.size
    proposals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        surya_rows = surya.detect(
            [image],
            threshold=SURYA_THRESHOLD,
            batch_size=1,
        )[0]
    for row in surya_rows:
        if row["label"] not in VISUAL_LABELS:
            continue
        box = clip_box(row["bbox"], width, height)
        if valid_proposal(box, width, height):
            proposals.append({
                "source": "surya",
                "label": str(row["label"]),
                "score": float(row["score"]),
                "box": box,
            })
    prediction = horae.predict(
        source=image,
        imgsz=640,
        conf=HORAE_THRESHOLD,
        iou=0.6,
        device="cpu",
        verbose=False,
    )[0]
    if prediction.boxes is not None:
        for class_id, confidence, raw_box in zip(
            prediction.boxes.cls,
            prediction.boxes.conf,
            prediction.boxes.xyxy,
        ):
            label = str(horae.names[int(class_id)])
            if label not in HORAE_CLASSES:
                continue
            box = clip_box(raw_box.tolist(), width, height)
            if valid_proposal(box, width, height):
                proposals.append({
                    "source": "horae",
                    "label": label,
                    "score": float(confidence),
                    "box": box,
                })
    return proposals


def merge_proposals(
    proposals: Sequence[dict],
    width: int,
    height: int,
) -> list[dict]:
    source_priority = {
        "surya": 4,
        "pigment": 3,
        "horae": 2,
    }
    ordered = sorted(
        proposals,
        key=lambda row: (
            source_priority.get(row["source"], 0),
            row["score"],
            math.sqrt(box_area(row["box"]) / (width * height)),
        ),
        reverse=True,
    )
    merged: list[dict] = []
    for proposal in ordered:
        box = expand_box(proposal["box"], width, height)
        absorbed = False
        for target in merged:
            overlap = intersection(box, target["box"])
            smaller = min(box_area(box), box_area(target["box"]))
            union = box_area(box) + box_area(target["box"]) - overlap
            if (
                overlap / max(smaller, 1.0) >= 0.62
                or overlap / max(union, 1.0) >= 0.34
            ):
                union_box = [
                    min(target["box"][0], box[0]),
                    min(target["box"][1], box[1]),
                    max(target["box"][2], box[2]),
                    max(target["box"][3], box[3]),
                ]
                spatial_expansion = (
                    box_area(union_box)
                    / max(box_area(target["box"]), 1.0)
                )
                same_source = proposal["source"] in target["sources"]
                if (
                    proposal["source"] == "pigment"
                    or (
                        not same_source
                        and spatial_expansion <= 1.35
                    )
                ):
                    target["box"] = union_box
                target["sources"] = sorted(set(
                    target["sources"] + [proposal["source"]]
                ))
                target["labels"] = sorted(set(
                    target["labels"] + [proposal["label"]]
                ))
                target["score"] = max(
                    target["score"], proposal["score"]
                )
                absorbed = True
                break
        if not absorbed:
            merged.append({
                "box": box,
                "sources": [proposal["source"]],
                "labels": [proposal["label"]],
                "score": proposal["score"],
            })
    merged.sort(
        key=lambda row: (
            row["score"]
            + math.sqrt(box_area(row["box"]) / (width * height)),
            box_area(row["box"]),
        ),
        reverse=True,
    )
    return merged[:MAX_PROPOSALS]


def box_mask(
    box: Sequence[float],
    shape: tuple[int, int],
) -> np.ndarray:
    height, width = shape
    left, top, right, bottom = [
        int(round(value)) for value in box
    ]
    result = np.zeros(shape, dtype=bool)
    result[
        max(0, top):min(height, bottom),
        max(0, left):min(width, right),
    ] = True
    return result


def substantial_components(
    mask: np.ndarray,
    retain_small: bool = False,
) -> np.ndarray:
    height, width = mask.shape
    minimum_area = max(32, round(width * height * 0.00012))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    result = np.zeros_like(mask)
    for index in range(1, count):
        left, top, box_width, box_height, area = [
            int(value) for value in stats[index]
        ]
        if not retain_small and area < minimum_area:
            continue
        right = left + box_width
        bottom = top + box_height
        vertical_edge = (
            (left < width * 0.02 or right > width * 0.98)
            and box_height > height * 0.35
            and box_width < width * 0.14
        )
        horizontal_edge = (
            (top < height * 0.02 or bottom > height * 0.98)
            and box_width > width * 0.50
            and box_height < height * 0.16
        )
        page_frame = (
            left < width * 0.02
            and right > width * 0.98
            and top < height * 0.02
            and bottom > height * 0.98
            and area / max(1, box_width * box_height) < 0.25
        )
        if vertical_edge or horizontal_edge or page_frame:
            continue
        result[labels == index] = True
    return result


def complete_sam_mask(
    image: Image.Image,
    proposals: Sequence[dict],
    sam: SAM,
    foreground: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    shape = (image.height, image.width)
    if not proposals:
        return np.zeros(shape, dtype=bool), []
    boxes = [row["box"] for row in proposals]
    prediction = sam.predict(
        source=image,
        bboxes=boxes,
        device="cpu",
        retina_masks=True,
        verbose=False,
    )[0]
    if prediction.masks is None:
        raw_masks = []
    else:
        raw_masks = [
            mask > 0.5
            for mask in prediction.masks.data.cpu().numpy()
        ]
    union = np.zeros(shape, dtype=bool)
    audits = []
    for index, proposal in enumerate(proposals):
        proposal_mask = box_mask(
            expand_box(
                proposal["box"],
                image.width,
                image.height,
                fraction=0.025,
            ),
            shape,
        )
        raw = (
            raw_masks[index]
            if index < len(raw_masks)
            else np.zeros(shape, dtype=bool)
        )
        raw &= proposal_mask
        raw_fraction = (
            float(raw.sum()) / max(1.0, box_area(proposal["box"]))
        )
        complement = proposal_mask & ~raw
        raw_foreground_density = (
            float(foreground[raw].mean()) if raw.any() else 0.0
        )
        complement_foreground_density = (
            float(foreground[complement].mean())
            if complement.any()
            else 0.0
        )
        complement_foreground_richer = (
            raw_fraction > 0.55
            and complement_foreground_density
            > raw_foreground_density * 1.35
        )
        completed = substantial_components(
            raw & foreground,
            retain_small="Diagram" in proposal["labels"],
        )
        unsupported = not completed.any()
        if not unsupported:
            completed = ndimage.binary_fill_holes(completed)
            radius = max(2, round(min(image.size) * 0.004))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (radius * 2 + 1, radius * 2 + 1),
            )
            completed = cv2.morphologyEx(
                completed.astype(np.uint8),
                cv2.MORPH_CLOSE,
                kernel,
            ).astype(bool)
            completed = cv2.dilate(
                completed.astype(np.uint8),
                kernel,
            ).astype(bool)
            completed &= proposal_mask
        union |= completed
        audits.append({
            **proposal,
            "raw_sam_fraction_of_box": raw_fraction,
            "raw_foreground_density": raw_foreground_density,
            "complement_foreground_density": (
                complement_foreground_density
            ),
            "complement_foreground_richer": (
                complement_foreground_richer
            ),
            "unsupported_mask": unsupported,
            "completed_fraction_of_page": float(completed.mean()),
        })
    return union, audits


def grid_density(mask: np.ndarray, grid: int = 8) -> list[float]:
    return [
        round(float(cell.mean()), 8)
        for row in np.array_split(mask, grid, axis=0)
        for cell in np.array_split(row, grid, axis=1)
    ]


def projection(mask: np.ndarray, bins: int = 16) -> list[float]:
    return [
        round(float(row.mean()), 8)
        for row in np.array_split(mask, bins, axis=0)
    ] + [
        round(float(column.mean()), 8)
        for column in np.array_split(mask, bins, axis=1)
    ]


def skeletonize(mask: np.ndarray) -> np.ndarray:
    working = mask.astype(np.uint8) * 255
    skeleton = np.zeros_like(working)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(working):
        eroded = cv2.erode(working, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton |= cv2.subtract(working, opened)
        working = eroded
    return skeleton > 0


def topology_features(
    object_mask: np.ndarray,
    ink_mask: np.ndarray,
    pigment_mask: np.ndarray,
    proposals: Sequence[dict],
) -> list[float]:
    size = (256, 256)
    object_small = cv2.resize(
        object_mask.astype(np.uint8),
        size,
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    ink_small = cv2.resize(
        ink_mask.astype(np.uint8),
        size,
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    pigment_small = cv2.resize(
        pigment_mask.astype(np.uint8),
        size,
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    skeleton = skeletonize(ink_small)
    neighbors = cv2.filter2D(
        skeleton.astype(np.uint8),
        -1,
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
    ) - skeleton.astype(np.uint8)
    component_count = max(
        0,
        cv2.connectedComponents(
            ink_small.astype(np.uint8),
            connectivity=8,
        )[0] - 1,
    )
    yy, xx = np.indices(object_small.shape)
    if object_small.any():
        points_y, points_x = np.nonzero(object_small)
        centroid_x = float(xx[object_small].mean() / 255 * 2 - 1)
        centroid_y = float(yy[object_small].mean() / 255 * 2 - 1)
        width = float((points_x.max() - points_x.min() + 1) / 256)
        height = float((points_y.max() - points_y.min() + 1) / 256)
    else:
        centroid_x = centroid_y = width = height = 0.0
    global_values = [
        float(bool(proposals)),
        float(len(proposals) / MAX_PROPOSALS),
        float(object_small.mean()),
        float(ink_small.mean()),
        float(pigment_small.mean()),
        float(skeleton.mean()),
        float(component_count / 128),
        float(np.sum(skeleton & (neighbors == 1)) / 256),
        float(np.sum(skeleton & (neighbors >= 3)) / 256),
        centroid_x,
        centroid_y,
        width,
        height,
        float(math.log((width + 1e-4) / (height + 1e-4))),
        1.0 - float(np.mean(object_small != np.fliplr(object_small))),
        1.0 - float(np.mean(object_small != np.flipud(object_small))),
    ]
    return [
        round(value, 8) for value in global_values
    ] + (
        grid_density(object_small)
        + grid_density(ink_small)
        + grid_density(skeleton)
        + grid_density(pigment_small)
        + projection(ink_small)
        + projection(skeleton)
    )


def object_views(
    image: Image.Image,
    object_mask: np.ndarray,
    foreground: np.ndarray,
    pigment: np.ndarray,
) -> tuple[dict[str, Image.Image], np.ndarray]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    object_ink = foreground & object_mask
    guarded_rgb = np.full_like(rgb, 255)
    guarded_rgb[object_mask] = rgb[object_mask]
    silhouette = np.full_like(rgb, 255)
    silhouette[object_ink] = 0
    if object_mask.any():
        points_y, points_x = np.nonzero(object_mask)
        padding = max(4, round(min(image.size) * 0.01))
        left = max(0, int(points_x.min()) - padding)
        right = min(image.width, int(points_x.max()) + padding + 1)
        top = max(0, int(points_y.min()) - padding)
        bottom = min(image.height, int(points_y.max()) + padding + 1)
    else:
        left, top, right, bottom = 0, 0, image.width, image.height
    return {
        "full_rgb": Image.fromarray(guarded_rgb),
        "tight_rgb": Image.fromarray(guarded_rgb).crop(
            (left, top, right, bottom)
        ),
        "full_silhouette": Image.fromarray(silhouette),
        "tight_silhouette": Image.fromarray(silhouette).crop(
            (left, top, right, bottom)
        ),
    }, object_ink


def proposal_overlay(
    image: Image.Image,
    proposals: Sequence[dict],
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    colors = {
        "surya": (20, 160, 70),
        "pigment": (20, 100, 220),
        "horae": (220, 80, 30),
        "tall_ink": (160, 60, 190),
    }
    for proposal in proposals:
        source = "+".join(proposal["sources"])
        color = colors.get(proposal["sources"][0], (30, 30, 30))
        draw.rectangle(proposal["box"], outline=color, width=4)
        draw.text(
            (proposal["box"][0] + 4, proposal["box"][1] + 4),
            source,
            fill=color,
            stroke_width=2,
            stroke_fill="white",
        )
    return overlay


def save_qc(
    qc: Path,
    rows: dict[str, tuple[Image.Image, ...]],
) -> None:
    cell = 224
    label_height = 20
    montage = Image.new(
        "RGB",
        (cell * 4, (cell + label_height) * len(QC_FOLIOS)),
        "white",
    )
    draw = ImageDraw.Draw(montage)
    headings = ("source", "proposals", "object_rgb", "topology")
    for row_index, folio in enumerate(QC_FOLIOS):
        if folio not in rows:
            continue
        for column, image in enumerate(rows[folio]):
            rendered = ImageOps.pad(
                image.convert("RGB"),
                (cell, cell),
                method=Image.Resampling.LANCZOS,
                color=(255, 255, 255),
            )
            left = column * cell
            top = row_index * (cell + label_height)
            montage.paste(rendered, (left, top))
            draw.text(
                (left + 3, top + cell + 2),
                f"{folio} {headings[column]}",
                fill="black",
            )
    qc.parent.mkdir(parents=True, exist_ok=True)
    montage.save(qc)


def build(
    pdf: Path,
    canvases: Path,
    corpus: Path,
    foldout_map: Path,
    output: Path,
    qc: Path,
    cache: Path,
    batch_size: int,
    progress: bool,
    only_folios: set[str] | None = None,
) -> dict:
    from surya.common.rfdetr_torch import RfDetrTorch

    labels = guarded.canvas_order(canvases)
    targets, mapping_audit = target_map(labels, corpus, foldout_map)
    if only_folios:
        targets = {
            index: [
                target for target in values
                if target["folio"] in only_folios
            ]
            for index, values in targets.items()
        }
        targets = {
            index: values for index, values in targets.items() if values
        }
        mapping_audit["test_only_folios"] = sorted(only_folios)
    surya_path = cache / "surya_layout2"
    horae_path = cache / "horae_yolo12s" / "best.pt"
    sam_path = cache / "sam2" / "sam2.1_t.pt"
    dino_repo = cache / "dinov2" / "repository"
    dino_weights = cache / "dinov2" / "dinov2_vits14_pretrain.pth"
    required = (
        surya_path / "config.json",
        surya_path / "rfdetr_layout.pth",
        horae_path,
        sam_path,
        dino_repo / ".git",
        dino_weights,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "run setup_public_object_attack.sh first: " + ", ".join(missing)
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        surya = RfDetrTorch(
            str(surya_path),
            num_threads=4,
            device="cpu",
        )
    horae = YOLO(str(horae_path))
    sam = SAM(str(sam_path))
    embedding_model = dino.load_model(
        str(dino_repo),
        str(dino_weights),
    )

    document = fitz.open(pdf)
    if len(labels) != document.page_count:
        raise ValueError(
            f"canvas/PDF mismatch: {len(labels)} != {document.page_count}"
        )
    records = []
    pending: list[tuple[int, str, torch.Tensor]] = []
    qc_rows: dict[str, tuple[Image.Image, ...]] = {}
    proposal_sources: Counter = Counter()
    detector_pages: Counter = Counter()

    def flush() -> None:
        if not pending:
            return
        cls, patches = dino.embed_batch(
            embedding_model,
            [item[2] for item in pending],
        )
        for index, (record_index, view, _tensor) in enumerate(pending):
            features = records[record_index]["features"]
            features[f"{view}_cls"] = [
                round(float(value), 8) for value in cls[index]
            ]
            features[f"{view}_patch_mean"] = [
                round(float(value), 8) for value in patches[index]
            ]
        pending.clear()

    for pdf_index in sorted(targets):
        page = document[pdf_index]
        images = page.get_images(full=True)
        if not images:
            raise ValueError(f"no raster image on PDF page {pdf_index}")
        payload = document.extract_image(images[0][0])
        source_image = Image.open(
            io.BytesIO(payload["image"])
        ).convert("RGB")
        for target in targets[pdf_index]:
            image = crop_normalized(source_image, target["crop"])
            foreground, pigment, _rgb = foreground_masks(image)
            raw_proposals = detector_proposals(image, surya, horae)
            raw_proposals.extend(component_proposals(
                pigment,
                "pigment",
                0.50,
                minimum_area_fraction=0.00045,
                minimum_height_fraction=0.025,
            ))
            proposals = merge_proposals(
                raw_proposals,
                image.width,
                image.height,
            )
            object_mask, proposal_audit = complete_sam_mask(
                image,
                proposals,
                sam,
                foreground,
            )
            pigment_region = np.zeros_like(pigment)
            for proposal in proposals:
                if "pigment" in proposal["sources"]:
                    pigment_region |= box_mask(
                        proposal["box"],
                        pigment.shape,
                    )
            selected_pigment = pigment & pigment_region
            support_radius = max(
                2,
                round(min(image.size) * 0.006),
            )
            support_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (support_radius * 2 + 1, support_radius * 2 + 1),
            )
            pigment_support = cv2.dilate(
                selected_pigment.astype(np.uint8),
                support_kernel,
            ).astype(bool)
            object_mask |= selected_pigment
            object_mask |= foreground & pigment_support
            views, object_ink = object_views(
                image,
                object_mask,
                foreground,
                pigment,
            )
            topology = topology_features(
                object_mask,
                object_ink,
                pigment & object_mask,
                proposals,
            )
            leakage = foreground & ~object_mask
            for proposal in proposals:
                for source in proposal["sources"]:
                    proposal_sources[source] += 1
                    detector_pages[(target["folio"], source)] += 1
            record_index = len(records)
            records.append({
                "folio": target["folio"],
                "pdf_page_index": pdf_index,
                "source_label": labels[pdf_index],
                "source_crop": target["crop"],
                "foldout_mapped": target["foldout_mapped"],
                "shared_canvas": target["shared_canvas"],
                "exclude_from_gate": target["shared_canvas"],
                "eligible_illustration": bool(proposals),
                "audit": {
                    "image_size": list(image.size),
                    "raw_proposals": len(raw_proposals),
                    "raw_proposal_details": raw_proposals,
                    "merged_proposals": len(proposals),
                    "object_fraction": round(
                        float(object_mask.mean()), 8
                    ),
                    "object_ink_fraction": round(
                        float(object_ink.mean()), 8
                    ),
                    "pigment_fraction": round(
                        float(pigment.mean()), 8
                    ),
                    "proposals": proposal_audit,
                },
                "features": {
                    "combined_guarded": topology,
                },
                "leakage_diagnostic": {
                    "small_component_grid": grid_density(leakage),
                },
            })
            if target["folio"] in QC_FOLIOS:
                qc_rows[target["folio"]] = (
                    image.copy(),
                    proposal_overlay(image, proposals),
                    views["full_rgb"].copy(),
                    views["full_silhouette"].copy(),
                )
            for view in dino.VIEWS:
                pending.append((
                    record_index,
                    view,
                    dino.tensor(views[view]),
                ))
                if len(pending) >= batch_size:
                    flush()
            if progress and len(records) % 20 == 0:
                print(
                    f"processed object pages={len(records)}",
                    flush=True,
                )
    flush()
    document.close()
    save_qc(qc, qc_rows)

    expected = {
        f"{view}_{pooling}"
        for view in dino.VIEWS
        for pooling in ("cls", "patch_mean")
    } | {"combined_guarded"}
    for record in records:
        if set(record["features"]) != expected:
            raise ValueError(
                f"incomplete features for {record['folio']}: "
                f"{set(record['features'])}"
            )
    result = {
        "experiment": "public_complete_object_dinov2_embeddings",
        "parameters": {
            "maximum_image_side": MAX_IMAGE_SIDE,
            "surya_visual_labels": sorted(VISUAL_LABELS),
            "surya_confidence_threshold": SURYA_THRESHOLD,
            "horae_classes": sorted(HORAE_CLASSES),
            "horae_confidence_threshold": HORAE_THRESHOLD,
            "maximum_merged_proposals": MAX_PROPOSALS,
            "proposal_sources": [
                "Surya visual boxes",
                "HORAE decoration boxes",
                "directional pigment connected components",
            ],
            "mask_completion": (
                "SAM box mask; binary hole fill; close and dilate; "
                "clip to padded proposal; retain foreground in a narrow "
                "band around selected pigment; reapply foreground for "
                "topology"
            ),
            "dino_views": list(dino.VIEWS),
            "topology_features": (
                "global mask/ink/skeleton graph statistics; 8x8 grids "
                "for object, ink, skeleton, pigment; 16-bin ink and "
                "skeleton projections"
            ),
            "text_guard": (
                "RGB and silhouette are white outside completed object "
                "masks; excluded leakage diagnostic uses outside-mask "
                "foreground only"
            ),
        },
        "models": {
            "surya_layout2": {
                "repository": "https://github.com/datalab-to/surya",
                "weights": "https://huggingface.co/datalab-to/surya_layout2",
                "revision": (
                    "0aee81d5fd9275c0582e545bf3a56944b1e75679"
                ),
                "weights_sha256": sha256(
                    surya_path / "rfdetr_layout.pth"
                ),
                "license": "AI Pubs OpenRAIL-M",
            },
            "horae_yolo12s": {
                "record": "https://doi.org/10.5281/zenodo.17279775",
                "weights_sha256": sha256(horae_path),
                "training_pages": 10187,
                "license": "CC BY 4.0",
            },
            "sam2_1_hiera_tiny": {
                "repository": "https://github.com/facebookresearch/sam2",
                "weights": (
                    "https://dl.fbaipublicfiles.com/"
                    "segment_anything_2/092824/"
                    "sam2.1_hiera_tiny.pt"
                ),
                "weights_sha256": sha256(sam_path),
                "license": "Apache-2.0",
            },
            "dinov2_vits14": {
                "repository": dino.MODEL_REPOSITORY,
                "weights": dino.MODEL_WEIGHTS,
                "weights_sha256": sha256(dino_weights),
                "license": "Apache-2.0",
            },
        },
        "assets": {
            asset_name(pdf): sha256(pdf),
            asset_name(canvases): sha256(canvases),
            asset_name(corpus): sha256(corpus),
            asset_name(foldout_map): sha256(foldout_map),
        },
        "mapping_audit": mapping_audit,
        "qc_montage": asset_name(qc),
        "records": records,
        "summary": {
            "page_sides": len(records),
            "unique_folios": len({
                record["folio"] for record in records
            }),
            "eligible_illustration_sides": sum(
                record["eligible_illustration"] for record in records
            ),
            "blank_object_sides": sum(
                not record["eligible_illustration"]
                for record in records
            ),
            "foldout_mapped_sides": sum(
                record["foldout_mapped"] for record in records
            ),
            "shared_canvas_sides": sum(
                record["shared_canvas"] for record in records
            ),
            "merged_proposals_by_source": dict(proposal_sources),
            "pages_with_source": dict(Counter(
                source for _folio, source in detector_pages
            )),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=PDF)
    parser.add_argument("--canvases", type=Path, default=CANVASES)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--foldout-map", type=Path, default=FOLDOUT_MAP)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--qc", type=Path, default=QC)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--folios",
        help="comma-separated test-only folios; omit for the declared full run",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(
        args.pdf,
        args.canvases,
        args.corpus,
        args.foldout_map,
        args.output,
        args.qc,
        args.cache,
        args.batch_size,
        progress=not args.quiet,
        only_folios=(
            set(args.folios.split(",")) if args.folios else None
        ),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"WROTE {args.output}")
    print(f"WROTE {args.qc}")


if __name__ == "__main__":
    main()
