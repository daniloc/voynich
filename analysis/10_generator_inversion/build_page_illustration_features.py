#!/usr/bin/env python3
"""
Build text-guarded visual features for every Voynich page side.

The pinned PDF and IIIF canvas list have the same order after the four
non-page edge views are removed.  Components are detected at 384 x 512 pixels
after subtracting a local paper-background estimate, then projected to a
192 x 256 analysis grid.  Connected dark/pigmented components are retained
only when their vertical span is too tall to be an ordinary text word.  The
primary feature families therefore describe large illustration silhouettes
and pigments, not adjacent glyph shapes.

The output also carries a small-component grid as a leakage diagnostic.  It is
never part of the declared illustration candidate family.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[2]
PDF = ROOT / "images" / "facsimile" / "Voynich_Manuscript.pdf"
CANVASES = ROOT / "data" / "iiif" / "iiif_canvases.json"
OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_page_illustration_features.json"
)
QC = (
    ROOT
    / "images"
    / "crops"
    / "generator_inversion_illustration_masks.png"
)
WIDTH = 192
HEIGHT = 256
DETECT_WIDTH = WIDTH * 2
DETECT_HEIGHT = HEIGHT * 2
GRID = 8
EDGE_VIEWS = {"[Head]", "[Tail]", "[Fore-edge]", "[Spine]"}
QC_FOLIOS = (
    "f4r",
    "f23r",
    "f43r",
    "f57v",
    "f75r",
    "f88r",
    "f103r",
    "f116r",
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


def normalized(
    image: Image.Image, width: int = WIDTH, height: int = HEIGHT
) -> np.ndarray:
    return np.asarray(
        image.convert("RGB").resize(
            (width, height), Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    )


def grid_density(mask: np.ndarray) -> list[float]:
    values = []
    for row in np.array_split(mask, GRID, axis=0):
        for cell in np.array_split(row, GRID, axis=1):
            values.append(round(float(np.mean(cell)), 8))
    return values


def projection(mask: np.ndarray) -> list[float]:
    rows = [
        round(float(np.mean(row)), 8)
        for row in np.array_split(mask, 16, axis=0)
    ]
    columns = [
        round(float(np.mean(column)), 8)
        for column in np.array_split(mask, 16, axis=1)
    ]
    return rows + columns


def connected_masks(
    foreground: np.ndarray,
    active_margin_x: int,
    active_margin_y: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    visited = np.zeros(foreground.shape, dtype=bool)
    illustration = np.zeros(foreground.shape, dtype=bool)
    text_proxy = np.zeros(foreground.shape, dtype=bool)
    components = []
    height, width = foreground.shape
    for start_y, start_x in zip(*np.nonzero(foreground)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels = []
        min_y = max_y = int(start_y)
        min_x = max_x = int(start_x)
        while stack:
            point_y, point_x = stack.pop()
            pixels.append((point_y, point_x))
            min_y = min(min_y, point_y)
            max_y = max(max_y, point_y)
            min_x = min(min_x, point_x)
            max_x = max(max_x, point_x)
            for near_y in range(
                max(0, point_y - 1), min(height, point_y + 2)
            ):
                for near_x in range(
                    max(0, point_x - 1), min(width, point_x + 2)
                ):
                    if (
                        foreground[near_y, near_x]
                        and not visited[near_y, near_x]
                    ):
                        visited[near_y, near_x] = True
                        stack.append((near_y, near_x))
        box_height = max_y - min_y + 1
        box_width = max_x - min_x + 1
        area = len(pixels)
        near_page_edge = (
            min_x <= max(active_margin_x, round(width * 0.075))
            or max_x >= width - max(active_margin_x, round(width * 0.075)) - 1
            or min_y <= max(active_margin_y, round(height * 0.05))
            or max_y >= height - max(active_margin_y, round(height * 0.05)) - 1
        )
        border_artifact = near_page_edge and (
            area >= 500 or box_height >= 80 or box_width >= 120
        )
        is_illustration = (
            area >= 24 and box_height >= 18 and not border_artifact
        )
        target = illustration if is_illustration else text_proxy
        for point_y, point_x in pixels:
            target[point_y, point_x] = True
        components.append({
            "area": area,
            "height": box_height,
            "width": box_width,
            "illustration": int(is_illustration),
            "border_artifact": int(border_artifact),
        })
    return illustration, text_proxy, components


def resized_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (WIDTH, HEIGHT), Image.Resampling.NEAREST
        )
    ) > 0


def guarded_views(
    rgb: np.ndarray,
    illustration: np.ndarray,
) -> dict[str, Image.Image]:
    source = np.clip(rgb, 0, 255).astype(np.uint8)
    guarded = np.full_like(source, 255)
    guarded[illustration] = source[illustration]
    silhouette = np.full_like(source, 255)
    silhouette[illustration] = 0

    if illustration.any():
        points_y, points_x = np.nonzero(illustration)
        padding = 5
        left = max(0, int(points_x.min()) - padding)
        right = min(WIDTH, int(points_x.max()) + padding + 1)
        top = max(0, int(points_y.min()) - padding)
        bottom = min(HEIGHT, int(points_y.max()) + padding + 1)
    else:
        left, top, right, bottom = 0, 0, WIDTH, HEIGHT
    return {
        "full_rgb": Image.fromarray(guarded),
        "tight_rgb": Image.fromarray(guarded).crop(
            (left, top, right, bottom)
        ),
        "full_silhouette": Image.fromarray(silhouette),
        "tight_silhouette": Image.fromarray(silhouette).crop(
            (left, top, right, bottom)
        ),
    }


def feature_record(
    image: Image.Image,
) -> tuple[dict, Image.Image, dict[str, Image.Image]]:
    detect_rgb = normalized(image, DETECT_WIDTH, DETECT_HEIGHT)
    gray = (
        0.299 * detect_rgb[..., 0]
        + 0.587 * detect_rgb[..., 1]
        + 0.114 * detect_rgb[..., 2]
    )
    local_paper = np.asarray(
        Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius=4)
        ),
        dtype=np.float32,
    )
    local_darkness = local_paper - gray
    paper = float(np.percentile(gray, 85))
    ink = (local_darkness > 16.0) & (gray < 230.0)

    red_green = detect_rgb[..., 0] - detect_rgb[..., 1]
    red_blue = detect_rgb[..., 0] - detect_rgb[..., 2]
    green_red = -red_green
    green_blue = detect_rgb[..., 1] - detect_rgb[..., 2]
    blue_red = -red_blue
    blue_green = -green_blue
    yellow_blue = (
        (detect_rgb[..., 0] + detect_rgb[..., 1]) / 2.0
        - detect_rgb[..., 2]
    )
    paper_pixels = gray >= np.percentile(gray, 55)
    def paper_median(values: np.ndarray) -> float:
        return float(np.median(values[paper_pixels]))

    red = (
        (red_green > paper_median(red_green) + 11.0)
        & (red_blue > paper_median(red_blue) + 9.0)
        & (gray < 230.0)
    )
    green = (
        (green_red > paper_median(green_red) + 8.0)
        & (green_blue > paper_median(green_blue) + 3.0)
        & (gray < 230.0)
    )
    blue = (
        (blue_red > paper_median(blue_red) + 8.0)
        & (blue_green > paper_median(blue_green) + 3.0)
        & (gray < 230.0)
    )
    gold = (
        (yellow_blue > paper_median(yellow_blue) + 11.0)
        & (gray < 230.0)
    )
    pigment = red | green | blue | gold
    foreground = ink | pigment
    margin_x = round(DETECT_WIDTH * 0.035)
    margin_y = round(DETECT_HEIGHT * 0.02)
    foreground[:margin_y, :] = False
    foreground[-margin_y:, :] = False
    foreground[:, :margin_x] = False
    foreground[:, -margin_x:] = False
    illustration_detect, text_proxy_detect, components = connected_masks(
        foreground, margin_x, margin_y
    )
    rgb = normalized(image)
    illustration = resized_mask(illustration_detect)
    text_proxy = resized_mask(text_proxy_detect)
    red = resized_mask(red)
    green = resized_mask(green)
    blue = resized_mask(blue)
    gold = resized_mask(gold)
    pigment = red | green | blue | gold
    illustration_components = [
        component
        for component in components
        if component["illustration"]
    ]
    component_areas = [
        component["area"] for component in illustration_components
    ]
    component_heights = [
        component["height"] for component in illustration_components
    ]
    component_widths = [
        component["width"] for component in illustration_components
    ]
    yy, xx = np.indices(illustration.shape)
    if illustration.any():
        centroid_x = float(
            xx[illustration].mean() / (WIDTH - 1) * 2 - 1
        )
        centroid_y = float(
            yy[illustration].mean() / (HEIGHT - 1) * 2 - 1
        )
    else:
        centroid_x = centroid_y = 0.0
    global_guarded = [
        float(np.mean(illustration)),
        max(component_areas, default=0)
        / float(DETECT_WIDTH * DETECT_HEIGHT),
        float(len(illustration_components)),
        max(component_heights, default=0) / DETECT_HEIGHT,
        max(component_widths, default=0) / DETECT_WIDTH,
        centroid_x,
        centroid_y,
        1.0 - float(np.mean(illustration != np.fliplr(illustration))),
        1.0 - float(np.mean(illustration != np.flipud(illustration))),
        float(np.mean(red & illustration)),
        float(np.mean(green & illustration)),
        float(np.mean(blue & illustration)),
        float(np.mean(gold & illustration)),
        float(np.mean(pigment & illustration)),
    ]
    silhouette = grid_density(illustration) + projection(illustration)
    pigment_grid = (
        grid_density(red & illustration)
        + grid_density(green & illustration)
        + grid_density(blue & illustration)
        + grid_density(gold & illustration)
    )
    combined = global_guarded + silhouette + pigment_grid
    record = {
        "eligible_illustration": bool(
            np.mean(illustration) >= 0.0008
            and max(component_heights, default=0) >= 18
        ),
        "audit": {
            "paper_gray": round(paper, 8),
            "raw_ink_fraction": round(float(np.mean(ink)), 8),
            "illustration_fraction": round(
                float(np.mean(illustration)), 8
            ),
            "text_proxy_fraction": round(
                float(np.mean(text_proxy)), 8
            ),
            "illustration_components": len(illustration_components),
            "small_components": (
                len(components) - len(illustration_components)
            ),
            "border_artifacts": sum(
                component["border_artifact"] for component in components
            ),
        },
        "features": {
            "global_guarded": [
                round(value, 8) for value in global_guarded
            ],
            "silhouette_guarded": silhouette,
            "pigment_guarded": pigment_grid,
            "combined_guarded": [
                round(value, 8) for value in combined
            ],
        },
        "leakage_diagnostic": {
            "small_component_grid": grid_density(text_proxy),
        },
    }
    overlay = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    overlay[..., :] = np.clip(rgb, 0, 255).astype(np.uint8)
    overlay[illustration] = (
        0.35 * overlay[illustration]
        + 0.65 * np.array([20, 190, 70])
    ).astype(np.uint8)
    overlay[text_proxy] = (
        0.45 * overlay[text_proxy]
        + 0.55 * np.array([210, 40, 40])
    ).astype(np.uint8)
    return (
        record,
        Image.fromarray(overlay),
        guarded_views(rgb, illustration),
    )


def canvas_order(path: Path) -> list[str]:
    canvases = json.loads(path.read_text(encoding="utf-8"))
    return [
        str(row[0])
        for row in canvases
        if str(row[0]) not in EDGE_VIEWS
    ]


def build(pdf: Path, canvases: Path, output: Path, qc: Path) -> dict:
    labels = canvas_order(canvases)
    document = fitz.open(pdf)
    if len(labels) != document.page_count:
        raise ValueError(
            f"canvas/PDF mismatch: {len(labels)} != {document.page_count}"
        )
    records = []
    qc_images = {}
    for index, label in enumerate(labels):
        page = document[index]
        images = page.get_images(full=True)
        if not images:
            continue
        payload = document.extract_image(images[0][0])
        image = Image.open(io.BytesIO(payload["image"])).convert("RGB")
        record, overlay, _guarded = feature_record(image)
        if label and label[0].isdigit():
            folio = f"f{label}"
            records.append({
                "folio": folio,
                "pdf_page_index": index,
                **record,
            })
            if folio in QC_FOLIOS:
                qc_images[folio] = overlay
    document.close()

    qc.parent.mkdir(parents=True, exist_ok=True)
    cell_width = WIDTH
    cell_height = HEIGHT + 20
    montage = Image.new(
        "RGB",
        (cell_width * 4, cell_height * 2),
        "white",
    )
    from PIL import ImageDraw

    draw = ImageDraw.Draw(montage)
    for index, folio in enumerate(QC_FOLIOS):
        if folio not in qc_images:
            continue
        left = (index % 4) * cell_width
        top = (index // 4) * cell_height
        montage.paste(qc_images[folio], (left, top))
        draw.text((left + 4, top + HEIGHT + 2), folio, fill="black")
    montage.save(qc)

    result = {
        "experiment": "text_guarded_page_illustration_features",
        "parameters": {
            "normalized_size": [WIDTH, HEIGHT],
            "detection_size": [DETECT_WIDTH, DETECT_HEIGHT],
            "grid": GRID,
            "minimum_component_area": 24,
            "illustration_component_rule": "height>=18",
            "local_darkness_threshold": 16.0,
            "border_artifact_rule": (
                "enters 7.5% x / 5% y edge zone AND "
                "(area>=500 OR height>=80 OR width>=120)"
            ),
            "primary_feature_families": [
                "global_guarded",
                "silhouette_guarded",
                "pigment_guarded",
                "combined_guarded",
            ],
            "excluded_from_primary": ["small_component_grid"],
        },
        "assets": {
            asset_name(pdf): sha256(pdf),
            asset_name(canvases): sha256(canvases),
        },
        "qc_montage": str(qc.relative_to(ROOT)),
        "records": records,
        "summary": {
            "page_sides": len(records),
            "eligible_illustration_sides": sum(
                row["eligible_illustration"] for row in records
            ),
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
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--qc", type=Path, default=QC)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args.pdf, args.canvases, args.output, args.qc)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"WROTE {args.output}")
    print(f"WROTE {args.qc}")


if __name__ == "__main__":
    main()
