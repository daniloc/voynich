#!/usr/bin/env python3
"""
Embed text-masked Voynich drawings with the pinned DINOv2-S/14 backbone.

The masking and PDF/IIIF folio alignment are inherited from
build_page_illustration_features.py.  Four guarded views are embedded: full
page layout and a tight union-of-illustrations crop, each as original color
and a black silhouette.  Ordinary glyph-sized foreground and rejected scan
edges are white, so the target writing is not presented to the backbone.

DINOv2 is externally pretrained and frozen.  This script performs no fitting
on Voynich text or images.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import fitz
import numpy as np
import torch
from PIL import Image, ImageOps

import build_page_illustration_features as guarded


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "data"
    / "intermediate"
    / "generator_inversion_guarded_dinov2_embeddings.json"
)
QC = (
    ROOT
    / "images"
    / "crops"
    / "generator_inversion_dinov2_inputs.png"
)
MODEL_REPOSITORY = (
    "facebookresearch/dinov2:"
    "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)
MODEL_NAME = "dinov2_vits14"
MODEL_WEIGHTS = (
    "https://dl.fbaipublicfiles.com/dinov2/"
    "dinov2_vits14/dinov2_vits14_pretrain.pth"
)
EXPECTED_WEIGHT_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe"
    "8ea9ca46a7e1d8d00217f29aef60cd9"
)
INPUT_SIZE = 224
VIEWS = (
    "full_rgb",
    "tight_rgb",
    "full_silhouette",
    "tight_silhouette",
)
QC_FOLIOS = ("f23r", "f57v", "f75r", "f88r")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_revision(repo: str) -> str:
    path = Path(repo)
    if not path.exists():
        return MODEL_REPOSITORY.rsplit(":", 1)[1]
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_model(repo: str, weights: str) -> torch.nn.Module:
    local = Path(repo).exists()
    arguments = {
        "source": "local" if local else "github",
        "pretrained": True,
        "weights": weights,
    }
    if not local:
        arguments.update({
            "trust_repo": True,
            "skip_validation": True,
        })
    model = torch.hub.load(repo, MODEL_NAME, **arguments)
    model.eval()
    return model


def tensor(image: Image.Image) -> torch.Tensor:
    padded = ImageOps.pad(
        image.convert("RGB"),
        (INPUT_SIZE, INPUT_SIZE),
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )
    values = np.asarray(padded, dtype=np.float32) / 255.0
    values = (values - MEAN) / STD
    return torch.from_numpy(values.transpose(2, 0, 1))


def embed_batch(
    model: torch.nn.Module,
    values: list[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    batch = torch.stack(values)
    with torch.inference_mode():
        features = model.forward_features(batch)
    cls = features["x_norm_clstoken"].cpu().numpy()
    patches = features["x_norm_patchtokens"].mean(dim=1).cpu().numpy()
    return cls, patches


def build(
    pdf: Path,
    canvases: Path,
    output: Path,
    qc: Path,
    repo: str,
    weights: str,
    batch_size: int,
    progress: bool,
) -> dict:
    weight_path = Path(weights)
    if weight_path.exists():
        weight_hash = sha256(weight_path)
        if weight_hash != EXPECTED_WEIGHT_SHA256:
            raise ValueError(
                f"unexpected DINOv2 weight hash: {weight_hash}"
            )
    else:
        weight_hash = EXPECTED_WEIGHT_SHA256

    revision = repo_revision(repo)
    if revision != MODEL_REPOSITORY.rsplit(":", 1)[1]:
        raise ValueError(f"unexpected DINOv2 revision: {revision}")
    model = load_model(repo, weights)

    labels = guarded.canvas_order(canvases)
    document = fitz.open(pdf)
    if len(labels) != document.page_count:
        raise ValueError(
            f"canvas/PDF mismatch: {len(labels)} != {document.page_count}"
        )
    pending: list[tuple[int, str, torch.Tensor]] = []
    records: list[dict] = []
    qc_views: dict[str, dict[str, Image.Image]] = {}

    def flush() -> None:
        if not pending:
            return
        cls, patches = embed_batch(
            model, [item[2] for item in pending]
        )
        for index, (target_index, view, _value) in enumerate(pending):
            target = records[target_index]["features"]
            target[f"{view}_cls"] = [
                round(float(value), 8) for value in cls[index]
            ]
            target[f"{view}_patch_mean"] = [
                round(float(value), 8) for value in patches[index]
            ]
        pending.clear()

    for pdf_index, label in enumerate(labels):
        page = document[pdf_index]
        images = page.get_images(full=True)
        if not images:
            continue
        payload = document.extract_image(images[0][0])
        image = Image.open(io.BytesIO(payload["image"])).convert("RGB")
        _record, _overlay, views = guarded.feature_record(image)
        if not label or not label[0].isdigit():
            continue
        folio = f"f{label}"
        target_index = len(records)
        records.append({
            "folio": folio,
            "pdf_page_index": pdf_index,
            "features": {},
        })
        if folio in QC_FOLIOS:
            qc_views[folio] = {
                view: views[view].copy()
                for view in ("full_rgb", "tight_rgb")
            }
        for view in VIEWS:
            pending.append((target_index, view, tensor(views[view])))
            if len(pending) >= batch_size:
                flush()
        if progress and len(records) % 20 == 0:
            print(f"embedded pages={len(records)}", flush=True)
    flush()
    document.close()

    expected = {
        f"{view}_{pooling}"
        for view in VIEWS
        for pooling in ("cls", "patch_mean")
    }
    for record in records:
        if set(record["features"]) != expected:
            raise ValueError(
                f"incomplete embeddings for {record['folio']}: "
                f"{set(record['features'])}"
            )
    from PIL import ImageDraw

    cell = INPUT_SIZE
    label_height = 18
    montage = Image.new(
        "RGB",
        (cell * len(QC_FOLIOS), (cell + label_height) * 2),
        "white",
    )
    draw = ImageDraw.Draw(montage)
    for column, folio in enumerate(QC_FOLIOS):
        for row, view in enumerate(("full_rgb", "tight_rgb")):
            image = ImageOps.pad(
                qc_views[folio][view].convert("RGB"),
                (cell, cell),
                method=Image.Resampling.LANCZOS,
                color=(255, 255, 255),
            )
            left = column * cell
            top = row * (cell + label_height)
            montage.paste(image, (left, top))
            draw.text(
                (left + 4, top + cell + 2),
                f"{folio} {view}",
                fill="black",
            )
    qc.parent.mkdir(parents=True, exist_ok=True)
    montage.save(qc)

    result = {
        "experiment": "guarded_dinov2_page_embeddings",
        "model": {
            "repository": "https://github.com/facebookresearch/dinov2",
            "revision": revision,
            "name": MODEL_NAME,
            "weights": MODEL_WEIGHTS,
            "weights_sha256": weight_hash,
            "embedding_dimensions": 384,
            "license": "Apache-2.0",
            "input_size": INPUT_SIZE,
            "normalization_mean": MEAN.tolist(),
            "normalization_std": STD.tolist(),
            "frozen": True,
        },
        "guard": {
            "builder": (
                "analysis/10_generator_inversion/"
                "build_page_illustration_features.py"
            ),
            "views": list(VIEWS),
            "ordinary_glyph_sized_components_presented_to_model": False,
        },
        "assets": {
            guarded.asset_name(pdf): sha256(pdf),
            guarded.asset_name(canvases): sha256(canvases),
        },
        "qc_montage": str(qc.relative_to(ROOT)),
        "records": records,
        "summary": {
            "page_sides": len(records),
            "feature_families": len(expected),
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
    parser.add_argument("--pdf", type=Path, default=guarded.PDF)
    parser.add_argument("--canvases", type=Path, default=guarded.CANVASES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--qc", type=Path, default=QC)
    parser.add_argument("--repo", default=MODEL_REPOSITORY)
    parser.add_argument("--weights", default=MODEL_WEIGHTS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(
        args.pdf,
        args.canvases,
        args.output,
        args.qc,
        args.repo,
        args.weights,
        args.batch_size,
        progress=not args.quiet,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"WROTE {args.output}")
    print(f"WROTE {args.qc}")


if __name__ == "__main__":
    main()
