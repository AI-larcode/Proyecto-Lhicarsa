#!/usr/bin/env python3
"""
prepare_feedback_dataset.py
===========================

Convierte las correcciones humanas guardadas por ``wheel_feedback_gui.py`` en
un dataset YOLO-seg listo para fine-tuning incremental.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import sys
from pathlib import Path

import cv2

logger = logging.getLogger("prepare_feedback_dataset")
REPO_ROOT = Path(__file__).resolve().parents[2]

_FNAME_VIEW_RE = re.compile(r"_(?P<view>Frontal|Izq|Der)_", re.IGNORECASE)


def _entry_matches_view(entry: dict, view_filter: str | None) -> bool:
    """Devuelve True si la entrada del manifest coincide con el filtro de vista."""
    if not view_filter:
        return True
    image_name = entry.get("image", "")
    m = _FNAME_VIEW_RE.search(image_name)
    return m is not None and m.group("view").casefold() == view_filter.casefold()


def resolve_repo_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    raw = Path(path_str)
    if raw.is_absolute() and raw.exists():
        return raw

    candidates = [REPO_ROOT / raw]
    mapping = {
        "Lhicarsa_imagenes": REPO_ROOT / "data" / "images" / "Lhicarsa_imagenes",
        "resultados": REPO_ROOT / "artifacts" / "segmentacion" / "resultados",
        "resultados_iter_v1": REPO_ROOT / "artifacts" / "segmentacion" / "resultados_iter_v1",
        "resultados_strict2": REPO_ROOT / "artifacts" / "segmentacion" / "resultados_strict2",
        "feedback_humano": REPO_ROOT / "data" / "feedback" / "humano",
    }
    parts = raw.parts
    if parts and parts[0] in mapping:
        candidates.append(mapping[parts[0]].joinpath(*parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_manifest(feedback_dir: Path) -> dict[str, dict]:
    manifest_path = feedback_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No existe el manifest de feedback: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def mask_to_segments(mask_path: Path) -> list[list[float]]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"No se pudo leer la máscara: {mask_path}")
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments: list[list[float]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        epsilon = 0.002 * cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        if len(approx) < 3:
            continue
        coords: list[float] = []
        for point in approx.reshape(-1, 2):
            x, y = point.tolist()
            coords.extend([x / w, y / h])
        segments.append(coords)
    return segments


def write_label(label_path: Path, segments: list[list[float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for coords in segments:
            f.write("0 " + " ".join(f"{v:.6f}" for v in coords) + "\n")


def build_split(items: list[dict], seed: int, val_ratio: float) -> dict[str, list[dict]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_ratio))
    val_items = shuffled[:n_val]
    train_items = shuffled[n_val:]
    return {"train": train_items, "val": val_items}


def write_data_yaml(output_dir: Path) -> None:
    content = "\n".join([
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: rueda",
        "",
    ])
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepara un dataset YOLO-seg a partir de correcciones humanas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--feedback-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--view", type=str, default="Frontal",
                   help="Incluir solo correcciones de esta vista (Frontal/Izq/Der). "
                        "Por defecto solo se incluyen imágenes frontales.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--copy-images", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    manifest = load_manifest(args.feedback_dir)
    items = [
        entry for entry in manifest.values()
        if entry.get("corrected_mask") and _entry_matches_view(entry, args.view)
    ]
    if args.view:
        logger.info("Filtro de vista '%s' aplicado al manifest de feedback.", args.view)
    if not items:
        logger.error("No hay correcciones humanas guardadas.")
        return 1

    splits = build_split(items, seed=args.seed, val_ratio=args.val_ratio)
    for split_name, split_items in splits.items():
        logger.info("%s: %d muestras", split_name, len(split_items))
        for entry in split_items:
            image_path = resolve_repo_path(entry.get("image"))
            mask_path = resolve_repo_path(entry.get("corrected_mask"))
            if image_path is None or mask_path is None:
                logger.warning("Entrada inválida en el manifest: %s", entry)
                continue
            target_image = args.output / "images" / split_name / image_path.name
            target_label = args.output / "labels" / split_name / f"{image_path.stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            if args.copy_images:
                shutil.copy2(image_path, target_image)
            else:
                if target_image.exists() or target_image.is_symlink():
                    target_image.unlink()
                target_image.symlink_to(image_path.resolve())
            segments = mask_to_segments(mask_path)
            write_label(target_label, segments)

    write_data_yaml(args.output)
    logger.info("Dataset de feedback preparado en %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
