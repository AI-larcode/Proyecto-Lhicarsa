#!/usr/bin/env python3
"""
prepare_yolo_seg_dataset.py
===========================

Convierte las máscaras generadas por ``yoloe_seg_predict.py`` en un dataset de
segmentación compatible con Ultralytics YOLO.

El script busca pares imagen/máscara, extrae el contorno principal de cada
máscara binaria y genera:
  * ``images/{train,val,test}``
  * ``labels/{train,val,test}``
  * ``data.yaml``

Las etiquetas se escriben en formato YOLO-seg con una única clase: ``rueda``.
"""
from __future__ import annotations

import argparse
import logging
import random
import re
import shutil
import sys
from pathlib import Path

import cv2

logger = logging.getLogger("prepare_yolo_seg_dataset")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

_FNAME_VIEW_RE = re.compile(r"_(?P<view>Frontal|Izq|Der)_", re.IGNORECASE)


def _matches_view(image_path: Path, view_filter: str | None) -> bool:
    """Devuelve True si la imagen coincide con el filtro de vista (o no hay filtro)."""
    if not view_filter:
        return True
    m = _FNAME_VIEW_RE.search(image_path.stem)
    return m is not None and m.group("view").casefold() == view_filter.casefold()


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def iter_images(source: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in source.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )


def find_mask(mask_dir: Path, image_path: Path) -> Path | None:
    candidate = mask_dir / f"{image_path.stem}_mask.png"
    return candidate if candidate.is_file() else None


def mask_to_yolo_segments(mask_path: Path) -> tuple[int, int, list[list[float]]]:
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
    return w, h, segments


def write_label(label_path: Path, segments: list[list[float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as f:
        for coords in segments:
            f.write("0 " + " ".join(f"{v:.6f}" for v in coords) + "\n")


def build_split(items: list[tuple[Path, Path]], seed: int, val_ratio: float, test_ratio: float) -> dict[str, list[tuple[Path, Path]]]:
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_test = min(n_test, n_total)
    n_val = min(n_val, max(0, n_total - n_test))
    test_items = shuffled[:n_test]
    val_items = shuffled[n_test:n_test + n_val]
    train_items = shuffled[n_test + n_val:]
    return {"train": train_items, "val": val_items, "test": test_items}


def write_data_yaml(output_dir: Path) -> None:
    data_yaml = output_dir / "data.yaml"
    content = "\n".join([
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        "  0: rueda",
        "",
    ])
    data_yaml.write_text(content, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepara un dataset YOLO-seg a partir de imágenes y máscaras binarias.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images", type=Path, required=True, help="Directorio con imágenes originales.")
    p.add_argument("--masks", type=Path, required=True, help="Directorio con máscaras *_mask.png.")
    p.add_argument("--output", type=Path, required=True, help="Directorio de salida del dataset YOLO.")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--view", type=str, default="Frontal",
                   help="Incluir solo imágenes de esta vista (Frontal/Izq/Der). "
                        "Por defecto solo se incluyen imágenes frontales.")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--copy-images", action="store_true",
                   help="Copia imágenes en lugar de crear enlaces simbólicos.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    image_paths = iter_images(args.images, recursive=args.recursive)
    if args.view:
        before = len(image_paths)
        image_paths = [p for p in image_paths if _matches_view(p, args.view)]
        logger.info("Filtro de vista '%s': %d → %d imágenes.", args.view, before, len(image_paths))
    items: list[tuple[Path, Path]] = []
    for image_path in image_paths:
        mask_path = find_mask(args.masks, image_path)
        if mask_path is None:
            continue
        _, _, segments = mask_to_yolo_segments(mask_path)
        if not segments:
            continue
        items.append((image_path, mask_path))

    if not items:
        logger.error("No se encontraron pares imagen/máscara válidos.")
        return 1

    logger.info("Pares válidos encontrados: %d", len(items))
    splits = build_split(items, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    for split_name, split_items in splits.items():
        logger.info("%s: %d muestras", split_name, len(split_items))
        for image_path, mask_path in split_items:
            target_image = args.output / "images" / split_name / image_path.name
            target_label = args.output / "labels" / split_name / f"{image_path.stem}.txt"
            target_image.parent.mkdir(parents=True, exist_ok=True)
            if args.copy_images:
                shutil.copy2(image_path, target_image)
            else:
                if target_image.exists() or target_image.is_symlink():
                    target_image.unlink()
                target_image.symlink_to(image_path.resolve())
            _, _, segments = mask_to_yolo_segments(mask_path)
            write_label(target_label, segments)

    write_data_yaml(args.output)
    logger.info("Dataset YOLO-seg preparado en %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
