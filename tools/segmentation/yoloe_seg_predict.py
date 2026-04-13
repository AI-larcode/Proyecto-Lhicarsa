#!/usr/bin/env python3
"""
yoloe_seg_predict.py
====================

Segmentación de neumáticos con YOLO (Ultralytics) usando un modelo de
segmentación preentrenado y extracción de características geométricas para
clasificación posterior del estado de revisión.

Salidas posibles:
    - Máscara binaria PNG por imagen (--no-mask para desactivar).
    - Overlay PNG (imagen + detección + máscara) (--no-overlay para desactivar).
    - Informe JSON global con metadatos de inferencia (--report).
    - CSV de características geométricas listo para entrenar un clasificador (--features).

Ejemplo
-------
Procesar una carpeta y volcar características para entrenar el clasificador:
    python yoloe_seg_predict.py \\
        --weights ./models/base/yoloe-11s-seg.pt \\
        --source  ./data/images/Lhicarsa_imagenes \\
        --output  ./artifacts/segmentacion/inferencia \\
        --recursive \\
        --features ./artifacts/segmentacion/inferencia/features.csv \\
        --report   ./artifacts/segmentacion/inferencia/informe.json \\
        --no-overlay
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

IMG_EXTENSIONS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
)

logger = logging.getLogger("yolo_seg")
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Estructuras de datos
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    image: str
    width: int
    height: int
    num_detections: int
    largest_mask_area_px: int
    largest_mask_ratio: float
    inference_time_ms: float
    overlay_path: str | None = None
    mask_path: str | None = None


@dataclass
class GeometricFeatures:
    """Descriptores geométricos invariantes a escala extraídos de la máscara."""
    # Tamaño / escala
    area_norm: float
    perimeter_norm: float
    equiv_diameter_norm: float
    bbox_w_norm: float
    bbox_h_norm: float
    # Forma
    circularity: float
    solidity: float
    extent: float
    aspect_ratio: float
    ellipse_axis_ratio: float
    ellipse_angle_deg: float
    # Huella / aplastamiento
    contact_patch_norm: float
    bottom_flatness: float
    bottom_curvature: float
    # Distribución de masa
    lower_half_ratio: float
    centroid_y_norm: float
    # Momentos de Hu (log-transformados)
    hu1: float; hu2: float; hu3: float; hu4: float
    hu5: float; hu6: float; hu7: float


@dataclass
class ImageMetadata:
    truck_id: int | None = None    # Se reutiliza para guardar el número de rueda
    view: str | None = None        # Vista de captura: Frontal, Izq o Der
    pressure_bar: float | None = None
    truck_type: str | None = None  # 'CargaLateral' | 'RecolectorTrasera' | 'RecolectorLateral'
    brand: str | None = None       # 'Iveco' | 'Scania'
    composite_id: int | None = None  # hash único tipo+marca+nº rueda


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def select_device(requested: str = "auto") -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def es_referencia_modelo_valida(model_ref: str) -> bool:
    if Path(model_ref).exists():
        return True
    return "/" not in model_ref and "\\" not in model_ref


def usa_modelo_yoloe(model_ref: str) -> bool:
    return "yoloe" in Path(model_ref).name.casefold()


def iter_images(source: Path, recursive: bool) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() not in IMG_EXTENSIONS:
            raise ValueError(f"Extensión no soportada: {source.suffix}")
        yield source
        return
    if not source.is_dir():
        raise FileNotFoundError(f"No existe la ruta: {source}")
    pattern = "**/*" if recursive else "*"
    for p in sorted(source.glob(pattern)):
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
            yield p


# ---------------------------------------------------------------------------
# Parseo de metadatos del nombre de fichero
# ---------------------------------------------------------------------------

# Convención de nombrado admitida:
#   TipoDeCamion_Marca_NumeroRueda_Vista_Presion_Fecha.ext
# Ejemplos:
#   CargaLateral_Scania_1_Der_4'75_05-03-2026.jpg
#   RecolectorTrasera_Iveco_4_Izq_6'25_30-01-2026.JPG
#   RecolectorLateral_Scania_2_Frontal_7_05-03-2026.JPG
_FNAME_RE_CON_VISTA = re.compile(
    r"^(?P<type>CargaLateral|RecolectorTrasera|RecolectorLateral)_"
    r"(?P<brand>Iveco|Scania)_"
    r"(?P<wheel>\d+)_"
    r"(?P<view>Frontal|Izq|Der)_"
    r"(?P<pressure>\d+(?:[_']\d+)?)_"
    r"(?P<date>\d{2}-?\d{2}-?\d{4})$",
    re.IGNORECASE,
)

# Mapa estable tipo+marca → offset para composite_id
_COMPOSITE_OFFSETS = {
    ("cargalateral", "scania"): 100,
    ("recolectortrasera", "scania"): 200,
    ("recolectorlateral", "scania"): 300,
    ("recolectortrasera", "iveco"): 400,
}


def _decode_pressure(token: str) -> float | None:
    """4'75 -> 4.75 ; 4_75 -> 4.75 ; 7 -> 7.0 ; 55 -> 5.5 ; 625 -> 6.25 ..."""
    if not token:
        return None
    # Acepta ' o _ como separador decimal
    for sep in ("'", "_"):
        if sep in token:
            try:
                return float(token.replace(sep, "."))
            except ValueError:
                return None
    if not token.isdigit():
        return None
    n = len(token)
    if n == 1: return float(token)
    if n == 2: return float(f"{token[0]}.{token[1]}")
    if n == 3: return float(f"{token[0]}.{token[1:]}")
    return None


def parse_filename(path: Path) -> ImageMetadata:
    m = _FNAME_RE_CON_VISTA.search(path.stem)
    if not m:
        return ImageMetadata()
    truck_type = m.group("type")
    brand = m.group("brand")
    wheel_num = int(m.group("wheel"))
    key = (truck_type.casefold(), brand.casefold())
    offset = _COMPOSITE_OFFSETS.get(key, 0)
    return ImageMetadata(
        truck_id=wheel_num,
        view=m.groupdict().get("view"),
        pressure_bar=_decode_pressure(m.group("pressure")),
        truck_type=truck_type,
        brand=brand,
        composite_id=offset + wheel_num,
    )


# ---------------------------------------------------------------------------
# Procesamiento de imagen
# ---------------------------------------------------------------------------

def crop_center_aspect(img: np.ndarray, target_ratio: float = 4 / 3) -> np.ndarray:
    h, w = img.shape[:2]
    current = w / h
    if abs(current - target_ratio) < 1e-3:
        return img
    if current > target_ratio:
        new_w = int(target_ratio * h)
        offset = (w - new_w) // 2
        return img[:, offset:offset + new_w]
    new_h = int(w / target_ratio)
    offset = (h - new_h) // 2
    return img[offset:offset + new_h, :]


def preprocess_image(path: Path, width: int, height: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"No se pudo leer la imagen: {path}")
    img = crop_center_aspect(img, target_ratio=width / height)
    if (img.shape[1], img.shape[0]) != (width, height):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# Postprocesado de máscaras
# ---------------------------------------------------------------------------

def _mask_wheel_score(mask: np.ndarray) -> tuple[bool, float, int]:
    """Evalúa si una máscara parece una rueda y devuelve una puntuación."""
    h, w = mask.shape[:2]
    img_area = float(h * w)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, -1.0, 0

    cnt = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(cnt))
    if area <= 0:
        return False, -1.0, 0

    perimeter = cv2.arcLength(cnt, closed=True)
    x, y, bw, bh = cv2.boundingRect(cnt)
    area_ratio = area / img_area if img_area else 0.0
    aspect_ratio = (bw / bh) if bh else 0.0
    fill_ratio = area / float(max(1, bw * bh))
    circularity = float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    # Reglas duras para descartar camiones completos o regiones poco plausibles.
    es_rueda_plausible = (
        0.025 <= area_ratio <= 0.50
        and 0.55 <= aspect_ratio <= 1.60
        and 0.35 <= fill_ratio <= 0.92
        and circularity >= 0.45
    )

    # Priorizamos formas compactas y casi circulares, manteniendo preferencia por mayor área.
    score = (
        area_ratio * 3.0
        + min(circularity, 1.0) * 1.5
        + max(0.0, 1.0 - abs(1.0 - aspect_ratio)) * 1.0
        + fill_ratio * 0.5
    )
    return es_rueda_plausible, score, area


def extract_largest_mask(masks_tensor, out_size: tuple[int, int]) -> tuple[np.ndarray, int]:
    W, H = out_size
    clean = np.zeros((H, W), dtype=np.uint8)
    if masks_tensor is None or len(masks_tensor) == 0:
        return clean, 0
    masks = masks_tensor.cpu().numpy().astype(np.uint8)
    best_area = 0
    best_mask: np.ndarray | None = None
    best_score = -1.0
    for m in masks:
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        es_rueda_plausible, score, area = _mask_wheel_score(m)
        if not es_rueda_plausible:
            continue
        if score > best_score or (abs(score - best_score) < 1e-6 and area > best_area):
            best_score = score
            best_area = area
            best_mask = m
    if best_mask is not None:
        clean[best_mask == 1] = 255
    return clean, best_area


# ---------------------------------------------------------------------------
# Extracción de características geométricas
# ---------------------------------------------------------------------------

def compute_geometric_features(mask: np.ndarray) -> GeometricFeatures | None:
    """Calcula descriptores geométricos de una máscara binaria 0/255."""
    if mask is None or mask.max() == 0:
        return None
    H, W = mask.shape[:2]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)
    if area <= 0 or perimeter <= 0:
        return None

    x, y, bw, bh = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) or 1.0

    if len(cnt) >= 5:
        (_, _), (axis_a, axis_b), angle = cv2.fitEllipse(cnt)
        axes = sorted((axis_a, axis_b))
        ellipse_axis_ratio = float(axes[0] / axes[1]) if axes[1] > 0 else 1.0
        ellipse_angle = float(angle) % 180.0
    else:
        ellipse_axis_ratio, ellipse_angle = 1.0, 0.0

    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] <= 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    lower_area = float(np.count_nonzero(mask[int(cy):, :]))
    total_area = float(np.count_nonzero(mask))
    lower_half_ratio = lower_area / total_area if total_area else 0.0
    centroid_y_norm = (cy - y) / bh if bh else 0.5

    contact_patch_norm, bottom_flatness, bottom_curvature = _bottom_edge_features(
        mask, x, bw
    )

    hu = cv2.HuMoments(M).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    norm = float(np.sqrt(W * H))
    return GeometricFeatures(
        area_norm=float(area / (W * H)),
        perimeter_norm=float(perimeter / norm),
        equiv_diameter_norm=float(np.sqrt(4 * area / np.pi) / norm),
        bbox_w_norm=float(bw / W),
        bbox_h_norm=float(bh / H),
        circularity=float(4 * np.pi * area / (perimeter ** 2)),
        solidity=float(area / hull_area),
        extent=float(area / (bw * bh)) if bw * bh else 0.0,
        aspect_ratio=float(bw / bh) if bh else 0.0,
        ellipse_axis_ratio=ellipse_axis_ratio,
        ellipse_angle_deg=ellipse_angle,
        contact_patch_norm=contact_patch_norm,
        bottom_flatness=bottom_flatness,
        bottom_curvature=bottom_curvature,
        lower_half_ratio=lower_half_ratio,
        centroid_y_norm=float(centroid_y_norm),
        hu1=float(hu_log[0]), hu2=float(hu_log[1]), hu3=float(hu_log[2]),
        hu4=float(hu_log[3]), hu5=float(hu_log[4]), hu6=float(hu_log[5]),
        hu7=float(hu_log[6]),
    )


def _bottom_edge_features(mask, x0, bw) -> tuple[float, float, float]:
    """Tres descriptores del borde inferior: huella, aplanamiento y curvatura."""
    if bw <= 2:
        return 0.0, 0.0, 0.0

    bottom_y = np.full(bw, np.nan, dtype=np.float32)
    for i, col in enumerate(range(x0, x0 + bw)):
        ys = np.flatnonzero(mask[:, col])
        if ys.size:
            bottom_y[i] = ys.max()

    valid = ~np.isnan(bottom_y)
    if valid.sum() < 5:
        return 0.0, 0.0, 0.0

    by = bottom_y[valid]
    xs = np.arange(bw, dtype=np.float32)[valid]

    y_max = by.max()
    flat_mask = (y_max - by) <= 1.5
    contact_run = _longest_true_run(flat_mask)
    contact_patch_norm = float(contact_run / bw)

    var_norm = float(by.var() / max(1.0, bw ** 2))
    bottom_flatness = float(np.exp(-var_norm * 50.0))

    try:
        coefs = np.polyfit(xs, by, deg=2)
        bottom_curvature = float(abs(coefs[0]))
    except (np.linalg.LinAlgError, ValueError):
        bottom_curvature = 0.0

    return contact_patch_norm, bottom_flatness, bottom_curvature


def _longest_true_run(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    best = run = 0
    for v in arr:
        if v:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


# ---------------------------------------------------------------------------
# Visualización / guardado
# ---------------------------------------------------------------------------

def save_visualization(overlay_bgr: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Imagen + Detección"); ax[0].axis("off")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Máscara binaria"); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def show_visualization(overlay_bgr: np.ndarray, mask: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    fig.canvas.manager.set_window_title(title)
    ax[0].imshow(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Imagen + Detección"); ax[0].axis("off")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Máscara binaria"); ax[1].axis("off")
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# CSV de características
# ---------------------------------------------------------------------------

FEATURE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(GeometricFeatures))
META_FIELDS: tuple[str, ...] = (
    "image", "truck_id", "view", "pressure_bar",
    "truck_type", "brand", "composite_id",
    "width", "height", "num_detections",
    "largest_mask_area_px", "largest_mask_ratio",
)


def init_features_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(META_FIELDS + FEATURE_FIELDS)


def append_features_row(
    path: Path,
    pred: PredictionResult,
    meta: ImageMetadata,
    feats: GeometricFeatures | None,
) -> None:
    base = [
        pred.image, meta.truck_id, meta.view, meta.pressure_bar,
        meta.truck_type, meta.brand, meta.composite_id,
        pred.width, pred.height, pred.num_detections,
        pred.largest_mask_area_px, pred.largest_mask_ratio,
    ]
    if feats is None:
        feat_vals = [""] * len(FEATURE_FIELDS)
    else:
        d = asdict(feats)
        feat_vals = [d[k] for k in FEATURE_FIELDS]
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(base + feat_vals)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_inference(model, img_bgr, conf, iou, retina):
    t0 = time.perf_counter()
    results = model.predict(
        source=img_bgr, retina_masks=retina,
        conf=conf, iou=iou, verbose=False,
    )
    return results[0], (time.perf_counter() - t0) * 1000.0


def process_image(
    model, img_path: Path, width: int, height: int,
    conf: float, iou: float, retina: bool,
    output_dir: Path | None,
    save_overlay: bool, save_mask: bool, show: bool,
    features_csv: Path | None,
    usar_prompts: bool = False,
    prompts_alternativos: list[str] | None = None,
) -> tuple[PredictionResult, GeometricFeatures | None, ImageMetadata]:
    meta = parse_filename(img_path)
    logger.info(
        "→ %s  [tipo=%s marca=%s rueda=%s vista=%s P=%s bar cid=%s]",
        img_path.name, meta.truck_type, meta.brand,
        meta.truck_id, meta.view, meta.pressure_bar, meta.composite_id,
    )

    img = preprocess_image(img_path, width, height)
    result, elapsed_ms = run_inference(model, img, conf=conf, iou=iou, retina=retina)

    masks_tensor = result.masks.data if result.masks is not None else None
    mask, area = extract_largest_mask(masks_tensor, out_size=(width, height))
    num_det = 0 if masks_tensor is None else int(masks_tensor.shape[0])

    if area == 0 and usar_prompts and prompts_alternativos:
        for fb_prompt in prompts_alternativos:
            logger.info("   reintentando con prompt='%s' conf=%.2f", fb_prompt, conf * 0.6)
            model.set_classes([fb_prompt])
            fb_result, fb_ms = run_inference(
                model, img, conf=conf * 0.6, iou=iou, retina=retina,
            )
            elapsed_ms += fb_ms
            fb_masks = fb_result.masks.data if fb_result.masks is not None else None
            mask, area = extract_largest_mask(fb_masks, out_size=(width, height))
            fb_det = 0 if fb_masks is None else int(fb_masks.shape[0])
            num_det = max(num_det, fb_det)
            if area > 0:
                result = fb_result
                logger.info("   prompt alternativo '%s' encontró detección (área=%d)", fb_prompt, area)
                break

    ratio = area / float(width * height) if width * height else 0.0

    feats = compute_geometric_features(mask) if area > 0 else None

    overlay_path = mask_path = None
    if output_dir is not None and (save_overlay or save_mask):
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = img_path.stem
        if save_mask:
            mask_path = output_dir / f"{stem}_mask.png"
            cv2.imwrite(str(mask_path), mask)
        if save_overlay:
            overlay_path = output_dir / f"{stem}_overlay.png"
            save_visualization(result.plot(), mask, overlay_path)
    if show:
        show_visualization(result.plot(), mask, title=img_path.name)

    pred = PredictionResult(
        image=str(img_path), width=width, height=height,
        num_detections=num_det,
        largest_mask_area_px=area, largest_mask_ratio=ratio,
        inference_time_ms=elapsed_ms,
        overlay_path=str(overlay_path) if overlay_path else None,
        mask_path=str(mask_path) if mask_path else None,
    )

    if feats is not None:
        logger.info(
            "   det=%d  area=%.3f  circ=%.3f  contact=%.3f  flat=%.3f  inf=%.1fms",
            num_det, feats.area_norm, feats.circularity,
            feats.contact_patch_norm, feats.bottom_flatness, elapsed_ms,
        )
    else:
        logger.info("   sin máscara válida (det=%d)  inf=%.1fms", num_det, elapsed_ms)

    if features_csv is not None:
        append_features_row(features_csv, pred, meta, feats)

    return pred, feats, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Segmentación de neumáticos con YOLO + extracción de características geométricas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights", type=str, default=str(REPO_ROOT / "models" / "base" / "yoloe-11s-seg.pt"),
                   help="Ruta o referencia del modelo de segmentación YOLO.")
    p.add_argument("--source",  type=Path, required=True)
    p.add_argument("--output",  type=Path, default=REPO_ROOT / "artifacts" / "segmentacion" / "inferencia")
    p.add_argument("--prompt",  type=str,  default="truck wheel | tire | wheel",
                   help="Texto guía para modelos YOLOE.")
    p.add_argument("--width",   type=int,  default=640)
    p.add_argument("--height",  type=int,  default=480)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--iou",     type=float, default=0.45)
    p.add_argument("--device",  type=str,  default="auto")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--retina-masks",   dest="retina", action="store_true",  default=True)
    p.add_argument("--no-retina-masks", dest="retina", action="store_false")
    p.add_argument("--no-overlay", dest="save_overlay", action="store_false", default=True)
    p.add_argument("--no-mask",    dest="save_mask",    action="store_false", default=True)
    p.add_argument("--show", action="store_true")
    p.add_argument("--report",   type=Path, default=None)
    p.add_argument("--features", type=Path, default=None,
                   help="Ruta al CSV con características geométricas para entrenar el clasificador.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    if not es_referencia_modelo_valida(args.weights):
        logger.error("No se encuentra el modelo o la referencia no es válida: %s", args.weights); return 2
    if not args.source.exists():
        logger.error("La fuente no existe: %s", args.source); return 2

    usar_yoloe = usa_modelo_yoloe(args.weights)
    if usar_yoloe:
        from ultralytics import YOLOE
    else:
        from ultralytics import YOLO

    device = select_device(args.device)
    logger.info("Cargando modelo %s en dispositivo '%s'…", args.weights, device)
    model = YOLOE(args.weights) if usar_yoloe else YOLO(args.weights)
    try:
        model.to(device)
    except Exception as e:
        logger.warning("No se pudo mover el modelo al dispositivo %s: %s", device, e)

    primary_classes = [c.strip() for c in args.prompt.split("|") if c.strip()]
    if usar_yoloe:
        logger.info("Modelo YOLOE detectado. Clases por prompt: %s", primary_classes)
        model.set_classes(primary_classes)
    elif args.prompt:
        logger.warning(
            "Modelo YOLO estándar detectado. Se ignora --prompt y es posible que detecte "
            "camiones completos en lugar de ruedas. Para ruedas, usa un peso YOLOE."
        )

    images = list(iter_images(args.source, recursive=args.recursive))
    if not images:
        logger.error("No se encontraron imágenes en %s", args.source); return 1
    logger.info("Imágenes a procesar: %d", len(images))

    if args.features is not None:
        init_features_csv(args.features)
        logger.info("CSV de características: %s", args.features)

    summary: list[PredictionResult] = []
    for img_path in images:
        try:
            if usar_yoloe:
                model.set_classes(primary_classes)
            pred, _, _ = process_image(
                model=model, img_path=img_path,
                width=args.width, height=args.height,
                conf=args.conf, iou=args.iou, retina=args.retina,
                output_dir=args.output,
                save_overlay=args.save_overlay, save_mask=args.save_mask,
                show=args.show, features_csv=args.features,
                usar_prompts=usar_yoloe,
                prompts_alternativos=["black rubber tire", "truck wheel", "tire"] if usar_yoloe else None,
            )
            summary.append(pred)
        except Exception as e:
            logger.exception("Fallo procesando %s: %s", img_path, e)

    if args.report is not None and summary:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in summary], f, indent=2, ensure_ascii=False)
        logger.info("Informe JSON guardado en %s", args.report)

    if summary:
        avg_ms = sum(r.inference_time_ms for r in summary) / len(summary)
        n_det = sum(1 for r in summary if r.num_detections > 0)
        logger.info(
            "Hecho. %d/%d con detección. Tiempo medio: %.1f ms",
            n_det, len(summary), avg_ms,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
