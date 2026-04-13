#!/usr/bin/env python3
"""
run_iterative_pressure_pipeline.py
=================================

Orquesta una iteración completa del flujo:
1. Construcción del dataset YOLO-seg a partir de máscaras existentes.
2. Fine-tuning del segmentador de ruedas.
3. Resegmentación de todas las imágenes con el mejor modelo ajustado.
4. Entrenamiento del clasificador de revisión de presión con las nuevas características.

El script no asume privilegios especiales. Usa el mismo intérprete con el que
se ejecuta, así que conviene lanzarlo desde tu entorno real del proyecto.
"""
from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("iterative_pressure_pipeline")
REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_YOLO_SCRIPT = REPO_ROOT / "tools" / "datasets" / "prepare_yolo_seg_dataset.py"
TRAIN_SEG_SCRIPT = REPO_ROOT / "tools" / "training" / "train_yolo_wheel_seg.py"
SEGMENT_SCRIPT = REPO_ROOT / "tools" / "segmentation" / "yoloe_seg_predict.py"
TRAIN_CLASSIFIER_SCRIPT = REPO_ROOT / "tools" / "training" / "train_pressure_classifier.py"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_step(cmd: list[str], cwd: Path) -> None:
    logger.info("Ejecutando: %s", shlex.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def resolve_best_weights(project: Path, name: str) -> Path:
    candidates = [project / name / "weights" / "best.pt"]
    project_tail = project.name
    candidates.extend([
        REPO_ROOT / "runs" / "segment" / project_tail / name / "weights" / "best.pt",
        REPO_ROOT / "experiments" / "runs" / "segment" / project_tail / name / "weights" / "best.pt",
        REPO_ROOT / "experiments" / "runs" / "segment" / name / "weights" / "best.pt",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ejecuta una iteración completa de mejora de segmentación y presión.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images", type=Path, default=REPO_ROOT / "data" / "images" / "Lhicarsa_imagenes",
                   help="Directorio con las imágenes originales.")
    p.add_argument("--bootstrap-masks", type=Path, default=REPO_ROOT / "artifacts" / "segmentacion" / "resultados_strict2",
                   help="Directorio con máscaras iniciales *_mask.png.")
    p.add_argument("--dataset-output", type=Path, default=REPO_ROOT / "data" / "datasets" / "ruedas_yolo_iter")
    p.add_argument("--seg-model-base", type=str, default=str(REPO_ROOT / "models" / "base" / "yolo26n-seg.pt"),
                   help="Modelo base para el fine-tuning del segmentador.")
    p.add_argument("--seg-epochs", type=int, default=60)
    p.add_argument("--seg-imgsz", type=int, default=640)
    p.add_argument("--seg-batch", type=int, default=8)
    p.add_argument("--seg-device", type=str, default="0")
    p.add_argument("--seg-workers", type=int, default=8)
    p.add_argument("--seg-project", type=Path, default=REPO_ROOT / "experiments" / "segmentacion_ruedas")
    p.add_argument("--seg-name", type=str, default="iteracion_ruedas")
    p.add_argument("--resegment-output", type=Path, default=REPO_ROOT / "artifacts" / "segmentacion" / "resultados_iter")
    p.add_argument("--classifier-output", type=Path, default=REPO_ROOT / "models" / "classification" / "modelo_revision_iter.joblib")
    p.add_argument("--classifier-report", type=Path, default=REPO_ROOT / "artifacts" / "reports" / "report_revision_iter.json")
    p.add_argument("--view", type=str, default="Frontal",
                   help="Procesar solo imágenes de esta vista (Frontal/Izq/Der). "
                        "Por defecto solo se procesan imágenes frontales.")
    p.add_argument("--copy-images", action="store_true",
                   help="Copia imágenes al dataset YOLO en lugar de usar enlaces simbólicos.")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-seg-train", action="store_true")
    p.add_argument("--skip-resegment", action="store_true")
    p.add_argument("--skip-classifier", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    root = REPO_ROOT
    py = sys.executable

    if not args.skip_prepare:
        cmd = [
            py, str(PREPARE_YOLO_SCRIPT),
            "--images", str(args.images),
            "--masks", str(args.bootstrap_masks),
            "--output", str(args.dataset_output),
            "--seed", str(args.seed),
        ]
        if args.view:
            cmd.extend(["--view", args.view])
        if args.recursive:
            cmd.append("--recursive")
        if args.copy_images:
            cmd.append("--copy-images")
        run_step(cmd, root)

    if not args.skip_seg_train:
        cmd = [
            py, str(TRAIN_SEG_SCRIPT),
            "--data", str(args.dataset_output / "data.yaml"),
            "--model", args.seg_model_base,
            "--epochs", str(args.seg_epochs),
            "--imgsz", str(args.seg_imgsz),
            "--batch", str(args.seg_batch),
            "--device", args.seg_device,
            "--workers", str(args.seg_workers),
            "--project", str(args.seg_project),
            "--name", args.seg_name,
            "--seed", str(args.seed),
            "--exist-ok",
        ]
        run_step(cmd, root)

    best_weights = resolve_best_weights(args.seg_project, args.seg_name)
    if not best_weights.is_file():
        raise FileNotFoundError("No se encontró el best.pt del segmentador ajustado.")

    if not args.skip_resegment:
        cmd = [
            py, str(SEGMENT_SCRIPT),
            "--weights", str(best_weights),
            "--source", str(args.images),
            "--output", str(args.resegment_output),
            "--features", str(args.resegment_output / "features.csv"),
        ]
        if args.view:
            cmd.extend(["--view", args.view])
        if args.recursive:
            cmd.append("--recursive")
        run_step(cmd, root)

    if not args.skip_classifier:
        cmd = [
            py, str(TRAIN_CLASSIFIER_SCRIPT),
            "--features", str(args.resegment_output / "features.csv"),
            "--one-hot-type",
            "--output", str(args.classifier_output),
            "--report-json", str(args.classifier_report),
        ]
        if args.view:
            cmd.extend(["--view", args.view])
        run_step(cmd, root)

    logger.info("Iteración completada.")
    logger.info("Pesos afinados: %s", best_weights)
    if not args.skip_classifier:
        logger.info("Clasificador de revisión: %s", args.classifier_output)
        logger.info("Informe de clasificación: %s", args.classifier_report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
