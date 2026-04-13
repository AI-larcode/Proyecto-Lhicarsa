#!/usr/bin/env python3
"""
run_feedback_finetune.py
========================

Convierte correcciones humanas en dataset YOLO-seg y lanza un fine-tuning
incremental del segmentador de ruedas.
"""
from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("run_feedback_finetune")
REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_FEEDBACK_SCRIPT = REPO_ROOT / "tools" / "datasets" / "prepare_feedback_dataset.py"
TRAIN_SEG_SCRIPT = REPO_ROOT / "tools" / "training" / "train_yolo_wheel_seg.py"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_step(cmd: list[str], cwd: Path) -> None:
    logger.info("Ejecutando: %s", shlex.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lanza fine-tuning incremental a partir de feedback humano.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--feedback-dir", type=Path, default=REPO_ROOT / "data" / "feedback" / "humano")
    p.add_argument("--dataset-output", type=Path, default=REPO_ROOT / "data" / "datasets" / "feedback_ruedas")
    p.add_argument("--view", type=str, default="Frontal",
                   help="Incluir solo correcciones de esta vista (Frontal/Izq/Der). "
                        "Por defecto solo se usan imágenes frontales.")
    p.add_argument("--model", type=str, required=True,
                   help="Modelo base o best.pt sobre el que continuar el entrenamiento.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", type=Path, default=REPO_ROOT / "experiments" / "segmentacion_ruedas")
    p.add_argument("--name", type=str, default="feedback_incremental")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--copy-images", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    root = REPO_ROOT
    py = sys.executable

    run_step(
        [
            py, str(PREPARE_FEEDBACK_SCRIPT),
            "--feedback-dir", str(args.feedback_dir),
            "--output", str(args.dataset_output),
            "--seed", str(args.seed),
            *(["--view", args.view] if args.view else []),
            *(["--copy-images"] if args.copy_images else []),
        ],
        root,
    )

    run_step(
        [
            py, str(TRAIN_SEG_SCRIPT),
            "--data", str(args.dataset_output / "data.yaml"),
            "--model", args.model,
            "--epochs", str(args.epochs),
            "--imgsz", str(args.imgsz),
            "--batch", str(args.batch),
            "--device", args.device,
            "--workers", str(args.workers),
            "--project", str(args.project),
            "--name", args.name,
            "--seed", str(args.seed),
            "--exist-ok",
        ],
        root,
    )
    logger.info("Fine-tuning con feedback completado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
