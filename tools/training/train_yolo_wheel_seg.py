#!/usr/bin/env python3
"""
train_yolo_wheel_seg.py
=======================

Lanza entrenamiento o fine-tuning de un segmentador YOLO para ruedas a partir
de un dataset YOLO-seg.

Uso típico:
    python train_yolo_wheel_seg.py \\
        --data ./data/datasets/ruedas_yolo/data.yaml \\
        --model ./models/base/yolo26n-seg.pt \\
        --epochs 80
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("train_yolo_wheel_seg")
REPO_ROOT = Path(__file__).resolve().parents[2]


class SafeThreadPool:
    """Sustituto mínimo de ThreadPool sin dependencias de multiprocessing."""

    def __init__(self, processes: int | None = None, initializer=None, initargs=()) -> None:
        self.processes = processes or 1
        self.initializer = initializer
        self.initargs = initargs
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    def __enter__(self) -> "SafeThreadPool":
        if self.initializer is not None:
            self.initializer(*self.initargs)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.processes)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def imap(self, func, iterable):
        if self._executor is None:
            raise RuntimeError("El pool no está inicializado.")
        futures = [self._executor.submit(func, item) for item in iterable]
        for future in futures:
            yield future.result()


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrena o ajusta un modelo YOLO de segmentación para ruedas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=Path, required=True, help="Ruta al data.yaml del dataset YOLO-seg.")
    p.add_argument("--model", type=str, default=str(REPO_ROOT / "models" / "base" / "yolo26n-seg.pt"),
                   help="Modelo base para entrenamiento o fine-tuning.")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=Path, default=REPO_ROOT / "experiments" / "segmentacion_ruedas")
    p.add_argument("--name", type=str, default="finetune_ruedas")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--close-mosaic", type=int, default=10)
    p.add_argument("--lr0", type=float, default=0.005)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--exist-ok", action="store_true",
                   help="Permite reutilizar un directorio de salida existente.")
    p.add_argument("--from-scratch", action="store_true",
                   help="Construye el modelo desde YAML si la referencia termina en .yaml.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    if not args.data.is_file():
        logger.error("No existe el dataset YAML: %s", args.data)
        return 2

    from ultralytics import YOLO
    import ultralytics.data.dataset as yolo_dataset

    # El sandbox bloquea semáforos de multiprocessing; usamos un pool basado en hilos.
    yolo_dataset.ThreadPool = SafeThreadPool
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    if args.from_scratch and not args.model.endswith(".yaml"):
        logger.error("--from-scratch requiere un modelo YAML, por ejemplo yolo11n-seg.yaml")
        return 2

    logger.info("Cargando modelo base: %s", args.model)
    model = YOLO(args.model)
    logger.info("Iniciando entrenamiento de segmentación de ruedas.")
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        close_mosaic=args.close_mosaic,
        lr0=args.lr0,
        dropout=args.dropout,
        exist_ok=args.exist_ok,
    )
    logger.info("Entrenamiento finalizado: %s", results.save_dir if hasattr(results, "save_dir") else results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
