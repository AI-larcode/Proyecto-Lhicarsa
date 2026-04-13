#!/usr/bin/env python3
"""
wheel_feedback_gui.py
=====================

Interfaz gráfica para revisar y corregir máscaras de segmentación de ruedas.

Permite:
  * Cargar una imagen original y su máscara predicha.
  * Pintar o borrar sobre la máscara con un pincel.
  * Marcar una imagen como "sin rueda válida".
  * Restaurar la máscara original.
  * Guardar la corrección en un directorio de feedback reutilizable para
    fine-tuning posterior.

Atajos:
  * n / Flecha derecha: siguiente imagen
  * p / Flecha izquierda: imagen anterior
  * s: guardar
  * a: modo añadir
  * e: modo borrar
  * r: restaurar máscara original
  * x: marcar sin rueda
  * + / -: cambiar tamaño del pincel
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "wheel_feedback_gui.py requiere Pillow. Instálalo en el entorno del proyecto."
    ) from exc


logger = logging.getLogger("wheel_feedback_gui")
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CANVAS_W = 1100
CANVAS_H = 760
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Item:
    image_path: Path
    mask_path: Path | None


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


def alpha_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    if mask is None:
        return overlay
    red = np.zeros_like(image_bgr)
    red[:, :, 2] = 255
    alpha = (mask > 0).astype(np.uint8)[:, :, None]
    overlay = np.where(alpha, cv2.addWeighted(image_bgr, 0.55, red, 0.45, 0), image_bgr)
    return overlay


class FeedbackApp:
    def __init__(
        self,
        root: tk.Tk,
        items: list[Item],
        feedback_dir: Path,
        start_index: int = 0,
    ) -> None:
        self.root = root
        self.items = items
        self.feedback_dir = feedback_dir
        self.index = max(0, min(start_index, len(items) - 1))
        self.brush_radius = 18
        self.mode = "add"
        self.current_image_bgr: np.ndarray | None = None
        self.original_mask: np.ndarray | None = None
        self.current_mask: np.ndarray | None = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.photo: ImageTk.PhotoImage | None = None
        self.dragging = False
        self.manifest_path = self.feedback_dir / "manifest.json"
        self.manifest = self._load_manifest()

        self._build_ui()
        self._bind_events()
        self._load_current()

    def _load_manifest(self) -> dict[str, dict]:
        if self.manifest_path.is_file():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self) -> None:
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _build_ui(self) -> None:
        self.root.title("Corrección humana de segmentación de ruedas")
        self.root.geometry("1450x900")

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(main, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H, bg="#111111", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.status_var, wraplength=300, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 12))

        self.mode_var = tk.StringVar(value="add")
        ttk.Label(right, text="Modo").pack(anchor=tk.W)
        ttk.Radiobutton(right, text="Añadir máscara", variable=self.mode_var, value="add", command=self._set_mode).pack(anchor=tk.W)
        ttk.Radiobutton(right, text="Borrar máscara", variable=self.mode_var, value="erase", command=self._set_mode).pack(anchor=tk.W)

        ttk.Label(right, text="Tamaño pincel").pack(anchor=tk.W, pady=(12, 0))
        self.brush_var = tk.IntVar(value=self.brush_radius)
        self.brush_scale = ttk.Scale(
            right,
            from_=4,
            to=80,
            orient=tk.HORIZONTAL,
            command=self._on_brush_change,
        )
        self.brush_scale.set(self.brush_radius)
        self.brush_scale.pack(fill=tk.X)
        self.brush_label = ttk.Label(right, text=f"{self.brush_radius}px")
        self.brush_label.pack(anchor=tk.W)

        ttk.Separator(right).pack(fill=tk.X, pady=12)

        ttk.Button(right, text="Guardar corrección (S)", command=self.save_current).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Restaurar original (R)", command=self.restore_original).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Marcar sin rueda (X)", command=self.mark_empty).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Anterior (P)", command=self.prev_item).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Siguiente (N)", command=self.next_item).pack(fill=tk.X, pady=2)

        ttk.Separator(right).pack(fill=tk.X, pady=12)
        ttk.Label(
            right,
            text=(
                "Flujo recomendado:\n"
                "1. Corrige la máscara.\n"
                "2. Guarda.\n"
                "3. Exporta feedback para fine-tuning."
            ),
            justify=tk.LEFT,
            wraplength=300,
        ).pack(anchor=tk.W)

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>", self._start_paint)
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._stop_paint)
        self.root.bind("<Right>", lambda _: self.next_item())
        self.root.bind("<Left>", lambda _: self.prev_item())
        self.root.bind("n", lambda _: self.next_item())
        self.root.bind("p", lambda _: self.prev_item())
        self.root.bind("s", lambda _: self.save_current())
        self.root.bind("a", lambda _: self._set_mode_value("add"))
        self.root.bind("e", lambda _: self._set_mode_value("erase"))
        self.root.bind("r", lambda _: self.restore_original())
        self.root.bind("x", lambda _: self.mark_empty())
        self.root.bind("<plus>", lambda _: self._change_brush(2))
        self.root.bind("<minus>", lambda _: self._change_brush(-2))

    def _set_mode(self) -> None:
        self.mode = self.mode_var.get()

    def _set_mode_value(self, value: str) -> None:
        self.mode_var.set(value)
        self._set_mode()

    def _on_brush_change(self, value: str) -> None:
        self.brush_radius = max(1, int(float(value)))
        if hasattr(self, "brush_label"):
            self.brush_label.config(text=f"{self.brush_radius}px")

    def _change_brush(self, delta: int) -> None:
        self.brush_radius = min(80, max(4, self.brush_radius + delta))
        self.brush_scale.set(self.brush_radius)
        self.brush_label.config(text=f"{self.brush_radius}px")

    def _load_current(self) -> None:
        item = self.items[self.index]
        self.current_image_bgr = cv2.imread(str(item.image_path))
        if self.current_image_bgr is None:
            raise RuntimeError(f"No se pudo leer la imagen: {item.image_path}")

        feedback_mask_path = self.feedback_dir / "masks" / f"{item.image_path.stem}_mask.png"
        if feedback_mask_path.is_file():
            mask = cv2.imread(str(feedback_mask_path), cv2.IMREAD_GRAYSCALE)
        elif item.mask_path and item.mask_path.is_file():
            mask = cv2.imread(str(item.mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros(self.current_image_bgr.shape[:2], dtype=np.uint8)

        if mask is None:
            mask = np.zeros(self.current_image_bgr.shape[:2], dtype=np.uint8)

        if mask.shape[:2] != self.current_image_bgr.shape[:2]:
            mask = cv2.resize(mask, (self.current_image_bgr.shape[1], self.current_image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        self.original_mask = mask.copy()
        self.current_mask = mask.copy()
        self._refresh_canvas()

    def _refresh_canvas(self) -> None:
        assert self.current_image_bgr is not None
        assert self.current_mask is not None

        overlay = alpha_overlay(self.current_image_bgr, self.current_mask)
        image_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        scale = min(CANVAS_W / w, CANVAS_H / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        self.display_scale = scale
        self.display_offset = ((CANVAS_W - new_w) // 2, (CANVAS_H - new_h) // 2)

        pil_img = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(self.display_offset[0], self.display_offset[1], anchor=tk.NW, image=self.photo)

        item = self.items[self.index]
        entry = self.manifest.get(item.image_path.name, {})
        estado = entry.get("status", "sin_guardar")
        self.status_var.set(
            f"Imagen {self.index + 1}/{len(self.items)}\n"
            f"{item.image_path.name}\n"
            f"Modo: {'añadir' if self.mode == 'add' else 'borrar'} | pincel={self.brush_radius}px\n"
            f"Estado feedback: {estado}"
        )

    def _canvas_to_image(self, x: int, y: int) -> tuple[int, int] | None:
        assert self.current_image_bgr is not None
        off_x, off_y = self.display_offset
        rel_x = x - off_x
        rel_y = y - off_y
        if rel_x < 0 or rel_y < 0:
            return None
        img_x = int(rel_x / self.display_scale)
        img_y = int(rel_y / self.display_scale)
        h, w = self.current_image_bgr.shape[:2]
        if not (0 <= img_x < w and 0 <= img_y < h):
            return None
        return img_x, img_y

    def _start_paint(self, event) -> None:
        self.dragging = True
        self._paint(event)

    def _paint(self, event) -> None:
        if not self.dragging or self.current_mask is None:
            return
        pt = self._canvas_to_image(event.x, event.y)
        if pt is None:
            return
        color = 255 if self.mode == "add" else 0
        cv2.circle(self.current_mask, pt, self.brush_radius, color, thickness=-1)
        self._refresh_canvas()

    def _stop_paint(self, _event) -> None:
        self.dragging = False

    def restore_original(self) -> None:
        if self.original_mask is not None:
            self.current_mask = self.original_mask.copy()
            self._refresh_canvas()

    def mark_empty(self) -> None:
        if self.current_mask is None:
            return
        self.current_mask[:] = 0
        self._refresh_canvas()

    def save_current(self) -> None:
        assert self.current_mask is not None
        item = self.items[self.index]
        masks_dir = self.feedback_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        target_mask = masks_dir / f"{item.image_path.stem}_mask.png"
        cv2.imwrite(str(target_mask), self.current_mask)

        nonzero = int(np.count_nonzero(self.current_mask))
        status = "empty" if nonzero == 0 else "corrected"
        self.manifest[item.image_path.name] = {
            "image": str(item.image_path.relative_to(REPO_ROOT)),
            "base_mask": str(item.mask_path.relative_to(REPO_ROOT)) if item.mask_path else None,
            "corrected_mask": str(target_mask.relative_to(REPO_ROOT)),
            "status": status,
            "nonzero_px": nonzero,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_manifest()
        self._refresh_canvas()

    def next_item(self) -> None:
        if self.index < len(self.items) - 1:
            self.index += 1
            self._load_current()

    def prev_item(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._load_current()


def build_items(images_dir: Path, masks_dir: Path, recursive: bool) -> list[Item]:
    items = []
    for image_path in iter_images(images_dir, recursive=recursive):
        mask_path = masks_dir / f"{image_path.stem}_mask.png"
        items.append(Item(image_path=image_path, mask_path=mask_path if mask_path.is_file() else None))
    return items


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interfaz de corrección humana para máscaras de ruedas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images", type=Path, required=True, help="Directorio con imágenes originales.")
    p.add_argument("--masks", type=Path, required=True, help="Directorio con máscaras predichas.")
    p.add_argument("--feedback-dir", type=Path, default=REPO_ROOT / "data" / "feedback" / "humano",
                   help="Directorio donde se guardan las correcciones humanas.")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    items = build_items(args.images, args.masks, recursive=args.recursive)
    if not items:
        logger.error("No se encontraron imágenes para revisar.")
        return 1

    root = tk.Tk()
    app = FeedbackApp(root, items, feedback_dir=args.feedback_dir, start_index=args.start_index)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
