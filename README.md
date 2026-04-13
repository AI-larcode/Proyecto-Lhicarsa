# Proyecto Lhicarsa

Flujo actual para segmentacion de ruedas, correccion humana y clasificacion de revision de presion.

## Estructura

- `data/images/Lhicarsa_imagenes`: imagenes originales.
- `data/datasets`: datasets YOLO-seg preparados.
- `data/feedback/humano`: correcciones humanas y `manifest.json`.
- `artifacts/segmentacion`: resultados de inferencia, mascaras y overlays.
- `artifacts/reports`: informes JSON de entrenamiento y evaluacion.
- `models/base`: pesos base de YOLO y YOLOE.
- `models/classification`: clasificadores `joblib`.
- `experiments`: salidas de fine-tuning y experimentos largos.
- `tools/segmentation`: scripts de inferencia de segmentacion.
- `tools/datasets`: scripts para preparar datasets.
- `tools/training`: scripts de entrenamiento.
- `apps/feedback`: interfaz grafica de correccion.
- `pipelines`: ejecuciones encadenadas de entrenamiento y mejora.
- `legacy`: material antiguo que ya no forma parte del flujo actual.

## Entorno

El proyecto usa `uv` para gestionar dependencias.

```bash
uv sync
```

## Flujo principal

1. Generar segmentaciones iniciales:

```bash
uv run python tools/segmentation/yoloe_seg_predict.py \
  --source ./data/images/Lhicarsa_imagenes \
  --output ./artifacts/segmentacion/resultados_manual \
  --recursive \
  --features ./artifacts/segmentacion/resultados_manual/features.csv
```

2. Preparar dataset YOLO-seg:

```bash
uv run python tools/datasets/prepare_yolo_seg_dataset.py \
  --images ./data/images/Lhicarsa_imagenes \
  --masks ./artifacts/segmentacion/resultados_strict2 \
  --output ./data/datasets/ruedas_yolo_strict \
  --recursive
```

3. Fine-tuning del segmentador:

```bash
uv run python tools/training/train_yolo_wheel_seg.py \
  --data ./data/datasets/ruedas_yolo_strict/data.yaml \
  --model ./models/base/yolo26n-seg.pt \
  --epochs 60
```

4. Entrenar el clasificador de revision:

```bash
uv run python tools/training/train_pressure_classifier.py \
  --features ./artifacts/segmentacion/resultados_strict2/features.csv \
  --one-hot-view \
  --one-hot-type
```

## Feedback humano

Abrir la interfaz grafica:

```bash
uv run python apps/feedback/wheel_feedback_gui.py \
  --images ./data/images/Lhicarsa_imagenes \
  --masks ./artifacts/segmentacion/resultados_iter_v1 \
  --feedback-dir ./data/feedback/humano \
  --recursive
```

Atajos principales:

- `n` o flecha derecha: siguiente imagen
- `p` o flecha izquierda: imagen anterior
- `a`: anadir mascara
- `e`: borrar mascara
- `s`: guardar
- `r`: restaurar mascara original
- `x`: marcar sin rueda

## Pipelines

Iteracion completa:

```bash
uv run python pipelines/run_iterative_pressure_pipeline.py --recursive
```

Fine-tuning incremental con feedback:

```bash
uv run python pipelines/run_feedback_finetune.py \
  --model ./experiments/runs/segment/runs_seg_ruedas/iteracion_ruedas_v1/weights/best.pt
```

## Notas

- `tools/training/train_pressure_classifier.py` sustituye al antiguo flujo de regresion continua.
- `legacy/` conserva scripts previos para referencia, pero no forman parte de la fase actual.
