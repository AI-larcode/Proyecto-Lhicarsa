#!/usr/bin/env python3
"""
train_pressure_classifier.py
============================

Entrena un clasificador para decidir si una rueda requiere revisión por baja
presión a partir del CSV de características geométricas generado por
``yoloe_seg_predict.py --features``.

Estrategia:
  * Validación cruzada con GroupKFold por identificador de rueda.
  * Modelo: HistGradientBoostingClassifier.
  * Evaluación operativa: precisión, recall, F1, ROC-AUC y matriz de confusión.
  * Comparación automática de varios umbrales de presión para decidir qué
    definición de ``requiere_revision`` funciona mejor.
  * Modelo final reentrenado sobre todos los datos con el umbral elegido.

Ejemplo
-------
    python train_pressure_classifier.py \\
        --features ./artifacts/segmentacion/resultados_strict2/features.csv \\
        --output   ./models/classification/modelo_revision.joblib \\
        --threshold 6.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

# Columnas que NO se usan como características
NON_FEATURE_COLS = {
    "image", "truck_id", "view", "pressure_bar",
    "truck_type", "brand", "composite_id",
    "width", "height", "num_detections",
    "largest_mask_area_px", "largest_mask_ratio",
    "view_one_hot",
}

logger = logging.getLogger("train_pressure")
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Entrada/salida y limpieza
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dataset(
    csv_path: Path,
    view_filter: str | None,
    drop_views: list[str],
    one_hot_view: bool,
    one_hot_type: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Carga el CSV, descarta filas no etiquetadas y aplica filtros de vista."""
    df = pd.read_csv(csv_path)
    n0 = len(df)
    df = df.dropna(subset=["pressure_bar", "composite_id"])
    df = df[df["largest_mask_area_px"].fillna(0) > 0]
    if view_filter:
        df = df[df["view"].str.casefold() == view_filter.casefold()]
    if drop_views:
        df = df[~df["view"].str.casefold().isin([v.casefold() for v in drop_views])]
    n1 = len(df)
    logger.info("Filas: %d → %d tras limpieza/filtros.", n0, n1)
    if n1 == 0:
        raise RuntimeError("No quedan datos tras los filtros.")

    df = df.reset_index(drop=True)
    df["composite_id"] = df["composite_id"].astype(int)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    if one_hot_view:
        view_dum = pd.get_dummies(df["view"], prefix="view", dtype=float)
        df = pd.concat([df, view_dum], axis=1)
        feature_cols += list(view_dum.columns)

    if one_hot_type:
        for col_name, prefix in [("truck_type", "ttype"), ("brand", "brand")]:
            if col_name in df.columns:
                dum = pd.get_dummies(df[col_name], prefix=prefix, dtype=float)
                df = pd.concat([df, dum], axis=1)
                feature_cols += list(dum.columns)

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=feature_cols)
    logger.info("Características (%d): %s", len(feature_cols), feature_cols)
    return df, feature_cols


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------

def make_model(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        max_iter=400,
        learning_rate=0.05,
        max_depth=None,
        max_leaf_nodes=15,
        min_samples_leaf=2,
        l2_regularization=0.1,
        random_state=seed,
    )


def make_binary_target(pressure: pd.Series, threshold: float) -> np.ndarray:
    """1 si la rueda requiere revisión por presión baja."""
    return (pressure <= threshold).astype(int).to_numpy()


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, float | int] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def cross_validate_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
    n_splits: int | None = None,
    seed: int = 0,
    decision_threshold: float = 0.5,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Valida el clasificador por grupos de rueda."""
    X = df[feature_cols].to_numpy()
    y = make_binary_target(df["pressure_bar"], threshold=threshold)
    groups = df["composite_id"].to_numpy()

    positivos = int(y.sum())
    negativos = int(len(y) - positivos)
    logger.info(
        "Umbral %.2f bar -> revisar=%d, no_revisar=%d",
        threshold, positivos, negativos,
    )
    if positivos == 0 or negativos == 0:
        raise RuntimeError(
            f"El umbral {threshold:.2f} genera una sola clase. Ajusta los umbrales."
        )

    n_groups = len(np.unique(groups))
    k = n_splits or min(5, n_groups)
    if k < 2:
        raise RuntimeError(
            f"Se necesitan ≥2 ruedas distintas para la validación cruzada por grupo (hay {n_groups})."
        )
    logger.info("GroupKFold con k=%d (ruedas únicas=%d).", k, n_groups)

    gkf = GroupKFold(n_splits=k)
    oof_score = np.full(len(y), fill_value=np.nan, dtype=float)
    oof_pred = np.zeros(len(y), dtype=int)
    fold_metrics = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        model = make_model(seed=seed + fold)
        model.fit(X[tr], y[tr])
        score = model.predict_proba(X[te])[:, 1]
        pred = (score >= decision_threshold).astype(int)
        oof_score[te] = score
        oof_pred[te] = pred

        metrics = _classification_metrics(y[te], pred, score)
        metrics["fold"] = fold
        metrics["test_trucks"] = sorted(set(int(g) for g in groups[te]))
        metrics["n_test"] = int(len(te))
        fold_metrics.append(metrics)
        logger.info(
            "  pliegue %d  ruedas_prueba=%s  n=%d  precision=%.3f  recall=%.3f  f1=%.3f  roc_auc=%.3f",
            fold, metrics["test_trucks"], metrics["n_test"],
            metrics["precision"], metrics["recall"], metrics["f1"], metrics["roc_auc"],
        )

    overall = _classification_metrics(y, oof_pred, oof_score)
    overall["positive_rate"] = float(y.mean())
    overall["fold_metrics"] = fold_metrics
    logger.info(
        "CV global  precision=%.3f  recall=%.3f  f1=%.3f  roc_auc=%.3f  tp=%d fp=%d tn=%d fn=%d",
        overall["precision"], overall["recall"], overall["f1"], overall["roc_auc"],
        overall["tp"], overall["fp"], overall["tn"], overall["fn"],
    )

    oof_df = pd.DataFrame({
        "composite_id": groups,
        "pressure_bar": df["pressure_bar"].to_numpy(),
        "y_true": y,
        "y_score": oof_score,
        "y_pred": oof_pred,
    })
    return overall, oof_df


def feature_importance(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
    n_repeats: int = 30,
    seed: int = 0,
) -> pd.DataFrame:
    """Importancia por permutación sobre el clasificador entrenado en todos los datos."""
    X = df[feature_cols].to_numpy()
    y = make_binary_target(df["pressure_bar"], threshold=threshold)
    model = make_model(seed=seed).fit(X, y)
    res = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, scoring="f1",
    )
    return (
        pd.DataFrame({
            "feature": feature_cols,
            "importance_mean": res.importances_mean,
            "importance_std": res.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def per_pressure_summary(oof_df: pd.DataFrame) -> pd.DataFrame:
    g = oof_df.groupby(oof_df["pressure_bar"].round(2))
    return (
        pd.DataFrame({
            "n": g.size(),
            "prob_media_revision": g["y_score"].mean().round(3),
            "recall_revision": g["y_pred"].mean().round(3),
        })
        .reset_index()
        .rename(columns={"pressure_bar": "pressure_bar"})
    )


def evaluate_thresholds(
    df: pd.DataFrame,
    feature_cols: list[str],
    thresholds: list[float],
    n_splits: int | None,
    seed: int,
    decision_threshold: float,
) -> tuple[list[dict[str, object]], float]:
    results: list[dict[str, object]] = []
    for threshold in thresholds:
        overall, _ = cross_validate_classifier(
            df,
            feature_cols,
            threshold=threshold,
            n_splits=n_splits,
            seed=seed,
            decision_threshold=decision_threshold,
        )
        result = {
            "threshold": threshold,
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
            "roc_auc": overall["roc_auc"],
            "positive_rate": overall["positive_rate"],
        }
        results.append(result)

    ranked = sorted(
        results,
        key=lambda item: (item["f1"], item["recall"], item["roc_auc"]),
        reverse=True,
    )
    best_threshold = float(ranked[0]["threshold"])
    return results, best_threshold


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrena un clasificador de revisión de presión sobre características geométricas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--features", type=Path, required=True,
                   help="CSV generado por yoloe_seg_predict.py --features")
    p.add_argument("--output", type=Path, default=REPO_ROOT / "models" / "classification" / "modelo_revision.joblib",
                   help="Ruta del clasificador final (joblib).")
    p.add_argument("--view", type=str, default="Frontal",
                   help="Filtrar a una sola vista (Frontal/Izq/Der). "
                        "Por defecto solo se usan imágenes frontales.")
    p.add_argument("--drop-view", action="append", default=[],
                   help="Vista(s) a excluir (puede repetirse).")
    p.add_argument("--one-hot-view", action="store_true",
                   help="Codificar la vista como variable one-hot.")
    p.add_argument("--one-hot-type", action="store_true",
                   help="Codificar truck_type y brand como variables one-hot.")
    p.add_argument("--folds", type=int, default=None,
                   help="Nº de pliegues GroupKFold (por defecto: min(5, nº ruedas)).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threshold", type=float, default=None,
                   help="Umbral fijo de presión. Si no se indica, se comparan varios.")
    p.add_argument("--candidate-thresholds", type=float, nargs="+",
                   default=[5.5, 6.0, 6.25],
                   help="Umbrales candidatos si no se fija --threshold.")
    p.add_argument("--decision-threshold", type=float, default=0.5,
                   help="Umbral de probabilidad para clasificar una rueda como revisar.")
    p.add_argument("--report-json", type=Path, default=REPO_ROOT / "artifacts" / "reports" / "report_revision.json",
                   help="Ruta opcional para volcar el informe completo en JSON.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    if not args.features.is_file():
        logger.error("No se encuentra el CSV: %s", args.features)
        return 2

    df, feature_cols = load_dataset(
        args.features,
        view_filter=args.view,
        drop_views=args.drop_view,
        one_hot_view=args.one_hot_view,
        one_hot_type=args.one_hot_type,
    )
    logger.info(
        "Distribución por presión:\n%s",
        df["pressure_bar"].value_counts().sort_index().to_string(),
    )
    logger.info(
        "Ruedas disponibles (composite_id): %s",
        sorted(df["composite_id"].unique().tolist()),
    )

    if args.threshold is None:
        threshold_results, best_threshold = evaluate_thresholds(
            df,
            feature_cols,
            thresholds=args.candidate_thresholds,
            n_splits=args.folds,
            seed=args.seed,
            decision_threshold=args.decision_threshold,
        )
        logger.info(
            "Comparativa de umbrales:\n%s",
            pd.DataFrame(threshold_results).sort_values("threshold").to_string(index=False),
        )
        logger.info("Mejor umbral seleccionado: %.2f bar", best_threshold)
    else:
        threshold_results = []
        best_threshold = args.threshold

    overall, oof_df = cross_validate_classifier(
        df,
        feature_cols,
        threshold=best_threshold,
        n_splits=args.folds,
        seed=args.seed,
        decision_threshold=args.decision_threshold,
    )

    per_pressure = per_pressure_summary(oof_df)
    logger.info(
        "Resumen por presión:\n%s",
        per_pressure.to_string(index=False),
    )

    importance = feature_importance(
        df,
        feature_cols,
        threshold=best_threshold,
        seed=args.seed,
    )
    logger.info(
        "Características principales (importancia por permutación):\n%s",
        importance.head(10).to_string(index=False),
    )

    final_model = make_model(seed=args.seed)
    y_final = make_binary_target(df["pressure_bar"], threshold=best_threshold)
    final_model.fit(df[feature_cols].to_numpy(), y_final)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": final_model,
        "feature_cols": feature_cols,
        "view_filter": args.view,
        "one_hot_view": args.one_hot_view,
        "one_hot_type": args.one_hot_type,
        "trained_on_n": int(len(df)),
        "trucks": sorted(df["composite_id"].unique().tolist()),
        "pressure_threshold": best_threshold,
        "decision_threshold": args.decision_threshold,
        "cv_metrics": {k: v for k, v in overall.items() if k != "fold_metrics"},
    }
    joblib.dump(payload, args.output)
    logger.info("Modelo guardado en %s", args.output)

    if args.report_json is not None:
        report = {
            "n_samples": int(len(df)),
            "feature_cols": feature_cols,
            "trucks": sorted(df["composite_id"].unique().tolist()),
            "selected_pressure_threshold": best_threshold,
            "decision_threshold": args.decision_threshold,
            "threshold_comparison": threshold_results,
            "cv_overall": {k: v for k, v in overall.items() if k != "fold_metrics"},
            "cv_folds": overall["fold_metrics"],
            "per_pressure": per_pressure.to_dict(orient="records"),
            "permutation_importance": importance.to_dict(orient="records"),
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Informe JSON guardado en %s", args.report_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
