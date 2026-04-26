from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scoring import CALIBRATION_FEATURE_COLUMNS
from storage import DB_DEFAULT_PATH, load_calibration_dataset


def _fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    n_rows, n_cols = X.shape
    X_aug = np.hstack([np.ones((n_rows, 1)), X])
    reg = np.eye(n_cols + 1) * float(alpha)
    reg[0, 0] = 0.0  # do not penalize intercept
    beta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
    intercept = float(beta[0])
    weights = beta[1:].astype(float)
    return intercept, weights


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _export_label_template(dataset: pd.DataFrame, export_path: Path) -> None:
    template = dataset[
        [
            "session_id",
            "mode_key",
            "started_at_utc",
            "final_risk_index",
            "profile_reliability_score",
            "profile_reliability_label",
        ]
    ].copy()
    template["target_risk_score"] = np.nan
    export_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(export_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate risk-score weights from stored sessions + external labels."
    )
    parser.add_argument("--db", default=DB_DEFAULT_PATH, help="Path to sqlite DB with stored sessions")
    parser.add_argument(
        "--labels",
        default="data/calibration_labels.csv",
        help="CSV with columns: session_id,target_risk_score",
    )
    parser.add_argument(
        "--output",
        default="calibration/score_weights.json",
        help="Where to save calibrated model",
    )
    parser.add_argument("--alpha", type=float, default=2.0, help="Ridge regularization alpha")
    parser.add_argument("--min-samples", type=int, default=30, help="Minimum labeled sessions for fitting")
    parser.add_argument(
        "--export-template",
        default="data/calibration_labels_template.csv",
        help="Export template for manual labels before fitting",
    )
    args = parser.parse_args()

    dataset = load_calibration_dataset(db_path=args.db)
    if dataset.empty:
        print("No completed sessions found in DB.")
        return 1

    if args.export_template:
        export_path = Path(args.export_template)
        _export_label_template(dataset, export_path)
        print(f"Label template exported: {export_path}")

    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"Labels file not found: {labels_path}")
        print("Fill the exported template and rerun calibration.")
        return 1

    labels = pd.read_csv(labels_path)
    required_cols = {"session_id", "target_risk_score"}
    missing_cols = required_cols.difference(labels.columns)
    if missing_cols:
        print(f"Labels file missing columns: {sorted(missing_cols)}")
        return 1

    merged = dataset.merge(labels[list(required_cols)], on="session_id", how="inner")
    merged["target_risk_score"] = pd.to_numeric(merged["target_risk_score"], errors="coerce")
    merged = merged.dropna(subset=["target_risk_score"]).copy()
    merged["target_risk_score"] = merged["target_risk_score"].clip(0.0, 100.0)

    if len(merged) < int(args.min_samples):
        print(f"Not enough labeled sessions: {len(merged)} < min-samples {args.min_samples}")
        return 1

    for col in CALIBRATION_FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0.0

    X = merged[CALIBRATION_FEATURE_COLUMNS].astype(float).to_numpy()
    y = merged["target_risk_score"].astype(float).to_numpy()
    intercept, weights = _fit_ridge(X, y, alpha=float(args.alpha))
    y_hat = np.clip(intercept + X @ weights, 0.0, 100.0)
    fit_metrics = _metrics(y, y_hat)

    model = {
        "model_version": "calibrated_linear_v1",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(merged)),
        "target_column": "target_risk_score",
        "alpha": float(args.alpha),
        "feature_names": CALIBRATION_FEATURE_COLUMNS,
        "intercept": float(intercept),
        "weights": [float(v) for v in weights],
        "blend_with_heuristic": 0.20,
        "fit_metrics": fit_metrics,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Calibrated model saved: {out_path}")
    print(f"Samples: {len(merged)} | RMSE: {fit_metrics['rmse']:.3f} | MAE: {fit_metrics['mae']:.3f} | R2: {fit_metrics['r2']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
