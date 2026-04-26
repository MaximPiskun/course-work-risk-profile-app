from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WEIGHT_COLS = ["w_cash", "w_bonds", "w_stocks", "w_gold"]
DEFAULT_SCORE_MODEL_VERSION = "heuristic_v2"
DEFAULT_CALIBRATION_MODEL_PATH = "calibration/score_weights.json"
SCORING_SCIENTIFIC_BASIS = [
    "Kahneman, D., & Tversky, A. (1979). Prospect Theory: An Analysis of Decision under Risk. Econometrica, 47(2), 263-291.",
    "Benartzi, S., & Thaler, R. H. (1995). Myopic Loss Aversion and the Equity Premium Puzzle. The Quarterly Journal of Economics, 110(1), 73-92.",
    "Grable, J. E., & Lytton, R. H. (1999). Financial risk tolerance revisited: the development of a risk assessment instrument. Financial Services Review, 8(3), 163-181.",
    "Barber, B. M., & Odean, T. (2000). Trading Is Hazardous to Your Wealth. The Journal of Finance, 55(2), 773-806.",
]
CALIBRATION_FEATURE_COLUMNS = [
    "avg_risky_weight",
    "max_drawdown_tolerated_before_derisk",
    "panic_actions_count",
    "recovery_risk_restore",
    "churn",
    "early_exit_flag",
    "avg_discomfort",
    "decision_flip_rate",
    "calm_market_rebalance_rate",
    "loss_aversion_proxy",
    "chaos_index",
    "investment_horizon_years_norm",
    "observations_count_norm",
]


@dataclass(frozen=True)
class BehaviorFeatures:
    avg_risky_weight: float
    max_drawdown_tolerated_before_derisk: float
    panic_actions_count: int
    recovery_risk_restore: float
    churn: float
    early_exit_flag: int
    avg_discomfort: float
    decision_flip_rate: float
    calm_market_rebalance_rate: float
    loss_aversion_proxy: float
    chaos_index: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


@lru_cache(maxsize=1)
def _load_calibration_model_from_disk(path: str = DEFAULT_CALIBRATION_MODEL_PATH) -> dict[str, Any] | None:
    model_path = Path(path)
    if not model_path.exists():
        return None
    raw = json.loads(model_path.read_text(encoding="utf-8"))
    feature_names = raw.get("feature_names")
    weights = raw.get("weights")
    if not isinstance(feature_names, list) or not isinstance(weights, list):
        return None
    if len(feature_names) != len(weights):
        return None
    return raw


def _build_calibration_feature_map(
    features: BehaviorFeatures,
    investment_horizon_years: int | None,
    observations_count: int,
) -> dict[str, float]:
    horizon = max(int(investment_horizon_years or 0), 0)
    return {
        "avg_risky_weight": float(features.avg_risky_weight),
        "max_drawdown_tolerated_before_derisk": float(features.max_drawdown_tolerated_before_derisk),
        "panic_actions_count": float(features.panic_actions_count),
        "recovery_risk_restore": float(features.recovery_risk_restore),
        "churn": float(features.churn),
        "early_exit_flag": float(features.early_exit_flag),
        "avg_discomfort": float(features.avg_discomfort),
        "decision_flip_rate": float(features.decision_flip_rate),
        "calm_market_rebalance_rate": float(features.calm_market_rebalance_rate),
        "loss_aversion_proxy": float(features.loss_aversion_proxy),
        "chaos_index": float(features.chaos_index),
        "investment_horizon_years_norm": float(np.clip(horizon / 20.0, 0.0, 1.0)),
        "observations_count_norm": float(np.clip(observations_count / 24.0, 0.0, 1.0)),
    }


def _predict_from_calibration_model(feature_map: dict[str, float], model: dict[str, Any]) -> float | None:
    feature_names = model.get("feature_names")
    weights = model.get("weights")
    if not isinstance(feature_names, list) or not isinstance(weights, list):
        return None
    if len(feature_names) != len(weights):
        return None
    if any(name not in feature_map for name in feature_names):
        return None

    x = np.array([_safe_float(feature_map[name]) for name in feature_names], dtype=float)
    w = np.array([_safe_float(v) for v in weights], dtype=float)
    intercept = _safe_float(model.get("intercept", 0.0))
    pred = intercept + float(np.dot(x, w))
    return _clip_score(pred)


def _risky_weight_series(log_df: pd.DataFrame) -> pd.Series:
    return log_df["w_stocks"].astype(float) + 0.5 * log_df["w_gold"].astype(float)


def _risky_delta_series(risky_weight: pd.Series) -> pd.Series:
    return risky_weight.diff().fillna(0.0)


def _first_derisk_drawdown(log_df: pd.DataFrame, risky_weight: pd.Series) -> float:
    risky_delta = _risky_delta_series(risky_weight)
    derisk = (risky_delta <= -0.15) | (log_df["go_to_cash"].astype(int) == 1)
    if derisk.any():
        idx = derisk.idxmax()
        return abs(float(log_df.loc[idx, "drawdown"]))
    return abs(float(log_df["drawdown"].min()))


def _panic_actions_count(log_df: pd.DataFrame, risky_weight: pd.Series) -> int:
    shock_mask = (log_df["portfolio_return"] <= -0.05) | (log_df["ret_stocks"] <= -0.10)
    shock_months = log_df.loc[shock_mask, "month_index"].astype(int).tolist()
    if not shock_months:
        return 0

    risk_delta = _risky_delta_series(risky_weight)
    panic_count = 0
    for month in shock_months:
        window_mask = log_df["month_index"].astype(int).isin([month, month + 1])
        panic_mask = (log_df["go_to_cash"].astype(int) == 1) | (risk_delta <= -0.15)
        if bool((window_mask & panic_mask).any()):
            panic_count += 1
    return int(panic_count)


def _recovery_risk_restore(log_df: pd.DataFrame, risky_weight: pd.Series) -> float:
    shock_mask = (log_df["portfolio_return"] <= -0.05) | (log_df["ret_stocks"] <= -0.10)
    if not shock_mask.any():
        return 1.0

    first_shock_idx = int(log_df.index[shock_mask][0])
    if first_shock_idx == 0:
        pre_shock_risk = float(risky_weight.iloc[0])
    else:
        pre_shock_risk = float(risky_weight.iloc[first_shock_idx - 1])

    post_shock = log_df.iloc[first_shock_idx + 1 :].copy()
    if post_shock.empty:
        return 0.0

    stock_recovery = (1.0 + post_shock["ret_stocks"]).cumprod() - 1.0
    recovered = post_shock.loc[stock_recovery >= 0.05]
    if recovered.empty:
        return 0.0

    recovery_index = int(recovered.index[0])
    restored_risk = float(risky_weight.loc[recovery_index:].max())
    threshold = 0.8 * max(pre_shock_risk, 0.05)
    return 1.0 if restored_risk >= threshold else 0.0


def _decision_flip_rate(risky_delta: pd.Series, threshold: float = 0.05) -> float:
    active = risky_delta.loc[risky_delta.abs() >= threshold]
    if len(active) < 2:
        return 0.0
    signs = np.sign(active.to_numpy())
    flips = np.sum(signs[1:] * signs[:-1] < 0)
    return float(flips / max(len(signs) - 1, 1))


def _calm_market_rebalance_rate(log_df: pd.DataFrame, risky_delta: pd.Series) -> float:
    calm_mask = (log_df["ret_stocks"].astype(float).abs() < 0.02) & (
        log_df["portfolio_return"].astype(float).abs() < 0.02
    )
    calm_count = int(calm_mask.sum())
    if calm_count == 0:
        return 0.0

    sharp_shift = (risky_delta.abs() >= 0.12) | (log_df["go_to_cash"].astype(int) == 1)
    return float((sharp_shift & calm_mask).sum() / calm_count)


def _loss_aversion_proxy(log_df: pd.DataFrame, risky_delta: pd.Series) -> float:
    """Higher value means stronger de-risking in losses vs gains."""
    loss_mask = (log_df["portfolio_return"] <= -0.05) | (log_df["ret_stocks"] <= -0.10)
    gain_mask = (log_df["portfolio_return"] >= 0.03) | (log_df["ret_stocks"] >= 0.06)

    loss_count = int(loss_mask.sum())
    gain_count = int(gain_mask.sum())
    if loss_count == 0:
        return 0.0

    derisk = (risky_delta <= -0.12) | (log_df["go_to_cash"].astype(int) == 1)
    loss_derisk_rate = float((derisk & loss_mask).sum() / loss_count)
    gain_derisk_rate = float((derisk & gain_mask).sum() / max(gain_count, 1))

    return float(np.clip(loss_derisk_rate - gain_derisk_rate, 0.0, 1.0))


def _chaos_index(churn: float, decision_flip_rate: float, calm_market_rebalance_rate: float) -> float:
    churn_norm = float(np.clip(churn / 0.65, 0.0, 1.0))
    return float(
        np.clip(
            0.45 * churn_norm + 0.35 * decision_flip_rate + 0.20 * calm_market_rebalance_rate,
            0.0,
            1.0,
        )
    )


def compute_behavior_features(log_df: pd.DataFrame) -> BehaviorFeatures:
    required = {
        "month_index",
        "ret_stocks",
        "portfolio_return",
        "drawdown",
        "go_to_cash",
        "end_early",
        "discomfort_1_5",
        *WEIGHT_COLS,
    }
    missing = required.difference(log_df.columns)
    if missing:
        raise ValueError(f"log_df is missing required columns: {sorted(missing)}")
    if log_df.empty:
        raise ValueError("log_df cannot be empty")

    safe_df = log_df.sort_values("month_index").reset_index(drop=True).copy()
    risky_weight = _risky_weight_series(safe_df)
    risky_delta = _risky_delta_series(risky_weight)
    avg_risky_weight = float(risky_weight.mean())
    dd_tolerated = _first_derisk_drawdown(safe_df, risky_weight)
    panic_count = _panic_actions_count(safe_df, risky_weight)
    restored = _recovery_risk_restore(safe_df, risky_weight)

    churn = float(safe_df[WEIGHT_COLS].diff().abs().sum(axis=1).fillna(0.0).mean())
    early_exit = int(safe_df["end_early"].astype(int).max())
    avg_discomfort = float(safe_df["discomfort_1_5"].astype(float).mean())
    flip_rate = _decision_flip_rate(risky_delta)
    calm_rebalance_rate = _calm_market_rebalance_rate(safe_df, risky_delta)
    loss_aversion_proxy = _loss_aversion_proxy(safe_df, risky_delta)
    chaos_index = _chaos_index(churn, flip_rate, calm_rebalance_rate)

    return BehaviorFeatures(
        avg_risky_weight=avg_risky_weight,
        max_drawdown_tolerated_before_derisk=dd_tolerated,
        panic_actions_count=panic_count,
        recovery_risk_restore=restored,
        churn=churn,
        early_exit_flag=early_exit,
        avg_discomfort=avg_discomfort,
        decision_flip_rate=flip_rate,
        calm_market_rebalance_rate=calm_rebalance_rate,
        loss_aversion_proxy=loss_aversion_proxy,
        chaos_index=chaos_index,
    )


def compute_risk_tolerance_score(
    features: BehaviorFeatures,
    investment_horizon_years: int | None = None,
) -> float:
    # Structured around risk capacity + behavior under losses:
    # strategic risk budget, drawdown tolerance, and post-shock restoration.
    risky_component = 45.0 * np.clip(features.avg_risky_weight / 0.80, 0.0, 1.0)
    dd_component = 30.0 * np.clip(features.max_drawdown_tolerated_before_derisk / 0.25, 0.0, 1.0)
    recovery_component = 15.0 * np.clip(features.recovery_risk_restore, 0.0, 1.0)
    horizon_bonus = 0.0
    if investment_horizon_years is not None:
        horizon_bonus = 10.0 * np.clip((investment_horizon_years - 1) / 14, 0.0, 1.0)
    return _clip_score(float(risky_component + dd_component + recovery_component + horizon_bonus))


def compute_behavioral_reactivity_score(features: BehaviorFeatures) -> float:
    # Higher score = stronger short-term behavioral reactions:
    # panic in losses, high turnover, and unstable decision direction.
    panic_component = 40.0 * np.clip(features.panic_actions_count / 2.0, 0.0, 1.0)
    churn_component = 15.0 * np.clip(features.churn / 0.60, 0.0, 1.0)
    flip_component = 15.0 * np.clip(features.decision_flip_rate, 0.0, 1.0)
    loss_aversion_component = 10.0 * np.clip(features.loss_aversion_proxy, 0.0, 1.0)
    discomfort_component = 10.0 * np.clip((features.avg_discomfort - 1.0) / 4.0, 0.0, 1.0)
    no_restore_component = 10.0 * (1.0 - np.clip(features.recovery_risk_restore, 0.0, 1.0))
    chaos_component = 10.0 * np.clip(features.chaos_index, 0.0, 1.0)
    early_exit_component = 15.0 * features.early_exit_flag
    return _clip_score(
        panic_component
        + churn_component
        + flip_component
        + loss_aversion_component
        + discomfort_component
        + no_restore_component
        + chaos_component
        + early_exit_component
    )


def combine_final_risk_index(
    risk_tolerance_score: float,
    behavioral_reactivity_score: float,
    chaos_index: float,
) -> float:
    # Behavioral reactivity and chaotic execution reduce actionable risk level.
    adjusted = (
        0.72 * risk_tolerance_score
        + 0.20 * (100.0 - behavioral_reactivity_score)
        + 0.08 * (100.0 - 100.0 * np.clip(chaos_index, 0.0, 1.0))
    )
    return _clip_score(adjusted)


def map_risk_group_3(score: float) -> str:
    if score <= 33:
        return "Conservative"
    if score <= 66:
        return "Moderate"
    return "Aggressive"


def map_risk_group_5(score: float) -> str:
    if score <= 20:
        return "Very Conservative"
    if score <= 40:
        return "Conservative"
    if score <= 60:
        return "Balanced"
    if score <= 80:
        return "Growth"
    return "Aggressive"


def compute_profile_reliability_score(features: BehaviorFeatures, observations_count: int) -> float:
    coverage_component = np.clip(observations_count / 12.0, 0.0, 1.0)
    stability_component = 1.0 - np.clip(features.chaos_index, 0.0, 1.0)
    completion_component = 1.0 - 0.7 * features.early_exit_flag

    score = 100.0 * (
        0.45 * coverage_component
        + 0.40 * stability_component
        + 0.15 * completion_component
    )
    if observations_count < 6:
        score -= 15.0
    return _clip_score(float(score))


def map_profile_reliability_label(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 45:
        return "Medium"
    return "Low"


def score_portfolio(
    log_df: pd.DataFrame,
    onboarding: dict[str, Any] | None = None,
    calibration_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features = compute_behavior_features(log_df)
    horizon = None
    if onboarding is not None:
        horizon = int(onboarding.get("investment_horizon_years", 0) or 0)
    risk_tolerance = compute_risk_tolerance_score(features, horizon)
    behavioral = compute_behavioral_reactivity_score(features)
    raw_final_index = combine_final_risk_index(risk_tolerance, behavioral, features.chaos_index)
    reliability_score = compute_profile_reliability_score(features, observations_count=len(log_df))
    reliability_label = map_profile_reliability_label(reliability_score)
    feature_map = _build_calibration_feature_map(features, horizon, observations_count=len(log_df))

    model = calibration_model if calibration_model is not None else _load_calibration_model_from_disk()
    calibration_applied = False
    score_model_version = DEFAULT_SCORE_MODEL_VERSION
    final_index = raw_final_index
    calibrated_index = None
    blend_with_heuristic = 0.0

    if model is not None:
        calibrated_index = _predict_from_calibration_model(feature_map, model)
        blend_with_heuristic = float(np.clip(_safe_float(model.get("blend_with_heuristic", 0.0)), 0.0, 1.0))
        if calibrated_index is not None:
            final_index = _clip_score(blend_with_heuristic * raw_final_index + (1.0 - blend_with_heuristic) * calibrated_index)
            calibration_applied = True
            score_model_version = str(model.get("model_version", "calibrated_linear_v1"))

    return {
        "features": features.to_dict(),
        "calibration_features": feature_map,
        "risk_tolerance_score": risk_tolerance,
        "behavioral_reactivity_score": behavioral,
        "raw_final_risk_index": raw_final_index,
        "final_risk_index": final_index,
        "risk_group_3": map_risk_group_3(final_index),
        "risk_group_5": map_risk_group_5(final_index),
        "profile_reliability_score": reliability_score,
        "profile_reliability_label": reliability_label,
        "score_model_version": score_model_version,
        "calibration_applied": calibration_applied,
        "calibrated_final_risk_index": calibrated_index,
        "calibration_blend_with_heuristic": blend_with_heuristic,
        "scientific_basis": SCORING_SCIENTIFIC_BASIS,
    }
