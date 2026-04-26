from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

WEIGHT_COLS = ["w_cash", "w_bonds", "w_stocks", "w_gold"]


@dataclass(frozen=True)
class BehaviorFeatures:
    avg_risky_weight: float
    max_drawdown_tolerated_before_derisk: float
    panic_actions_count: int
    recovery_risk_restore: float
    churn: float
    early_exit_flag: int
    avg_discomfort: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip_score(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _risky_weight_series(log_df: pd.DataFrame) -> pd.Series:
    return log_df["w_stocks"].astype(float) + 0.5 * log_df["w_gold"].astype(float)


def _first_derisk_drawdown(log_df: pd.DataFrame, risky_weight: pd.Series) -> float:
    risky_delta = risky_weight.diff().fillna(0.0)
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

    risk_delta = risky_weight.diff().fillna(0.0)
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
    avg_risky_weight = float(risky_weight.mean())
    dd_tolerated = _first_derisk_drawdown(safe_df, risky_weight)
    panic_count = _panic_actions_count(safe_df, risky_weight)
    restored = _recovery_risk_restore(safe_df, risky_weight)

    churn = float(safe_df[WEIGHT_COLS].diff().abs().sum(axis=1).fillna(0.0).mean())
    early_exit = int(safe_df["end_early"].astype(int).max())
    avg_discomfort = float(safe_df["discomfort_1_5"].astype(float).mean())

    return BehaviorFeatures(
        avg_risky_weight=avg_risky_weight,
        max_drawdown_tolerated_before_derisk=dd_tolerated,
        panic_actions_count=panic_count,
        recovery_risk_restore=restored,
        churn=churn,
        early_exit_flag=early_exit,
        avg_discomfort=avg_discomfort,
    )


def compute_risk_tolerance_score(
    features: BehaviorFeatures,
    investment_horizon_years: int | None = None,
) -> float:
    # Main driver is average risky allocation.
    risky_component = 60.0 * np.clip(features.avg_risky_weight / 0.80, 0.0, 1.0)
    # Drawdown tolerance rewards users who keep risk through losses.
    dd_component = 30.0 * np.clip(features.max_drawdown_tolerated_before_derisk / 0.25, 0.0, 1.0)
    horizon_bonus = 0.0
    if investment_horizon_years is not None:
        horizon_bonus = 10.0 * np.clip((investment_horizon_years - 1) / 14, 0.0, 1.0)
    return _clip_score(float(risky_component + dd_component + horizon_bonus))


def compute_behavioral_reactivity_score(features: BehaviorFeatures) -> float:
    # Higher score = stronger short-term behavioral reactions.
    panic_component = 45.0 * np.clip(features.panic_actions_count / 3.0, 0.0, 1.0)
    churn_component = 25.0 * np.clip(features.churn / 0.60, 0.0, 1.0)
    early_exit_component = 20.0 * features.early_exit_flag
    no_restore_component = 10.0 * (1.0 - features.recovery_risk_restore)
    discomfort_component = 10.0 * np.clip((features.avg_discomfort - 1.0) / 4.0, 0.0, 1.0)
    return _clip_score(
        panic_component
        + churn_component
        + early_exit_component
        + no_restore_component
        + discomfort_component
    )


def combine_final_risk_index(risk_tolerance_score: float, behavioral_reactivity_score: float) -> float:
    # Behavioral reactivity reduces final actionable risk level.
    adjusted = 0.75 * risk_tolerance_score + 0.25 * (100.0 - behavioral_reactivity_score)
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


def score_portfolio(log_df: pd.DataFrame, onboarding: dict[str, Any] | None = None) -> dict[str, Any]:
    features = compute_behavior_features(log_df)
    horizon = None
    if onboarding is not None:
        horizon = int(onboarding.get("investment_horizon_years", 0) or 0)
    risk_tolerance = compute_risk_tolerance_score(features, horizon)
    behavioral = compute_behavioral_reactivity_score(features)
    final_index = combine_final_risk_index(risk_tolerance, behavioral)

    return {
        "features": features.to_dict(),
        "risk_tolerance_score": risk_tolerance,
        "behavioral_reactivity_score": behavioral,
        "final_risk_index": final_index,
        "risk_group_3": map_risk_group_3(final_index),
        "risk_group_5": map_risk_group_5(final_index),
    }

