from __future__ import annotations

import pandas as pd

from scoring import score_portfolio


def _build_base_row(month: int) -> dict:
    return {
        "month_index": month,
        "ret_cash": 0.001,
        "ret_bonds": 0.002,
        "ret_stocks": 0.010,
        "ret_gold": 0.003,
        "w_cash": 0.10,
        "w_bonds": 0.10,
        "w_stocks": 0.70,
        "w_gold": 0.10,
        "portfolio_return": 0.008,
        "portfolio_value": 100000 * (1.008**month),
        "drawdown": -0.01 if month % 6 == 0 else 0.0,
        "volatility_estimate": 0.14,
        "rebalance": 0,
        "go_to_cash": 0,
        "reduce_risk": 0,
        "end_early": 0,
        "discomfort_1_5": 2,
    }


def test_scoring_high_risk_stable_behavior() -> None:
    rows = [_build_base_row(m) for m in range(1, 13)]
    df = pd.DataFrame(rows)

    out = score_portfolio(df, onboarding={"investment_horizon_years": 15}, calibration_model={})
    assert out["risk_tolerance_score"] > 65
    assert out["behavioral_reactivity_score"] < 35
    assert out["risk_group_3"] in {"Moderate", "Aggressive"}
    assert out["score_model_version"] == "heuristic_v2"
    assert out["calibration_applied"] is False


def test_scoring_panic_behavior() -> None:
    rows = []
    for m in range(1, 7):
        row = _build_base_row(m)
        if m == 4:
            row["ret_stocks"] = -0.22
            row["portfolio_return"] = -0.15
            row["drawdown"] = -0.20
        if m >= 5:
            row["w_cash"] = 0.90
            row["w_bonds"] = 0.05
            row["w_stocks"] = 0.03
            row["w_gold"] = 0.02
            row["go_to_cash"] = 1 if m == 5 else 0
            row["reduce_risk"] = 1
            row["discomfort_1_5"] = 5
        if m == 6:
            row["end_early"] = 1
        rows.append(row)

    df = pd.DataFrame(rows)
    out = score_portfolio(df, onboarding={"investment_horizon_years": 2}, calibration_model={})
    assert out["risk_tolerance_score"] < 55
    assert out["behavioral_reactivity_score"] > 55
    assert out["risk_group_3"] in {"Conservative", "Moderate"}
    assert out["profile_reliability_label"] in {"Low", "Medium"}


def test_scoring_uses_calibrated_model_when_available() -> None:
    rows = [_build_base_row(m) for m in range(1, 13)]
    df = pd.DataFrame(rows)

    model = {
        "model_version": "calibrated_linear_v1_test",
        "feature_names": [
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
        ],
        "weights": [0.0] * 13,
        "intercept": 80.0,
        "blend_with_heuristic": 0.0,
    }

    out = score_portfolio(df, onboarding={"investment_horizon_years": 15}, calibration_model=model)
    assert out["calibration_applied"] is True
    assert out["score_model_version"] == "calibrated_linear_v1_test"
    assert out["final_risk_index"] == 80.0


def test_reliability_decreases_for_short_and_early_exit_paths() -> None:
    long_rows = [_build_base_row(m) for m in range(1, 13)]
    long_df = pd.DataFrame(long_rows)
    long_out = score_portfolio(long_df, onboarding={"investment_horizon_years": 10}, calibration_model={})

    short_rows = []
    for m in range(1, 5):
        row = _build_base_row(m)
        row["end_early"] = 1 if m == 4 else 0
        row["discomfort_1_5"] = 5
        short_rows.append(row)
    short_df = pd.DataFrame(short_rows)
    short_out = score_portfolio(short_df, onboarding={"investment_horizon_years": 10}, calibration_model={})

    assert short_out["profile_reliability_score"] < long_out["profile_reliability_score"]
    assert short_out["profile_reliability_label"] in {"Low", "Medium"}


def test_scoring_extreme_profiles_are_reachable() -> None:
    aggressive_rows = []
    for m in range(1, 40):
        aggressive_rows.append(
            {
                "month_index": m,
                "ret_cash": 0.0003,
                "ret_bonds": 0.002,
                "ret_stocks": 0.01,
                "ret_gold": 0.004,
                "w_cash": 0.0,
                "w_bonds": 0.0,
                "w_stocks": 1.0,
                "w_gold": 0.0,
                "portfolio_return": 0.012,
                "portfolio_value": 100_000 * (1.012**m),
                "drawdown": -0.30,
                "volatility_estimate": 0.14,
                "rebalance": 0,
                "go_to_cash": 0,
                "reduce_risk": 0,
                "end_early": 0,
                "discomfort_1_5": 1,
            }
        )
    aggressive_out = score_portfolio(
        pd.DataFrame(aggressive_rows),
        onboarding={"investment_horizon_years": 30},
        calibration_model={},
    )

    conservative_rows = []
    for m in range(1, 8):
        conservative_rows.append(
            {
                "month_index": m,
                "ret_cash": 0.0,
                "ret_bonds": -0.02,
                "ret_stocks": -0.20,
                "ret_gold": -0.05,
                "w_cash": 1.0,
                "w_bonds": 0.0,
                "w_stocks": 0.0,
                "w_gold": 0.0,
                "portfolio_return": -0.18,
                "portfolio_value": 100_000 * (0.95**m),
                "drawdown": -0.02,
                "volatility_estimate": 0.25,
                "rebalance": 0,
                "go_to_cash": 1,
                "reduce_risk": 1,
                "end_early": 1 if m == 7 else 0,
                "discomfort_1_5": 5,
            }
        )
    conservative_out = score_portfolio(
        pd.DataFrame(conservative_rows),
        onboarding={"investment_horizon_years": 1},
        calibration_model={},
    )

    low_reliability_rows = []
    alt_weights = [
        (1.0, 0.0, 0.0, 0.0),
        (0.05, 0.05, 0.85, 0.05),
        (1.0, 0.0, 0.0, 0.0),
        (0.05, 0.05, 0.85, 0.05),
        (1.0, 0.0, 0.0, 0.0),
    ]
    for m, w in enumerate(alt_weights, start=1):
        low_reliability_rows.append(
            {
                "month_index": m,
                "ret_cash": 0.0003,
                "ret_bonds": -0.01 if m % 2 else 0.01,
                "ret_stocks": -0.18 if m % 2 else 0.12,
                "ret_gold": -0.04 if m % 2 else 0.05,
                "w_cash": w[0],
                "w_bonds": w[1],
                "w_stocks": w[2],
                "w_gold": w[3],
                "portfolio_return": -0.12 if m % 2 else 0.08,
                "portfolio_value": 100_000 * (0.99**m),
                "drawdown": -0.25 if m % 2 else -0.05,
                "volatility_estimate": 0.3,
                "rebalance": 1 if m % 2 == 0 else 0,
                "go_to_cash": 1 if m in (1, 3, 5) else 0,
                "reduce_risk": 1 if m % 2 else 0,
                "end_early": 1 if m == 5 else 0,
                "discomfort_1_5": 5,
            }
        )
    low_reliability_out = score_portfolio(
        pd.DataFrame(low_reliability_rows),
        onboarding={"investment_horizon_years": 1},
        calibration_model={},
    )

    assert aggressive_out["risk_group_5"] == "Aggressive"
    assert conservative_out["risk_group_5"] == "Very Conservative"
    assert low_reliability_out["profile_reliability_label"] == "Low"
