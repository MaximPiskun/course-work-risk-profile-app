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

    out = score_portfolio(df, onboarding={"investment_horizon_years": 15})
    assert out["risk_tolerance_score"] > 65
    assert out["behavioral_reactivity_score"] < 35
    assert out["risk_group_3"] in {"Moderate", "Aggressive"}


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
    out = score_portfolio(df, onboarding={"investment_horizon_years": 2})
    assert out["risk_tolerance_score"] < 55
    assert out["behavioral_reactivity_score"] > 60
    assert out["risk_group_3"] in {"Conservative", "Moderate"}
