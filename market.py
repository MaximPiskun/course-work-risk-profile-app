from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

ASSETS = ["Cash", "Bonds", "Stocks", "Gold"]


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    mean: dict[str, float]
    vol: dict[str, float]
    crash_probability: float = 0.0
    crash_stock_impact: float = 0.0
    crash_bond_impact: float = 0.0
    crash_gold_impact: float = 0.0
    small_drawdown_probability: float = 0.0
    mean_reversion_strength: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REGIME_LIBRARY: dict[str, RegimeSpec] = {
    "Bull": RegimeSpec(
        name="Bull",
        mean={"Cash": 0.0012, "Bonds": 0.0028, "Stocks": 0.0100, "Gold": 0.0030},
        vol={"Cash": 0.0004, "Bonds": 0.0120, "Stocks": 0.0450, "Gold": 0.0280},
        small_drawdown_probability=0.14,
    ),
    "Crisis": RegimeSpec(
        name="Crisis",
        mean={"Cash": 0.0010, "Bonds": 0.0022, "Stocks": 0.0010, "Gold": 0.0025},
        vol={"Cash": 0.0005, "Bonds": 0.0100, "Stocks": 0.0800, "Gold": 0.0450},
        crash_probability=0.08,
        crash_stock_impact=-0.24,
        crash_bond_impact=-0.012,
        crash_gold_impact=0.03,
    ),
    "Sideways": RegimeSpec(
        name="Sideways",
        mean={"Cash": 0.0010, "Bonds": 0.0018, "Stocks": 0.0015, "Gold": 0.0018},
        vol={"Cash": 0.0003, "Bonds": 0.0100, "Stocks": 0.0500, "Gold": 0.0320},
        small_drawdown_probability=0.24,
        mean_reversion_strength=0.35,
    ),
}


def _standardized_t_shocks(rng: np.random.Generator, size: int, df: int = 5) -> np.ndarray:
    """Student-t shocks rescaled to approx unit variance."""
    raw = rng.standard_t(df=df, size=size)
    scale = np.sqrt(df / (df - 2))
    return raw / scale


def _clamp_returns(returns: dict[str, float]) -> dict[str, float]:
    return {k: float(np.clip(v, -0.95, 0.95)) for k, v in returns.items()}


def _generate_crash_month(rng: np.random.Generator, months: int) -> int:
    if months <= 1:
        return 1
    if months <= 6:
        return int(rng.integers(1, months + 1))

    low = 6
    high = max(low, min(months - 3, 16))
    return int(rng.integers(low, high + 1))


def _normalize_regime_weights(
    available_regimes: list[str],
    regime_weights: dict[str, float] | None,
) -> np.ndarray:
    if not regime_weights:
        return np.repeat(1.0 / len(available_regimes), len(available_regimes))

    raw = np.array([max(0.0, float(regime_weights.get(name, 0.0))) for name in available_regimes], dtype=float)
    total = float(raw.sum())
    if total <= 0:
        return np.repeat(1.0 / len(available_regimes), len(available_regimes))
    return raw / total


def _build_regime_switch_plan(
    total_steps: int,
    rng: np.random.Generator,
    segment_range: tuple[int, int],
    regime_weights: dict[str, float] | None = None,
) -> list[tuple[str, int]]:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")

    seg_low, seg_high = segment_range
    if seg_low < 1 or seg_high < seg_low:
        raise ValueError("segment_range must satisfy 1 <= low <= high")

    available_regimes = list(REGIME_LIBRARY.keys())
    remaining = total_steps
    prev_regime: str | None = None
    plan: list[tuple[str, int]] = []

    while remaining > 0:
        seg_len = int(rng.integers(seg_low, seg_high + 1))
        seg_len = min(seg_len, remaining)

        candidates = available_regimes
        if prev_regime is not None and len(available_regimes) > 1:
            candidates = [name for name in available_regimes if name != prev_regime]

        probs = _normalize_regime_weights(candidates, regime_weights)
        next_regime = str(rng.choice(candidates, p=probs))

        plan.append((next_regime, seg_len))
        prev_regime = next_regime
        remaining -= seg_len

    return plan


def generate_episode_returns(
    regime: str,
    months: int = 36,
    seed: int = 42,
    use_student_t: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if regime not in REGIME_LIBRARY:
        raise ValueError(f"Unknown regime: {regime}")
    if months < 1:
        raise ValueError("months must be >= 1")

    spec = REGIME_LIBRARY[regime]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    crash_month: int | None = None
    if regime == "Crisis":
        crash_month = _generate_crash_month(rng, months)

    prev_stock = 0.0
    prev_gold = 0.0

    for month in range(1, months + 1):
        if use_student_t:
            shocks = _standardized_t_shocks(rng, len(ASSETS))
        else:
            shocks = rng.normal(0.0, 1.0, size=len(ASSETS))

        base_returns: dict[str, float] = {}
        for i, asset in enumerate(ASSETS):
            base_returns[asset] = spec.mean[asset] + spec.vol[asset] * shocks[i]

        if regime == "Bull" and rng.random() < spec.small_drawdown_probability:
            base_returns["Stocks"] += rng.normal(-0.040, 0.015)
            base_returns["Bonds"] += rng.normal(-0.003, 0.003)

        if regime == "Crisis":
            # Elevated turbulence around the crash window.
            if crash_month is not None and abs(month - crash_month) <= 2:
                base_returns["Stocks"] += rng.normal(0.0, 0.06)
                base_returns["Bonds"] += rng.normal(0.0, 0.006)
                base_returns["Gold"] += rng.normal(0.0, 0.03)

            if crash_month is not None and month == crash_month:
                base_returns["Stocks"] += spec.crash_stock_impact + rng.normal(0.0, 0.04)
                base_returns["Bonds"] += spec.crash_bond_impact + rng.normal(0.0, 0.015)
                base_returns["Gold"] += spec.crash_gold_impact + rng.normal(0.0, 0.02)
            elif rng.random() < spec.crash_probability:
                base_returns["Stocks"] += rng.normal(-0.11, 0.04)
                base_returns["Bonds"] += rng.normal(-0.005, 0.004)
                base_returns["Gold"] += rng.normal(0.01, 0.02)

        if regime == "Sideways":
            base_returns["Stocks"] += -spec.mean_reversion_strength * prev_stock
            base_returns["Gold"] += -0.25 * prev_gold
            if rng.random() < spec.small_drawdown_probability:
                base_returns["Stocks"] += rng.normal(-0.030, 0.010)
                base_returns["Bonds"] += rng.normal(-0.0025, 0.002)

        base_returns = _clamp_returns(base_returns)
        prev_stock = base_returns["Stocks"]
        prev_gold = base_returns["Gold"]

        row: dict[str, Any] = {
            "month_index": month,
            "regime": regime,
            "ret_cash": base_returns["Cash"],
            "ret_bonds": base_returns["Bonds"],
            "ret_stocks": base_returns["Stocks"],
            "ret_gold": base_returns["Gold"],
        }
        rows.append(row)

    returns_df = pd.DataFrame(rows)
    metadata = {
        "regime": regime,
        "months": months,
        "seed": seed,
        "use_student_t": use_student_t,
        "crash_month": crash_month,
        "regime_parameters": spec.to_dict(),
    }
    return returns_df, metadata


def generate_episode(
    months: int = 36,
    seed: int = 42,
    regime: str | None = None,
    use_student_t: bool = True,
    steps_range: tuple[int, int] | None = None,
    segment_range: tuple[int, int] | None = None,
    regime_weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selector_rng = np.random.default_rng(seed)
    total_steps = int(months)

    if steps_range is not None:
        low, high = steps_range
        if low < 1 or high < low:
            raise ValueError("steps_range must satisfy 1 <= low <= high")
        total_steps = int(selector_rng.integers(low, high + 1))

    if regime is not None:
        returns_seed = int(selector_rng.integers(1, 1_000_000_000))
        returns_df, meta = generate_episode_returns(
            regime=regime,
            months=total_steps,
            seed=returns_seed,
            use_student_t=use_student_t,
        )
        meta["episode_seed"] = seed
        meta["returns_seed"] = returns_seed
        meta["dynamic_regimes"] = False
        return returns_df, meta

    if segment_range is None:
        chosen_regime = list(REGIME_LIBRARY.keys())[int(selector_rng.integers(0, len(REGIME_LIBRARY)))]
        returns_seed = int(selector_rng.integers(1, 1_000_000_000))
        returns_df, meta = generate_episode_returns(
            regime=chosen_regime,
            months=total_steps,
            seed=returns_seed,
            use_student_t=use_student_t,
        )
        meta["episode_seed"] = seed
        meta["returns_seed"] = returns_seed
        meta["dynamic_regimes"] = False
        return returns_df, meta

    plan = _build_regime_switch_plan(
        total_steps=total_steps,
        rng=selector_rng,
        segment_range=segment_range,
        regime_weights=regime_weights,
    )

    chunk_frames: list[pd.DataFrame] = []
    phase_path: list[dict[str, Any]] = []
    month_cursor = 1

    for phase_index, (phase_regime, phase_len) in enumerate(plan, start=1):
        phase_seed = int(selector_rng.integers(1, 1_000_000_000))
        phase_df, phase_meta = generate_episode_returns(
            regime=phase_regime,
            months=phase_len,
            seed=phase_seed,
            use_student_t=use_student_t,
        )
        phase_df = phase_df.copy()
        phase_df["month_index"] = np.arange(month_cursor, month_cursor + phase_len)
        phase_df["phase_index"] = phase_index
        phase_df["phase_label"] = f"Фаза {phase_index}"
        phase_df["phase_step"] = np.arange(1, phase_len + 1)
        month_cursor += phase_len
        chunk_frames.append(phase_df)

        phase_path.append(
            {
                "phase_index": phase_index,
                "phase_label": f"Фаза {phase_index}",
                "regime": phase_regime,
                "length": phase_len,
                "seed": phase_seed,
                "crash_month_in_phase": phase_meta.get("crash_month"),
            }
        )

    returns_df = pd.concat(chunk_frames, ignore_index=True)
    meta = {
        "regime": "Mixed",
        "months": total_steps,
        "seed": seed,
        "use_student_t": use_student_t,
        "episode_seed": seed,
        "dynamic_regimes": True,
        "segment_range": {"min": int(segment_range[0]), "max": int(segment_range[1])},
        "regime_weights": regime_weights or {},
        "regime_path": phase_path,
    }
    return returns_df, meta


def regime_options() -> list[str]:
    return list(REGIME_LIBRARY.keys())
