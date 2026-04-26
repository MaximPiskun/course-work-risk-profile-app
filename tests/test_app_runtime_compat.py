from __future__ import annotations

import app


def test_generate_episode_call_is_backward_compatible(monkeypatch) -> None:
    calls: list[dict] = []

    def old_generate_episode_signature(*, months: int, seed: int, regime: str | None, use_student_t: bool):
        calls.append(
            {
                "months": months,
                "seed": seed,
                "regime": regime,
                "use_student_t": use_student_t,
            }
        )
        return [], {"ok": True}

    monkeypatch.setattr(app, "generate_episode", old_generate_episode_signature)

    cfg = {
        "months": 24,
        "regime": None,
        "steps_range": (18, 24),
        "segment_range": (4, 7),
        "regime_weights": {"Bull": 0.34, "Sideways": 0.33, "Crisis": 0.33},
    }

    returns_df, meta = app._call_generate_episode_with_compat(cfg, seed=42)
    assert returns_df == []
    assert meta["ok"] is True
    assert len(calls) == 1
    assert calls[0] == {
        "months": 24,
        "seed": 42,
        "regime": None,
        "use_student_t": True,
    }


def test_mode_configs_generate_long_dynamic_live_episodes() -> None:
    for mode_key, cfg in app.MODE_CONFIGS.items():
        returns_df, meta = app._call_generate_episode_with_compat(cfg, seed=42)

        assert meta.get("dynamic_regimes") is True
        assert len(returns_df) >= int(cfg["steps_range"][0])
        assert len(returns_df) <= int(cfg["steps_range"][1])
        assert len(set(returns_df["regime"].tolist())) >= 2, mode_key
