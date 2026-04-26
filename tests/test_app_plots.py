from __future__ import annotations

import matplotlib.figure
import pandas as pd

import app


def test_plot_builders_return_figures() -> None:
    values = [100_000, 101_200, 99_800, 102_500, 103_100]
    fig_value = app._build_value_plot(values)
    fig_dd = app._build_drawdown_plot(values)

    assert isinstance(fig_value, matplotlib.figure.Figure)
    assert isinstance(fig_dd, matplotlib.figure.Figure)
    assert len(fig_value.axes) == 1
    assert len(fig_dd.axes) == 1


def test_allocation_and_asset_plots_return_figures() -> None:
    log_df = pd.DataFrame(
        {
            "month_index": [1, 2, 3],
            "w_cash": [0.3, 0.4, 0.2],
            "w_bonds": [0.35, 0.3, 0.4],
            "w_stocks": [0.3, 0.25, 0.35],
            "w_gold": [0.05, 0.05, 0.05],
        }
    )
    fig_alloc = app._build_allocation_plot(log_df)
    assert isinstance(fig_alloc, matplotlib.figure.Figure)
    assert len(fig_alloc.axes) == 1

    returns_df = pd.DataFrame(
        {
            "ret_cash": [0.001, 0.001, 0.001],
            "ret_bonds": [0.002, -0.001, 0.003],
            "ret_stocks": [0.01, -0.03, 0.02],
            "ret_gold": [0.005, 0.0, -0.002],
        }
    )
    asset_index_df = app._build_asset_index_df(returns_df, upto_month=3)
    fig_assets = app._build_asset_paths_plot(asset_index_df, "test")
    snapshot = app._build_asset_snapshot(asset_index_df)

    assert isinstance(fig_assets, matplotlib.figure.Figure)
    assert len(fig_assets.axes) == 1
    assert set(snapshot.columns) == {"Актив", "Было (индекс)", "Сейчас (индекс)", "Изменение"}
