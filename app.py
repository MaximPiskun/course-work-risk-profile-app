from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from market import generate_episode, regime_options
from scoring import score_portfolio

ASSETS = ["Cash", "Bonds", "Stocks", "Gold"]
FLOW_STEPS = [
    ("landing", "1. Добро пожаловать"),
    ("questionnaire", "2. Анкета"),
    ("summary", "3. Предварительная интерпретация"),
    ("mode", "4. Выбор режима"),
    ("simulation", "5. Симуляция"),
    ("results", "6. Итоги"),
]
FLOW_INDEX = {key: idx for idx, (key, _) in enumerate(FLOW_STEPS)}

MODE_CONFIGS: dict[str, dict[str, Any]] = {
    "standard": {
        "title": "Стандартный режим",
        "description": "Сбалансированный темп и типичная волатильность. Подходит для большинства пользователей.",
        "duration": "36 месяцев (≈10-12 минут)",
        "months": 36,
        "regime": None,
        "intensity": "Средняя",
    },
    "stress": {
        "title": "Стресс-сценарий",
        "description": "Повышенная турбулентность и выраженные просадки. Показывает реакцию на стресс.",
        "duration": "36 месяцев (≈10-12 минут)",
        "months": 36,
        "regime": "Crisis",
        "intensity": "Высокая",
    },
    "quick": {
        "title": "Быстрый режим",
        "description": "Укороченная версия для быстрого прохождения и первичной оценки поведения.",
        "duration": "18 месяцев (≈5-7 минут)",
        "months": 18,
        "regime": None,
        "intensity": "Ниже средней",
    },
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1180px;}
          .app-hero, .soft-card {
            border: 1px solid #e9edf5;
            border-radius: 14px;
            background: #ffffff;
            padding: 1.1rem 1.2rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
          }
          .app-hero {padding: 1.4rem 1.4rem; background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);}
          .step-label {
            border: 1px solid #e9edf5;
            border-radius: 999px;
            background: #fbfcff;
            padding: 0.3rem 0.6rem;
            font-size: 0.82rem;
            text-align: center;
            white-space: nowrap;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "screen": "landing",
        "questionnaire_step": 0,
        "q_age": 35,
        "q_education": "Бакалавриат",
        "q_monthly_income": 5000.0,
        "q_monthly_expenses": 3000.0,
        "q_savings_amount": 25000.0,
        "q_horizon_years": 10,
        "q_investment_goal": "Сбалансированный рост",
        "q_random_seed": 42,
        "onboarding_profile": None,
        "baseline_summary": None,
        "selected_mode": None,
        "simulation_months": 36,
        "episode_returns": None,
        "episode_meta": None,
        "episode_id": None,
        "month_index": 0,
        "current_weights": {"Cash": 0.30, "Bonds": 0.35, "Stocks": 0.30, "Gold": 0.05},
        "portfolio_values": [],
        "portfolio_returns": [],
        "logs": [],
        "completed": False,
        "ended_early": False,
        "flash_message": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def hard_reset() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("q_") or key in {
            "screen",
            "questionnaire_step",
            "onboarding_profile",
            "baseline_summary",
            "selected_mode",
            "simulation_months",
            "episode_returns",
            "episode_meta",
            "episode_id",
            "month_index",
            "current_weights",
            "portfolio_values",
            "portfolio_returns",
            "logs",
            "completed",
            "ended_early",
            "flash_message",
            "sim_action",
            "sim_reduce_pct",
            "sim_uncomfortable",
            "sim_bonds_pct",
            "sim_stocks_pct",
            "sim_gold_pct",
        }:
            del st.session_state[key]
    initialize_state()


def render_step_tracker() -> None:
    idx = FLOW_INDEX[st.session_state.screen]
    st.progress((idx + 1) / len(FLOW_STEPS))
    cols = st.columns(len(FLOW_STEPS))
    for i, (_, label) in enumerate(FLOW_STEPS):
        text = f"<strong>{label}</strong>" if i == idx else label
        cols[i].markdown(f"<div class='step-label'>{text}</div>", unsafe_allow_html=True)


def collect_onboarding_profile() -> dict[str, Any]:
    return {
        "age": int(st.session_state.q_age),
        "education": st.session_state.q_education,
        "monthly_income": float(st.session_state.q_monthly_income),
        "monthly_expenses": float(st.session_state.q_monthly_expenses),
        "savings_amount": float(st.session_state.q_savings_amount),
        "investment_horizon_years": int(st.session_state.q_horizon_years),
        "investment_goal": st.session_state.q_investment_goal,
        "random_seed": int(st.session_state.q_random_seed),
    }


def estimate_baseline_profile(onboarding: dict[str, Any]) -> dict[str, Any]:
    income = float(onboarding["monthly_income"])
    expenses = float(onboarding["monthly_expenses"])
    savings = float(onboarding["savings_amount"])
    horizon = int(onboarding["investment_horizon_years"])
    goal = str(onboarding["investment_goal"])

    income_buffer = (income - expenses) / max(income, 1.0)
    reserve_months = savings / max(expenses, 1.0)

    score = 0.0
    score += 45.0 * float(np.clip((income_buffer + 0.2) / 0.7, 0.0, 1.0))
    score += 35.0 * float(np.clip(reserve_months / 12.0, 0.0, 1.0))
    score += 20.0 * float(np.clip((horizon - 1) / 14.0, 0.0, 1.0))

    if goal == "Сохранение капитала":
        score -= 8.0
    elif goal == "Долгосрочный рост":
        score += 4.0
    elif goal == "Ускоренный рост":
        score += 8.0

    score = float(np.clip(score, 0.0, 100.0))
    if score <= 33:
        level = "Консервативный"
    elif score <= 66:
        level = "Умеренный"
    else:
        level = "Агрессивный"

    notes: list[str] = []
    if reserve_months < 6:
        notes.append("Финансовая подушка ниже 6 месяцев расходов, это может снижать устойчивость к волатильности.")
    if income_buffer < 0.1:
        notes.append("Свободный денежный поток ограничен: просадки могут переноситься психологически сложнее.")
    if horizon < 3:
        notes.append("Горизонт короткий, поэтому рискованные колебания могут быть менее комфортными.")
    if not notes:
        notes.append("Анкета указывает на достаточный запас по горизонту и финансовой устойчивости.")

    return {
        "baseline_capacity_score": score,
        "baseline_capacity_level": level,
        "interpretation": (
            f"По анкете просматривается {level.lower()} базовый профиль. "
            "Теперь посмотрим, как вы будете действовать в динамике рынка."
        ),
        "notes": notes,
    }


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        return {"Cash": 1.0, "Bonds": 0.0, "Stocks": 0.0, "Gold": 0.0}
    return {k: float(v / total) for k, v in weights.items()}


def _reduce_risk(weights: dict[str, float], reduction_pct: float) -> dict[str, float]:
    reduction = float(np.clip(reduction_pct, 0.0, 1.0))
    updated = dict(weights)
    moved_to_cash = 0.0
    for risky in ["Stocks", "Gold"]:
        old_value = updated[risky]
        new_value = old_value * (1.0 - reduction)
        moved_to_cash += old_value - new_value
        updated[risky] = new_value
    updated["Cash"] += moved_to_cash
    return _normalize_weights(updated)


def _current_drawdown(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    peak = np.maximum.accumulate(arr)
    return float((arr / peak - 1.0)[-1])


def _max_drawdown(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    peak = np.maximum.accumulate(arr)
    return float((arr / peak - 1.0).min())


def _vol_estimate(returns: list[float], window: int = 6) -> float:
    if len(returns) < 2:
        return 0.0
    return float(pd.Series(returns[-window:]).std(ddof=1) * np.sqrt(12))

def _build_value_plot(values: list[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 3.1))
    x = np.arange(len(values))
    ax.plot(x, values, linewidth=2.0, color="#2a5a9e")
    ax.set_title("Траектория стоимости портфеля")
    ax.set_xlabel("Период")
    ax.set_ylabel("Стоимость")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _build_drawdown_plot(values: list[float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 3.1))
    arr = np.array(values, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = arr / peak - 1.0
    x = np.arange(len(values))
    ax.fill_between(x, dd, 0, alpha=0.25, color="#9db7e8")
    ax.plot(x, dd, linewidth=1.8, color="#2a5a9e")
    ax.set_title("Просадка")
    ax.set_xlabel("Период")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _build_allocation_plot(log_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 3.1))
    if log_df.empty:
        ax.set_title("История аллокации")
        fig.tight_layout()
        return fig

    x = log_df["month_index"].to_numpy()
    stacks = [
        log_df["w_cash"].to_numpy(),
        log_df["w_bonds"].to_numpy(),
        log_df["w_stocks"].to_numpy(),
        log_df["w_gold"].to_numpy(),
    ]
    ax.stackplot(x, stacks, labels=ASSETS, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_title("История аллокации")
    ax.legend(loc="upper right", ncols=4, fontsize=8)
    fig.tight_layout()
    return fig


def _build_asset_index_df(returns_df: pd.DataFrame, upto_month: int | None = None) -> pd.DataFrame:
    """Build normalized asset index paths starting from 100."""
    if returns_df.empty:
        return pd.DataFrame(columns=["month_index", "Cash", "Bonds", "Stocks", "Gold"])

    frame = returns_df.copy()
    if upto_month is not None:
        frame = frame.iloc[: max(upto_month, 0)]

    if frame.empty:
        return pd.DataFrame(
            {
                "month_index": [0],
                "Cash": [100.0],
                "Bonds": [100.0],
                "Stocks": [100.0],
                "Gold": [100.0],
            }
        )

    out = pd.DataFrame({"month_index": np.arange(1, len(frame) + 1)})
    out["Cash"] = 100.0 * (1.0 + frame["ret_cash"].astype(float)).cumprod()
    out["Bonds"] = 100.0 * (1.0 + frame["ret_bonds"].astype(float)).cumprod()
    out["Stocks"] = 100.0 * (1.0 + frame["ret_stocks"].astype(float)).cumprod()
    out["Gold"] = 100.0 * (1.0 + frame["ret_gold"].astype(float)).cumprod()

    start_row = pd.DataFrame(
        {"month_index": [0], "Cash": [100.0], "Bonds": [100.0], "Stocks": [100.0], "Gold": [100.0]}
    )
    return pd.concat([start_row, out], ignore_index=True)


def _build_asset_paths_plot(asset_index_df: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 3.3))
    for asset in ["Cash", "Bonds", "Stocks", "Gold"]:
        ax.plot(asset_index_df["month_index"], asset_index_df[asset], linewidth=1.9, label=asset)
    ax.set_title(title)
    ax.set_xlabel("Период")
    ax.set_ylabel("Индекс (старт = 100)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", ncols=2, fontsize=8)
    fig.tight_layout()
    return fig


def _build_asset_snapshot(asset_index_df: pd.DataFrame) -> pd.DataFrame:
    start = asset_index_df.iloc[0]
    current = asset_index_df.iloc[-1]
    rows = []
    for asset in ["Cash", "Bonds", "Stocks", "Gold"]:
        start_val = float(start[asset])
        current_val = float(current[asset])
        rows.append(
            {
                "Актив": asset,
                "Было (индекс)": f"{start_val:.1f}",
                "Сейчас (индекс)": f"{current_val:.1f}",
                "Изменение": f"{(current_val / start_val - 1.0):.2%}",
            }
        )
    return pd.DataFrame(rows)


def build_market_commentary(regime: str, month_idx: int, drawdown: float, last_stock_ret: float | None) -> str:
    if month_idx == 0:
        if regime == "Crisis":
            return "Старт стрессового эпизода: рынок нестабилен, вероятны резкие движения и локальные шоки."
        if regime == "Bull":
            return "Старт эпизода роста: общий фон позитивный, но возможны краткосрочные откаты."
        return "Старт бокового эпизода: возможны частые небольшие колебания в обе стороны."

    if last_stock_ret is None:
        return "Рынок смешанный: оценивайте баланс между защитой капитала и участием в росте."

    if drawdown <= -0.15 or last_stock_ret <= -0.08:
        return "Рынок снижается второй месяц подряд. Волатильность выросла, участники становятся осторожнее."
    if last_stock_ret >= 0.06 and drawdown > -0.06:
        return "Наблюдается восстановление после снижения. Рынок демонстрирует признаки стабилизации."
    if -0.02 <= last_stock_ret <= 0.02:
        return "Движение сдержанное: рынок остается в смешанном состоянии без выраженного тренда."
    return "Рынок волатилен, но без однозначного направления. Решение зависит от вашей комфортной просадки."


def _action_label(rebalance: int, go_to_cash: int, reduce_risk: int, end_early: int) -> str:
    if end_early:
        return "Завершение досрочно"
    if go_to_cash:
        return "Переход в кэш"
    if reduce_risk:
        return "Снижение риска"
    if rebalance:
        return "Ребалансировка"
    return "Сохранение структуры"


def start_episode(mode_key: str) -> None:
    onboarding = st.session_state.onboarding_profile
    cfg = MODE_CONFIGS[mode_key]
    seed = int(onboarding["random_seed"])

    returns_df, meta = generate_episode(
        months=int(cfg["months"]),
        seed=seed,
        regime=cfg["regime"],
        use_student_t=True,
    )

    meta["mode_key"] = mode_key
    meta["mode_title"] = cfg["title"]

    st.session_state.selected_mode = mode_key
    st.session_state.simulation_months = int(cfg["months"])
    st.session_state.episode_returns = returns_df
    st.session_state.episode_meta = meta
    st.session_state.episode_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{mode_key}_{seed}"

    initial_value = float(max(1000.0, onboarding["savings_amount"]))
    st.session_state.month_index = 0
    st.session_state.current_weights = {"Cash": 0.30, "Bonds": 0.35, "Stocks": 0.30, "Gold": 0.05}
    st.session_state.portfolio_values = [initial_value]
    st.session_state.portfolio_returns = []
    st.session_state.logs = []
    st.session_state.completed = False
    st.session_state.ended_early = False
    st.session_state.flash_message = "Режим выбран. Симуляция готова к старту."
    st.session_state.screen = "simulation"


def validate_questionnaire_step(step: int) -> tuple[bool, str]:
    if step == 0 and int(st.session_state.q_age) < 18:
        return False, "Возраст должен быть не меньше 18 лет."

    if step == 1:
        income = float(st.session_state.q_monthly_income)
        expenses = float(st.session_state.q_monthly_expenses)
        savings = float(st.session_state.q_savings_amount)
        if income <= 0:
            return False, "Укажите ежемесячный доход больше нуля."
        if expenses < 0 or savings < 0:
            return False, "Расходы и накопления не могут быть отрицательными."
        if expenses > income * 3:
            return False, "Расходы выглядят слишком высокими относительно дохода. Проверьте ввод."

    if step == 2:
        if int(st.session_state.q_horizon_years) < 1:
            return False, "Горизонт инвестирования должен быть не меньше 1 года."
        if int(st.session_state.q_random_seed) < 1:
            return False, "Seed должен быть положительным числом."

    return True, ""


def render_landing() -> None:
    st.markdown(
        """
        <div class="app-hero">
          <h2 style="margin-bottom:0.4rem;">Симуляция инвестиционного поведения</h2>
          <p style="margin-top:0;">Короткий исследовательский сценарий для оценки склонности к риску и реакции на просадки.</p>
        </div>
        <div class="soft-card">
          <h4 style="margin-top:0;">Как это работает</h4>
          <ul>
            <li>Вы отвечаете на несколько вопросов о целях и финансовом профиле.</li>
            <li>Проходите симуляцию рынка с ежемесячными решениями по портфелю.</li>
            <li>Получаете итоговый профиль риска и поведенческий разбор.</li>
          </ul>
          <p>Обычно прохождение занимает 7-12 минут.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Начать", type="primary", use_container_width=True):
        st.session_state.screen = "questionnaire"
        st.rerun()

    st.caption("Это не инвестиционная рекомендация, а исследовательская симуляция.")


def render_questionnaire() -> None:
    step = int(st.session_state.questionnaire_step)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Короткая анкета")
    st.caption("Сначала ответьте на несколько вопросов о себе. Это займет 1-2 минуты.")
    st.progress((step + 1) / 3)
    st.write(f"Шаг {step + 1} из 3")

    if step == 0:
        st.markdown("#### 1) Базовая информация")
        st.number_input("Возраст", min_value=18, max_value=90, step=1, key="q_age")
        st.selectbox(
            "Образование",
            ["Среднее", "Бакалавриат", "Магистратура", "Аспирантура/PhD", "Другое"],
            key="q_education",
        )

    elif step == 1:
        st.markdown("#### 2) Финансовая ситуация")
        st.number_input("Ежемесячный доход", min_value=0.0, step=100.0, key="q_monthly_income")
        st.number_input("Ежемесячные расходы", min_value=0.0, step=100.0, key="q_monthly_expenses")
        st.number_input("Объем накоплений", min_value=0.0, step=500.0, key="q_savings_amount")
        st.caption("Подсказка: расходы и накопления помогают оценить способность переносить волатильность.")

    else:
        st.markdown("#### 3) Горизонт и цель")
        st.number_input("Горизонт инвестирования (лет)", min_value=1, max_value=40, step=1, key="q_horizon_years")
        st.selectbox(
            "Инвестиционная цель",
            ["Сохранение капитала", "Стабильный доход", "Сбалансированный рост", "Долгосрочный рост", "Ускоренный рост"],
            key="q_investment_goal",
        )
        st.number_input("Random seed для воспроизводимости", min_value=1, step=1, key="q_random_seed")

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("Назад", use_container_width=True):
            if step == 0:
                st.session_state.screen = "landing"
            else:
                st.session_state.questionnaire_step = step - 1
            st.rerun()

    with col_next:
        if st.button("Продолжить", type="primary", use_container_width=True):
            valid, message = validate_questionnaire_step(step)
            if not valid:
                st.warning(message)
                st.stop()

            if step < 2:
                st.session_state.questionnaire_step = step + 1
            else:
                onboarding = collect_onboarding_profile()
                st.session_state.onboarding_profile = onboarding
                st.session_state.baseline_summary = estimate_baseline_profile(onboarding)
                st.session_state.flash_message = "Ваши ответы сохранены."
                st.session_state.screen = "summary"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

def render_summary() -> None:
    onboarding = st.session_state.onboarding_profile
    baseline = st.session_state.baseline_summary

    if onboarding is None or baseline is None:
        st.info("Сначала заполните анкету.")
        if st.button("Перейти к анкете", use_container_width=True):
            st.session_state.screen = "questionnaire"
            st.rerun()
        return

    if st.session_state.flash_message:
        st.success(st.session_state.flash_message)
        st.session_state.flash_message = ""

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Предварительная интерпретация анкеты")
    st.caption("Это только первый слой оценки. Финальный профиль формируется после поведения в рыночной динамике.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Горизонт", f"{onboarding['investment_horizon_years']} лет")
    c2.metric("Базовая емкость к риску", f"{baseline['baseline_capacity_score']:.1f}/100")
    c3.metric("Предварительный уровень", baseline["baseline_capacity_level"])

    st.write(baseline["interpretation"])
    st.write("Важные заметки:")
    for note in baseline["notes"]:
        st.write(f"- {note}")

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("Назад", use_container_width=True):
            st.session_state.screen = "questionnaire"
            st.session_state.questionnaire_step = 2
            st.rerun()
    with col_next:
        if st.button("Выбрать режим", type="primary", use_container_width=True):
            st.session_state.screen = "mode"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_mode_selection() -> None:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Выберите режим прохождения")
    st.caption("Выбор влияет на длительность и интенсивность сценария.")

    cols = st.columns(3)
    for col, mode_key in zip(cols, ["standard", "stress", "quick"]):
        cfg = MODE_CONFIGS[mode_key]
        with col:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown(f"**{cfg['title']}**")
            st.write(cfg["description"])
            st.caption(f"Длительность: {cfg['duration']}")
            st.caption(f"Интенсивность: {cfg['intensity']}")
            if st.button("Выбрать", key=f"choose_{mode_key}", type="primary", use_container_width=True):
                start_episode(mode_key)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Назад", use_container_width=True):
        st.session_state.screen = "summary"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_simulation() -> None:
    if st.session_state.onboarding_profile is None or st.session_state.episode_returns is None:
        st.warning("Сначала заполните анкету и выберите режим.")
        if st.button("Перейти к выбору режима", use_container_width=True):
            st.session_state.screen = "mode"
            st.rerun()
        return

    if st.session_state.flash_message:
        st.success(st.session_state.flash_message)
        st.session_state.flash_message = ""

    month_idx = int(st.session_state.month_index)
    months = int(st.session_state.simulation_months)
    if month_idx >= months:
        st.session_state.completed = True
        st.session_state.screen = "results"
        st.rerun()

    returns_df: pd.DataFrame = st.session_state.episode_returns
    current_values = st.session_state.portfolio_values
    current_value = float(current_values[-1])
    initial_value = float(current_values[0])
    cumulative_return = current_value / initial_value - 1.0
    current_dd = _current_drawdown(current_values)
    max_dd = _max_drawdown(current_values)
    vol_est = _vol_estimate(st.session_state.portfolio_returns)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Основная симуляция")
    st.caption(f"Режим: {st.session_state.episode_meta.get('mode_title', '-')}. Период {month_idx + 1} из {months}.")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Период", f"{month_idx + 1}/{months}")
    m2.metric("Стоимость портфеля", f"{current_value:,.0f}")
    m3.metric("Доходность с начала", f"{cumulative_return:.2%}")
    m4.metric("Текущая просадка", f"{current_dd:.2%}")
    m5.metric("Макс. просадка", f"{max_dd:.2%}")

    st.caption(
        "Нейтральное раскрытие риска: результаты симуляции не гарантируют будущую доходность. "
        "Снижение риска может уменьшать волатильность, но фиксирует часть убытка."
    )

    left, right = st.columns([1.45, 1.0])

    with left:
        st.markdown("#### A. Рыночный контекст")
        regime = st.session_state.episode_meta["regime"]
        if month_idx > 0:
            prev = returns_df.iloc[month_idx - 1]
            last_stock = float(prev["ret_stocks"])
            snapshot = pd.DataFrame(
                {
                    "Актив": ["Cash", "Bonds", "Stocks", "Gold"],
                    "Доходность за прошлый период": [
                        f"{float(prev['ret_cash']):.2%}",
                        f"{float(prev['ret_bonds']):.2%}",
                        f"{float(prev['ret_stocks']):.2%}",
                        f"{float(prev['ret_gold']):.2%}",
                    ],
                }
            )
        else:
            last_stock = None
            snapshot = pd.DataFrame({"Актив": ["Cash", "Bonds", "Stocks", "Gold"], "Доходность за прошлый период": ["-", "-", "-", "-"]})

        st.info(build_market_commentary(regime, month_idx, current_dd, last_stock))
        st.write(f"Текущий рыночный режим: **{regime}**")
        st.dataframe(snapshot, use_container_width=True, hide_index=True)

        with st.expander("Что это значит: просадка и волатильность"):
            st.write("Просадка показывает снижение от предыдущего максимума, а волатильность — амплитуду колебаний доходности.")
            st.write(
                "В этой модели доходность облигаций = купонный поток + рыночная переоценка. "
                "Поэтому при росте ставок облигации могут временно показывать отрицательную доходность."
            )

        asset_index_df = _build_asset_index_df(returns_df, upto_month=month_idx)
        asset_snapshot = _build_asset_snapshot(asset_index_df)

        t1, t2, t3, t4 = st.tabs(["Портфель", "Просадка", "Аллокация", "Активы"])
        with t1:
            st.pyplot(_build_value_plot(st.session_state.portfolio_values), clear_figure=True)
        with t2:
            st.pyplot(_build_drawdown_plot(st.session_state.portfolio_values), clear_figure=True)
        with t3:
            st.pyplot(_build_allocation_plot(pd.DataFrame(st.session_state.logs)), clear_figure=True)
        with t4:
            st.pyplot(
                _build_asset_paths_plot(asset_index_df, "Динамика активов: где были раньше и где сейчас"),
                clear_figure=True,
            )
            st.dataframe(asset_snapshot, use_container_width=True, hide_index=True)
            st.caption("Индексы нормированы: старт каждого актива = 100.")

    with right:
        st.markdown("#### B. Панель решения")
        st.caption("Выберите действие на этот период. Этот блок является ключевым для оценки поведения.")
        cw = st.session_state.current_weights
        st.write("Текущая структура портфеля:")
        weights_view = pd.DataFrame(
            {
                "Актив": ["Cash", "Bonds", "Stocks", "Gold"],
                "Доля": [f"{cw['Cash']:.0%}", f"{cw['Bonds']:.0%}", f"{cw['Stocks']:.0%}", f"{cw['Gold']:.0%}"],
            }
        )
        st.dataframe(weights_view, use_container_width=True, hide_index=True)

        action = st.radio(
            "Ваше действие",
            ["Оставить текущую структуру", "Ребалансировать", "Снизить риск на X%", "Пауза: перейти в кэш"],
            key="sim_action",
        )

        if action == "Ребалансировать":
            b = st.slider("Bonds, %", 0, 100, int(round(cw["Bonds"] * 100)), key="sim_bonds_pct")
            s = st.slider("Stocks, %", 0, 100, int(round(cw["Stocks"] * 100)), key="sim_stocks_pct")
            g = st.slider("Gold, %", 0, 100, int(round(cw["Gold"] * 100)), key="sim_gold_pct")
            total = b + s + g
            if total <= 100:
                st.info(f"Cash автоматически: {100 - total}%")
            else:
                st.warning("Сумма Bonds + Stocks + Gold должна быть <= 100%.")

        reduce_pct = st.slider("Размер снижения риска, %", 5, 60, 20, 5, key="sim_reduce_pct")
        discomfort = st.slider("Насколько вам сейчас некомфортно?", 1, 5, 3, key="sim_uncomfortable")

        c_apply, c_early = st.columns(2)
        apply_clicked = c_apply.button("Сохранить решение и продолжить", type="primary", use_container_width=True)
        early_clicked = c_early.button("Завершить досрочно", use_container_width=True)

        if early_clicked:
            st.session_state.logs.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "episode_id": st.session_state.episode_id,
                    "month_index": int(month_idx + 1),
                    "regime": regime,
                    "ret_cash": 0.0,
                    "ret_bonds": 0.0,
                    "ret_stocks": 0.0,
                    "ret_gold": 0.0,
                    "w_cash": float(cw["Cash"]),
                    "w_bonds": float(cw["Bonds"]),
                    "w_stocks": float(cw["Stocks"]),
                    "w_gold": float(cw["Gold"]),
                    "portfolio_return": 0.0,
                    "portfolio_value": current_value,
                    "drawdown": current_dd,
                    "volatility_estimate": vol_est,
                    "rebalance": 0,
                    "go_to_cash": 0,
                    "reduce_risk": 0,
                    "end_early": 1,
                    "discomfort_1_5": int(discomfort),
                    "action_label": _action_label(0, 0, 0, 1),
                }
            )
            st.session_state.completed = True
            st.session_state.ended_early = True
            st.session_state.screen = "results"
            st.session_state.flash_message = "Симуляция завершена досрочно."
            st.rerun()

        if apply_clicked:
            rebalance = 0
            go_to_cash = 0
            reduce_risk_flag = 0

            if action == "Оставить текущую структуру":
                chosen_w = dict(cw)
            elif action == "Ребалансировать":
                b = int(st.session_state.sim_bonds_pct)
                s = int(st.session_state.sim_stocks_pct)
                g = int(st.session_state.sim_gold_pct)
                total = b + s + g
                if total > 100:
                    st.error("Невозможно сохранить: сумма Bonds + Stocks + Gold должна быть <= 100%.")
                    st.stop()
                chosen_w = _normalize_weights({"Cash": (100 - total) / 100, "Bonds": b / 100, "Stocks": s / 100, "Gold": g / 100})
                rebalance = 1
            elif action == "Снизить риск на X%":
                chosen_w = _reduce_risk(cw, float(st.session_state.sim_reduce_pct) / 100.0)
                reduce_risk_flag = 1
            else:
                chosen_w = {"Cash": 1.0, "Bonds": 0.0, "Stocks": 0.0, "Gold": 0.0}
                go_to_cash = 1

            row = returns_df.iloc[month_idx]
            portfolio_return = (
                chosen_w["Cash"] * float(row["ret_cash"])
                + chosen_w["Bonds"] * float(row["ret_bonds"])
                + chosen_w["Stocks"] * float(row["ret_stocks"])
                + chosen_w["Gold"] * float(row["ret_gold"])
            )
            new_value = current_value * (1.0 + portfolio_return)

            st.session_state.portfolio_values.append(float(new_value))
            st.session_state.portfolio_returns.append(float(portfolio_return))
            new_dd = _current_drawdown(st.session_state.portfolio_values)
            new_vol = _vol_estimate(st.session_state.portfolio_returns)

            st.session_state.logs.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "episode_id": st.session_state.episode_id,
                    "month_index": int(month_idx + 1),
                    "regime": regime,
                    "ret_cash": float(row["ret_cash"]),
                    "ret_bonds": float(row["ret_bonds"]),
                    "ret_stocks": float(row["ret_stocks"]),
                    "ret_gold": float(row["ret_gold"]),
                    "w_cash": float(chosen_w["Cash"]),
                    "w_bonds": float(chosen_w["Bonds"]),
                    "w_stocks": float(chosen_w["Stocks"]),
                    "w_gold": float(chosen_w["Gold"]),
                    "portfolio_return": float(portfolio_return),
                    "portfolio_value": float(new_value),
                    "drawdown": float(new_dd),
                    "volatility_estimate": float(new_vol),
                    "rebalance": int(rebalance),
                    "go_to_cash": int(go_to_cash),
                    "reduce_risk": int(reduce_risk_flag),
                    "end_early": 0,
                    "discomfort_1_5": int(discomfort),
                    "action_label": _action_label(rebalance, go_to_cash, reduce_risk_flag, 0),
                }
            )

            st.session_state.current_weights = chosen_w
            st.session_state.month_index = month_idx + 1
            if st.session_state.month_index >= months:
                st.session_state.screen = "results"
                st.session_state.completed = True
                st.session_state.flash_message = "Симуляция завершена, можно перейти к результатам."
            else:
                st.session_state.flash_message = "Решение сохранено. Переходим к следующему периоду."
            st.rerun()

    st.markdown("---")
    st.markdown("#### C. Лента событий и решений")
    if not st.session_state.logs:
        st.info("История пока пуста. Примите первое решение, чтобы увидеть динамику.")
    else:
        history = pd.DataFrame(st.session_state.logs)
        view = history[["month_index", "action_label", "portfolio_return", "portfolio_value", "drawdown", "discomfort_1_5"]].copy()
        view.columns = ["Период", "Действие", "Доходность", "Стоимость", "Просадка", "Некомфортность (1-5)"]
        st.dataframe(view.tail(10), use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

def render_results() -> None:
    if st.session_state.onboarding_profile is None or not st.session_state.logs:
        st.info("Результаты появятся после прохождения симуляции.")
        if st.button("Перейти к симуляции", use_container_width=True):
            st.session_state.screen = "simulation"
            st.rerun()
        return

    log_df = pd.DataFrame(st.session_state.logs)
    scores = score_portfolio(log_df, onboarding=st.session_state.onboarding_profile)
    features = scores["features"]
    baseline = st.session_state.baseline_summary

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.subheader("Итоговый профиль")

    c1, c2, c3 = st.columns(3)
    c1.metric("Долгосрочная толерантность к риску", f"{scores['risk_tolerance_score']:.1f}/100")
    c2.metric("Поведенческая реактивность", f"{scores['behavioral_reactivity_score']:.1f}/100")
    c3.metric("Итоговый индекс риска", f"{scores['final_risk_index']:.1f}/100")

    st.write(f"Группа риска (3 уровня): **{scores['risk_group_3']}**")
    st.write(f"Группа риска (5 уровней): **{scores['risk_group_5']}**")

    st.markdown("### Как вы вели себя в стрессовых моментах")
    st.write(f"- Средняя доля рискованных активов: {features['avg_risky_weight']:.2f}")
    st.write(f"- Просадка до первого заметного снижения риска: {features['max_drawdown_tolerated_before_derisk']:.2%}")
    st.write(f"- Количество панических действий: {features['panic_actions_count']}")
    st.write(f"- Восстановление риска после отскока: {int(features['recovery_risk_restore'])}")
    st.write(f"- Churn (частота изменений структуры): {features['churn']:.2f}")

    st.markdown("### Где анкета и поведение совпали / не совпали")
    if baseline is not None:
        declared = baseline["baseline_capacity_level"]
        observed = scores["risk_group_3"]
        st.write(f"- По анкете: **{declared}**")
        st.write(f"- По симуляции: **{observed}**")
        if (declared == "Консервативный" and observed == "Conservative") or (declared == "Умеренный" and observed == "Moderate") or (declared == "Агрессивный" and observed == "Aggressive"):
            st.info("Анкета и фактические решения в целом согласованы.")
        else:
            st.info("Есть расхождение между заявленным профилем и поведением в динамике рынка.")

    st.markdown("### Ключевые сигналы")
    signals: list[str] = []
    if features["panic_actions_count"] >= 2:
        signals.append("Частые защитные действия после негативных шоков.")
    if features["churn"] >= 0.45:
        signals.append("Высокая частота перераспределений (повышенная реактивность).")
    if features["early_exit_flag"] == 1:
        signals.append("Присутствует досрочное завершение эпизода.")
    if features["avg_risky_weight"] >= 0.55 and features["panic_actions_count"] == 0:
        signals.append("Устойчивое удержание рискованной доли даже при волатильности.")
    if not signals:
        signals.append("Поведение без выраженных крайностей по текущему эпизоду.")
    for s in signals:
        st.write(f"- {s}")

    st.markdown("### Графики итогов")
    st.pyplot(_build_value_plot(st.session_state.portfolio_values), clear_figure=True)
    st.pyplot(_build_drawdown_plot(st.session_state.portfolio_values), clear_figure=True)

    returns_df: pd.DataFrame | None = st.session_state.episode_returns
    if returns_df is not None and not returns_df.empty:
        realized_months = int(min(st.session_state.month_index, len(returns_df)))
        asset_index_df = _build_asset_index_df(returns_df, upto_month=realized_months)
        st.markdown("### Динамика активов (было и сейчас)")
        st.pyplot(
            _build_asset_paths_plot(asset_index_df, "Индексы активов за пройденный период"),
            clear_figure=True,
        )
        st.dataframe(_build_asset_snapshot(asset_index_df), use_container_width=True, hide_index=True)
        st.caption("Индекс 100 = стартовая точка актива в начале эпизода.")

    important = log_df[
        (log_df["go_to_cash"] == 1)
        | (log_df["reduce_risk"] == 1)
        | (log_df["rebalance"] == 1)
        | (log_df["portfolio_return"] <= -0.04)
    ].copy()
    st.markdown("### Важные решения по ходу симуляции")
    if important.empty:
        st.info("Выраженных событий не зафиксировано.")
    else:
        view = important[["month_index", "action_label", "portfolio_return", "drawdown", "discomfort_1_5"]].copy()
        view.columns = ["Период", "Событие", "Доходность", "Просадка", "Некомфортность (1-5)"]
        st.dataframe(view, use_container_width=True, hide_index=True)

    st.markdown("### Скачать результаты")
    payload = {
        "onboarding": st.session_state.onboarding_profile,
        "baseline_summary": st.session_state.baseline_summary,
        "episode_meta": st.session_state.episode_meta,
        "selected_mode": st.session_state.selected_mode,
        "scores": scores,
        "logs": log_df.to_dict(orient="records"),
    }

    d1, d2 = st.columns(2)
    d1.download_button("Скачать JSON", data=json.dumps(payload, ensure_ascii=False, indent=2), file_name="portfolio_simulation_results.json", mime="application/json", use_container_width=True)
    d2.download_button("Скачать CSV", data=log_df.to_csv(index=False), file_name="portfolio_decision_log.csv", mime="text/csv", use_container_width=True)

    n1, n2 = st.columns(2)
    if n1.button("Назад к симуляции", use_container_width=True):
        st.session_state.screen = "simulation"
        st.rerun()
    if n2.button("Начать заново", type="primary", use_container_width=True):
        hard_reset()
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def render_active_screen() -> None:
    screen = st.session_state.screen
    if screen == "landing":
        render_landing()
    elif screen == "questionnaire":
        render_questionnaire()
    elif screen == "summary":
        render_summary()
    elif screen == "mode":
        render_mode_selection()
    elif screen == "simulation":
        render_simulation()
    else:
        render_results()


def render_sidebar() -> None:
    st.sidebar.title("Навигация")
    readable = dict(FLOW_STEPS)
    st.sidebar.write(f"Текущий этап: **{readable[st.session_state.screen]}**")

    if st.session_state.onboarding_profile is not None and st.sidebar.button("К анкете", use_container_width=True):
        st.session_state.screen = "questionnaire"
        st.session_state.questionnaire_step = 0
        st.rerun()

    if st.session_state.onboarding_profile is not None and st.sidebar.button("К выбору режима", use_container_width=True):
        st.session_state.screen = "mode"
        st.rerun()

    if st.session_state.logs and st.sidebar.button("Посмотреть результаты", use_container_width=True):
        st.session_state.screen = "results"
        st.rerun()

    if st.sidebar.button("Сбросить прогресс", use_container_width=True):
        hard_reset()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.caption("Режимы сценариев")
    st.sidebar.caption(", ".join(regime_options()))


def main() -> None:
    st.set_page_config(page_title="Risk Profiling Simulation", layout="wide")
    inject_styles()
    initialize_state()
    render_sidebar()
    render_step_tracker()
    render_active_screen()


if __name__ == "__main__":
    main()
