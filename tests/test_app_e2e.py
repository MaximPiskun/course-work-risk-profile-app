from __future__ import annotations

import time

from streamlit.testing.v1 import AppTest

BTN_START = "Начать"
BTN_NEXT = "Продолжить"
BTN_CHOOSE_MODE = "Выбрать режим"
BTN_CHOOSE = "Выбрать"
BTN_ENTER_DECISION = "Принять решение (пауза)"
BTN_CANCEL_DECISION = "Вернуться к наблюдению"
BTN_APPLY = "Применить решение и продолжить"
BTN_SHOW_RESULTS = "Посмотреть результаты"
BTN_GROWTH = "Рост"
BTN_EARLY = "Завершить досрочно"
BTN_TO_MODE = "К выбору режима"
BTN_TO_QUESTIONNAIRE = "К анкете"
BTN_RESET = "Сбросить прогресс"
RADIO_REBALANCE = "Ребалансировать"
RADIO_ACTION_LABEL = "Ваше действие"
SUBHEADER_SIM = "Основная симуляция"
SUBHEADER_RESULTS = "Итоговый профиль"
SUBHEADER_MODE = "Выберите режим прохождения"
SUBHEADER_QUESTIONNAIRE = "Короткая анкета"


def _click_button(at: AppTest, label: str) -> None:
    matches = [b for b in at.button if b.label == label]
    assert matches, f"Button not found: {label}. Available: {[b.label for b in at.button]}"
    matches[0].click()
    at.run()


def _go_to_simulation(at: AppTest) -> None:
    at.run()
    _click_button(at, BTN_START)
    _click_button(at, BTN_NEXT)
    _click_button(at, BTN_NEXT)
    _click_button(at, BTN_NEXT)
    _click_button(at, BTN_CHOOSE_MODE)
    _click_button(at, BTN_CHOOSE)
    subheaders = [s.value for s in at.subheader]
    assert SUBHEADER_SIM in subheaders


def test_e2e_happy_path_to_results() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    at.session_state["decision_mode_active"] = True
    at.session_state["sim_paused"] = True
    at.run()

    before = int(at.session_state["month_index"])
    _click_button(at, BTN_APPLY)
    after = int(at.session_state["month_index"])
    assert after >= before + 1 or at.session_state["screen"] == "results"

    _click_button(at, BTN_SHOW_RESULTS)
    subheaders = [s.value for s in at.subheader]
    assert SUBHEADER_RESULTS in subheaders


def test_e2e_decision_mode_pauses_and_resume_returns_live() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    at.session_state["decision_mode_active"] = True
    at.session_state["sim_paused"] = True
    at.run()

    action_radios = [r for r in at.radio if r.label == RADIO_ACTION_LABEL]
    assert action_radios

    paused_month = int(at.session_state["month_index"])
    time.sleep(1.2)
    at.run()
    assert int(at.session_state["month_index"]) == paused_month

    at.session_state["decision_mode_active"] = False
    at.session_state["sim_paused"] = False
    at.run()

    action_radios = [r for r in at.radio if r.label == RADIO_ACTION_LABEL]
    assert not action_radios


def test_e2e_enter_decision_button_enables_pause_and_freezes_ticks() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    before_click = int(at.session_state["month_index"])
    _click_button(at, BTN_ENTER_DECISION)

    assert bool(at.session_state["decision_mode_active"]) is True
    assert bool(at.session_state["sim_paused"]) is True
    assert any(r.label == RADIO_ACTION_LABEL for r in at.radio)

    paused_month = int(at.session_state["month_index"])
    time.sleep(1.2)
    at.run()
    assert int(at.session_state["month_index"]) == paused_month == before_click


def test_e2e_sticky_resume_button_returns_to_live_mode() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    _click_button(at, BTN_ENTER_DECISION)
    _click_button(at, BTN_CANCEL_DECISION)

    assert bool(at.session_state["decision_mode_active"]) is False
    assert bool(at.session_state["sim_paused"]) is False


def test_e2e_decision_controls_appear_only_in_decision_mode() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    action_radios = [r for r in at.radio if r.label == RADIO_ACTION_LABEL]
    assert not action_radios

    at.session_state["decision_mode_active"] = True
    at.session_state["sim_paused"] = True
    at.run()

    action_radios = [r for r in at.radio if r.label == RADIO_ACTION_LABEL]
    assert action_radios


def test_e2e_rebalance_contains_bonds_slider_after_decision_click() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    _click_button(at, BTN_ENTER_DECISION)
    action_radios = [r for r in at.radio if r.label == RADIO_ACTION_LABEL]
    assert action_radios
    action_radios[0].set_value(RADIO_REBALANCE)
    at.run()

    slider_labels = {s.label for s in at.slider}
    assert {"Bonds, %", "Stocks, %", "Gold, %"}.issubset(slider_labels)


def test_e2e_rebalance_preset_and_invalid_sum_blocks_apply() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    at.session_state["decision_mode_active"] = True
    at.session_state["sim_paused"] = True
    at.run()

    action_radios = [r for r in at.radio if r.label == RADIO_ACTION_LABEL]
    assert action_radios, "No action radio on simulation screen"
    action_radios[0].set_value(RADIO_REBALANCE)
    at.run()

    _click_button(at, BTN_GROWTH)
    slider_vals = {s.label: s.value for s in at.slider}
    assert slider_vals.get("Bonds, %") == 20
    assert slider_vals.get("Stocks, %") == 60
    assert slider_vals.get("Gold, %") == 10

    for slider in at.slider:
        if slider.label in {"Bonds, %", "Stocks, %", "Gold, %"}:
            slider.set_value(100)
    at.run()

    apply_buttons = [b for b in at.button if b.label == BTN_APPLY]
    assert apply_buttons
    assert all(b.disabled for b in apply_buttons)


def test_e2e_apply_decision_turns_off_pause_and_keeps_live_running() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    _click_button(at, BTN_ENTER_DECISION)
    paused_month = int(at.session_state["month_index"])
    _click_button(at, BTN_APPLY)

    assert bool(at.session_state["decision_mode_active"]) is False
    assert bool(at.session_state["sim_paused"]) is False

    after_apply = int(at.session_state["month_index"])
    assert after_apply >= paused_month + 1 or at.session_state["screen"] == "results"

    if at.session_state["screen"] == "simulation":
        time.sleep(1.2)
        at.run()
        assert int(at.session_state["month_index"]) >= after_apply


def test_e2e_live_controls_present() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    slider_labels = [s.label for s in at.slider]
    assert "Скорость live-тика, сек" in slider_labels

    metric_labels = [m.label for m in at.metric]
    assert "До следующего тика" in metric_labels


def test_e2e_live_tick_advances_after_interval() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    before = int(at.session_state["month_index"])
    time.sleep(1.2)
    at.run()
    after = int(at.session_state["month_index"])

    assert after >= before + 1


def test_e2e_early_finish_button_present() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    assert any(b.label == BTN_EARLY for b in at.button)


def test_e2e_early_finish_navigates_to_results() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    _click_button(at, BTN_EARLY)
    subheaders = [s.value for s in at.subheader]
    assert SUBHEADER_RESULTS in subheaders


def test_e2e_sidebar_navigation_and_reset() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    _click_button(at, BTN_TO_MODE)
    subheaders = [s.value for s in at.subheader]
    assert SUBHEADER_MODE in subheaders

    _click_button(at, BTN_TO_QUESTIONNAIRE)
    subheaders = [s.value for s in at.subheader]
    assert SUBHEADER_QUESTIONNAIRE in subheaders

    _click_button(at, BTN_RESET)
    button_labels = [b.label for b in at.button]
    assert BTN_START in button_labels


def test_e2e_simulation_tabs_present() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    tab_labels = [t.label for t in at.get("tab")]
    assert {"Портфель", "Просадка", "Аллокация", "Активы"}.issubset(set(tab_labels))


def test_e2e_no_mode_subheader_leak_on_simulation_screen() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    subheaders = [s.value for s in at.subheader]
    assert SUBHEADER_SIM in subheaders
    assert SUBHEADER_MODE not in subheaders



def test_e2e_mode_screen_contains_preplay_instructions() -> None:
    at = AppTest.from_file("app.py")
    at.run()
    _click_button(at, BTN_START)
    _click_button(at, BTN_NEXT)
    _click_button(at, BTN_NEXT)
    _click_button(at, BTN_NEXT)
    _click_button(at, BTN_CHOOSE_MODE)

    info_texts = [x.value for x in at.info]
    expected = "Перед стартом:"
    assert any(expected in t for t in info_texts)


def test_e2e_mode_screen_contains_quick_brief_block() -> None:
    at = AppTest.from_file("app.py")
    at.run()
    _click_button(at, BTN_START)
    _click_button(at, BTN_NEXT)
    _click_button(at, BTN_NEXT)
    _click_button(at, BTN_NEXT)
    _click_button(at, BTN_CHOOSE_MODE)

    markdown_texts = [m.value for m in at.markdown]
    assert any("20-30" in t for t in markdown_texts)
    assert any("mode-cards-grid" in t for t in markdown_texts)


def test_e2e_simulation_layout_wrappers_present() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    markdown_texts = [m.value for m in at.markdown]
    assert any("sim-sticky-panel" in t for t in markdown_texts)
    assert any("sim-metrics-grid" in t for t in markdown_texts)
    assert any("sim-main-grid" in t for t in markdown_texts)


def test_e2e_live_screen_shows_decision_hint_when_not_in_decision_mode() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    info_texts = [x.value for x in at.info]
    expected = "Нажмите «Принять решение (пауза)»"
    assert any(expected in t for t in info_texts)


def test_e2e_repeated_live_reruns_do_not_raise_exceptions() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    for _ in range(8):
        at.run()
        subheaders = [s.value for s in at.subheader]
        assert SUBHEADER_SIM in subheaders
        assert SUBHEADER_MODE not in subheaders
        assert len(at.exception) == 0


def test_e2e_chart_controls_block_present_near_graphs() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    markdown_texts = [m.value for m in at.markdown]
    assert any("sim-chart-controls" in t for t in markdown_texts)
    assert any(b.label == BTN_ENTER_DECISION for b in at.button)


def test_e2e_decision_toggle_is_stable_over_multiple_clicks() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    enter_buttons = [b for b in at.button if b.label == BTN_ENTER_DECISION]
    assert len(enter_buttons) == 1

    for _ in range(4):
        _click_button(at, BTN_ENTER_DECISION)
        assert bool(at.session_state["decision_mode_active"]) is True
        assert bool(at.session_state["sim_paused"]) is True

        cancel_buttons = [b for b in at.button if b.label == BTN_CANCEL_DECISION]
        assert len(cancel_buttons) == 1
        _click_button(at, BTN_CANCEL_DECISION)
        assert bool(at.session_state["decision_mode_active"]) is False
        assert bool(at.session_state["sim_paused"]) is False
