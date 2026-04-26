from __future__ import annotations

from streamlit.testing.v1 import AppTest

BTN_START = "\u041d\u0430\u0447\u0430\u0442\u044c"
BTN_NEXT = "\u041f\u0440\u043e\u0434\u043e\u043b\u0436\u0438\u0442\u044c"
BTN_CHOOSE_MODE = "\u0412\u044b\u0431\u0440\u0430\u0442\u044c \u0440\u0435\u0436\u0438\u043c"
BTN_CHOOSE = "\u0412\u044b\u0431\u0440\u0430\u0442\u044c"
BTN_APPLY = "\u041f\u0440\u0438\u043d\u044f\u0442\u044c \u0440\u0435\u0448\u0435\u043d\u0438\u0435 \u0438 \u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0438\u0442\u044c"
BTN_SHOW_RESULTS = "\u041f\u043e\u0441\u043c\u043e\u0442\u0440\u0435\u0442\u044c \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b"
BTN_PAUSE = "\u041f\u043e\u0441\u0442\u0430\u0432\u0438\u0442\u044c \u043d\u0430 \u043f\u0430\u0443\u0437\u0443"
BTN_RESUME = "\u0412\u043e\u0437\u043e\u0431\u043d\u043e\u0432\u0438\u0442\u044c \u0441\u0438\u043c\u0443\u043b\u044f\u0446\u0438\u044e"
BTN_GROWTH = "\u0420\u043e\u0441\u0442"
BTN_EARLY = "\u0417\u0430\u0432\u0435\u0440\u0448\u0438\u0442\u044c \u0434\u043e\u0441\u0440\u043e\u0447\u043d\u043e"
BTN_TO_MODE = "\u041a \u0432\u044b\u0431\u043e\u0440\u0443 \u0440\u0435\u0436\u0438\u043c\u0430"
BTN_TO_QUESTIONNAIRE = "\u041a \u0430\u043d\u043a\u0435\u0442\u0435"
BTN_RESET = "\u0421\u0431\u0440\u043e\u0441\u0438\u0442\u044c \u043f\u0440\u043e\u0433\u0440\u0435\u0441\u0441"
RADIO_REBALANCE = "\u0420\u0435\u0431\u0430\u043b\u0430\u043d\u0441\u0438\u0440\u043e\u0432\u0430\u0442\u044c"
SUBHEADER_SIM = "\u041e\u0441\u043d\u043e\u0432\u043d\u0430\u044f \u0441\u0438\u043c\u0443\u043b\u044f\u0446\u0438\u044f"
SUBHEADER_RESULTS = "\u0418\u0442\u043e\u0433\u043e\u0432\u044b\u0439 \u043f\u0440\u043e\u0444\u0438\u043b\u044c"
SUBHEADER_MODE = "\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0440\u0435\u0436\u0438\u043c \u043f\u0440\u043e\u0445\u043e\u0436\u0434\u0435\u043d\u0438\u044f"
SUBHEADER_QUESTIONNAIRE = "\u041a\u043e\u0440\u043e\u0442\u043a\u0430\u044f \u0430\u043d\u043a\u0435\u0442\u0430"


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
    assert SUBHEADER_SIM in [s.value for s in at.subheader]


def test_e2e_happy_path_to_results() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    _click_button(at, BTN_APPLY)
    _click_button(at, BTN_SHOW_RESULTS)

    subheaders = [s.value for s in at.subheader]
    assert SUBHEADER_RESULTS in subheaders

    captions = [c.value for c in at.caption]
    assert any("\u041c\u043e\u0434\u0435\u043b\u044c:" in text for text in captions)


def test_e2e_pause_disables_apply_and_resume_enables() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    _click_button(at, BTN_PAUSE)
    apply_buttons = [b for b in at.button if b.label == BTN_APPLY]
    assert apply_buttons
    assert all(b.disabled for b in apply_buttons)

    _click_button(at, BTN_RESUME)
    apply_buttons = [b for b in at.button if b.label == BTN_APPLY]
    assert apply_buttons
    assert all(not b.disabled for b in apply_buttons)


def test_e2e_rebalance_preset_and_invalid_sum_blocks_apply() -> None:
    at = AppTest.from_file("app.py")
    _go_to_simulation(at)

    assert at.radio, "No radio controls on simulation screen"
    at.radio[0].set_value(RADIO_REBALANCE)
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


def test_e2e_early_finish_opens_results() -> None:
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
