# Portfolio Risk Tolerance Simulation (Offline Prototype)

Educational/research Streamlit app for:
- long-term risk tolerance estimation,
- short-term behavioral reactivity to losses,
- transparent score model (0-100) with behavioral quality checks,
- reproducible simulation episodes with fixed random seed.

## Features
- **Page 1: Onboarding** (minimum profile)
  - age
  - monthly income and expenses
  - savings amount
  - education
  - investment horizon
  - investment goal
  - random seed + optional scenario selection
- **Page 2: Simulation** (variable dynamic steps)
  - choose allocation weights (Cash/Bonds/Stocks/Gold)
  - quick rebalance presets (Conservative / Balanced / Growth)
  - random market phases with random phase length (Bull/Sideways/Crisis)
  - actions: keep, rebalance, reduce risk, go to cash, end early
  - explicit pause/resume mode while preparing a rebalance decision
  - neutral risk disclosure
  - decision logging per step
  - charts: portfolio value, drawdown, allocation history
- **Page 3: Results**
  - risk tolerance score (0-100)
  - behavioral reactivity score (0-100)
  - final risk index + 3-level and 5-level groups
  - personal feedback cards (what happened, what it means, what to do)
  - profile reliability score and chaotic behavior warning
  - feature-level explanation of score drivers
  - reproducibility section (seed + scenario parameters)
  - downloads: JSON and CSV
  - score model version and calibration status

- **Data capture for calibration**
  - local SQLite storage of sessions, decision events, and final scores
  - export + fit pipeline for recalibrating score weights on real labeled sessions

## Project Structure
- `app.py` - Streamlit UI and simulation loop
- `market.py` - regime-based return generator
- `scoring.py` - transparent scoring heuristics and group mapping
- `storage.py` - local SQLite persistence for sessions/events/scores
- `scripts/calibrate_scoring.py` - score-weight calibration utility
- `tests/test_scoring.py` - unit tests for scoring behavior
- `requirements.txt`

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests
```bash
pytest -q
```

## Calibration Workflow
1. Run app and collect sessions (they are written to `risk_profile_sessions.db`).
2. Export labeling template and try fit:
```bash
python scripts/calibrate_scoring.py --db risk_profile_sessions.db
```
3. Fill `data/calibration_labels.csv` with `session_id,target_risk_score`.
4. Re-run script to produce calibrated model at `calibration/score_weights.json`.
5. App will auto-load calibrated weights on next run.

## Notes
- No external APIs are used.
- All data are generated locally from regime parameters.
- The prototype is educational/research only and not investment advice.

## Scientific Basis (used in scoring proxies)
- Kahneman, D., & Tversky, A. (1979). *Prospect Theory: An Analysis of Decision under Risk*. Econometrica, 47(2), 263-291.
- Benartzi, S., & Thaler, R. H. (1995). *Myopic Loss Aversion and the Equity Premium Puzzle*. The Quarterly Journal of Economics, 110(1), 73-92.
- Grable, J. E., & Lytton, R. H. (1999). *Financial risk tolerance revisited: the development of a risk assessment instrument*. Financial Services Review, 8(3), 163-181.
- Barber, B. M., & Odean, T. (2000). *Trading Is Hazardous to Your Wealth*. The Journal of Finance, 55(2), 773-806.
