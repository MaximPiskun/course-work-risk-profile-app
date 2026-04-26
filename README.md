# Portfolio Risk Tolerance Simulation (Offline Prototype)

Educational/research Streamlit app for:
- long-term risk tolerance estimation,
- short-term behavioral reactivity to losses,
- transparent heuristic scoring (0-100),
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
- **Page 2: Simulation** (36 monthly steps)
  - choose allocation weights (Cash/Bonds/Stocks/Gold)
  - actions: keep, rebalance, reduce risk, go to cash, end early
  - neutral risk disclosure
  - decision logging per month
  - charts: portfolio value, drawdown, allocation history
- **Page 3: Results**
  - risk tolerance score (0-100)
  - behavioral reactivity score (0-100)
  - final risk index + 3-level and 5-level groups
  - feature-level explanation of score drivers
  - reproducibility section (seed + scenario parameters)
  - downloads: JSON and CSV

## Project Structure
- `app.py` - Streamlit UI and simulation loop
- `market.py` - regime-based return generator
- `scoring.py` - transparent scoring heuristics and group mapping
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

## Notes
- No external APIs are used.
- All data are generated locally from regime parameters.
- The prototype is educational/research only and not investment advice.
