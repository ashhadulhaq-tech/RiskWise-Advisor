# AI Stock Advisor — University Project

## What this is
A stock price-direction predictor (Random Forest on technical indicators)
combined with a risk-profiling questionnaire that tailors stock
recommendations to the user's risk tolerance — now running on **real
Pakistan Stock Exchange (PSX) historical data** for 6 tickers: AICL, FFC,
LUCK, MEBL, OGDC, SYS (10 years, July 2016–July 2026).

## Data source
Historical data was manually downloaded from PSX's Data Portal
(dps.psx.com.pk), per PSX's own terms of use, which permit personal/
academic single-copy downloads but prohibit automated scraping — see
"Data source & licensing" below for the full reasoning. The exported file
lives in `raw_uploads/` and is parsed by `src/psx_ingest.py`.

## Setup (run these once)
    pip install -r requirements.txt

## Run the full pipeline (do this first, in order)
    python src/psx_ingest.py        # parses raw_uploads/*.xlsx into data/{TICKER}.csv
    python src/features.py          # builds technical indicators
    python src/model.py             # trains models + logs accuracy metrics

Or run all three (plus re-ingestion) in one step:
    python scripts/update_data.py

## Run the tests
    pytest tests/ -v

## Launch the app
    streamlit run app/app.py

It will open automatically at http://localhost:8501

## Keeping data fresh
Since PSX prohibits automated scraping, "daily auto-update" here means:
periodically re-download an updated export from PSX's Data Portal (a
manual, personal-use step, same as the original file), drop it into
`raw_uploads/`, then run `scripts/update_data.py` — it re-ingests
whatever `.xlsx` is in that folder, rebuilds features, and retrains
models. `.github/workflows/daily_refresh.yml` automates the
rebuild-from-existing-file part on a schedule; it does NOT fetch new
PSX data itself, for the licensing reasons above. If genuine automated
daily fetching is needed later, the real path is a licensed PSX data
feed (marketdatarequest@psx.com.pk) or the `psxdata` library IF you've
independently confirmed it's used within PSX's terms — `update_data.py`
has a stubbed integration point for that (`try_fetch_via_psxdata`),
currently inactive since `psxdata` isn't installed by default.

## Project structure
    raw_uploads/  -> manually downloaded PSX Excel export(s)
    data/         -> parsed stock data + engineered features (CSV)
    models/       -> trained models (.pkl) + evaluation metrics (.json)
    logs/         -> app.log (all pipeline/app activity)
    src/          -> config.py, psx_ingest.py, features.py, model.py, risk_engine.py, data_pipeline.py
    app/          -> app.py (Streamlit UI)
    scripts/      -> update_data.py (refresh pipeline)
    tests/        -> pytest test suite
    .github/      -> GitHub Actions workflow for scheduled rebuilds

## Risk categorization methodology
Stocks are classified into fixed volatility bands (not ranked relative to
each other): low <15% annualized volatility, moderate 15-30%, high >30%.
On the real PSX data, none of the 6 tickers currently fall under 15% —
all are moderate-to-high — which is itself a useful, honest finding for
your report (PSX blue-chip volatility running higher than typical
US-blue-chip-calibrated thresholds). Thresholds are configurable in
src/config.py. Beta (systematic risk relative to the market) is also
computed; the current market benchmark is an equal-weighted average of
the 6 tracked tickers' own returns, used as a documented placeholder
until a real KSE-100 index feed is available.

## Data source & licensing (important — read before extending this)
PSX's Data Portal explicitly prohibits automated scraping and
redistribution of market data without a license (see their Terms of Use
and "Unauthorized use of PSX data" notice on dps.psx.com.pk). Their terms
DO permit downloading a single copy for personal, non-commercial use —
which is what was done here. This project therefore does NOT scrape PSX
automatically; it ingests a manually-downloaded export instead. This is
a deliberate, documented design decision — worth stating explicitly in
your report as evidence of respecting data-provider terms rather than
working around them.

## Notes for the report
- Model evaluation uses a chronological (time-based) train/test split —
  never random — to avoid lookahead bias, standard practice for time series.
- Accuracy is compared against a majority-class baseline, not judged alone.
  On real PSX data, average model accuracy came out to ~49.5% vs. a ~51%
  baseline — a genuinely honest result consistent with the Efficient
  Market Hypothesis, not a bug.
- Risk profiling deliberately uses a transparent scored questionnaire (not
  ML), matching real-world robo-advisor practice (Betterment, Wealthfront)
  for auditability and explainability.
- Risk bands are fixed/absolute, not relative percentiles — a deliberate
  fix after an early version used relative ranking, which made labels
  dependent on which other stocks happened to be in the dataset.
