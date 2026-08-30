# AI Stock Advisor — University Project

## What this is
A stock price-direction predictor (Random Forest on technical indicators)
combined with a risk-profiling questionnaire that tailors stock
recommendations to the user's risk tolerance, running on **real Pakistan
Stock Exchange (PSX) historical data**, fetched and kept up to date
**automatically** — no manual Excel downloads required.

## Data source & how it's fetched
`src/psx_api_fetch.py` fetches historical + daily OHLCV data via the
`psxdata` Python library, which works by reading PSX's public Data Portal
(dps.psx.com.pk). **Important trade-off to know about**: PSX's own Terms
of Use technically prohibit automated/programmatic scraping of their site
without a license. This project uses psxdata anyway, as a deliberate,
informed choice in favor of full automation over the manual-download
approach this project used previously (`src/psx_ingest.py`, still present
as a fully ToS-compliant fallback if you'd rather not rely on psxdata —
see "Manual fallback" below).

## Setup (run these once)
    pip install -r requirements.txt

## First-time setup — run these IN ORDER, don't skip step 1
    python scripts/verify_data_source.py   # REQUIRED FIRST — tests the data
                                            # source on 5 real symbols and
                                            # reports what dates are actually
                                            # available. If this fails or
                                            # looks wrong, stop and fix
                                            # psx_api_fetch.py before continuing
                                            # (see that script's own docstring).
    python scripts/update_data.py          # fetches full KSE-100 history
                                            # (up to 10 years per ticker),
                                            # builds features, trains models

The first real run of `update_data.py` will take a while — it's fetching
and training for potentially ~100 tickers, sequentially and politely
rate-limited (see `REQUEST_DELAY_SECONDS` in `config.py`), not in
parallel. Expect this to take a meaningful amount of time, not seconds.

## Every run after that
    python scripts/update_data.py

Same command — it automatically detects this isn't a first run (each
ticker already has stored data), and fetches ONLY the days since each
ticker's last stored date instead of the full history again. This is
what should run daily via a scheduler — see "Keeping data fresh" below.

## Run the tests
    pytest tests/ -v

## Launch the app
    streamlit run app/app.py

It will open automatically at http://localhost:8501. The app always reads
whatever's currently in `data/`, so once `update_data.py` has run (once,
or on a schedule), predictions and recommendations reflect the latest
data with no separate step needed.

## Keeping data fresh — now genuinely automatic
`.github/workflows/daily_refresh.yml` runs `scripts/update_data.py` once a
day via GitHub Actions and commits the result. If the app is deployed on
Streamlit Community Cloud (which redeploys on every push), this keeps the
LIVE app's data and predictions current with zero manual work — a real
change from the earlier manual-re-download workflow this project used
before.

## Manual fallback (if you'd rather not use psxdata)
`src/psx_ingest.py` still works exactly as before: download an export
from PSX's Data Portal yourself (personal/non-commercial use, fully
within PSX's stated terms), drop it in `raw_uploads/`, run
`python src/psx_ingest.py`. It now writes the same `.parquet` format as
the automated path, so both approaches are fully interchangeable — you
can use one, the other, or switch between them.

## Project structure
    raw_uploads/  -> manual Excel export(s), only used by the fallback path
    data/         -> stock data + engineered features, Parquet format
    models/       -> trained models (.pkl) + evaluation metrics (.json)
    logs/         -> app.log (all pipeline/app activity)
    src/          -> config.py, psx_api_fetch.py, psx_ingest.py, features.py, model.py, risk_engine.py
    app/          -> app.py (Streamlit UI)
    scripts/      -> verify_data_source.py, update_data.py
    tests/        -> pytest test suite
    .github/      -> GitHub Actions workflow for scheduled automated refresh

## Risk categorization methodology
Stocks are classified into fixed volatility bands (not ranked relative to
each other): low <15% annualized volatility, moderate 15-30%, high >30%.
On the real PSX data, none of the 6 tickers currently fall under 15% —
all are moderate-to-high — which is itself a useful, honest finding for
your report (PSX blue-chip volatility running higher than typical
US-blue-chip-calibrated thresholds). Thresholds are configurable in
src/config.py. Beta (systematic risk relative to the market) is computed
against the **real KSE-100 index** (see "Real KSE-100 index data" below) —
not an approximation.

## Data source & licensing (important — read before extending this)
This project's PRIMARY data path (`src/psx_api_fetch.py`, via `psxdata`)
scrapes PSX's public Data Portal programmatically. PSX's Terms of Use
technically prohibit automated scraping/systematic retrieval without a
license (see their "Unauthorized use of PSX data" notice on
dps.psx.com.pk). This project uses psxdata anyway — a deliberate,
informed trade-off made in favor of full automation (no manual downloads,
daily auto-updates) after weighing it against the fully-compliant but
manual alternative. State this explicitly and honestly in your report;
don't present the automated pipeline as unambiguously ToS-compliant, because
it isn't. If you'd rather not make that trade-off, `src/psx_ingest.py` +
manually downloading exports yourself is fully within PSX's stated terms
(personal, non-commercial single-copy use) and produces identical output —
see "Manual fallback" above.

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
- The automated data pipeline (psxdata) was chosen knowingly over a fully
  ToS-compliant manual alternative — this trade-off, and the reasoning
  behind it, is itself worth a paragraph in your report's methodology or
  limitations section.

## New in this version

### Fully automated data pipeline (up to all KSE-100 stocks)
`scripts/update_data.py` now fetches data itself, via `src/psx_api_fetch.py`
— no manual downloads needed. It fetches the live KSE-100 constituent
list from PSX and, for each ticker, either backfills up to `HISTORY_YEARS`
(10 by default, `config.py`) years of history on first run, or fetches
only the days since that ticker's last stored date on every run after
that. `config.get_available_tickers()` auto-detects however many tickers
have been processed, so the app scales from 6 to ~100 tickers with zero
code changes — purely a function of how many the fetch succeeds for.
`src/psx_ingest.py` (manual Excel) still works as a fully interchangeable
fallback — both write the same `.parquet` format.

Sequential, rate-limited fetching (`REQUEST_DELAY_SECONDS` in `config.py`)
means a full ~100-ticker first run will take a meaningful amount of time,
not seconds — this is deliberate, to stay a low-load, polite consumer of
a scraped free source, not a bug.

**Testing status**: `psx_api_fetch.py` and the live fetch logic in
`update_data.py` were written against psxdata's documented API but could
not be executed against a live network by the person who wrote them (no
internet access in that environment). Run `scripts/verify_data_source.py`
first — genuinely first, before anything else — the very first time you
set this up.

### 15-question risk questionnaire (up from 6)
Expanded to cover more dimensions: financial dependents, debt load,
liquidity needs, concentration comfort, reaction to hype, past loss
experience, guaranteed-vs-uncertain return preference, market knowledge,
and investment goals — alongside the original 6 (loss reaction, horizon,
experience, income stability, growth preference, emotional reaction to
swings). Scoring logic is unchanged (percentage-based), so thresholds
still work correctly at any question count.

### Profile persistence via a shareable code (no login required)
After completing the questionnaire, the app generates a short text code
(`RW1-...`) encoding your answers. Save it, and paste it back into the
"Already have a saved profile code?" box on a future visit to skip
re-answering. This is deliberately NOT a real user account/database —
implemented this way specifically because Streamlit Community Cloud's
free tier does not guarantee persistent server-side storage across app
restarts, so anything written to a local file or database could be lost
without warning. The profile code sidesteps that entirely: it lives in
the user's hands, not the server's disk. If genuine account-based
persistence across devices without re-entering a code is needed later,
that requires a real hosted database (e.g. a free-tier Postgres via
Supabase/Neon) — a bigger addition intentionally left out for now.

### Expected return, realized return, and alpha (not "valuation")
For each recommended stock, the app now shows:
- **Realized annual return**: the stock's actual historical average annual
  return, computed from its full price history
- **CAPM expected return**: `risk_free_rate + beta × (market_return − risk_free_rate)`
  — the return the stock's risk level (beta) would predict it "should"
  deliver, per the Capital Asset Pricing Model
- **Alpha**: realized − expected. Positive alpha = historically
  outperforming what its risk level would predict; negative = underperforming

**Important honesty note**: this is a risk-adjusted *performance* signal,
not a fundamental *valuation* judgment. True over/undervaluation (in the
sense of "is this stock priced correctly given the company's earnings")
requires fundamental data — P/E ratio, EPS, book value — which isn't
present in OHLCV price history at all. This app does not have access to
that data and does not claim to. `RISK_FREE_RATE` in `config.py` is a
configurable placeholder (currently 11%, representative of Pakistan's
higher-rate environment) — update it if you have a current, cited figure.
The market benchmark used for beta/CAPM is the **real KSE-100 index**
(see "Real KSE-100 index data" below), not a proxy.

### Beta and correlation together
Beta (magnitude of co-movement with the market) and correlation (how
closely a stock tracks the market's direction, independent of magnitude)
are now both shown — they answer different questions and can diverge
(e.g. a stock can have high beta but low correlation if it swings hard
but not always in sync with the market).

## Teacher feedback round — additions for viva readiness

Addressing 5 points raised after presenting the 6-stock version:

1. **"On what basis is it predicting?"** — Stock Explorer tab now shows a
   feature-importance bar chart per stock (which technical indicators the
   model actually relied on), plus a plain-language explanation of the
   top signals.
2. **"What sources besides price history?"** — Answered honestly, not
   padded: currently **only price and volume** feed the model. Sector
   classification was added as static reference metadata (verified against
   public company records) for context, but is descriptive, not a model
   input — one stock per sector isn't enough data to build a real
   sector-index feature. See Tab 4 for the full, explicit "what this app
   does/doesn't use" statement.
3. **"Make analysis visible for viva"** — new **Tab 4 (Analysis &
   Methodology)** consolidates data sources, sector overview, the
   correlation matrix, the beta caveat, and the evaluation methodology all
   in one place, specifically so this doesn't require hunting across tabs
   during a defense.
4. **"Beta vs. KSE and other stocks, industry stats"** — a real
   stock-vs-stock **correlation matrix** (`get_pairwise_correlation_matrix`)
   was added, computed from tracked stocks' own data. **Beta vs. the real
   KSE-100 index is now implemented** (see "Real KSE-100 index data"
   below) — `get_market_returns()` uses actual KSE-100 index history when
   available, replacing the earlier 6-stock proxy.
5. **"Long-term ROI, stability, price in 1/3/6/12 months"** —
   `risk_engine.project_future_price()` gives a trend-extrapolation
   projection (historical CAGR + volatility-based range) at each horizon.
   **Deliberately not a new ML model**: a multi-month forecasting model
   trained the same way as the daily predictor would very likely show
   similarly weak accuracy at that horizon, and presenting a shaky number
   as a "forecast" would invite more scrutiny in a viva, not less. The
   trend-extrapolation framing is standard, transparent, and honestly
   widens its uncertainty range the further out it projects.

## Real KSE-100 index data (resolves the earlier beta limitation)
As of the latest data update, `raw_uploads/` includes a "KSE 100" sheet
with genuine historical KSE-100 index prices (same source/format as the
6 stocks). `src/psx_ingest.py` ingests it as `data/KSE100.parquet` /
`KSE100_features.parquet` — same pipeline as any stock, EXCEPT
`config.get_available_tickers()` explicitly excludes it from the
recommendable-stock list (`config.MARKET_INDEX_TICKER = "KSE100"`),
since it's a benchmark, not a stock to recommend.

`risk_engine.get_market_returns()` now prefers this real index data,
logging clearly which source a given run actually used — falling back
to the old equal-weighted 6-stock proxy only if `KSE100_features.parquet`
is missing (e.g. on a stale/older data ingest). Beta, correlation, CAPM
expected return, and alpha throughout the app now reflect genuine
market-relative figures, not an approximation — Tab 4 in the app states
this plainly.

## External data integration (Tier 1-3) — final addition before submission

Building on an external data workbook (sector classification, economic
factor hypotheses, macro/commodity series, company fundamentals), added
in three honestly-scoped tiers rather than one undifferentiated feature:

### Tier 1 — Sector & economic rationale (static, safe, complete)
`src/external_data.py` (`load_stock_industry`, `load_factor_mapping`)
replaces the earlier built-in sector labels with richer external data:
industry classification plus each stock's hypothesized primary/secondary
external factor and WHY (e.g., "OGDC ← Brent: oil revenue/profitability").
Shown in Tab 4 as an explicit "Economic rationale" table — this is
genuine economic reasoning behind what gets tested, not a black-box
correlation search.

### Tier 2 — Macro/commodity experiment (real data, honestly scoped)
`src/macro_experiment.py` trains a price-only model and a price+macro
model on the SAME shorter window and compares them fairly. **Real
constraint, stated everywhere this is shown**: macro/commodity data
(SBP rate, KIBOR, Brent, Coal, Natural Gas, Fertilizer price — NASDAQ100
is present but entirely empty in the source and excluded from the model)
only covers ~24 months, vs. the main model's 10 years of daily price
data — so this is kept as a separate "supplementary experiment," never
merged into or compared against the main model's headline numbers.
Monthly values are forward-filled to daily; results are reported
honestly either way, including for stocks where adding macro data made
predictions worse, not just where it helped.

### Tier 3 — Real P/E ratio (partial, but genuine fundamental valuation)
`compute_pe_ratios()` computes actual P/E = current price ÷ most recently
published EPS — real fundamental valuation, something this project
explicitly could not do before (only price-derived alpha/CAPM). **Only
available for AICL, OGDC, MEBL** — SYS/LUCK/FFC have no published
financials in the source data and are shown as such, not estimated.
Look-ahead bias is handled by assuming a configurable ~90-day publication
lag after fiscal year-end (the source gives fiscal year, not an exact
publication date) — this assumption is stated explicitly in the UI, not
hidden in the calculation.

### Where this shows up
All three tiers live in **Tab 4 (Analysis & Methodology)** — sector
overview, economic rationale, company fundamentals/P/E, and the macro
experiment comparison, in that order, right before the existing
correlation matrix and beta sections. The main model and its predictions
elsewhere in the app are UNCHANGED by any of this — still price/volume
only, still the same ~50% honest accuracy story.

## Multi-page structure with login (latest addition)

The app was restructured from a single page with tabs into a proper
multi-page app, per instructor feedback requesting an intro page and
login flow before the main app content:

- **Welcome** (intro page) — what the app does, shown to everyone
- **Login** — a simple username/password gate
- **1. Risk Questionnaire → 4. Analysis & Methodology** — the real app,
  only reachable after logging in

### How the login works — read this before presenting
This is a **simple demo-level login gate, not real authentication**.
One shared username/password (default: `student` / `riskwise2026`,
shown on the login screen itself if you haven't set your own), no
per-user accounts, no password hashing. This was a deliberate scope
choice: it satisfies "the app should have a login" visually and
functionally for a university presentation, without the added setup of
registering a real OAuth provider (Google/Microsoft) for what's ultimately
a class project. Say this plainly if asked in a viva — it's a legitimate,
disclosed limitation, not something to be caught out on.

**To set your own login** (recommended before a real presentation, so
the default password isn't visible on screen): copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill
in your own username/password. This file is gitignored — never commit
real secrets to a public repo.

### Upgrading to real login later
Streamlit has a built-in `st.login()` supporting real OAuth (Google,
Microsoft, etc.) from version 1.42+. That requires registering an OAuth
app with the provider and configuring client credentials — genuinely more
setup than this project's scope, but the natural next step if this ever
needs real per-user accounts.

### File structure of the app itself
    app/
    ├── app.py                        -> entry point: session state, login
    │                                    gate, builds the page list, calls
    │                                    st.navigation()
    └── page_modules/
        ├── shared.py                 -> cached wrappers used by multiple pages
        ├── intro.py                  -> Welcome page
        ├── login.py                  -> Login page
        ├── questionnaire.py          -> Page 1
        ├── recommendations.py        -> Page 2
        ├── stock_explorer.py         -> Page 3
        └── methodology.py            -> Page 4

Deploying to Streamlit Community Cloud: the "Main file path" setting
stays `app/app.py` — unchanged from before, no redeploy configuration
needed, just push the new files.
