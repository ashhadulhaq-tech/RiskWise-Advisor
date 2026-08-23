"""
scripts/verify_data_source.py - RUN THIS FIRST, before anything else
========================================================================
Tests the psxdata integration against a handful of real PSX symbols and
reports exactly what came back: row counts, earliest/latest available
date, and any errors. This exists specifically because the person who
wrote src/psx_api_fetch.py could not run it against a live network
themselves (no internet access in that environment) — so this script is
your actual first confirmation that the integration works, not a leap
of faith.

Usage:
    pip install -r requirements.txt
    python scripts/verify_data_source.py

If this fails or the data looks wrong (e.g. column names don't match,
dates look garbled, symbols return nothing), STOP — do not proceed to
scripts/update_data.py. Fix psx_api_fetch.py's _standardize() function
to match whatever psxdata actually returns on your machine first. The
psxdata GitHub README (https://github.com/mtauha/psxdata) and its
examples/ folder are the source of truth for its real current output
format, which may have shifted since this was written.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import get_logger
import psx_api_fetch

logger = get_logger(__name__)

# A deliberately small, well-known mix of PSX tickers to sanity-check
# against — large-cap, liquid, long-listed names most likely to have deep
# history if the data source has it at all. Includes the 6 tickers this
# project already has manually-downloaded data for, so you can directly
# compare psxdata's output against the known-good Excel-derived data.
TEST_SYMBOLS = ["LUCK", "OGDC", "MEBL", "FFC", "SYS"]


def main():
    print("=" * 70)
    print("PSX DATA SOURCE VERIFICATION")
    print("=" * 70)

    print("\n[1/2] Testing psxdata.indices('KSE100') — current index constituents...")
    try:
        tickers = psx_api_fetch.get_kse100_tickers()
        print(f"  -> SUCCESS: {len(tickers)} tickers returned.")
        print(f"  -> First 10: {tickers[:10]}")
        if len(tickers) < 50:
            print(f"  -> ⚠️  WARNING: expected ~100 constituents, got {len(tickers)}. "
                  f"Check psxdata's indices() output format — may have changed.")
    except Exception as e:
        print(f"  -> FAILED: {e}")
        print("  -> Cannot get the live KSE-100 list. The rest of this script will "
              "still test individual symbols, but scripts/update_data.py's automatic "
              "'fetch all KSE-100' mode won't work until this is fixed.")

    print(f"\n[2/2] Testing psxdata.stocks() on {len(TEST_SYMBOLS)} symbols: {TEST_SYMBOLS}")
    print(f"      Requesting up to 10 years of history for each...\n")

    results = []
    for symbol in TEST_SYMBOLS:
        print(f"  Fetching {symbol}...")
        try:
            df = psx_api_fetch.fetch_history(symbol)
            if df.empty:
                print(f"    -> NO DATA returned for {symbol}.")
                results.append((symbol, "NO DATA", None, None, 0))
                continue

            earliest = df["date"].min().date()
            latest = df["date"].max().date()
            n_rows = len(df)
            years_covered = (df["date"].max() - df["date"].min()).days / 365.25

            print(f"    -> {n_rows} rows | {earliest} to {latest} "
                  f"(~{years_covered:.1f} years)")
            results.append((symbol, "OK", earliest, latest, n_rows))
        except Exception as e:
            print(f"    -> ERROR: {e}")
            results.append((symbol, f"ERROR: {e}", None, None, 0))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Symbol':<8} {'Status':<12} {'Earliest':<12} {'Latest':<12} {'Rows':<6}")
    for symbol, status, earliest, latest, n_rows in results:
        print(f"{symbol:<8} {status:<12} {str(earliest or '-'):<12} "
              f"{str(latest or '-'):<12} {n_rows:<6}")

    successes = [r for r in results if r[1] == "OK"]
    print(f"\n{len(successes)}/{len(TEST_SYMBOLS)} symbols returned usable data.")

    if len(successes) == 0:
        print("\n❌ NOTHING WORKED. Do not proceed to scripts/update_data.py yet.")
        print("   Likely causes: psxdata's scraping target changed (PSX redesigned")
        print("   their site), psxdata isn't installed correctly, or a network/")
        print("   firewall issue. Check the error messages above.")
        sys.exit(1)
    elif len(successes) < len(TEST_SYMBOLS):
        print("\n⚠️  PARTIAL SUCCESS. Some symbols worked, some didn't — this is")
        print("   plausible (e.g. thinly-traded or newly-listed symbols can genuinely")
        print("   have less history) but double check the failures aren't a bug.")
    else:
        min_years = min((r[3] - r[2]).days / 365.25 for r in successes if r[2] and r[3])
        print(f"\n✅ ALL TEST SYMBOLS WORKED. Shortest history available: "
              f"~{min_years:.1f} years.")
        if min_years < 9:
            print(f"   Note: requested {psx_api_fetch.HISTORY_YEARS if hasattr(psx_api_fetch, 'HISTORY_YEARS') else 10} "
                  f"years but got less — this may just be a genuine data availability "
                  f"limit on PSX's side (common for a ~10-year-old scraping target), "
                  f"not necessarily a bug.")
        print("\n   You're clear to run: python scripts/update_data.py")


if __name__ == "__main__":
    main()
