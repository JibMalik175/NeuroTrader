"""
funding_cost_analysis.py — G3: does funding erase the futures edge?
───────────────────────────────────────────────────────────────────
Our deployment target is USDT-M perpetual futures (maker fees + shorts).
Perpetuals charge a FUNDING rate every 8h: when the rate is positive
(normal/contango), longs PAY and shorts RECEIVE; negative flips it.
fee_sensitivity.py ignores this — this script measures whether that was safe.

Pulls real historical funding rates from Binance's public endpoint
(fapi/v1/fundingRate, no API key needed), then computes the expected
funding cost per trade for our actual average hold times, separately for
longs and shorts, over the validation and test date ranges.

Usage:
  python scripts/funding_cost_analysis.py --val data/BTC_USDT_1h_val.parquet \
      --test data/BTC_USDT_1h_test.parquet --avg-hold-candles 13
"""

import argparse
import time

import numpy as np
import pandas as pd
import requests

FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"
SYMBOL = "BTCUSDT"


def fetch_funding(start_ms: int, end_ms: int) -> pd.DataFrame:
    """Paginated fetch of all funding events in [start, end] (8h cadence)."""
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        r = requests.get(FAPI, params={
            "symbol": SYMBOL, "startTime": cursor, "endTime": end_ms, "limit": 1000,
        }, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)  # stay polite to the public endpoint
    df = pd.DataFrame(rows)
    df["rate"] = df["fundingRate"].astype(float)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return df[["time", "rate"]]


def analyze(name: str, fund: pd.DataFrame, avg_hold_candles: float, candle_hours: float):
    """Expected funding cost per trade for the avg hold, by direction."""
    hold_hours = avg_hold_candles * candle_hours
    intervals = hold_hours / 8.0  # funding events crossed per trade (expected)

    mean_rate = fund["rate"].mean()
    pos_share = (fund["rate"] > 0).mean()

    # expected per-trade funding: longs pay +rate, shorts pay -rate
    long_cost = intervals * mean_rate * 100      # % of notional
    short_cost = intervals * -mean_rate * 100
    worst_abs = intervals * fund["rate"].abs().quantile(0.95) * 100

    print(f"\n[{name}] {fund['time'].min():%Y-%m-%d} -> {fund['time'].max():%Y-%m-%d}"
          f"  ({len(fund)} funding events)")
    print(f"  mean rate/8h      : {mean_rate*100:+.4f}%  (positive {pos_share:.0%} of the time)")
    print(f"  avg hold          : {avg_hold_candles:.1f} candles = {hold_hours:.0f}h "
          f"≈ {intervals:.1f} funding intervals")
    print(f"  expected cost/trade: LONG {long_cost:+.4f}%   SHORT {short_cost:+.4f}%")
    print(f"  95th-pct |cost|    : {worst_abs:.4f}% per trade")
    return long_cost, short_cost


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--avg-hold-candles", type=float, default=13.0,
                   help="average hold from the deployment candidate (p2_8/p2_9 ≈ 13)")
    p.add_argument("--candle-hours", type=float, default=1.0)
    args = p.parse_args()

    results = []
    for name, path in [("VAL", args.val), ("TEST", args.test)]:
        df = pd.read_parquet(path)
        idx = df.index if hasattr(df.index, "tz") else pd.to_datetime(df.index, utc=True)
        start = int(idx.min().timestamp() * 1000)
        end = int(idx.max().timestamp() * 1000)
        fund = fetch_funding(start, end)
        results.append(analyze(name, fund, args.avg_hold_candles, args.candle_hours))

    print("\n== G3 VERDICT ==")
    print("  Compare against the deployment candidates' net edge per trade at 0.04% RT:")
    print("  p2_8 besttrain net expectancy ≈ +0.18%/trade val, p2_9 ≈ +0.15%/trade.")
    worst = max(abs(c) for pair in results for c in pair)
    print(f"  Worst expected per-trade funding drag measured: {worst:.4f}%")
    if worst < 0.05:
        print("  → funding is a minor haircut at our hold times, NOT an edge-killer.")
    else:
        print("  → funding is MATERIAL — model it into fee_sensitivity before deploying.")


if __name__ == "__main__":
    main()
