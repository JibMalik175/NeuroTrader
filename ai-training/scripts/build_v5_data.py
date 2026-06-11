"""
build_v5_data.py — H3: add perpetual funding-rate features (V5) to the splits
──────────────────────────────────────────────────────────────────────────────
Funding is the market's positioning thermometer: persistently positive =
crowded longs (bull euphoria), negative = crowded shorts (capitulation).
OctoBot reads sentiment from Reddit/Trends (not backfillable); funding is the
rigorous equivalent — Binance serves the full history since 2019.

Features (all causal):
  funding_rate — the LAST PAID rate, ffilled until the next funding event.
                 A rate is used only AFTER its fundingTime (no lookahead).
                 Normalized: rate / 0.001 clipped to [-1, 1] (0.1% = extreme).
  funding_ma   — 30-event (~10 day) rolling mean of the paid rate, same scale.
  funding_z    — rate's z-score vs its 90-event (~30 day) window, /3, clipped.

Open interest is EXCLUDED: Binance only serves ~30 days of OI history, which
cannot honestly backfill a 4-year training set.

Usage:
  python scripts/build_v5_data.py \
      --splits data/BTC_USDT_1h_train.parquet data/BTC_USDT_1h_val.parquet data/BTC_USDT_1h_test.parquet
Writes <name>_v5.parquet next to each input.
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"
SYMBOL = "BTCUSDT"
NORM = 0.001          # 0.1%/8h = extreme funding → ±1.0
MA_EVENTS = 30        # ~10 days of 8h events
Z_EVENTS = 90         # ~30 days


def fetch_funding(start_ms: int, end_ms: int) -> pd.DataFrame:
    rows, cursor = [], start_ms
    while cursor < end_ms:
        r = requests.get(FAPI, params={"symbol": SYMBOL, "startTime": cursor,
                                       "endTime": end_ms, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df["rate"] = df["fundingRate"].astype(float)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    return df.set_index("time")[["rate"]].sort_index()


def funding_features(fund: pd.DataFrame) -> pd.DataFrame:
    f = fund.copy()
    f["funding_rate"] = np.clip(f["rate"] / NORM, -1.0, 1.0)
    f["funding_ma"] = np.clip(
        f["rate"].rolling(MA_EVENTS, min_periods=MA_EVENTS // 3).mean() / NORM, -1.0, 1.0)
    mu = f["rate"].rolling(Z_EVENTS, min_periods=Z_EVENTS // 3).mean()
    sd = f["rate"].rolling(Z_EVENTS, min_periods=Z_EVENTS // 3).std() + 1e-9
    f["funding_z"] = np.clip((f["rate"] - mu) / sd / 3.0, -1.0, 1.0)
    return f[["funding_rate", "funding_ma", "funding_z"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", required=True)
    args = p.parse_args()

    dfs = {path: pd.read_parquet(path) for path in args.splits}
    for path, df in dfs.items():
        if not isinstance(df.index, pd.DatetimeIndex):
            sys.exit(f"{path}: needs a DatetimeIndex (got {type(df.index).__name__})")

    lo = min(df.index.min() for df in dfs.values())
    hi = max(df.index.max() for df in dfs.values())
    # fetch one extra Z-window of history so the first rows have warm features
    pad_ms = Z_EVENTS * 8 * 3600 * 1000
    print(f"[FETCH] funding {lo} -> {hi} (+{Z_EVENTS} events warmup)")
    fund = fetch_funding(int(lo.timestamp() * 1000) - pad_ms,
                         int(hi.timestamp() * 1000))
    print(f"[FETCH] {len(fund)} funding events")

    feats = funding_features(fund)

    for path, df in dfs.items():
        # asof-merge: each candle gets the latest funding event AT OR BEFORE its
        # own timestamp — strictly causal by construction.
        merged = pd.merge_asof(
            df.sort_index(), feats.sort_index(),
            left_index=True, right_index=True, direction="backward")
        n_nan = int(merged[["funding_rate", "funding_ma", "funding_z"]].isna().sum().sum())
        if n_nan:
            print(f"[WARN] {path}: {n_nan} NaN feature cells (pre-history) — dropping rows")
            merged = merged.dropna(subset=["funding_rate", "funding_ma", "funding_z"])
        out = path.replace(".parquet", "_v5.parquet")
        merged.to_parquet(out)
        print(f"[SAVED] {out}  rows={len(merged)}  "
              f"funding_rate range [{merged['funding_rate'].min():.3f}, "
              f"{merged['funding_rate'].max():.3f}]")


if __name__ == "__main__":
    main()
