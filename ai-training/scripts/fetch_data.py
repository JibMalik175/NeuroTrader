"""
fetch_data.py
─────────────
Downloads historical OHLCV candlestick data from Binance's public API
using CCXT. No API keys required — this endpoint is public.

Usage:
    python fetch_data.py --symbol BTC/USDT --timeframe 1h --days 730
    python fetch_data.py --symbol ETH/USDT --timeframe 15m --days 365
"""

import ccxt
import pandas as pd
import numpy as np
import argparse
import time
import os
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TIMEFRAME_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

# Binance returns max 1000 candles per request
CANDLES_PER_REQUEST = 1000


# ── Exchange Setup ─────────────────────────────────────────────────────────────

def get_exchange() -> ccxt.binance:
    """
    Returns an authenticated-free Binance exchange instance.
    Rate limits are respected automatically via CCXT's built-in throttler.
    """
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
        },
    })
    return exchange


# ── Core Fetch Logic ──────────────────────────────────────────────────────────

def fetch_ohlcv_range(
    exchange: ccxt.binance,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """
    Fetches all OHLCV candles between start_ms and end_ms by paginating
    through Binance's 1000-candle limit per request.

    Returns a DataFrame with columns:
        timestamp, open, high, low, close, volume
    """
    tf_ms = TIMEFRAME_MS[timeframe]
    all_candles = []

    current_ms = start_ms
    total_candles = (end_ms - start_ms) // tf_ms
    pbar = tqdm(total=total_candles, desc=f"Fetching {symbol} [{timeframe}]", unit="candles")

    while current_ms < end_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_ms,
                limit=CANDLES_PER_REQUEST,
            )
        except ccxt.NetworkError as e:
            print(f"\n[WARN] Network error: {e}. Retrying in 5s...")
            time.sleep(5)
            continue
        except ccxt.ExchangeError as e:
            print(f"\n[ERROR] Exchange error: {e}. Aborting.")
            break

        if not candles:
            break

        all_candles.extend(candles)
        pbar.update(len(candles))

        # Advance cursor past the last returned candle
        last_ts = candles[-1][0]
        current_ms = last_ts + tf_ms

        # Avoid hammering the API beyond CCXT's built-in rate limiter
        time.sleep(exchange.rateLimit / 1000)

    pbar.close()

    if not all_candles:
        raise ValueError(f"No candles returned for {symbol} on {timeframe}")

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    # Convert timestamp from milliseconds to UTC datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)

    # Drop duplicates that can occur at pagination boundaries
    df = df[~df.index.duplicated(keep="first")]

    # Clip to exact requested range
    start_dt = pd.Timestamp(start_ms, unit="ms", tz="UTC")
    end_dt   = pd.Timestamp(end_ms,   unit="ms", tz="UTC")
    df = df.loc[start_dt:end_dt]

    return df.sort_index()


# ── Validation ────────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    """Sanity checks on the downloaded data before saving."""
    assert len(df) > 0, "DataFrame is empty"
    assert df.isnull().sum().sum() == 0, "DataFrame contains NaN values"
    assert (df["high"] >= df["low"]).all(), "Found candles where high < low"
    assert (df["volume"] >= 0).all(), "Found negative volume"

    # P2-4 fix: assert timestamps are strictly increasing.
    # Binance occasionally returns out-of-order candles at pagination boundaries.
    # If any slip through the drop_duplicates step, rolling windows in feature
    # engineering will silently produce incorrect values.
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            f"Timestamps are not strictly increasing for {symbol} [{timeframe}]. "
            "Data may be corrupted at a pagination boundary. Re-fetch."
        )

    # Check for large gaps (missing candles)
    tf_ms   = TIMEFRAME_MS[timeframe]
    diffs   = df.index.to_series().diff().dropna()
    max_gap = diffs.max()
    expected_gap = pd.Timedelta(milliseconds=tf_ms)

    if max_gap > expected_gap * 3:
        print(f"[WARN] Largest gap in data: {max_gap} (expected {expected_gap}). "
              "This may indicate exchange downtime or delisted pair.")

    print(f"[OK] {symbol} | {timeframe} | {len(df):,} candles | "
          f"{df.index[0]} -> {df.index[-1]}")


# ── Save ──────────────────────────────────────────────────────────────────────

def save_data(df: pd.DataFrame, symbol: str, timeframe: str) -> str:
    """Saves the DataFrame as a compressed Parquet file for efficient I/O."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Sanitize symbol for filesystem: BTC/USDT → BTC_USDT
    safe_symbol = symbol.replace("/", "_")
    filename = f"{safe_symbol}_{timeframe}.parquet"
    filepath = os.path.join(DATA_DIR, filename)

    df.to_parquet(filepath, compression="snappy")
    size_mb = os.path.getsize(filepath) / 1_048_576
    print(f"[SAVED] {filepath} ({size_mb:.2f} MB)")
    return filepath


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch Binance historical OHLCV data")
    parser.add_argument("--symbol",    type=str, default="BTC/USDT",
                        help="Trading pair, e.g. BTC/USDT, ETH/USDT")
    parser.add_argument("--timeframe", type=str, default="1h",
                        choices=list(TIMEFRAME_MS.keys()),
                        help="Candle timeframe")
    parser.add_argument("--days",      type=int, default=730,
                        help="Number of calendar days of history to download")
    args = parser.parse_args()

    if args.timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe: {args.timeframe}")

    now_ms    = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms  = now_ms - (args.days * 86_400_000)

    print(f"\n{'='*60}")
    print(f"  Symbol    : {args.symbol}")
    print(f"  Timeframe : {args.timeframe}")
    print(f"  Days      : {args.days}")
    print(f"  From      : {datetime.fromtimestamp(start_ms/1000, timezone.utc)}")
    print(f"  To        : {datetime.fromtimestamp(now_ms/1000, timezone.utc)}")
    print(f"{'='*60}\n")

    exchange = get_exchange()

    # Verify market exists on Binance before fetching
    exchange.load_markets()
    if args.symbol not in exchange.markets:
        raise ValueError(f"Symbol '{args.symbol}' not found on Binance. "
                         f"Check the pair name (e.g. BTC/USDT, not BTCUSDT).")

    df = fetch_ohlcv_range(
        exchange=exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_ms=start_ms,
        end_ms=now_ms,
    )

    validate_dataframe(df, args.symbol, args.timeframe)
    save_data(df, args.symbol, args.timeframe)


if __name__ == "__main__":
    main()
