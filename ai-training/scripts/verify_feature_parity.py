"""
verify_feature_parity.py
────────────────────────
FIX #3: Validates that the TypeScript indicators.ts produces
IDENTICAL feature vectors to the Python feature_engineering.py.

A discrepancy > 0.001 in ANY feature means the live bot is feeding
the ONNX model different data than it was trained on — "concept drift"
that causes erratic, unpredictable trading decisions.

How it works:
  1. Loads a small window of real historical candles
  2. Computes the feature vector with Python (the training source of truth)
  3. Serialises both the candles and the expected features to JSON
  4. A companion Node.js script reads the same candles, computes features
     using indicators.ts, and compares against the Python output
  5. Any delta > tolerance is flagged as a PARITY FAILURE

Usage:
  # Step 1: generate the reference data
  python scripts/verify_feature_parity.py --input data/BTC_USDT_1h.parquet

  # Step 2: run the Node.js checker (from execution-engine/)
  npx ts-node src/strategist/verifyParity.ts
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

WINDOW   = 500   # P2-3 fix: was 250 — EMA-200 needs 200+ warmup, leaving only 50 converged
                 # candles for comparison. At 500 the test candle has 300 fully-converged
                 # periods, making false PASS from unconverged EMA values impossible.
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")

# Must exactly match the TypeScript constants in indicators.ts
RSI_PERIOD     = 14
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
BB_PERIOD      = 20
BB_STD         = 2.0
ATR_PERIOD     = 14
EMA_SHORT_A    = 9
EMA_SHORT_B    = 21
EMA_LONG_A     = 50
EMA_LONG_B     = 200
VOL_MA_PERIOD  = 20


def compute_features(df: pd.DataFrame) -> dict:
    """Mirrors feature_engineering.py for a single window."""
    closes  = df["close"]
    highs   = df["high"]
    lows    = df["low"]
    volumes = df["volume"]

    # RSI → [-1, 1]
    rsi = (RSIIndicator(close=closes, window=RSI_PERIOD).rsi() - 50) / 50

    # MACD (normalised by rolling mean price)
    macd_obj  = MACD(close=closes, window_fast=MACD_FAST, window_slow=MACD_SLOW, window_sign=MACD_SIGNAL)
    price_ma  = closes.rolling(MACD_SLOW).mean()
    macd      = macd_obj.macd()       / price_ma
    macd_sig  = macd_obj.macd_signal()/ price_ma
    macd_hist = macd_obj.macd_diff()  / price_ma

    # Bollinger position
    bb        = BollingerBands(close=closes, window=BB_PERIOD, window_dev=BB_STD)
    bb_mid    = bb.bollinger_mavg()
    bb_upper  = bb.bollinger_hband()
    bb_lower  = bb.bollinger_lband()
    bb_band   = (bb_upper - bb_lower) / 2
    bb_pos    = (closes - bb_mid) / bb_band.replace(0, np.nan)
    bb_width  = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    # ATR ratio
    atr       = AverageTrueRange(high=highs, low=lows, close=closes, window=ATR_PERIOD).average_true_range()
    atr_ratio = atr / closes.replace(0, np.nan)

    # EMA crossovers
    ema9   = EMAIndicator(close=closes, window=EMA_SHORT_A).ema_indicator()
    ema21  = EMAIndicator(close=closes, window=EMA_SHORT_B).ema_indicator()
    ema50  = EMAIndicator(close=closes, window=EMA_LONG_A).ema_indicator()
    ema200 = EMAIndicator(close=closes, window=EMA_LONG_B).ema_indicator()
    ema_cross_short = (ema9 - ema21)   / closes.replace(0, np.nan)
    ema_cross_long  = (ema50 - ema200) / closes.replace(0, np.nan)

    # Volume ratio
    vol_ma    = volumes.rolling(VOL_MA_PERIOD).mean()
    vol_ratio = volumes / vol_ma.replace(0, np.nan)

    # Log returns
    log_r   = np.log(closes  / closes.shift(1))
    log_r_h = np.log(highs   / highs.shift(1))
    log_r_l = np.log(lows    / lows.shift(1))
    log_r_v = np.log(volumes.clip(lower=1e-8) / volumes.shift(1).clip(lower=1e-8))

    # Candle structure
    c_range  = (highs - lows).replace(0, np.nan)
    body     = (closes - df["open"]).abs() / c_range
    up_wick  = (highs - df[["open","close"]].max(axis=1)) / c_range
    lo_wick  = (df[["open","close"]].min(axis=1) - lows)  / c_range
    c_dir    = np.sign(closes - df["open"])

    # Return last row as a dict of named feature values (the "ground truth")
    last = df.index[-1]
    return {
        "log_return":       float(log_r.loc[last]),
        "log_return_h":     float(log_r_h.loc[last]),
        "log_return_l":     float(log_r_l.loc[last]),
        "log_return_v":     float(log_r_v.loc[last]),
        "body_ratio":       float(body.loc[last]),
        "upper_wick_ratio": float(up_wick.loc[last]),
        "lower_wick_ratio": float(lo_wick.loc[last]),
        "candle_direction": float(c_dir.loc[last]),
        "rsi":              float(rsi.loc[last]),
        "macd":             float(macd.loc[last]),
        "macd_signal":      float(macd_sig.loc[last]),
        "macd_hist":        float(macd_hist.loc[last]),
        "bb_position":      float(bb_pos.loc[last]),
        "bb_width":         float(bb_width.loc[last]),
        "atr_ratio":        float(atr_ratio.loc[last]),
        "ema_cross_short":  float(ema_cross_short.loc[last]),
        "ema_cross_long":   float(ema_cross_long.loc[last]),
        "volume_ratio":     float(vol_ratio.loc[last]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    # Take a representative window (most recent WINDOW candles)
    window_df = df.tail(WINDOW).copy()

    candles = [
        {
            "timestamp": int(ts.timestamp() * 1000),
            "open":   float(r["open"]),
            "high":   float(r["high"]),
            "low":    float(r["low"]),
            "close":  float(r["close"]),
            "volume": float(r["volume"]),
        }
        for ts, r in window_df.iterrows()
    ]

    expected = compute_features(window_df)

    # Sanitize NaNs for JSON serialisation
    for k, v in expected.items():
        if np.isnan(v) or np.isinf(v):
            expected[k] = 0.0

    output = {
        "description": "Parity test: Python ground truth for indicators.ts verification",
        "candle_count": len(candles),
        "last_candle_ts": candles[-1]["timestamp"],
        "candles": candles,
        "expected_features": expected,
        "tolerance": 0.001,
    }

    out_path = os.path.join(OUT_DIR, "parity_test.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[OK] Parity test data written to: {out_path}")
    print(f"     Candles exported : {len(candles)}")
    print(f"     Expected features:")
    for k, v in expected.items():
        print(f"       {k:<22}: {v:.8f}")
    print(f"\n  → Now run from execution-engine/:")
    print(f"       npx ts-node src/strategist/verifyParity.ts")


if __name__ == "__main__":
    main()
