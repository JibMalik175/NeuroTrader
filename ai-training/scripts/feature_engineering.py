"""
feature_engineering.py
───────────────────────
Transforms raw OHLCV parquet files into normalized feature matrices
ready for feeding into the DRL training environment.

Feature set (v2 — 24 features):
  - Normalized OHLCV (log-returns + min-max)
  - RSI (14)
  - MACD + Signal + Histogram
  - Bollinger Bands (20, 2σ) — position within bands
  - ATR (14) — normalized volatility
  - EMA crossovers (9/21, 50/200)
  - Volume ratio (current vs rolling avg)
  - Candle body/wick ratios (structural cues)
  - [NEW] Time-of-day encoding (hour_sin, hour_cos)
  - [NEW] Day-of-week encoding (day_sin, day_cos)
  - [NEW] ADX trend strength (normalized)
  - [NEW] OBV ratio (On-Balance Volume vs moving average)

v2 changes:
  - Added 6 new features (24 total, up from 18)
  - Time encoding captures intraday/weekly crypto patterns
  - ADX measures trend strength (high ADX = strong trend)
  - OBV ratio detects volume-confirmed moves

Usage:
    python feature_engineering.py --input ../data/BTC_USDT_1h.parquet
    python feature_engineering.py --input ../data/BTC_USDT_1h.parquet --output ../data/BTC_USDT_1h_features.parquet
"""

import pandas as pd
import numpy as np
import argparse
import os
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
# RobustScaler removed — scaling is handled by VecNormalize in train_agent.py


# ── Core Feature Builders ─────────────────────────────────────────────────────

def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Log returns are stationary and bounded — far better than raw prices
    for neural networks. Each value represents: ln(close_t / close_t-1)
    """
    df["log_return"]    = np.log(df["close"] / df["close"].shift(1))
    df["log_return_h"]  = np.log(df["high"]  / df["high"].shift(1))
    df["log_return_l"]  = np.log(df["low"]   / df["low"].shift(1))
    df["log_return_v"]  = np.log(df["volume"].clip(lower=1) / df["volume"].shift(1).clip(lower=1))
    return df


def add_candle_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes the geometric structure of each candle — the AI learns
    to recognize hammers, doji, engulfing, etc. from these ratios.
    """
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)

    # Body as fraction of total range (0 = doji, 1 = no wicks)
    df["body_ratio"]       = (df["close"] - df["open"]).abs() / candle_range

    # Upper wick / total range
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range

    # Lower wick / total range
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range

    # Direction: +1 bullish, -1 bearish
    df["candle_direction"]  = np.sign(df["close"] - df["open"])

    return df.fillna(0)


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """RSI scaled to [-1, 1]: 0.0 = neutral (50), +1.0 = overbought, -1.0 = oversold."""
    rsi = RSIIndicator(close=df["close"], window=window).rsi()
    df["rsi"] = (rsi - 50) / 50   # maps [0,100] → [-1, 1]
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD components normalized by the asset's recent price to make
    the values comparable across different price ranges.
    """
    macd_obj     = MACD(close=df["close"])
    price        = df["close"].rolling(26).mean().replace(0, np.nan)

    df["macd"]        = macd_obj.macd()        / price
    df["macd_signal"] = macd_obj.macd_signal() / price
    df["macd_hist"]   = macd_obj.macd_diff()   / price
    return df


def add_bollinger(df: pd.DataFrame, window: int = 20, std: float = 2.0) -> pd.DataFrame:
    """
    Instead of raw band values, we compute the position of price
    within the bands: -1 = at lower band, 0 = at midline, +1 = at upper band.
    Values outside [-1, 1] signal a breakout.
    """
    bb = BollingerBands(close=df["close"], window=window, window_dev=std)
    mid   = bb.bollinger_mavg()
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    band_width = (upper - lower).replace(0, np.nan)

    df["bb_position"] = (df["close"] - mid) / (band_width / 2)
    df["bb_width"]    = band_width / mid       # normalized bandwidth
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    ATR normalized by close price gives a volatility ratio.
    High values = volatile market, low = quiet/ranging.
    """
    atr = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=window
    ).average_true_range()
    df["atr_ratio"] = atr / df["close"]
    return df


def add_ema_crossovers(df: pd.DataFrame) -> pd.DataFrame:
    """
    EMA crossover signals encoded as normalized distances.
    Positive = fast EMA above slow EMA (bullish), negative = bearish.
    """
    ema9   = EMAIndicator(close=df["close"], window=9).ema_indicator()
    ema21  = EMAIndicator(close=df["close"], window=21).ema_indicator()
    ema50  = EMAIndicator(close=df["close"], window=50).ema_indicator()
    ema200 = EMAIndicator(close=df["close"], window=200).ema_indicator()

    price = df["close"].replace(0, np.nan)

    # Short-term cross: (EMA9 - EMA21) / price
    df["ema_cross_short"] = (ema9 - ema21) / price

    # Long-term cross: (EMA50 - EMA200) / price — "golden/death cross"
    df["ema_cross_long"]  = (ema50 - ema200) / price

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume ratio vs 20-period moving average. >1 = volume spike."""
    vol_ma = df["volume"].rolling(20).mean().replace(0, np.nan)
    df["volume_ratio"] = df["volume"] / vol_ma
    return df


# ── NEW Phase 2 Feature Builders ─────────────────────────────────────────────

def add_time_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cyclical time encoding using sine/cosine transforms.

    Why? Crypto markets have strong intraday and weekly patterns:
      - Volume spikes during US/EU market overlap (14:00–18:00 UTC)
      - Weekends are typically lower volatility
      - Sunday evening (UTC) often sees momentum shifts

    Sine/cosine encoding preserves the cyclical nature: hour 23 is
    close to hour 0, Saturday is close to Monday.
    """
    # Try to extract datetime index
    if hasattr(df.index, 'hour'):
        hour = df.index.hour
        dow  = df.index.dayofweek
    elif 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        hour = ts.dt.hour
        dow  = ts.dt.dayofweek
    elif 'open_time' in df.columns:
        ts = pd.to_datetime(df['open_time'], unit='ms')
        hour = ts.dt.hour
        dow  = ts.dt.dayofweek
    else:
        # G2 audit hardening: the old fallback fabricated hour/day from ROW
        # POSITION (arange % 24) — silently wrong for any frame that doesn't
        # start at hour 0, and train/serve-skewed for partial buffers. Fail
        # loud instead: time features need real timestamps.
        raise ValueError(
            "add_time_encoding: no datetime index, 'timestamp', or 'open_time' "
            "column — cannot compute real hour/day-of-week features. "
            "(Refusing to infer time from row position.)"
        )

    # Cyclical encoding — maps to [-1, 1] range naturally
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"]  = np.sin(2 * np.pi * dow / 7)
    df["day_cos"]  = np.cos(2 * np.pi * dow / 7)

    return df


def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average Directional Index — measures TREND STRENGTH (not direction).

    ADX > 25 = trending market (the agent should follow the trend)
    ADX < 20 = ranging market (the agent should be cautious)

    Normalized to [-1, 1] range: 0.0 = ADX at 25 (neutral).
    """
    adx_indicator = ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=window
    )
    adx_raw = adx_indicator.adx()

    # Normalize: ADX 0-100 → [-1, 1], centered at 25
    df["adx"] = (adx_raw - 25) / 25
    df["adx"] = df["adx"].clip(-1, 1)

    return df


def add_obv_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    On-Balance Volume ratio — detects volume-confirmed price moves.

    OBV accumulates volume on up-days and subtracts on down-days.
    The ratio of OBV to its moving average signals divergences:
      - OBV ratio > 1: volume is confirming the move (bullish signal)
      - OBV ratio < 1: volume is diverging from price (potential reversal)

    Normalized to roughly [-1, 1] range.
    """
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
    obv_values = obv.on_balance_volume()

    # OBV is cumulative, so we look at the ratio of OBV to its MA
    obv_ma = obv_values.rolling(window).mean()
    obv_std = obv_values.rolling(window).std().replace(0, np.nan)

    # Z-score of OBV: how many std deviations from its rolling mean
    df["obv_ratio"] = ((obv_values - obv_ma) / obv_std).clip(-3, 3) / 3  # normalize to [-1, 1]

    return df


def add_mtf_features(df: pd.DataFrame, candles_per_day: int = 96) -> pd.DataFrame:
    """
    Adds Multi-Timeframe (MTF) features (4h and 1d RSI/MACD) using causal rolling proxies.
    This bypasses pandas .resample() completely, which ensures:
      1. Zero look-ahead bias (only uses past data)
      2. 100% mathematical parity with the TypeScript real-time execution engine
         which streams tick-by-tick and uses these exact rolling window multipliers.
    """
    # Multipliers for 15m base timeframe (96 candles/day)
    # 4 hours = 16 candles. 1 day = 96 candles.
    multiplier_4h = int(4 * (candles_per_day / 24))
    multiplier_1d = candles_per_day

    # --- 4H Proxies ---
    rsi_4h = RSIIndicator(close=df["close"], window=14 * multiplier_4h).rsi()
    df["4h_rsi"] = (rsi_4h - 50) / 50

    macd_4h_obj = MACD(close=df["close"], window_slow=26 * multiplier_4h, window_fast=12 * multiplier_4h, window_sign=9 * multiplier_4h)
    price_4h = df["close"].rolling(26 * multiplier_4h).mean().replace(0, np.nan)
    df["4h_macd"] = macd_4h_obj.macd() / price_4h

    # --- 1D Proxies ---
    rsi_1d = RSIIndicator(close=df["close"], window=14 * multiplier_1d).rsi()
    df["1d_rsi"] = (rsi_1d - 50) / 50

    macd_1d_obj = MACD(close=df["close"], window_slow=26 * multiplier_1d, window_fast=12 * multiplier_1d, window_sign=9 * multiplier_1d)
    price_1d = df["close"].rolling(26 * multiplier_1d).mean().replace(0, np.nan)
    df["1d_macd"] = macd_1d_obj.macd() / price_1d

    # Fill NaNs with 0
    for col in ['4h_rsi', '4h_macd', '1d_rsi', '1d_macd']:
        df[col] = df[col].fillna(0)

    return df


def add_macro_features(df: pd.DataFrame, candles_per_day: int = 96) -> pd.DataFrame:
    """
    Long-term macro trend features using rolling windows.

    Parameters
    ----------
    candles_per_day : int
        Number of candles per calendar day in the input dataset.
        This MUST match your actual data timeframe or window sizes will be wrong.

        Common values:
          96  = 15-minute candles  (default)
          24  = 1-hour candles
          288 = 5-minute candles
          1   = daily candles

        Fix 3: Previously hardcoded to 96 (15m), which silently computed
        120-day windows when applied to 1h data. Now explicit and validated.

    Features added:
      dist_from_high     — distance below 30-day rolling high (always ≤ 0)
      macro_trend_sma    — % distance from 30-day SMA
      macro_volatility   — annualized 30-day return std dev
      macro_obv_ratio    — OBV z-score vs 30-day rolling mean
    """
    window_30d = 30 * candles_per_day

    # 1. Distance from 30-day High
    rolling_high = df["high"].rolling(window_30d, min_periods=window_30d // 2).max()
    df["dist_from_high"] = (df["close"] - rolling_high) / rolling_high

    # 2. Macro Moving Average (30-day SMA distance)
    sma_30d = df["close"].rolling(window_30d, min_periods=window_30d // 2).mean()
    df["macro_trend_sma"] = (df["close"] - sma_30d) / sma_30d

    # 3. Macro Volatility — annualized 30-day return std dev
    ret_30d_std = df["close"].pct_change().rolling(window_30d, min_periods=window_30d // 2).std()
    df["macro_volatility"] = ret_30d_std * np.sqrt(candles_per_day * 365)

    # 4. Long-term OBV z-score
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
    obv_ma  = obv.rolling(window_30d, min_periods=window_30d // 2).mean()
    obv_std = obv.rolling(window_30d, min_periods=window_30d // 2).std().replace(0, np.nan)
    df["macro_obv_ratio"] = ((obv - obv_ma) / obv_std).clip(-3, 3) / 3

    return df


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────

# v1 features (18)
FEATURE_COLS_V1 = [
    "log_return", "log_return_h", "log_return_l", "log_return_v",
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "candle_direction",
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_position", "bb_width",
    "atr_ratio",
    "ema_cross_short", "ema_cross_long",
    "volume_ratio",
]

# v2 features (24) — all v1 features + 6 new ones
FEATURE_COLS_V2 = FEATURE_COLS_V1 + [
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "adx",
    "obv_ratio",
]

# v3 features (28) — all v2 features + 4 MTF features
FEATURE_COLS_V3 = FEATURE_COLS_V2 + [
    "4h_rsi", "4h_macd",
    "1d_rsi", "1d_macd"
]

# v4 features (32) — all v3 features + 4 Macro features
FEATURE_COLS_V4 = FEATURE_COLS_V3 + [
    "dist_from_high", "macro_trend_sma", "macro_volatility", "macro_obv_ratio"
]

# Default to v4
FEATURE_COLS = FEATURE_COLS_V4


def build_features(df: pd.DataFrame, version: str = "v2", candles_per_day: int = 96,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Runs the full feature engineering pipeline on a raw OHLCV DataFrame.
    Returns a clean feature DataFrame with no NaN rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV DataFrame with columns: open, high, low, close, volume
    version : str
        Feature version: "v1" (18), "v2" (24), "v3" (28 MTF), "v4" (32 Macro)
    candles_per_day : int
        Candles per calendar day in the input data.
        96 = 15m (default), 24 = 1h, 288 = 5m.
        Used by add_macro_features for correct window sizing.
        Fix 3: this was hardcoded to 96 inside add_macro_features — now explicit.
    verbose : bool
        Print the summary block. Disable for programmatic callers (audits, tests).
    """
    df = df.copy()

    # v1 features (always computed)
    df = add_log_returns(df)
    df = add_candle_structure(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger(df)
    df = add_atr(df)
    df = add_ema_crossovers(df)
    df = add_volume_features(df)

    # v2 features
    if version in ["v2", "v3", "v4"]:
        df = add_time_encoding(df)
        df = add_adx(df)
        df = add_obv_ratio(df)

    # v3 features (MTF — using causal rolling proxies)
    if version in ["v3", "v4"]:
        df = add_mtf_features(df, candles_per_day=candles_per_day)

    # v4 features (Macro — now with parameterized window)
    if version == "v4":
        df = add_macro_features(df, candles_per_day=candles_per_day)

    if version == "v4":
        cols = FEATURE_COLS_V4
    elif version == "v3":
        cols = FEATURE_COLS_V3
    elif version == "v2":
        cols = FEATURE_COLS_V2
    else:
        cols = FEATURE_COLS_V1

    # Keep raw OHLCV alongside features (environment needs them for PnL calc)
    out = df[["open", "high", "low", "close", "volume"] + cols].copy()

    # P1-4 fix: track warmup rows explicitly before dropping
    rows_before = len(out)
    out.dropna(inplace=True)
    warmup_rows = rows_before - len(out)

    # G2 fix: don't crash on an empty result — happens when the input has fewer
    # rows than the feature warmup (e.g. a live buffer shorter than the longest
    # rolling window). Callers must check for emptiness.
    if verbose:
        print(f"[INFO] Feature matrix shape: {out.shape}")
        print(f"[INFO] Feature columns ({len(cols)}): {cols}")
        if out.empty:
            print(f"[WARN] ALL {rows_before} rows consumed by feature warmup — "
                  f"input is shorter than the longest rolling window")
        else:
            print(f"[INFO] Date range: {out.index[0]} -> {out.index[-1]}")
            print(f"[INFO] NaN count: {out.isnull().sum().sum()}")
            print(f"[INFO] Warmup rows dropped: {warmup_rows} "
                  f"({warmup_rows / rows_before * 100:.1f}% of raw data) "
                  f"— driven by longest rolling window (EMA-200 or macro features)")

    return out


# ── Train / Validation Split ──────────────────────────────────────────────────

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.75,
    val_ratio:   float = 0.15,
    # remainder is test (0.10)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological split — NEVER shuffle time-series data.
    Shuffling causes data leakage (future data in training set).

    Returns: (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val   = df.iloc[train_end:val_end]
    test  = df.iloc[val_end:]

    print(f"\n[SPLIT] Train : {len(train):>6,} rows  ({train.index[0]} -> {train.index[-1]})")
    print(f"[SPLIT] Val   : {len(val):>6,} rows  ({val.index[0]} -> {val.index[-1]})")
    print(f"[SPLIT] Test  : {len(test):>6,} rows  ({test.index[0]} -> {test.index[-1]})")

    return train, val, test


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build feature matrix from OHLCV parquet")
    parser.add_argument("--input",  type=str, required=True,
                        help="Path to raw OHLCV .parquet file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: replaces .parquet with _features.parquet)")
    parser.add_argument("--version", type=str, default="v4", choices=["v1", "v2", "v3", "v4"],
                        help="Feature version: v1 (18), v2 (24), v3 (28 MTF), v4 (32 Macro)")
    parser.add_argument("--candles-per-day", type=int, default=96,
                        help="Candles per calendar day: 96=15m (default), 24=1h, 288=5m. "
                             "Used for correct macro window sizing and Sharpe annualization.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"\n[LOADING] {args.input}")
    raw_df = pd.read_parquet(args.input)
    print(f"[RAW] Shape: {raw_df.shape}")
    print(f"[FEATURES] Using version: {args.version}, candles_per_day: {args.candles_per_day}")

    features_df = build_features(raw_df, version=args.version, candles_per_day=args.candles_per_day)

    # Save the full feature df
    output_path = args.output or args.input.replace(".parquet", "_features.parquet")
    features_df.to_parquet(output_path, compression="snappy")
    print(f"\n[SAVED] {output_path}")

    # Save the splits
    train, val, test = chronological_split(features_df)
    base = output_path.replace("_features.parquet", "")
    train.to_parquet(f"{base}_train.parquet", compression="snappy")
    val.to_parquet(f"{base}_val.parquet",     compression="snappy")
    test.to_parquet(f"{base}_test.parquet",   compression="snappy")
    print(f"\n[SAVED] Train/Val/Test splits written.")


if __name__ == "__main__":
    main()
