"""
audit_features.py â€” feature-pipeline bias audit (ported from Freqtrade's
optimize/analysis/{lookahead,recursive}.py ideas, G1/G2).

Two failure modes can silently invalidate every backtest result:

1. LOOKAHEAD BIAS â€” a feature at time T uses data from after T (centered
   windows, global stats, full-series normalization). Detection: compute
   features on the full series, then on the series truncated at T; the row at
   T must be identical. Any difference = the feature saw the future.

2. WARMUP (RECURSIVE) BIAS â€” recursive indicators (EMA, Wilder RSI/ADX) and
   long rolling windows give different values at time T depending on how much
   history preceded it. Training computes features over 4 years; the live
   watcher buffer holds only a few hundred candles. If the same candle gets
   different feature values live vs training, the model sees a distribution
   it was never trained on (train/serve skew). Detection: recompute features
   using only the last N candles for several N and compare the final row
   against the full-history values.

Usage:
  python scripts/audit_features.py --input data/BTC_USDT_1h.parquet --version v4 --candles-per-day 24
Exit code 1 if lookahead bias is detected (warmup bias is reported, not fatal â€”
it is fixed by raising the live warmup, not by changing features).
"""

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.feature_engineering import build_features  # noqa: E402

TOL = 1e-9          # lookahead: identical up to float noise
WARMUP_TOL = 0.01   # warmup: report features off by >1% of their full-history std


def feature_cols(df: pd.DataFrame, raw_cols) -> list:
    return [c for c in df.columns if c not in raw_cols]


def audit_lookahead(raw: pd.DataFrame, version: str, cpd: int, n_cuts: int = 8) -> bool:
    """Features at row T must not change when the future is removed."""
    full = build_features(raw.copy(), version=version, candles_per_day=cpd, verbose=False)
    cols = feature_cols(full, raw.columns)
    # row identity can shift because build_features dropna's the warmup; align on timestamp
    key = "timestamp" if "timestamp" in full.columns else None

    cuts = np.linspace(len(raw) // 2, len(raw) - 2, n_cuts, dtype=int)
    biased: dict[str, float] = {}
    for cut in cuts:
        trunc = build_features(raw.iloc[:cut].copy(), version=version, candles_per_day=cpd, verbose=False)
        if trunc.empty:
            continue
        last = trunc.iloc[-1]
        if key:
            match = full[full[key] == last[key]]
        else:
            match = full[full.index == trunc.index[-1]]
        if match.empty:
            continue
        ref = match.iloc[0]
        for c in cols:
            diff = abs(float(last[c]) - float(ref[c]))
            if diff > TOL:
                biased[c] = max(biased.get(c, 0.0), diff)

    print("\n== G1 LOOKAHEAD AUDIT ==")
    if biased:
        print(f"  BIAS DETECTED in {len(biased)} feature(s):")
        for c, d in sorted(biased.items(), key=lambda kv: -kv[1]):
            print(f"    {c:<22} max diff {d:.3e}")
        return False
    print(f"  clean â€” {len(cols)} features identical with the future removed "
          f"({n_cuts} cut points)")
    return True


def audit_warmup(raw: pd.DataFrame, version: str, cpd: int,
                 warmups=(200, 300, 500, 1000, 2000, 4000)) -> None:
    """How much history does the LIVE buffer need before features match training?"""
    full = build_features(raw.copy(), version=version, candles_per_day=cpd, verbose=False)
    cols = feature_cols(full, raw.columns)
    stds = full[cols].std().replace(0, np.nan)

    print("\n== G2 WARMUP (TRAIN/SERVE SKEW) AUDIT ==")
    print("  comparing the SAME final candle computed with N candles of history vs full history")
    print(f"  reporting features off by > {WARMUP_TOL:.0%} of their full-history std\n")
    print(f"  {'warmup N':>9}  {'#skewed':>8}  worst offenders (diff as x of feature std)")
    print("  " + "-" * 76)

    for n in warmups:
        if n >= len(raw):
            continue
        tail = build_features(raw.iloc[-n:].copy(), version=version, candles_per_day=cpd, verbose=False)
        if tail.empty:
            print(f"  {n:>9}  {'ALL':>8}  feature warmup consumes the whole buffer (all rows NaN)")
            continue
        last = tail.iloc[-1]
        ref = full.iloc[-1]
        skew = {}
        for c in cols:
            d = abs(float(last[c]) - float(ref[c]))
            rel = d / stds[c] if np.isfinite(stds[c]) else 0.0
            if rel > WARMUP_TOL:
                skew[c] = rel
        worst = ", ".join(f"{c}={v:.2f}x"
                          for c, v in sorted(skew.items(), key=lambda kv: -kv[1])[:4])
        print(f"  {n:>9}  {len(skew):>8}  {worst or 'â€” matches training'}")

    print("\n  Reading: the live watcher must buffer at least the first N with 0 skewed")
    print("  features, or the model receives observations it never saw in training.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--version", default="v4")
    p.add_argument("--candles-per-day", type=int, default=24)
    args = p.parse_args()

    # Keep the DatetimeIndex — resetting it would trip add_time_encoding's
    # missing-timestamp guard (and previously triggered its silent positional
    # fallback, which this audit caught as fake warmup skew on hour/day).
    raw = pd.read_parquet(args.input)
    print(f"[AUDIT] {args.input} | {len(raw):,} rows | features {args.version}")

    ok = audit_lookahead(raw, args.version, args.candles_per_day)
    audit_warmup(raw, args.version, args.candles_per_day)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

