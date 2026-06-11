"""
grid_backtest.py — H1 (OctoBot grid/staggered_orders idea): is a flat-regime
grid +EV after fees on the chop our router deliberately sits out?
─────────────────────────────────────────────────────────────────────────────
Concept: the RL router is ~95% flat BY DESIGN — it only trades trends. A grid
profits from oscillation without predicting anything: resting maker orders
every `spacing` below (buys) and above (sells) a center price; each filled
buy places a paired sell one spacing up (and vice versa), so every completed
ping-pong earns `spacing − 2×maker_fee`.

The catch (why this needs proof, not vibes): when the market BREAKS OUT of
the range, the grid is left holding inventory against the move. This harness
models that honestly:
  - grid is ACTIVE only while |macro_trend_sma| < flat threshold (same signal
    the router uses, so grid-time ≈ router-flat-time)
  - on regime flip, all inventory is liquidated at the close (the bleed)
  - fills are intrabar: a level fills when the candle's low/high crosses it,
    one fill per level per candle, maker fee on every fill

Honest simplifications (all conservative or neutral):
  - no compounding; PnL is % of a fixed budget = 2×levels notional units
  - liquidation pays TAKER fee (you're crossing in a breakout)
  - no partial fills, no queue position (level touched = filled — slightly
    optimistic; offset by charging full taker on every liquidation)

Usage:
  python scripts/grid_backtest.py --data data/BTC_USDT_1h_val.parquet \
      --levels 5 --spacing-atr 1.0 --flat-threshold 0.02
"""

import argparse
import sys

import numpy as np
import pandas as pd

MAKER_FEE = 0.0002
TAKER_FEE = 0.0005


def run_grid(df: pd.DataFrame, levels: int, spacing_atr: float,
             flat_threshold: float, confirm: int = 0) -> dict:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = (df["atr_ratio"].values * close)          # atr_ratio = ATR / price
    trend = np.abs(df["macro_trend_sma"].values)
    # Hysteresis against regime flicker (the raw flag liquidated 117x on val):
    # ENTER flat only after `confirm` consecutive calm candles, EXIT the moment
    # trend exceeds 1.5x the threshold — slow in, fast out.
    if confirm > 0:
        calm = trend < flat_threshold
        entered = np.zeros(len(df), dtype=bool)
        streak = 0
        active = False
        for i in range(len(df)):
            streak = streak + 1 if calm[i] else 0
            if not active and streak >= confirm:
                active = True
            elif active and trend[i] > flat_threshold * 1.5:
                active = False
                streak = 0
            entered[i] = active
        flat = entered
    else:
        flat = trend < flat_threshold

    budget_units = 2 * levels                        # notional units committed
    pnl = 0.0                                        # in units of notional
    round_trips = 0
    liquidations = 0
    liq_pnl = 0.0
    fills = 0
    active_candles = 0
    equity_curve = []

    grid_center = None
    spacing = None
    inventory: list[tuple[float, int]] = []          # (entry_price, dir)

    def liquidate(price: float) -> float:
        nonlocal inventory
        p = sum(d * (price - e) / e - TAKER_FEE for e, d in inventory)
        inventory = []
        return p

    for i in range(len(df)):
        if not flat[i]:
            if grid_center is not None:              # regime flipped → bleed
                hit = liquidate(close[i])
                pnl += hit
                liq_pnl += hit
                liquidations += 1
                grid_center = None
            equity_curve.append(pnl)
            continue

        if grid_center is None:                      # (re)activate the grid
            grid_center = close[i]
            spacing = max(spacing_atr * atr[i], grid_center * 1e-4)
            buy_levels = [grid_center - k * spacing for k in range(1, levels + 1)]
            sell_levels = [grid_center + k * spacing for k in range(1, levels + 1)]
            equity_curve.append(pnl)
            active_candles += 1
            continue

        active_candles += 1

        # fills: buys against the low, sells against the high. Levels are static
        # price lines; iterating each once per candle = one fill per level per
        # candle. A filled buy closes the oldest short (round trip) or adds a
        # long lot, capped at `levels` lots per side; mirror for sells.
        for lvl in buy_levels:
            if low[i] <= lvl:
                if any(d == -1 for _, d in inventory):
                    e, _ = inventory.pop(next(j for j, (_, d) in enumerate(inventory) if d == -1))
                    pnl += (e - lvl) / e - 2 * MAKER_FEE
                    round_trips += 1
                elif sum(1 for _, d in inventory if d == 1) < levels:
                    inventory.append((lvl, 1))
                fills += 1
        for lvl in sell_levels:
            if high[i] >= lvl:
                if any(d == 1 for _, d in inventory):
                    e, _ = inventory.pop(next(j for j, (_, d) in enumerate(inventory) if d == 1))
                    pnl += (lvl - e) / e - 2 * MAKER_FEE
                    round_trips += 1
                elif sum(1 for _, d in inventory if d == -1) < levels:
                    inventory.append((lvl, -1))
                fills += 1

        equity_curve.append(pnl)

    if grid_center is not None and inventory:        # end of data
        hit = liquidate(close[-1])
        pnl += hit
        liq_pnl += hit
        liquidations += 1

    eq = np.array(equity_curve) / budget_units * 100
    peak = np.maximum.accumulate(eq)
    max_dd = float((peak - eq).max()) if len(eq) else 0.0

    return {
        "net_pnl_pct": pnl / budget_units * 100,
        "grid_pnl_pct": (pnl - liq_pnl) / budget_units * 100,
        "liq_pnl_pct": liq_pnl / budget_units * 100,
        "round_trips": round_trips,
        "fills": fills,
        "liquidations": liquidations,
        "active_pct": active_candles / max(len(df), 1) * 100,
        "max_dd_pct": max_dd,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--levels", type=int, default=5)
    p.add_argument("--spacing-atr", type=float, default=1.0)
    p.add_argument("--flat-threshold", type=float, default=0.02)
    p.add_argument("--confirm", type=int, default=0,
                   help="hysteresis: consecutive calm candles required to enter")
    p.add_argument("--scan", action="store_true",
                   help="parameter scan over levels/spacing/threshold/confirm")
    args = p.parse_args()

    if args.scan:
        hdr = (f"{'dataset':<24}{'lv':>3}{'spc':>5}{'thr':>6}{'cnf':>4}{'act%':>6}"
               f"{'rt':>5}{'liq':>5}{'grid%':>7}{'liq%':>8}{'NET%':>7}{'DD%':>6}")
        print("H1 GRID PARAMETER SCAN")
        print(hdr)
        print("-" * len(hdr))
        for path in args.data:
            df = pd.read_parquet(path).reset_index(drop=True)
            name = path.split("/")[-1].split("\\")[-1][:22]
            for lv in (3, 5):
                for spc in (1.0, 2.0):
                    for thr in (0.015, 0.03):
                        for cnf in (24, 72):
                            r = run_grid(df, lv, spc, thr, cnf)
                            print(f"{name:<24}{lv:>3}{spc:>5.1f}{thr:>6.3f}{cnf:>4}"
                                  f"{r['active_pct']:>6.1f}{r['round_trips']:>5}"
                                  f"{r['liquidations']:>5}{r['grid_pnl_pct']:>7.2f}"
                                  f"{r['liq_pnl_pct']:>8.2f}{r['net_pnl_pct']:>7.2f}"
                                  f"{r['max_dd_pct']:>6.2f}")
        return

    hdr = (f"{'dataset':<28}{'active%':>8}{'rtrips':>8}{'liqs':>6}{'grid%':>8}"
           f"{'liq%':>8}{'NET%':>8}{'maxDD%':>8}")
    print(f"H1 GRID BACKTEST — levels={args.levels}, spacing={args.spacing_atr}xATR, "
          f"flat=|trend|<{args.flat_threshold}, maker {MAKER_FEE*100:.3f}%")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for path in args.data:
        df = pd.read_parquet(path).reset_index(drop=True)
        need = {"close", "high", "low", "atr_ratio", "macro_trend_sma"}
        if not need.issubset(df.columns):
            print(f"{path:<28} missing columns {need - set(df.columns)}")
            continue
        r = run_grid(df, args.levels, args.spacing_atr, args.flat_threshold)
        name = path.split("/")[-1].split("\\")[-1]
        print(f"{name:<28}{r['active_pct']:>8.1f}{r['round_trips']:>8}{r['liquidations']:>6}"
              f"{r['grid_pnl_pct']:>8.2f}{r['liq_pnl_pct']:>8.2f}{r['net_pnl_pct']:>8.2f}"
              f"{r['max_dd_pct']:>8.2f}")
    print("-" * len(hdr))
    print("grid% = ping-pong earnings | liq% = breakout bleed | NET must be > 0")
    print("to justify existing, and maxDD must be tolerable vs the router's own.")


if __name__ == "__main__":
    main()
