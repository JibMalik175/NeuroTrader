"""
baseline_backtest.py — RSI momentum baseline (Tutorial 2 insight).

The key finding: in crypto, buying RSI>70 (momentum) and exiting RSI<30
outperforms traditional mean-reversion RSI by ~6x using Wilder's smoothing.

Use this as your DRL benchmark. If your trained model cannot beat this
3-line rule after full training, something is wrong with the model.

Usage:
    python scripts/baseline_backtest.py --data data/BTC_USDT_15m_test.parquet
    python scripts/baseline_backtest.py --data data/BTC_USDT_15m_test.parquet --sweep
"""
import sys, os, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.config import ENV_CONFIG, EFFECTIVE_FEE_RATE

try:
    from ta.momentum import RSIIndicator
except ImportError:
    print("[ERROR] ta library not installed: pip install ta"); sys.exit(1)

def run_momentum_backtest(df, rsi_entry=70.0, rsi_exit=30.0,
                          fee_rate=None, position_frac=0.20, initial_bal=10_000.0, slippage=0.0002):
    fee_rate = fee_rate or EFFECTIVE_FEE_RATE
    rsi      = RSIIndicator(close=df["close"], window=14).rsi()
    rsi_prev = rsi.shift(1)
    entry_s  = (rsi_prev <= rsi_entry) & (rsi.values >= rsi_entry)  # crosses above
    exit_s   = (rsi_prev >= rsi_exit)  & (rsi.values < rsi_exit)    # crosses below

    balance = initial_bal; peak = initial_bal; in_pos = False
    entry_price = 0.0; pos_cost = 0.0
    trades = []; equity = [initial_bal]; max_dd = 0.0
    closes = df["close"].values

    for i in range(len(df)):
        p   = closes[i]
        pv  = balance + pos_cost * (p - entry_price) / entry_price if in_pos else balance
        peak = max(peak, pv)
        max_dd = max(max_dd, (peak - pv) / peak)
        equity.append(pv)

        if not in_pos and entry_s.iloc[i]:
            fill      = p * (1 + slippage)
            cash      = balance * position_frac
            fee       = cash * fee_rate
            balance  -= fee
            pos_cost  = cash - fee
            entry_price = fill
            in_pos    = True
        elif in_pos and exit_s.iloc[i]:
            fill      = p * (1 - slippage)
            pnl_pct   = (fill - entry_price) / entry_price
            gross     = pos_cost * (1 + pnl_pct)
            fee       = gross * fee_rate
            balance   = (balance - pos_cost) + gross - fee
            trades.append(pnl_pct)
            in_pos    = False; pos_cost = 0.0

    if in_pos:
        fill    = closes[-1] * (1 - slippage)
        pnl_pct = (fill - entry_price) / entry_price
        gross   = pos_cost * (1 + pnl_pct)
        balance = (balance - pos_cost) + gross - gross * fee_rate
        trades.append(pnl_pct)

    eq     = np.array(equity)
    sr     = np.diff(eq) / (eq[:-1] + 1e-10)
    cpd    = ENV_CONFIG.get("candles_per_day", 96)
    sharpe = float(sr.mean() / sr.std() * np.sqrt(cpd * 365)) if sr.std() > 1e-10 else 0.0
    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    bah    = (closes[-1] - closes[0]) / closes[0] * 100

    return {
        "total_return_pct": round((balance - initial_bal) / initial_bal * 100, 3),
        "bah_return_pct":   round(bah, 3),
        "alpha_pct":        round((balance - initial_bal) / initial_bal * 100 - bah, 3),
        "sharpe_ratio":     round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "total_trades":     len(trades),
        "win_rate_pct":     round(len(wins) / len(trades) * 100, 2) if trades else 0,
        "avg_win_pct":      round(np.mean(wins) * 100, 4)   if wins   else 0,
        "avg_loss_pct":     round(np.mean(losses) * 100, 4) if losses else 0,
        "profit_factor":    round(abs(sum(wins)) / abs(sum(losses)), 3) if losses else float("inf"),
        "fee_rate":         fee_rate,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      required=True)
    parser.add_argument("--rsi-entry", type=float, default=70.0)
    parser.add_argument("--rsi-exit",  type=float, default=30.0)
    parser.add_argument("--sweep",     action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.data).reset_index(drop=True)
    print(f"\n[BASELINE] {len(df):,} rows | fee={EFFECTIVE_FEE_RATE*100:.4f}%")

    if args.sweep:
        print(f"\n{'='*65}")
        print(f"  RSI Entry Threshold Sweep (exit fixed at {args.rsi_exit})")
        print(f"{'='*65}")
        print(f"  {'Entry':>8} {'Sharpe':>10} {'WR':>8} {'Trades':>8} {'Return':>10} {'Alpha':>8}")
        for entry in np.arange(55, 85, 5):
            r = run_momentum_backtest(df, rsi_entry=entry, rsi_exit=args.rsi_exit)
            m = " ←" if entry == 70 else ""
            print(f"  {entry:>8.0f} {r['sharpe_ratio']:>10.3f} {r['win_rate_pct']:>7.2f}% "
                  f"{r['total_trades']:>8} {r['total_return_pct']:>9.2f}% {r['alpha_pct']:>+7.2f}%{m}")
        print(f"\n  Your DRL model MUST beat the best row above to be worth deploying.")
        return

    r = run_momentum_backtest(df, rsi_entry=args.rsi_entry, rsi_exit=args.rsi_exit,
                              position_frac=ENV_CONFIG["position_fraction"],
                              initial_bal=ENV_CONFIG["initial_balance"])
    print(f"\n{'='*55}")
    print(f"  RSI MOMENTUM BASELINE (Wilder's smoothing)")
    print(f"{'='*55}")
    for k, v in r.items():
        print(f"  {k:<22}: {v}")
    print(f"{'='*55}")
    print(f"\n  ⚡ Your DRL model must beat Sharpe={r['sharpe_ratio']} and Return={r['total_return_pct']}%")
    print(f"  Run with --sweep to see full parameter sensitivity table.")

if __name__ == "__main__":
    main()
