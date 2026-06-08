"""
compare_runs.py — A/B comparison of training runs.

Reads models/<run>_training_log.json for each run and prints a single aligned
table of the validation metrics that actually matter for the fee-ceiling work:
net PF, net expectancy/trade, gross expectancy vs fee, trade count, hold, etc.

Usage:
    python scripts/compare_runs.py                       # auto: p0_/p2_ runs + baselines
    python scripts/compare_runs.py p2_tf1h p2_fee2x p2_fee3x
"""
import os, sys, json, glob

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Round-trip taker fee on notional (entry+exit). The bar gross expectancy must clear.
ROUND_TRIP_FEE_PCT = 0.20


def load(run):
    path = os.path.join(MODELS_DIR, f"{run}_training_log.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if not data:
        return None
    return data[-1]  # last window = production model


def main():
    runs = sys.argv[1:]
    if not runs:
        found = sorted(glob.glob(os.path.join(MODELS_DIR, "*_training_log.json")))
        runs = [os.path.basename(p).replace("_training_log.json", "") for p in found]
        # surface the timeframe/fee experiments first
        priority = [r for r in runs if r.startswith(("p0_", "p2_"))]
        runs = priority + [r for r in runs if r not in priority]

    hdr = (f"{'run':<22}{'Sharpe':>9}{'+-std':>7}{'trades':>7}"
           f"{'grossPF':>8}{'netPF':>7}{'gExp%':>8}{'nExp%':>8}"
           f"{'fees%':>7}{'net%':>8}{'hold':>6}{'H%':>6}")
    print(f"\nRound-trip fee bar: gross expectancy must exceed {ROUND_TRIP_FEE_PCT:.2f}%\n")
    print(hdr)
    print("-" * len(hdr))

    for run in runs:
        m = load(run)
        if not m:
            continue
        ad = m.get("action_distribution", {}) or {}
        tot = sum(ad.values()) or 1
        hpct = ad.get("0", 0) / tot * 100
        gexp = m.get("gross_expectancy_pct", 0)
        # flag whether gross edge clears the fee (the whole game)
        clears = "*" if gexp > ROUND_TRIP_FEE_PCT else " "
        npf = m.get("net_profit_factor", 0)
        prof = "<<" if npf > 1.0 else ""  # net-profitable marker
        print(f"{run:<22}{m.get('sharpe_ratio',0):>9.3f}{m.get('sharpe_std',0):>7.2f}"
              f"{m.get('total_trades',0):>7.1f}{m.get('gross_profit_factor',0):>8.3f}"
              f"{npf:>7.3f}{gexp:>7.3f}{clears}{m.get('net_expectancy_pct',0):>8.3f}"
              f"{m.get('fees_paid_pct',0):>7.2f}{m.get('net_realized_pnl_pct',0):>8.2f}"
              f"{m.get('avg_hold_candles',0):>6.1f}{hpct:>5.0f}% {prof}")

    print("\n  * = gross expectancy clears the round-trip fee (real edge after costs)")
    print("  << = net-profitable on validation (net PF > 1.0)")
    print("  NOTE: these are VALIDATION metrics (mean of 3 slices). Low trade counts")
    print("        (<25) make Sharpe/PF unreliable — trust runs with more trades + low std.\n")


if __name__ == "__main__":
    main()
