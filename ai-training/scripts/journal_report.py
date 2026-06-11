"""
journal_report.py — turn a month of decision-journal lines into the verdict
────────────────────────────────────────────────────────────────────────────
Reads execution-engine/logs/journal/*.jsonl (one JSON line per decision,
written by decisionJournal.ts) and produces the month-1 review:

  - every trade, chronologically, with entry/exit/pnl/fees/reason/duration
  - aggregate: trades, win rate, gross/net PnL, fees, profit factor,
    expectancy/trade, avg hold, equity curve + max drawdown
  - decision stats: signals seen, action distribution, entries blocked or
    skipped by reason (protections doing their job?)
  - backtest comparison block: the runbook gates (trade count within ~2x,
    expectancy sign matches, maker fill behavior)

Usage:
  python scripts/journal_report.py                      # default journal dir
  python scripts/journal_report.py --dir <path> --since 2026-06-15
"""

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np


def load_entries(journal_dir: str, since: str | None) -> list[dict]:
    entries = []
    for f in sorted(glob.glob(os.path.join(journal_dir, "*.jsonl"))):
        day = os.path.basename(f).replace(".jsonl", "")
        if since and day < since:
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # torn line from a crash — append-only means at most one
    return entries


def fmt_dur(ms: float) -> str:
    h = ms / 3_600_000
    return f"{h:.1f}h" if h < 48 else f"{h/24:.1f}d"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.join("..", "execution-engine", "logs", "journal"))
    p.add_argument("--since", default=None, help="YYYY-MM-DD lower bound")
    args = p.parse_args()

    entries = load_entries(args.dir, args.since)
    if not entries:
        print(f"No journal entries found in {args.dir}")
        return

    by_type = Counter(e["type"] for e in entries)
    signals = [e for e in entries if e["type"] == "signal"]
    exits = [e for e in entries if e["type"] == "exit"]
    blocked = [e for e in entries if e["type"] == "entry_blocked"]
    skipped = [e for e in entries if e["type"] == "entry_skipped"]

    print("=" * 74)
    print(f"PAPER-TRADING REVIEW  {entries[0]['ts'][:10]} -> {entries[-1]['ts'][:10]}")
    print("=" * 74)

    # ── decision stats ──────────────────────────────────────────────────────
    print(f"\nDECISIONS: {len(entries)} journal lines  "
          f"({', '.join(f'{k}={v}' for k, v in sorted(by_type.items()))})")
    if signals:
        dist = Counter(s["signal"] for s in signals)
        total = sum(dist.values())
        print("  signal mix : " + "  ".join(f"{k} {v / total:.0%}" for k, v in sorted(dist.items())))
        conf = [s["confidence"] for s in signals if s.get("confidence") is not None]
        if conf:
            print(f"  confidence : mean {np.mean(conf):.2f} | p90 {np.percentile(conf, 90):.2f}")
    if blocked:
        print("  blocked    : " + ", ".join(f"{r} x{c}" for r, c in
              Counter(b.get("reason", "?") for b in blocked).most_common()))
    if skipped:
        print("  skipped    : " + ", ".join(f"{r} x{c}" for r, c in
              Counter(s.get("reason", "?") for s in skipped).most_common()))

    # ── trades ──────────────────────────────────────────────────────────────
    if not exits:
        print("\nNO CLOSED TRADES YET — nothing to judge. (Expected pace: ~1 per 4-5 days.)")
        return

    print(f"\nTRADES ({len(exits)}):")
    print(f"  {'#':>3} {'closed (UTC)':<17} {'entry':>10} {'exit':>10} {'pnl%':>8} "
          f"{'fee$':>7} {'hold':>6}  reason")
    wins = 0
    for i, t in enumerate(exits, 1):
        pnl = t.get("pnlPct", 0.0) * 100
        wins += pnl > 0
        print(f"  {i:>3} {t['ts'][:16]:<17} {t.get('entryPrice', 0):>10.2f} "
              f"{t.get('exitPrice', 0):>10.2f} {pnl:>+8.3f} "
              f"{t.get('feePaid', 0):>7.3f} {fmt_dur(t.get('durationMs', 0)):>6}  "
              f"{t.get('exitReason', '?')}")

    pnls = np.array([t.get("pnlPct", 0.0) for t in exits]) * 100
    fees = sum(t.get("feePaid", 0.0) for t in exits)
    gains, losses = pnls[pnls > 0].sum(), -pnls[pnls < 0].sum()
    pf = gains / losses if losses > 0 else float("inf")
    balances = [t.get("newBalance") for t in exits if t.get("newBalance")]
    max_dd = 0.0
    if balances:
        b = np.array(balances)
        max_dd = float(((np.maximum.accumulate(b) - b) / np.maximum.accumulate(b)).max() * 100)

    print(f"\nAGGREGATE:")
    print(f"  win rate        : {wins}/{len(exits)} ({wins / len(exits):.0%})")
    print(f"  net PnL         : {pnls.sum():+.3f}%  (fees paid: ${fees:.2f})")
    print(f"  profit factor   : {pf:.3f}")
    print(f"  expectancy/trade: {pnls.mean():+.4f}%")
    holds = [t.get("durationMs", 0) / 3_600_000 for t in exits]
    print(f"  avg hold        : {np.mean(holds):.1f}h")
    print(f"  max drawdown    : {max_dd:.2f}% (on closed-trade equity)")

    # ── runbook gates ───────────────────────────────────────────────────────
    days = max((np.datetime64(entries[-1]["ts"][:10]) - np.datetime64(entries[0]["ts"][:10]))
               / np.timedelta64(1, "D"), 1)
    rate = len(exits) / days * 7
    print(f"\nRUNBOOK GATES (backtest expectation: ~1.5-2 trades/week, +0.15-0.21%/trade):")
    print(f"  trade pace      : {rate:.1f}/week  -> {'OK' if 0.7 <= rate <= 4 else 'INVESTIGATE'}")
    print(f"  expectancy sign : {'MATCHES backtest (+)' if pnls.mean() > 0 else 'NEGATIVE — does not match'}")
    print(f"  avg hold        : {np.mean(holds):.1f}h vs backtest ~13h -> "
          f"{'OK' if 4 <= np.mean(holds) <= 40 else 'INVESTIGATE'}")


if __name__ == "__main__":
    main()
