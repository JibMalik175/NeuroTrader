"""
fee_sensitivity.py — where does the model cross into net-profitable?
────────────────────────────────────────────────────────────────────
The capstone verdict (docs/CORE_TRAINING_FIX_PLAN.md, 06-10): the regime
router's gross edge (+0.132%/trade on val) is real but sits under the 0.20%
round-trip TAKER fee. Fees are an EXECUTION variable, not a model variable:

  per-side 0.1000%  = Binance spot taker, no discount   (training baseline)
  per-side 0.0750%  = spot taker with BNB 25% discount  (CONFIG.effectiveFeeRate)
  per-side 0.0500%  = futures taker
  per-side 0.0200%  = futures maker (post-only limit entries/exits)
  per-side 0.0000%  = zero-fee promo pairs (sanity bound)

This script re-evaluates a SAVED model across those scenarios on the val and
test splits (3 slices each, same protocol as training) and prints where net
PF crosses 1.0. The policy is held fixed; only the env's fee_rate changes.
(Fees leak into a few portfolio obs features, so trade counts can shift
slightly between scenarios — that's realistic, not a bug.)

Slippage is held at the training value (0.05%/side) for ALL scenarios, which
is CONSERVATIVE for maker fills (resting limit orders don't pay crossing
slippage), so the maker rows understate the true improvement.

Usage (defaults match the p2_8 regime-router run):
  python scripts/fee_sensitivity.py --model models/p2_8_regime_router_window1.zip \
      --vecnorm models/p2_8_regime_router_window1_vecnormalize.pkl \
      --val data/BTC_USDT_1h_val.parquet --test data/BTC_USDT_1h_test.parquet
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from environments.trading_env import TradingEnv
from scripts.config import ENV_CONFIG
from scripts import train_agent


FEE_SCENARIOS = [
    ("spot taker (baseline)", 0.00100),
    ("spot taker + BNB 25%",  0.00075),
    ("futures taker",         0.00050),
    ("futures maker",         0.00020),
    ("zero-fee (bound)",      0.00000),
]


def evaluate(model, df: pd.DataFrame, vec_norm) -> dict:
    """3-slice deterministic eval via the SAME run_validation used in training."""
    return train_agent.run_validation(model, df, vec_norm)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--model",   required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--val",     required=True)
    p.add_argument("--test",    default=None)
    # env flags must match how the model was TRAINED (defaults = p2_8 run)
    p.add_argument("--candles-per-day", type=int, default=24)
    p.add_argument("--allow-short",     action="store_true", default=True)
    p.add_argument("--reward-mode",     type=str, default="exit")
    p.add_argument("--cooldown",        type=int, default=12)
    p.add_argument("--regime-router",   action="store_true", default=True)
    args = p.parse_args()

    ENV_CONFIG["candles_per_day"]  = args.candles_per_day
    ENV_CONFIG["allow_short"]      = args.allow_short
    ENV_CONFIG["reward_mode"]      = args.reward_mode
    ENV_CONFIG["cooldown_candles"] = args.cooldown
    ENV_CONFIG["regime_router"]    = args.regime_router

    print(f"[LOAD] model   : {args.model}")
    model = RecurrentPPO.load(args.model)

    val_df  = pd.read_parquet(args.val).reset_index(drop=True)
    test_df = pd.read_parquet(args.test).reset_index(drop=True) if args.test else None

    # VecNormalize.load needs a live vec env with a matching obs space
    dummy = DummyVecEnv([lambda: Monitor(TradingEnv(val_df.iloc[:600], **ENV_CONFIG))])
    print(f"[LOAD] vecnorm : {args.vecnorm}")
    vec_norm = VecNormalize.load(args.vecnorm, dummy)
    vec_norm.training = False
    vec_norm.norm_reward = False

    rows = []
    for name, fee in FEE_SCENARIOS:
        ENV_CONFIG["fee_rate"] = fee
        rt = 2 * fee * 100  # round-trip %
        print(f"\n[EVAL] {name}: fee {fee*100:.3f}%/side ({rt:.2f}% round-trip)")

        vm = evaluate(model, val_df, vec_norm)
        tm = evaluate(model, test_df, vec_norm) if test_df is not None else {}
        rows.append((name, rt, vm, tm))

    hdr = (f"{'scenario':<24}{'RT fee%':>8}{'VAL netPF':>11}{'VAL net%':>10}"
           f"{'VAL gExp%':>11}{'trades':>8}{'TEST netPF':>12}{'TEST net%':>11}")
    print("\n" + "=" * len(hdr))
    print("FEE SENSITIVITY — policy fixed, only execution cost varies")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name, rt, vm, tm in rows:
        flag = " <<" if vm.get("net_profit_factor", 0) > 1.0 else ""
        print(f"{name:<24}{rt:>8.2f}"
              f"{vm.get('net_profit_factor', float('nan')):>11.3f}"
              f"{vm.get('net_realized_pnl_pct', float('nan')):>10.2f}"
              f"{vm.get('gross_expectancy_pct', float('nan')):>11.3f}"
              f"{vm.get('total_trades', float('nan')):>8.1f}"
              f"{tm.get('net_profit_factor', float('nan')):>12.3f}"
              f"{tm.get('net_realized_pnl_pct', float('nan')):>11.2f}"
              f"{flag}")
    print("-" * len(hdr))
    print("<< = net-profitable on validation (net PF > 1.0). Maker rows are")
    print("conservative: slippage is still charged at the taker rate.")


if __name__ == "__main__":
    main()
