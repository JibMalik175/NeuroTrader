"""
per_slice_check.py — deploy-bar gate: does EVERY slice hold up, or is one
lucky stretch carrying the averages?
─────────────────────────────────────────────────────────────────────────
All our headline numbers (fee sweeps, A/Bs) are MEANS of 3 chronological
slices per split. A strategy can post a great mean off one monster slice
and two duds — and means hide it. The deploy bar (PROGRESS_CHECKLIST.md)
demands: net PF > 1.0 on ALL validation slices, gross PF > 1.2, and
>=30 trades/slice at the deployment fee.

Prints one row PER SLICE for val and test at the deployment fee.

Usage:
  python scripts/per_slice_check.py \
      --model models/p2_8_regime_router_window1_besttrain.zip \
      --vecnorm models/p2_8_regime_router_window1_besttrain_vecnormalize.pkl \
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

N_SLICES = 3


def run_episode(model, vecnorm, env: TradingEnv) -> dict:
    obs, _ = env.reset()
    state = None
    ep_start = np.ones((1,), dtype=bool)
    done = False
    while not done:
        mobs = vecnorm.normalize_obs(np.array([obs], dtype=np.float32))[0]
        action, state = model.predict(mobs, state=state,
                                      episode_start=ep_start, deterministic=True)
        ep_start = np.zeros((1,), dtype=bool)
        obs, _, term, trunc, _ = env.step(int(action))
        done = term or trunc
    return env.get_episode_metrics()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--val",  required=True)
    p.add_argument("--test", default=None)
    p.add_argument("--fee", type=float, default=0.0002,
                   help="per-side deployment fee (default: futures maker 0.02%)")
    p.add_argument("--candles-per-day", type=int, default=24)
    p.add_argument("--cooldown", type=int, default=12)
    args = p.parse_args()

    ENV_CONFIG["candles_per_day"]  = args.candles_per_day
    ENV_CONFIG["allow_short"]      = True
    ENV_CONFIG["reward_mode"]      = "exit"
    ENV_CONFIG["cooldown_candles"] = args.cooldown
    ENV_CONFIG["regime_router"]    = True
    ENV_CONFIG["fee_rate"]         = args.fee

    print(f"[LOAD] {args.model}")
    model = RecurrentPPO.load(args.model)
    val_df = pd.read_parquet(args.val).reset_index(drop=True)
    test_df = pd.read_parquet(args.test).reset_index(drop=True) if args.test else None

    dummy = DummyVecEnv([lambda: Monitor(TradingEnv(val_df.iloc[:600], **ENV_CONFIG))])
    vecnorm = VecNormalize.load(args.vecnorm, dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False

    hdr = (f"{'slice':<12}{'trades':>8}{'grossPF':>9}{'netPF':>8}{'net%':>8}"
           f"{'gExp%':>8}{'sharpe':>8}{'maxDD%':>8}  verdict")
    print(f"\nPER-SLICE DEPLOY-BAR CHECK @ fee {args.fee*100:.3f}%/side "
          f"({2*args.fee*100:.2f}% RT)")
    print("bar: net PF > 1.0 every VAL slice | gross PF > 1.2 | >=30 trades/slice")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    all_pass = True
    for split, df in [("VAL", val_df), ("TEST", test_df)]:
        if df is None:
            continue
        size = len(df) // N_SLICES
        slices = [df.iloc[i * size:(i + 1) * size].reset_index(drop=True)
                  for i in range(N_SLICES)] if size >= 300 else [df]
        for i, s in enumerate(slices, 1):
            env = TradingEnv(s, **ENV_CONFIG)
            m = run_episode(model, vecnorm, env)
            npf = m.get("net_profit_factor", 0.0)
            gpf = m.get("gross_profit_factor", 0.0)
            ntr = m.get("total_trades", 0)
            ok_npf = npf > 1.0
            ok_gpf = gpf > 1.2
            ok_n   = ntr >= 30
            verdict = []
            if not ok_npf: verdict.append("netPF")
            if not ok_gpf: verdict.append("grossPF")
            if not ok_n:   verdict.append("trades")
            if split == "VAL" and verdict:
                all_pass = False
            tag = "PASS" if not verdict else "fail: " + ",".join(verdict)
            print(f"{split + ' #' + str(i):<12}{ntr:>8.0f}{gpf:>9.3f}{npf:>8.3f}"
                  f"{m.get('net_realized_pnl_pct', 0.0):>8.2f}"
                  f"{m.get('gross_expectancy_pct', 0.0):>8.3f}"
                  f"{m.get('sharpe_ratio', 0.0):>8.2f}"
                  f"{m.get('max_drawdown_pct', 0.0):>8.2f}  {tag}")
    print("-" * len(hdr))
    print("DEPLOY BAR (val slices): " + ("MET ✅" if all_pass else "NOT MET — see fails above"))
    print("TEST rows are informational (bear regime; judge vs buy&hold -34%).")


if __name__ == "__main__":
    main()
