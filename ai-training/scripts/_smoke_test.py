"""
_smoke_test.py
──────────────
Quick smoke test for the TradingEnv.
Runs 500 random steps and validates reward magnitudes, observation shape,
and reward component breakdown.

P3-3 fix: added --data and --timeframe CLI args so this works on 15m data
and any other timeframe — previously hardcoded to BTC_USDT_1h_train.parquet.

Usage:
    python scripts/_smoke_test.py
    python scripts/_smoke_test.py --data data/BTC_USDT_15m_train.parquet --timeframe 15m
"""
import sys, os, random, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from environments.trading_env import TradingEnv
# P3-1/P3-3 fix: use shared config — no more manual env kwargs that drift
from scripts.config import ENV_CONFIG, CANDLES_PER_DAY

def main():
    parser = argparse.ArgumentParser(description="TradingEnv smoke test")
    parser.add_argument(
        "--data", type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'BTC_USDT_15m_train.parquet'),
        help="Path to .parquet file (default: BTC_USDT_15m_train.parquet)",
    )
    parser.add_argument(
        "--timeframe", type=str, default="15m",
        choices=list(CANDLES_PER_DAY.keys()),
        help="Candle timeframe for correct Sharpe annualization",
    )
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERROR] Data file not found: {args.data}")
        sys.exit(1)

    # P3-1: build config from shared source, override candles_per_day per timeframe
    import copy
    cfg = copy.deepcopy(ENV_CONFIG)
    cfg["candles_per_day"] = CANDLES_PER_DAY[args.timeframe]

    print(f"\n[SMOKE TEST] Loading {args.data}")
    df = pd.read_parquet(args.data).reset_index(drop=True)
    print(f"[SMOKE TEST] {len(df):,} rows | timeframe={args.timeframe} | cpd={cfg['candles_per_day']}")

    env = TradingEnv(df, **cfg)
    obs, info = env.reset()

    print(f"\nObs shape  : {obs.shape}")
    print(f"Expected   : {env.observation_space.shape}")
    print(f"Shape match: {obs.shape == env.observation_space.shape}")
    print(f"Obs range  : [{obs.min():.4f}, {obs.max():.4f}]")
    print(f"Has NaN    : {np.isnan(obs).any()}")

    # Run N random steps
    rewards = []
    actions_count = {0: 0, 1: 0, 2: 0}
    done = False
    steps = 0
    random.seed(42)

    while not done and steps < args.steps:
        a = random.randint(0, 2)
        obs, r, terminated, truncated, info = env.step(a)
        rewards.append(r)
        actions_count[a] += 1
        done = terminated or truncated
        steps += 1

    r_arr = np.array(rewards)
    print(f"\n--- {args.steps}-step Random Policy Test ---")
    print(f"Steps run    : {steps}")
    print(f"Terminated   : {terminated}")
    print(f"Reward range : [{r_arr.min():.6f}, {r_arr.max():.6f}]")
    print(f"Reward mean  : {r_arr.mean():.6f}")
    print(f"Reward std   : {r_arr.std():.6f}")
    print(f"Max |reward| : {np.abs(r_arr).max():.6f}")
    print(f"Actions      : H={actions_count[0]} B={actions_count[1]} S={actions_count[2]}")
    print(f"Trades       : {len(env.trade_history)}")
    print(f"Balance      : {env.balance:.2f}")
    print(f"In position  : {env.position_held}")

    if "reward_components" in info:
        print(f"\nReward components (cumulative):")
        for k, v in info["reward_components"].items():
            print(f"  {k:<22}: {v:>+.6f}")

    print()
    ok = True
    if np.abs(r_arr).max() > 10.0:
        print("[WARN] ❌ Reward magnitude > 10.0 — scaling may be broken")
        ok = False
    else:
        print("[OK] ✅ Reward magnitudes sane (all < 10.0)")

    if r_arr.std() < 1e-10:
        print("[WARN] ❌ Reward is constant — flatline detected")
        ok = False
    else:
        print("[OK] ✅ Reward has variance — not flatlined")

    if obs.shape != env.observation_space.shape:
        print("[WARN] ❌ Observation shape mismatch")
        ok = False
    else:
        print("[OK] ✅ Observation shape correct")

    if np.isnan(obs).any():
        print("[WARN] ❌ NaN in final observation")
        ok = False
    else:
        print("[OK] ✅ No NaN in observations")

    print()
    if ok:
        print("[SMOKE TEST] ✅ ALL CHECKS PASSED")
    else:
        print("[SMOKE TEST] ❌ SOME CHECKS FAILED — review warnings above")
        sys.exit(1)

if __name__ == "__main__":
    main()
