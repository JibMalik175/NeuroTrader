"""
agreement_gate_eval.py — H4 (OctoBot evaluator-matrix idea, applied to our ensemble)
────────────────────────────────────────────────────────────────────────────────────
We have two independently-trained besttrain checkpoints that are BOTH slightly
profitable at deployment fees but err on different slices (p2_8: better val,
p2_9: better test). OctoBot's architecture never acts on one opinion; this
script tests that idea: step both models in lockstep over the SAME episode and
execute an action ONLY when they agree — disagreement = HOLD.

Hypothesis: agreement trades are higher-conviction → fewer trades, better
per-trade expectancy. Risk: too few trades to matter.

Each model keeps its own LSTM hidden state and its own VecNormalize obs stats
(they were trained in different normalization universes). Both observe the
same env/portfolio state — standard policy-ensemble practice.

Usage (defaults match the p2_8/p2_9 router configs):
  python scripts/agreement_gate_eval.py \
      --models models/p2_8_regime_router_window1_besttrain.zip models/p2_9_makerfee_window1_besttrain.zip \
      --vecnorms models/p2_8_regime_router_window1_besttrain_vecnormalize.pkl models/p2_9_makerfee_window1_besttrain_vecnormalize.pkl \
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

FEE_SCENARIOS = [
    ("spot taker 0.20% RT",   0.00100),
    ("futures maker 0.04% RT", 0.00020),
]
N_SLICES = 3


def run_agreement_episode(models, vecnorms, env: TradingEnv,
                          exit_mode: str = "any") -> dict:
    """
    One deterministic episode of consensus trading.
      ENTRIES (from flat): require unanimity — strict, high-conviction only.
      EXITS (in position): 'any' = close if EITHER model proposes the closing
        action (slow in, fast out — first run proved unanimous exits over-hold,
        avg hold 13→25 candles); 'unanimous' = original symmetric gate.
    """
    obs, _ = env.reset()
    states = [None] * len(models)
    ep_start = np.ones((1,), dtype=bool)
    done = False

    while not done:
        actions = []
        for i, (m, vn) in enumerate(zip(models, vecnorms)):
            mobs = vn.normalize_obs(np.array([obs], dtype=np.float32))[0]
            a, states[i] = m.predict(mobs, state=states[i],
                                     episode_start=ep_start, deterministic=True)
            actions.append(int(a))
        ep_start = np.zeros((1,), dtype=bool)

        unanimous = all(a == actions[0] for a in actions)
        pos = env.position_dir  # 0 flat, +1 long, -1 short
        if pos == 0:
            final = actions[0] if unanimous else 0
        else:
            closing = 2 if pos == 1 else 1  # long closes via SELL, short via BUY
            if exit_mode == "any" and closing in actions:
                final = closing
            else:
                final = actions[0] if unanimous else 0

        obs, _, term, trunc, _ = env.step(final)
        done = term or trunc

    return env.get_episode_metrics()


def eval_split(models, vecnorms, df: pd.DataFrame) -> dict:
    """3-slice mean, mirroring train_agent.run_validation's protocol."""
    size = len(df) // N_SLICES
    slices = [df.iloc[i * size:(i + 1) * size].reset_index(drop=True)
              for i in range(N_SLICES)] if size >= 300 else [df]
    all_m = []
    for s in slices:
        env = TradingEnv(s, **ENV_CONFIG)
        m = run_agreement_episode(models, vecnorms, env, exit_mode="any")
        if "error" not in m:
            all_m.append(m)
    keys = ["total_trades", "net_profit_factor", "gross_profit_factor",
            "net_realized_pnl_pct", "gross_expectancy_pct", "net_expectancy_pct",
            "sharpe_ratio", "avg_hold_candles"]
    out = {k: float(np.mean([m.get(k, 0.0) for m in all_m])) for k in keys}
    out["sharpe_std"] = float(np.std([m.get("sharpe_ratio", 0.0) for m in all_m]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models",   nargs="+", required=True)
    p.add_argument("--vecnorms", nargs="+", required=True)
    p.add_argument("--val",  required=True)
    p.add_argument("--test", default=None)
    p.add_argument("--candles-per-day", type=int, default=24)
    p.add_argument("--cooldown", type=int, default=12)
    args = p.parse_args()
    assert len(args.models) == len(args.vecnorms), "one vecnorm per model"

    # env flags matching how the checkpoints were trained
    ENV_CONFIG["candles_per_day"]  = args.candles_per_day
    ENV_CONFIG["allow_short"]      = True
    ENV_CONFIG["reward_mode"]      = "exit"
    ENV_CONFIG["cooldown_candles"] = args.cooldown
    ENV_CONFIG["regime_router"]    = True

    print(f"[LOAD] {len(args.models)} models")
    models = [RecurrentPPO.load(m) for m in args.models]

    val_df  = pd.read_parquet(args.val).reset_index(drop=True)
    test_df = pd.read_parquet(args.test).reset_index(drop=True) if args.test else None

    dummy = DummyVecEnv([lambda: Monitor(TradingEnv(val_df.iloc[:600], **ENV_CONFIG))])
    vecnorms = []
    for v in args.vecnorms:
        vn = VecNormalize.load(v, dummy)
        vn.training = False
        vn.norm_reward = False
        vecnorms.append(vn)

    hdr = (f"{'scenario':<26}{'split':>6}{'trades':>8}{'netPF':>8}{'net%':>8}"
           f"{'gExp%':>8}{'nExp%':>8}{'shrp_std':>9}{'hold':>6}")
    print("\n" + "=" * len(hdr))
    print("H4 AGREEMENT GATE — act only when ALL models agree (else HOLD)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for name, fee in FEE_SCENARIOS:
        ENV_CONFIG["fee_rate"] = fee
        for split, df in [("VAL", val_df), ("TEST", test_df)]:
            if df is None:
                continue
            r = eval_split(models, vecnorms, df)
            flag = " <<" if r["net_profit_factor"] > 1.0 else ""
            print(f"{name:<26}{split:>6}{r['total_trades']:>8.1f}"
                  f"{r['net_profit_factor']:>8.3f}{r['net_realized_pnl_pct']:>8.2f}"
                  f"{r['gross_expectancy_pct']:>8.3f}{r['net_expectancy_pct']:>8.3f}"
                  f"{r['sharpe_std']:>9.3f}{r['avg_hold_candles']:>6.1f}{flag}")
    print("-" * len(hdr))
    print("Compare vs solo (fee sweeps, 06-10): p2_8 val 1.657/test 1.147 @maker;")
    print("p2_9 val 1.441/test 1.285 @maker. << = net-profitable.")


if __name__ == "__main__":
    main()

