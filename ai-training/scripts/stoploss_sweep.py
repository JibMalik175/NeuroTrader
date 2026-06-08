"""
stoploss_sweep.py — Post-hoc stop-loss diagnostic (15-minute diagnostic).

Tests whether adding a hard stop-loss to the EXISTING trained model
improves Sharpe, answering the critical question:

  CASE 1: Few giant losers driving poor PF → SL will help dramatically
  REGIME: General underperformance       → SL will barely change Sharpe

This is a VALIDATION-ONLY script — no training occurs.
Uses the experiment_c model without any reward changes.

Usage:
    python scripts/stoploss_sweep.py \\
        --val   data/BTC_USDT_15m_val.parquet \\
        --model models/experiment_c_300k_best.zip \\
        --vecnorm models/experiment_c_300k_best_vecnormalize.pkl
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environments.trading_env import TradingEnv
from scripts.config import ENV_CONFIG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO


def run_validation_with_sl(model, val_df: pd.DataFrame, vec_norm, sl_pct=None) -> dict:
    """
    Runs deterministic 3-slice validation.
    If sl_pct is set, forces SELL whenever unrealized_pnl <= -sl_pct.
    The model's own policy is otherwise unchanged.
    """
    slice_size = len(val_df) // 3
    slices = [
        val_df.iloc[i * slice_size : (i + 1) * slice_size].reset_index(drop=True)
        for i in range(3)
    ]

    if vec_norm:
        vec_norm.training    = False
        vec_norm.norm_reward = False

    sharpes, returns, trades_list, sl_triggers = [], [], [], []

    for slice_df in slices:
        env  = TradingEnv(slice_df, **ENV_CONFIG)
        obs, _ = env.reset()
        done   = False
        lstm_state    = None
        episode_start = np.ones((1,), dtype=bool)
        sl_count      = 0

        while not done:
            policy_obs = obs
            if vec_norm:
                policy_obs = vec_norm.normalize_obs(np.array([obs], dtype=np.float32))[0]

            action, lstm_state = model.predict(
                policy_obs,
                state         = lstm_state,
                episode_start = episode_start,
                deterministic = True,
            )
            action        = int(action)
            episode_start = np.zeros((1,), dtype=bool)

            # ── Inject stop-loss override ──────────────────────────────────────
            if sl_pct is not None and env.position_held:
                current_price = env._get_close_price(env.current_step)
                unrealized    = (current_price - env.entry_price) / (env.entry_price + 1e-8)
                if unrealized <= -sl_pct:
                    action   = 2   # Force SELL
                    sl_count += 1

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        m = env.get_episode_metrics()
        sharpes.append(m.get("sharpe_ratio",    0.0))
        returns.append(m.get("total_return_pct", 0.0))
        trades_list.append(m.get("total_trades", 0))
        sl_triggers.append(sl_count)

    return {
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe":  float(np.std(sharpes)),
        "slices": [
            {
                "sharpe":       s,
                "return_pct":   r,
                "trades":       t,
                "sl_triggers":  sl,
            }
            for s, r, t, sl in zip(sharpes, returns, trades_list, sl_triggers)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Stop-loss sweep diagnostic")
    parser.add_argument("--val",      type=str, required=True,  help="Validation .parquet")
    parser.add_argument("--model",    type=str, required=True,  help="Model .zip path")
    parser.add_argument("--vecnorm",  type=str, default=None,   help="VecNormalize .pkl path")
    args = parser.parse_args()

    print(f"[LOADING] {args.val}")
    val_df = pd.read_parquet(args.val).reset_index(drop=True)
    print(f"  Val rows: {len(val_df):,}")

    print(f"[LOADING] {args.model}")
    model = RecurrentPPO.load(args.model)

    vec_norm = None
    if args.vecnorm and os.path.exists(args.vecnorm):
        # Need a dummy env to load VecNormalize — shape must match
        dummy_df  = val_df.iloc[:200].reset_index(drop=True)
        dummy_env = DummyVecEnv([lambda: TradingEnv(dummy_df, **ENV_CONFIG)])
        vec_norm  = VecNormalize.load(args.vecnorm, dummy_env)
        print(f"[LOADING] VecNormalize from {args.vecnorm}")
    else:
        print("[WARNING] No VecNormalize found — obs will not be normalized")

    sl_levels = [0.03, 0.05, 0.07, 0.10, None]

    print("\n" + "="*72)
    print("  STOP-LOSS SWEEP — experiment_c_300k model")
    print("  Q: Are losses from a few giant trades, or general regime fail?")
    print("  CASE 1 → Sharpe improves significantly with SL")
    print("  REGIME → Sharpe barely changes across all SL levels")
    print("="*72)

    header = f"{'SL Level':>10} | {'Mean Sharpe':>12} | {'±Std':>6} | {'S0':>8} | {'S1':>8} | {'S2':>8} | {'SL Fires':>9}"
    print(f"\n{header}")
    print("-" * len(header))

    results = {}
    for sl_pct in sl_levels:
        label  = f"-{sl_pct*100:.0f}%" if sl_pct is not None else "None"
        result = run_validation_with_sl(model, val_df, vec_norm, sl_pct)
        results[label] = result

        sl_fires = sum(s["sl_triggers"] for s in result["slices"])
        slices   = result["slices"]

        # Improvement over baseline (None)
        flag = ""
        if sl_pct is not None and "None" in results:
            base = results["None"]["mean_sharpe"]
            diff = result["mean_sharpe"] - base
            flag = f"  (+{diff:.3f})" if diff > 0 else f"  ({diff:.3f})"

        print(
            f"{label:>10} | {result['mean_sharpe']:>+12.3f} | {result['std_sharpe']:>6.3f} | "
            f"{slices[0]['sharpe']:>+8.3f} | {slices[1]['sharpe']:>+8.3f} | "
            f"{slices[2]['sharpe']:>+8.3f} | {sl_fires:>9}{flag}"
        )

    print("\n" + "="*72)

    # Auto-verdict
    base_sharpe = results["None"]["mean_sharpe"]
    best_sl     = max((v["mean_sharpe"] - base_sharpe, k)
                      for k, v in results.items() if k != "None")
    improvement = best_sl[0]
    best_level  = best_sl[1]

    print(f"\n  VERDICT:")
    if improvement > 0.5:
        print(f"  ✅ CASE 1 CONFIRMED — Best SL ({best_level}) improved Sharpe by +{improvement:.3f}")
        print(f"     Disposition effect on tail losses is real.")
        print(f"     Next: FIX-B + position_fraction=0.05, then proportional SL in training.")
    elif improvement > 0.2:
        print(f"  ⚠️  WEAK CASE 1 — Best SL ({best_level}) improved Sharpe by +{improvement:.3f}")
        print(f"     Some benefit from SL but not dominant. Mixed evidence.")
    else:
        print(f"  ❌ REGIME PROBLEM — Best SL ({best_level}) only improved Sharpe by +{improvement:.3f}")
        print(f"     Losses are spread across many trades, not concentrated in tail.")
        print(f"     Skip Experiment D. Focus on feature quality or more data.")
    print("="*72)


if __name__ == "__main__":
    main()
