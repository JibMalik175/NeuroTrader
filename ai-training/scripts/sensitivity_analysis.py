"""
sensitivity_analysis.py — Confidence threshold sensitivity sweep.
Sweeps MIN_CONFIDENCE from 0.35 to 0.90 and shows Sharpe vs threshold.
Inspired by Tutorial 2 parameter sensitivity heat maps.
Find the PLATEAU (stable Sharpe over a range), not the single peak.

Usage:
    python scripts/sensitivity_analysis.py \
        --model models/tradebot_ppo_best.zip \
        --data  data/BTC_USDT_15m_test.parquet
"""
import sys, os, argparse, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environments.trading_env import TradingEnv
from scripts.config import ENV_CONFIG

try:
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("[ERROR] stable_baselines3 not installed"); sys.exit(1)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def run_with_threshold(model, test_df, threshold, n_slices=3):
    slice_len = len(test_df) // n_slices
    slices = [test_df.iloc[i*slice_len:(i+1)*slice_len].reset_index(drop=True) for i in range(n_slices)]
    all_metrics = []
    for sl in slices:
        env = TradingEnv(sl, **ENV_CONFIG)
        obs, _ = env.reset()
        done, lstm_state, episode_start = False, None, np.ones((1,), dtype=bool)
        while not done:
            action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_start, deterministic=True)
            episode_start = np.zeros((1,), dtype=bool)
            # Apply confidence gate
            try:
                import torch
                with torch.no_grad():
                    obs_t = torch.tensor(obs[np.newaxis], dtype=torch.float32)
                    dist  = model.policy.get_distribution(obs_t)
                    probs = dist.distribution.probs.numpy()[0]
                    if probs.max() < threshold and int(action) != 0:
                        action = 0
            except Exception:
                pass
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        m = env.get_episode_metrics()
        if "error" not in m:
            all_metrics.append(m)
    if not all_metrics:
        return {"threshold": round(threshold, 3), "sharpe": -99, "sharpe_std": 99, "win_rate": 0, "total_trades": 0, "total_return": 0}
    return {
        "threshold":    round(threshold, 3),
        "sharpe":       round(float(np.mean([m["sharpe_ratio"]   for m in all_metrics])), 4),
        "sharpe_std":   round(float(np.std( [m["sharpe_ratio"]   for m in all_metrics])), 4),
        "win_rate":     round(float(np.mean([m["win_rate_pct"]   for m in all_metrics])), 2),
        "total_trades": round(float(np.mean([m["total_trades"]   for m in all_metrics])), 1),
        "total_return": round(float(np.mean([m["total_return_pct"] for m in all_metrics])), 3),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True)
    parser.add_argument("--data",    required=True)
    parser.add_argument("--min-thr", type=float, default=0.35)
    parser.add_argument("--max-thr", type=float, default=0.90)
    parser.add_argument("--steps",   type=int,   default=12)
    args = parser.parse_args()

    try:    model = RecurrentPPO.load(args.model, device="cpu")
    except: model = PPO.load(args.model, device="cpu")

    test_df     = pd.read_parquet(args.data).reset_index(drop=True)
    thresholds  = np.linspace(args.min_thr, args.max_thr, args.steps)
    results     = []

    print(f"\n{'='*70}")
    print(f"  Confidence Threshold Sensitivity | {args.min_thr:.2f}→{args.max_thr:.2f} | {args.steps} steps")
    print(f"{'='*70}")
    print(f"  {'Threshold':>10} {'Sharpe':>10} {'±Std':>8} {'WinRate':>9} {'Trades':>8} {'Return':>9}")
    print(f"  {'-'*60}")

    for thr in thresholds:
        r = run_with_threshold(model, test_df, thr)
        results.append(r)
        flag = " ← STABLE" if r["sharpe"] > 0 and r["sharpe_std"] < 0.5 else ""
        print(f"  {r['threshold']:>10.2f} {r['sharpe']:>10.4f} {r['sharpe_std']:>8.4f} "
              f"{r['win_rate']:>8.2f}% {r['total_trades']:>8.1f} {r['total_return']:>8.3f}%{flag}")

    positive = [r for r in results if r["sharpe"] > 0 and r["sharpe_std"] < 1.0]
    recommended = max(positive, key=lambda x: x["sharpe"])["threshold"] if positive else None

    if recommended:
        print(f"\n  ✅ Recommended threshold: {recommended:.2f}")
        print(f"  → Set MIN_CONFIDENCE={recommended:.2f} in your .env")
    else:
        print(f"\n  ❌ No profitable stable region found. Model needs more training.")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "sensitivity_report.json"), "w") as f:
        json.dump({"results": results, "recommended_threshold": recommended}, f, indent=2)

if __name__ == "__main__":
    main()
