"""
ensemble_predict.py
───────────────────
Runs inference using an ensemble of multiple trained models, aggregating
their predictions via majority voting. This reduces the variance of individual
policies and often yields a more stable trading strategy.

Usage:
    python scripts/ensemble_predict.py \
        --models models/model_seed1.zip models/model_seed2.zip models/model_seed3.zip \
        --data data/BTC_USDT_1h_test.parquet
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environments.trading_env import TradingEnv
# P0-4 fix: ENV_CONFIG is defined in config.py, not trading_env.py
# The original import caused an immediate ImportError on script launch.
from scripts.config import ENV_CONFIG

def load_models(model_paths: list[str]) -> list[PPO]:
    """Loads a list of trained PPO models."""
    models = []
    for path in model_paths:
        if not os.path.exists(path):
            print(f"[ERROR] Model not found: {path}")
            sys.exit(1)
        print(f"[LOAD] Loading {path}...")
        models.append(PPO.load(path))
    return models

def majority_vote(actions: list[int]) -> int:
    """Returns the most common action among the ensemble."""
    counts = np.bincount(actions)
    return np.argmax(counts)

def evaluate_ensemble(models: list[PPO], env: TradingEnv):
    """Evaluates the ensemble on the environment."""
    obs, info = env.reset()
    done = False
    
    action_counts = {0: 0, 1: 0, 2: 0}
    
    print("\n[INFERENCE] Running ensemble evaluation...")
    while not done:
        # Collect predictions from all models
        preds = []
        for model in models:
            action, _ = model.predict(obs, deterministic=True)
            preds.append(int(action))
            
        # Aggregate via majority voting
        final_action = majority_vote(preds)
        action_counts[final_action] += 1
        
        obs, reward, terminated, truncated, info = env.step(final_action)
        done = terminated or truncated
        
    # Print results
    pnl = env.balance - env.initial_balance
    pnl_pct = (pnl / env.initial_balance) * 100
    
    print("\n============================================================")
    print("  ENSEMBLE EVALUATION RESULTS")
    print("============================================================")
    print(f"  Models in ensemble : {len(models)}")
    print(f"  Final Balance      : ${env.balance:.2f} (Init: ${env.initial_balance:.2f})")
    print(f"  Net PnL            : {pnl_pct:+.2f}%")
    print(f"  Max Drawdown       : {env.max_drawdown_seen * 100:.2f}%")
    print(f"  Total Trades       : {len(env.trade_history)}")
    
    total_actions = sum(action_counts.values()) or 1
    print("\n  Ensemble Action Distribution:")
    print(f"    HOLD : {action_counts[0]:>4} ({action_counts[0]/total_actions*100:4.1f}%)")
    print(f"    BUY  : {action_counts[1]:>4} ({action_counts[1]/total_actions*100:4.1f}%)")
    print(f"    SELL : {action_counts[2]:>4} ({action_counts[2]/total_actions*100:4.1f}%)")
    print("============================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Ensemble Inference Script")
    parser.add_argument("--models", type=str, nargs='+', required=True,
                        help="Paths to trained model .zip files")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to feature parquet (e.g. test set)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"[ERROR] Data file not found: {args.data}")
        sys.exit(1)
        
    df = pd.read_parquet(args.data)
    
    # Initialize environment
    # Note: ensemble_predict does not train, so we don't need VecEnv or SubprocVecEnv
    # We can just use the raw TradingEnv
    env = TradingEnv(df, **ENV_CONFIG)
    
    # Load models
    models = load_models(args.models)
    
    # Run evaluation
    evaluate_ensemble(models, env)

if __name__ == "__main__":
    main()
