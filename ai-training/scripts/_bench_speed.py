"""
_bench_speed.py — Phase 0.3 training-throughput A/B harness.

Measures real model.learn() it/s (not just env step speed) across:
  - algo: PPO vs RecurrentPPO
  - n_envs: 4 vs 8
Each cell trains for a fixed small step budget on the fast env and reports it/s.

Run:
    venv\\Scripts\\python.exe scripts\\_bench_speed.py --steps 40000
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import torch
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environments.trading_env import TradingEnv
from scripts.config import ENV_CONFIG, PPO_HYPERPARAMS


def make_env(df, seed):
    def _init():
        return Monitor(TradingEnv(df, **ENV_CONFIG))
    return _init


def build(df, algo, n_envs, device):
    fns = [make_env(df, i) for i in range(n_envs)]
    try:
        raw = SubprocVecEnv(fns)
        venv_kind = "Subproc"
    except Exception:
        raw = DummyVecEnv(fns)
        venv_kind = "Dummy"
    venv = VecNormalize(raw, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    hp = dict(PPO_HYPERPARAMS)
    if algo == "recurrent":
        hp["policy_kwargs"] = {"lstm_hidden_size": 128, "n_lstm_layers": 1,
                               "net_arch": dict(pi=[128, 64], vf=[128, 64])}
        model = RecurrentPPO("MlpLstmPolicy", venv, device=device, verbose=0, **hp)
    else:
        model = PPO("MlpPolicy", venv, device=device, verbose=0, **hp)
    return model, venv, venv_kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--data", type=str, default="data/BTC_USDT_15m_train.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.data).reset_index(drop=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | steps/cell={args.steps:,} | train rows={len(df):,}\n")
    print(f"{'algo':<11}{'n_envs':<8}{'venv':<9}{'device':<7}{'it/s':>10}{'sec':>9}")
    print("-" * 54)

    cells = [
        ("recurrent", 4, "cuda"),
        ("recurrent", 8, "cuda"),
        ("ppo",       4, "cuda"),
        ("ppo",       8, "cuda"),
        ("ppo",       8, "cpu"),
    ]
    for algo, n_envs, dev in cells:
        dev = dev if (dev == "cpu" or torch.cuda.is_available()) else "cpu"
        model, venv, vk = build(df, algo, n_envs, dev)
        t = time.perf_counter()
        model.learn(total_timesteps=args.steps, progress_bar=False)
        dt = time.perf_counter() - t
        print(f"{algo:<11}{n_envs:<8}{vk:<9}{dev:<7}{args.steps/dt:>10,.0f}{dt:>9.1f}")
        venv.close()
        del model, venv


if __name__ == "__main__":
    main()
