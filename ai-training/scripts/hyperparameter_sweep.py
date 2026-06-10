"""
hyperparameter_sweep.py
───────────────────────
Optuna-based Bayesian hyperparameter sweep for the PPO trading agent.

Uses TPE (Tree-structured Parzen Estimator) to efficiently search over
PPO hypers + environment configuration, evaluating each trial on
validation-set Sharpe Ratio.

Features:
  - Bayesian search (TPESampler) over 10 hyperparameters
  - Median-based pruning to terminate unpromising trials early
  - SQLite persistence for pause/resume across sessions
  - JSON export of best-found parameters
  - Parallel vectorized environments per trial
  - Full VecNormalize wrapper (consistent with train_agent.py)

Usage:
    python scripts/hyperparameter_sweep.py \
        --train data/BTC_USDT_1h_train.parquet \
        --val   data/BTC_USDT_1h_val.parquet \
        --n-trials 50

Resume a previous sweep:
    python scripts/hyperparameter_sweep.py \
        --train data/BTC_USDT_1h_train.parquet \
        --val   data/BTC_USDT_1h_val.parquet \
        --n-trials 100 \
        --db sqlite:///optuna_sweep.db \
        --study-name tradebot_ppo_sweep
"""

import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Add parent directory to path so we can import environments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environments.trading_env import TradingEnv
from scripts.config import CANDLES_PER_DAY   # P0-5 / P3-1 fix

# Suppress noisy warnings during sweep
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ── Paths ─────────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOGS_DIR   = os.path.join(os.path.dirname(__file__), "..", "logs", "optuna")


# ── Network Architecture Presets ──────────────────────────────────────────────

NET_ARCH_MAP = {
    "small":  dict(pi=[128, 64],        vf=[128, 64]),
    "medium": dict(pi=[256, 256, 128],  vf=[256, 256, 128]),
    "large":  dict(pi=[512, 256, 128],  vf=[512, 256, 128]),
}


# ── Environment Factory ──────────────────────────────────────────────────────

def make_env(
    df:                pd.DataFrame,
    position_fraction: float = 0.20,
    max_drawdown_pct:  float = 0.50,
    candles_per_day:   int   = 96,    # P0-5 fix: was missing → Sharpe 2× inflated on 1h data
    seed:              int   = 0,
):
    """Creates a monitored TradingEnv with trial-specific env config."""
    def _init():
        env = TradingEnv(
            df,
            window_size=48,
            initial_balance=10_000.0,
            fee_rate=0.001,
            slippage_pct=0.0005,
            max_drawdown_pct=max_drawdown_pct,
            reward_scaling=1.0,
            position_fraction=position_fraction,
            candles_per_day=candles_per_day,
        )
        env = Monitor(env)
        return env
    return _init


# ── Pruning Callback ─────────────────────────────────────────────────────────

class OptunaTrialCallback(BaseCallback):
    """
    Reports intermediate reward to Optuna every `eval_freq` timesteps
    so that MedianPruner can terminate unpromising trials early.
    """
    def __init__(self, trial: optuna.Trial, eval_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.trial     = trial
        self.eval_freq = eval_freq
        self.ep_rewards: list[float] = []

    def _on_step(self) -> bool:
        # Collect completed episode rewards from Monitor wrappers
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])

        # Report to Optuna at regular intervals
        if self.num_timesteps % self.eval_freq == 0 and self.ep_rewards:
            mean_reward = float(np.mean(self.ep_rewards[-20:]))
            self.trial.report(mean_reward, step=self.num_timesteps)

            if self.trial.should_prune():
                raise optuna.TrialPruned(
                    f"Trial pruned at step {self.num_timesteps} "
                    f"(mean reward: {mean_reward:.4f})"
                )

        return True


# ── Validation Runner ────────────────────────────────────────────────────────

def run_validation(
    model:             PPO,
    val_df:            pd.DataFrame,
    vec_norm:          VecNormalize | None,
    position_fraction: float,
    max_drawdown_pct:  float,
    candles_per_day:   int = 96,      # P0-3/P0-5 fix: must match dataset timeframe
) -> dict:
    """
    P0-3 fix: Replaced single-episode validation with 3-slice method.

    Previously this ran one deterministic episode on the full val set, which
    could produce a Sharpe that was lucky or unlucky for that single market
    period. Hyperparameters were selected based on a single-regime result,
    making the sweep overfit to whatever trend happened to be in val_df.

    Now matches train_agent.py: splits val_df into 3 chronological slices,
    evaluates each independently, returns mean Sharpe with std. High std
    means regime-sensitive policy — Optuna will naturally penalise it.

    Also carries LSTM state between steps (P1-3 fix) for RecurrentPPO.
    """
    n_slices = 3
    slice_len = len(val_df) // n_slices
    slices = [
        val_df.iloc[i * slice_len : (i + 1) * slice_len].reset_index(drop=True)
        for i in range(n_slices)
    ]

    old_training    = vec_norm.training    if vec_norm else None
    old_norm_reward = vec_norm.norm_reward if vec_norm else None
    if vec_norm:
        vec_norm.training    = False
        vec_norm.norm_reward = False

    all_metrics: list[dict] = []

    try:
        for slice_df in slices:
            val_env = TradingEnv(
                slice_df,
                window_size=48,
                initial_balance=10_000.0,
                fee_rate=0.001,
                slippage_pct=0.0005,
                max_drawdown_pct=max_drawdown_pct,
                reward_scaling=1.0,
                position_fraction=position_fraction,
                candles_per_day=candles_per_day,
            )
            obs, _ = val_env.reset()
            done   = False

            # P1-3 fix: carry LSTM state — essential for RecurrentPPO
            lstm_state    = None
            episode_start = np.ones((1,), dtype=bool)

            while not done:
                policy_obs = obs
                if vec_norm:
                    policy_obs = vec_norm.normalize_obs(
                        np.array([obs], dtype=np.float32)
                    )[0]
                action, lstm_state = model.predict(
                    policy_obs,
                    state         = lstm_state,
                    episode_start = episode_start,
                    deterministic = True,
                )
                episode_start = np.zeros((1,), dtype=bool)
                obs, _, terminated, truncated, _ = val_env.step(int(action))
                done = terminated or truncated

            m = val_env.get_episode_metrics()
            if "error" not in m:
                all_metrics.append(m)

    finally:
        if vec_norm and old_training is not None:
            vec_norm.training    = old_training
            vec_norm.norm_reward = old_norm_reward

    if not all_metrics:
        return {"sharpe_ratio": -99.0, "error": "No trades in any validation slice"}

    # Aggregate: mean all numeric scalars across slices
    aggregated: dict = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if isinstance(m.get(key), (int, float))]
        if vals:
            aggregated[key] = round(float(np.mean(vals)), 4)

    sharpe_vals = [m.get("sharpe_ratio", 0) for m in all_metrics]
    aggregated["sharpe_std"] = round(float(np.std(sharpe_vals)), 4)
    aggregated["n_slices"]   = len(all_metrics)
    return aggregated


# ── Optuna Objective ─────────────────────────────────────────────────────────

def create_objective(
    train_df:        pd.DataFrame,
    val_df:          pd.DataFrame,
    timesteps:       int,
    n_envs:          int,
    candles_per_day: int = 96,   # P0-5 fix: pass through to make_env and run_validation
    objective_metric: str = "sharpe_ratio",  # F9: sharpe_ratio | sortino_ratio | calmar_ratio
):
    """
    Returns an Optuna objective function that:
      1. Samples PPO + env hyperparameters from the search space
      2. Trains PPO for `timesteps` steps
      3. Evaluates on validation set
      4. Returns validation Sharpe Ratio (maximize)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def objective(trial: optuna.Trial) -> float:
        # ── Sample Hyperparameters ────────────────────────────────────────────
        learning_rate    = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        ent_coef         = trial.suggest_float("ent_coef", 0.01, 0.1, log=True)
        gamma            = trial.suggest_float("gamma", 0.99, 0.999)
        n_steps          = trial.suggest_categorical("n_steps", [2048, 4096, 8192])
        clip_range       = trial.suggest_float("clip_range", 0.05, 0.3)
        batch_size       = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        n_epochs         = trial.suggest_categorical("n_epochs", [3, 5, 10])
        position_fraction = trial.suggest_float("position_fraction", 0.05, 0.50)
        max_drawdown_pct = trial.suggest_float("max_drawdown_pct", 0.30, 0.70)
        net_arch_type    = trial.suggest_categorical("net_arch_type", ["small", "medium", "large"])

        net_arch = NET_ARCH_MAP[net_arch_type]

        # ── Print Trial Config ────────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  Trial {trial.number:>3} | {net_arch_type} arch | "
              f"lr={learning_rate:.2e} ent={ent_coef:.3f} γ={gamma:.4f}")
        print(f"  n_steps={n_steps} batch={batch_size} epochs={n_epochs} "
              f"clip={clip_range:.3f}")
        print(f"  pos_frac={position_fraction:.2f} max_dd={max_drawdown_pct:.2f}")
        print(f"{'─'*65}")

        # ── Build Vectorized Environment ──────────────────────────────────────
        env_fns = [
            make_env(train_df, position_fraction, max_drawdown_pct,
                     candles_per_day=candles_per_day, seed=trial.number * 100 + i)
            for i in range(n_envs)
        ]

        try:
            raw_vec_env = SubprocVecEnv(env_fns)
        except Exception:
            raw_vec_env = DummyVecEnv(env_fns)

        vec_env = VecNormalize(
            raw_vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

        # ── Create Model ─────────────────────────────────────────────────────
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={
                "net_arch": net_arch,
                "activation_fn": torch.nn.Tanh,
            },
            device=device,
            verbose=0,
        )

        # ── Train ────────────────────────────────────────────────────────────
        callback = OptunaTrialCallback(trial, eval_freq=10_000)
        start_time = time.time()

        try:
            model.learn(
                total_timesteps=timesteps,
                callback=callback,
                progress_bar=False,
            )
        except optuna.TrialPruned:
            vec_env.close()
            raise

        elapsed = time.time() - start_time

        # ── Validate ─────────────────────────────────────────────────────────
        metrics = run_validation(
            model, val_df, vec_env,
            position_fraction, max_drawdown_pct,
            candles_per_day=candles_per_day,
        )

        vec_env.close()

        # Handle edge cases (no trades, errors)
        if "error" in metrics:
            print(f"  ⚠️  Trial {trial.number}: {metrics['error']}")
            return float("-inf")

        sharpe      = metrics.get("sharpe_ratio", -99.0)
        sortino     = metrics.get("sortino_ratio", -99.0)   # F9
        calmar      = metrics.get("calmar_ratio", 0.0)      # F9
        total_ret   = metrics.get("total_return_pct", 0.0)
        win_rate    = metrics.get("win_rate_pct", 0.0)
        n_trades    = metrics.get("total_trades", 0)
        max_dd      = metrics.get("max_drawdown_pct", 0.0)
        # G6 (Freqtrade hyperopt_loss_profit_drawdown): composite objective —
        # maximize return while penalizing the drawdown suffered to earn it.
        # Their default weighs drawdown at ~1:1 against profit; same here.
        if objective_metric == "profit_drawdown":
            obj_value = total_ret - max_dd
        else:
            obj_value = metrics.get(objective_metric, -99.0)  # F9: the metric we optimize

        print(f"\n  ✅ Trial {trial.number} complete ({elapsed:.0f}s)")
        print(f"     Obj[{objective_metric}]: {obj_value:>7.3f} | Sharpe: {sharpe:>7.3f} | "
              f"Sortino: {sortino:>7.3f} | Return: {total_ret:>6.2f}% | Trades: {n_trades}")

        # Store extra metrics as trial user attributes for later analysis
        trial.set_user_attr("sharpe_ratio", sharpe)
        trial.set_user_attr("sortino_ratio", sortino)
        trial.set_user_attr("calmar_ratio", calmar)
        trial.set_user_attr("total_return_pct", total_ret)
        trial.set_user_attr("win_rate_pct", win_rate)
        trial.set_user_attr("total_trades", n_trades)
        trial.set_user_attr("max_drawdown_pct", max_dd)
        trial.set_user_attr("training_time_s", round(elapsed, 1))

        # F9: optimize the chosen risk-adjusted metric (Sortino/Calmar more robust than Sharpe)
        return obj_value

    return objective


# ── Results Output ────────────────────────────────────────────────────────────

def print_study_results(study: optuna.Study):
    """Pretty-prints the final optimization results and best parameters."""

    print(f"\n{'='*65}")
    print(f"  OPTUNA SWEEP COMPLETE")
    print(f"{'='*65}")
    print(f"  Study name      : {study.study_name}")
    print(f"  Total trials    : {len(study.trials)}")
    print(f"  Completed       : {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Pruned          : {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Failed          : {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    if study.best_trial:
        best = study.best_trial
        print(f"\n  {'─'*61}")
        print(f"  BEST TRIAL (#{best.number})")
        print(f"  {'─'*61}")
        print(f"  Sharpe Ratio    : {best.value:.3f}")

        for attr_key in ["total_return_pct", "win_rate_pct", "total_trades",
                         "max_drawdown_pct", "training_time_s"]:
            if attr_key in best.user_attrs:
                label = attr_key.replace("_", " ").title()
                print(f"  {label:<18}: {best.user_attrs[attr_key]}")

        print(f"\n  Best Hyperparameters:")
        for key, val in best.params.items():
            if isinstance(val, float):
                print(f"    {key:<22}: {val:.6f}")
            else:
                print(f"    {key:<22}: {val}")

    # ── Top-5 Trials ──────────────────────────────────────────────────────────
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) > 1:
        ranked = sorted(completed, key=lambda t: t.value if t.value is not None else float("-inf"), reverse=True)
        print(f"\n  Top-5 Trials:")
        print(f"  {'#':<5} {'Sharpe':<10} {'Return%':<10} {'WR%':<8} {'Trades':<8} {'MaxDD%':<8}")
        print(f"  {'─'*49}")
        for t in ranked[:5]:
            ret = t.user_attrs.get("total_return_pct", "?")
            wr  = t.user_attrs.get("win_rate_pct", "?")
            nt  = t.user_attrs.get("total_trades", "?")
            dd  = t.user_attrs.get("max_drawdown_pct", "?")
            val = t.value if t.value is not None else float("-inf")
            print(f"  {t.number:<5} {val:<10.3f} {ret:<10} {wr:<8} {nt:<8} {dd:<8}")

    print(f"{'='*65}\n")


def save_best_params(study: optuna.Study, output_path: str):
    """Exports the best trial parameters to a JSON file."""
    if not study.best_trial:
        print("  [WARN] No completed trials — nothing to save.")
        return

    best = study.best_trial
    net_arch_type = best.params.get("net_arch_type", "medium")  # P3-4 fix: was .pop() — mutated trial dict

    result = {
        "study_name":       study.study_name,
        "best_trial":       best.number,
        "best_sharpe":      best.value,
        "ppo_hyperparams": {
            "learning_rate": best.params["learning_rate"],
            "n_steps":       best.params["n_steps"],
            "batch_size":    best.params["batch_size"],
            "n_epochs":      best.params["n_epochs"],
            "gamma":         best.params["gamma"],
            "clip_range":    best.params["clip_range"],
            "ent_coef":      best.params["ent_coef"],
            "gae_lambda":    0.95,
            "vf_coef":       0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch":      NET_ARCH_MAP[net_arch_type],
                "activation_fn": "Tanh",
            },
        },
        "env_config": {
            "window_size":       48,
            "initial_balance":   10_000.0,
            "fee_rate":          0.001,
            "slippage_pct":      0.0005,
            "max_drawdown_pct":  best.params["max_drawdown_pct"],
            "reward_scaling":    1.0,
            "position_fraction": best.params["position_fraction"],
        },
        "validation_metrics": {
            k: v for k, v in best.user_attrs.items()
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  [SAVED] Best params → {output_path}")


def print_train_command(study: optuna.Study, train_path: str, val_path: str):
    """Prints the exact command to train with the best-found hyperparameters."""
    if not study.best_trial:
        return

    best = study.best_trial
    p    = best.params

    print(f"\n{'='*65}")
    print(f"  TRAIN WITH BEST PARAMS")
    print(f"{'='*65}")
    print(f"  Copy-paste the command below to launch full training:\n")
    print(f"  python scripts/train_agent.py \\")
    print(f"      --train {train_path} \\")
    print(f"      --val {val_path} \\")
    print(f"      --timesteps 500000 \\")
    print(f"      --run-name sweep_best")
    print()
    print(f"  Then update PPO_HYPERPARAMS in train_agent.py with:")
    print(f"    learning_rate  = {p['learning_rate']:.6e}")
    print(f"    n_steps        = {p['n_steps']}")
    print(f"    batch_size     = {p['batch_size']}")
    print(f"    n_epochs       = {p['n_epochs']}")
    print(f"    gamma          = {p['gamma']:.6f}")
    print(f"    clip_range     = {p['clip_range']:.6f}")
    print(f"    ent_coef       = {p['ent_coef']:.6f}")
    print(f"    net_arch       = {NET_ARCH_MAP.get(p.get('net_arch_type', 'medium'))}")
    print()
    print(f"  And update ENV_CONFIG with:")
    print(f"    position_fraction = {p['position_fraction']:.4f}")
    print(f"    max_drawdown_pct  = {p['max_drawdown_pct']:.4f}")
    print(f"{'='*65}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter sweep for PPO trading agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick 10-trial sweep
  python scripts/hyperparameter_sweep.py \\
      --train data/BTC_USDT_1h_train.parquet \\
      --val   data/BTC_USDT_1h_val.parquet \\
      --n-trials 10 --timesteps 50000

  # Full 50-trial sweep (resume-safe)
  python scripts/hyperparameter_sweep.py \\
      --train data/BTC_USDT_1h_train.parquet \\
      --val   data/BTC_USDT_1h_val.parquet \\
      --n-trials 50 --db sqlite:///optuna_sweep.db
        """,
    )

    parser.add_argument("--train",      type=str, required=True,
                        help="Path to training .parquet file")
    parser.add_argument("--val",        type=str, required=True,
                        help="Path to validation .parquet file")
    parser.add_argument("--n-trials",   type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--timesteps",  type=int, default=100_000,
                        help="Training timesteps per trial (default: 100000)")
    parser.add_argument("--study-name", type=str, default="tradebot_ppo_sweep",
                        help="Optuna study name (default: tradebot_ppo_sweep)")
    parser.add_argument("--db",         type=str, default="sqlite:///optuna_sweep.db",
                        help="Optuna storage URI (default: sqlite:///optuna_sweep.db)")
    parser.add_argument("--n-envs",     type=int, default=4,
                        help="Parallel environments per trial (default: 4)")
    parser.add_argument("--objective",  type=str, default="sortino_ratio",
                        choices=["sharpe_ratio", "sortino_ratio", "calmar_ratio",
                                 "profit_drawdown"],
                        help="F9: metric to optimize. Default sortino_ratio (more robust than "
                             "Sharpe for our regime-sensitive, drawdown-prone problem).")
    parser.add_argument("--timeframe",  type=str, default="15m",
                        choices=list(CANDLES_PER_DAY.keys()),
                        help="Candle timeframe for correct Sharpe annualization (P0-5 fix)")

    args = parser.parse_args()

    # ── Load Data ─────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  OPTUNA HYPERPARAMETER SWEEP")
    print(f"{'='*65}")
    print(f"  Study         : {args.study_name}")
    print(f"  Storage       : {args.db}")
    print(f"  Trials        : {args.n_trials}")
    print(f"  Steps / trial : {args.timesteps:,}")
    print(f"  Parallel envs : {args.n_envs}")
    print(f"  Device        : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
    print(f"{'='*65}\n")

    print(f"[LOADING] {args.train}")
    train_df = pd.read_parquet(args.train).reset_index(drop=True)

    print(f"[LOADING] {args.val}")
    val_df = pd.read_parquet(args.val).reset_index(drop=True)

    print(f"[DATA] Train: {len(train_df):,} rows | Val: {len(val_df):,} rows\n")

    # ── Create / Load Study ───────────────────────────────────────────────────
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.db,
        direction="maximize",            # Maximize Sharpe Ratio
        sampler=TPESampler(
            seed=42,
            n_startup_trials=10,         # Pure random for first 10 trials
            multivariate=True,           # Model parameter correlations
        ),
        pruner=MedianPruner(
            n_startup_trials=5,          # Don't prune first 5 trials
            n_warmup_steps=20_000,       # Let each trial warm up before pruning
            interval_steps=10_000,
        ),
        load_if_exists=True,             # Resume from DB if study exists
    )

    existing_trials = len(study.trials)
    if existing_trials > 0:
        print(f"  [RESUME] Found {existing_trials} existing trials in study")
        if study.best_trial:
            print(f"  [RESUME] Current best Sharpe: {study.best_value:.3f} "
                  f"(trial #{study.best_trial.number})\n")

    # ── Run Optimization ──────────────────────────────────────────────────────
    cpd       = CANDLES_PER_DAY.get(args.timeframe, 96)
    objective = create_objective(train_df, val_df, args.timesteps, args.n_envs,
                                 candles_per_day=cpd, objective_metric=args.objective)

    sweep_start = time.time()

    study.optimize(
        objective,
        n_trials=args.n_trials,
        show_progress_bar=True,
        gc_after_trial=True,             # Free memory between trials
    )

    sweep_elapsed = time.time() - sweep_start

    # ── Output Results ────────────────────────────────────────────────────────
    print_study_results(study)

    output_path = os.path.join(MODELS_DIR, "best_hyperparams.json")
    save_best_params(study, output_path)

    print(f"  Total sweep time: {sweep_elapsed/60:.1f} minutes")
    print(f"  Avg per trial   : {sweep_elapsed/max(args.n_trials, 1):.1f}s")

    print_train_command(study, args.train, args.val)


if __name__ == "__main__":
    main()
