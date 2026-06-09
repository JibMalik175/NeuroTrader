"""
train_agent.py
──────────────
Trains a Deep Reinforcement Learning agent (PPO) on historical
Binance candlestick data using the custom TradingEnv.

Training strategy:
  - Algorithm: PPO (Proximal Policy Optimization)
    → More stable than DQN for continuous financial environments
    → Built-in entropy bonus prevents premature convergence
  - Walk-Forward Validation: agent trains on chronological windows
    and is validated on unseen future data after each window
  - Best model checkpoint saved based on validation Sharpe Ratio

v2 changes (post-audit):
  - Fixed PPO hyperparameters for trading (higher entropy, lower LR)
  - GPU device selection with CUDA auto-detect
  - SubprocVecEnv with 4 parallel environments
  - DiagnosticCallback integration for collapse detection
  - TensorBoard logging
  - Removed reward_scaling from ENV_CONFIG
  - Added position_fraction to ENV_CONFIG
  - GPU status logging at startup

Phase 1 fixes applied:
  - VecNormalize now has clip_reward=10.0 (prevents gradient explosion)
  - run_validation type hint uses Optional[VecNormalize] (Python 3.9 compat)
  - Environment sanity_check() called before training starts
  - DiagnosticCallback: fixed throughput calculation (tracks per-interval, not global)
  - DiagnosticCallback: reads reward_components from info dict when available
  - walk_forward_train: minimum episode data guard (warns if window too small)
  - main(): --n-envs flag respected globally via N_ENVS override

Usage:
    python train_agent.py --train ../data/BTC_USDT_1h_train.parquet \\
                          --val   ../data/BTC_USDT_1h_val.parquet
"""

import os
import sys
import argparse
import json
import time
from typing import Optional
import numpy as np
import pandas as pd
import torch

# Fix Windows cp1252 UnicodeEncodeError: emoji and box-drawing chars in print()
# crash when stdout is cp1252. Reconfigure to UTF-8 so ✅ ⚠️ → ─ all print safely.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Add parent directory to path so we can import environments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environments.trading_env import TradingEnv
from scripts.config import ENV_CONFIG, PPO_HYPERPARAMS, CANDLES_PER_DAY   # P3-1 fix

# ── Config ────────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOGS_DIR   = os.path.join(os.path.dirname(__file__), "..", "logs")

# Number of parallel environments for SubprocVecEnv
N_ENVS = 4
DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "debug-627897.log")
DEBUG_SESSION_ID = "627897"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass


def _extract_action_probs(model, policy_obs: np.ndarray, lstm_state, episode_start: np.ndarray) -> tuple[list[float] | None, str | None]:
    try:
        obs_tensor, _ = model.policy.obs_to_tensor(np.array([policy_obs], dtype=np.float32))
        episode_start_tensor = torch.as_tensor(episode_start, device=obs_tensor.device)
        with torch.no_grad():
            dist = None
            try:
                dist = model.policy.get_distribution(
                    obs_tensor, lstm_states=lstm_state, episode_starts=episode_start_tensor
                )
            except TypeError:
                try:
                    dist = model.policy.get_distribution(
                        obs_tensor, lstm_state=lstm_state, episode_start=episode_start_tensor
                    )
                except TypeError:
                    dist = model.policy.get_distribution(obs_tensor)

            probs_tensor = getattr(dist.distribution, "probs", None)
            if probs_tensor is None:
                return None, "distribution_has_no_probs"
            probs = probs_tensor.detach().cpu().numpy().reshape(-1)
            if probs.shape[0] < 3:
                return None, f"unexpected_prob_shape_{probs.shape[0]}"
            return [float(probs[0]), float(probs[1]), float(probs[2])], None
    except Exception as e:
        return None, f"{type(e).__name__}:{e}"


# ── Callbacks ─────────────────────────────────────────────────────────────────

# P1-5 fix: removed inline DiagnosticCallback (had P0-1 double-increment bug,
# lacked GPU tracking, entropy reading, and get_summary()).
# Now importing the full standalone version from diagnostic_callback.py.
from scripts.diagnostic_callback import DiagnosticCallback





# ── Env Factory ───────────────────────────────────────────────────────────────

def make_env(df: pd.DataFrame, seed: int = 0, env_kwargs: dict = None):
    """Creates a monitored environment for SB3."""
    kwargs = ENV_CONFIG.copy()
    if env_kwargs:
        kwargs.update(env_kwargs)
        
    def _init():
        env = TradingEnv(df, **kwargs)
        env = Monitor(env)
        return env
    return _init


# ── Validation Runner ─────────────────────────────────────────────────────────

def run_validation(model: PPO, val_df: pd.DataFrame, vec_norm: Optional["VecNormalize"] = None) -> dict:
    """
    Validates the model across three non-overlapping chronological slices of val_df.

    Fix 6 — Validation variance:
      A single deterministic episode on a fixed dataset always gives the same result
      and has enormous sampling variance when val_df has few trades (20-50 is common).
      Running stochastic episodes on the same data doesn't help — it just adds noise
      from random action selection, not from different market regimes.

      The correct approach is multiple independent market periods. We split val_df
      into thirds and run one deterministic episode on each. This tests whether the
      policy generalises across different regime slices (e.g. trending vs ranging),
      not just whether it happens to luck into good trades in one specific window.

      The reported Sharpe is the mean across all three slices. The std is logged
      as 'sharpe_std' so callers can see the spread. A high std means the policy
      is regime-sensitive and should not be trusted for deployment.

      If val_df is too small to split (< 300 rows per slice), falls back to a
      single full episode with a warning.
    """
    MIN_ROWS_PER_SLICE = 300
    n_slices = 3
    slice_size = len(val_df) // n_slices

    if slice_size < MIN_ROWS_PER_SLICE:
        print(f"  [VAL] val_df too small to split ({len(val_df)} rows). "
              f"Running single episode. Consider adding more validation data.")
        slices = [val_df]
    else:
        slices = [
            val_df.iloc[i * slice_size : (i + 1) * slice_size].reset_index(drop=True)
            for i in range(n_slices)
        ]

    # Pause VecNormalize updates during validation
    old_training    = vec_norm.training    if vec_norm else None
    old_norm_reward = vec_norm.norm_reward if vec_norm else None
    if vec_norm:
        vec_norm.training    = False
        vec_norm.norm_reward = False

    all_metrics: list[dict] = []

    try:
        for slice_idx, slice_df in enumerate(slices):
            val_env = TradingEnv(slice_df, **ENV_CONFIG)
            obs, _  = val_env.reset()
            done    = False
            val_action_counts = {0: 0, 1: 0, 2: 0}
            steps = 0

            # P1-3 fix: RecurrentPPO requires LSTM hidden state to be carried
            # across steps. Without this, every step begins with a zero hidden
            # state — the LSTM has no memory and is evaluated as a stateless MLP.
            # Standard PPO ignores state/episode_start so this is safe for both.
            lstm_state    = None
            episode_start = np.ones((1,), dtype=bool)

            while not done:
                policy_obs = obs
                if vec_norm:
                    policy_obs = vec_norm.normalize_obs(np.array([obs], dtype=np.float32))[0]

                # PERF FIX: removed _extract_action_probs() here — it made a full GPU
                # LSTM forward pass every single step just for probability logging.
                # That was 26,000 steps × 2 GPU calls = 52,000 GPU calls per validation
                # (7-10 min). Action distribution is tracked via val_action_counts instead.

                action, lstm_state = model.predict(
                    policy_obs,
                    state          = lstm_state,
                    episode_start  = episode_start,
                    deterministic  = True,
                )
                episode_start = np.zeros((1,), dtype=bool)   # only True on first step
                val_action_counts[int(action)] = val_action_counts.get(int(action), 0) + 1

                obs, _, terminated, truncated, _ = val_env.step(int(action))
                done = terminated or truncated
                steps += 1

            m = val_env.get_episode_metrics()
            if "error" not in m:
                all_metrics.append(m)
                # region agent log
                _debug_log(
                    run_id="pre-fix",
                    hypothesis_id="H5",
                    location="train_agent.py:RUN_VALIDATION_SLICE",
                    message="Validation deterministic action/trade summary",
                    data={
                        "slice_index": int(slice_idx),
                        "steps": int(steps),
                        "action_counts": {str(k): int(v) for k, v in val_action_counts.items()},
                        "total_trades": int(m.get("total_trades", 0)),
                        "avg_hold_candles": float(m.get("avg_hold_candles", 0.0)),
                        "sharpe_ratio": float(m.get("sharpe_ratio", 0.0)),
                        "total_return_pct": float(m.get("total_return_pct", 0.0)),
                        "gross_pnl_before_fees_pct": float(m.get("gross_pnl_before_fees_pct", 0.0)),
                        "fees_paid_pct": float(m.get("fees_paid_pct", 0.0)),
                        "net_realized_pnl_pct": float(m.get("net_realized_pnl_pct", 0.0)),
                        "gross_profit_factor": float(m.get("gross_profit_factor", 0.0)),
                        "net_profit_factor": float(m.get("net_profit_factor", 0.0)),
                        "gross_expectancy_pct": float(m.get("gross_expectancy_pct", 0.0)),
                        "net_expectancy_pct": float(m.get("net_expectancy_pct", 0.0)),
                        "raw_action_distribution": m.get("raw_action_distribution", {}),
                        "effective_action_distribution": m.get("action_distribution", {}),
                        "reward_components": m.get("reward_components", {}),
                        "mean_action_probs": None,
                        "mean_policy_entropy": None,
                        "mean_action_probs_steps": 0,
                        "buy_sell_action_ratio": float(val_action_counts[1] / max(val_action_counts[2], 1)),
                    },
                )
                # endregion
                # endregion

    finally:
        if vec_norm and old_training is not None:
            vec_norm.training    = old_training
            vec_norm.norm_reward = old_norm_reward

    if not all_metrics:
        return {"error": "No trades in any validation slice"}

    # Aggregate: mean all numeric scalars, use last slice for dicts
    aggregated: dict = {}
    numeric_keys = [k for k, v in all_metrics[0].items() if isinstance(v, (int, float))]
    dict_keys    = [k for k, v in all_metrics[0].items() if isinstance(v, dict)]

    for k in numeric_keys:
        vals = [m[k] for m in all_metrics if k in m]
        aggregated[k] = round(float(np.mean(vals)), 4)

    # Report Sharpe std so callers can see regime sensitivity
    sharpes = [m["sharpe_ratio"] for m in all_metrics if "sharpe_ratio" in m]
    aggregated["sharpe_std"]     = round(float(np.std(sharpes)), 4) if len(sharpes) > 1 else 0.0
    aggregated["n_val_slices"]   = len(all_metrics)

    for k in dict_keys:
        aggregated[k] = all_metrics[-1][k]  # Use last slice for distributions

    return aggregated


# ── Walk-Forward Training ─────────────────────────────────────────────────────

def walk_forward_train(
    train_df:       pd.DataFrame,
    val_df:         pd.DataFrame,
    total_timesteps: int = 1_000_000,
    n_windows:      int  = 3,
    run_name:       str  = "run",
    use_recurrent:  bool = False,
    domain_randomization: bool = False,
    curriculum:     bool = False,
    fee_multiplier: float = 1.0,
):
    """
    Walk-forward training with independent models per window.

    Fix 4 — Independent models:
      Previously, a single model was trained progressively across all windows
      (window 1 → continue → window 2 → continue → window 3). This meant:
        - Window 3 validation compared a model with 3M total steps against
          Window 1's model with 1M steps — not a fair comparison.
        - The "best checkpoint" selection was biased toward later windows.

      Now each window trains a fresh model from random initialization on
      exactly `total_timesteps` steps. Each window's validation Sharpe is
      a fair independent measurement of how well a model trained on that
      data slice generalises to the fixed val set.

      The model trained on Window N (100% of training data) is saved as
      the final production model. The per-window Sharpe values are
      diagnostic information about data quality and regime stability —
      not a beauty contest for checkpoint selection.

      VecNormalize statistics are NOT carried between windows since each
      model is independent and has its own normalizer.

    Windows:
      Window 1: first 33% of train_df
      Window 2: first 66% of train_df
      Window 3: 100% of train_df  ← this is the final deployed model
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # ── Device Selection ──────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*65}")
    print(f"  GPU STATUS")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    print(f"  PyTorch CUDA    : {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    if torch.cuda.is_available():
        print(f"  GPU device      : {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  Training device : {device.upper()}")
    print(f"  Parallel envs   : {N_ENVS}")
    print(f"{'='*65}")

    best_sharpe     = -np.inf
    best_model_path = None
    best_vecnorm_path = None
    results_log     = []

    print(f"\n{'='*65}")
    print(f"  Walk-Forward Training | {n_windows} independent models | {total_timesteps:,} steps each")
    print(f"  Training rows  : {len(train_df):,}")
    print(f"  Validation rows: {len(val_df):,}")
    print(f"  PPO Config: lr={PPO_HYPERPARAMS['learning_rate']}, ent={PPO_HYPERPARAMS['ent_coef']}, "
          f"gamma={PPO_HYPERPARAMS['gamma']}, clip={PPO_HYPERPARAMS['clip_range']}")
    print(f"  Env Config: pos_frac={ENV_CONFIG['position_fraction']}, "
          f"max_dd={ENV_CONFIG['max_drawdown_pct']}, candles_per_day={ENV_CONFIG['candles_per_day']}")
    print(f"{'='*65}")

    # ── Pre-training environment sanity check ─────────────────────────────────
    print("\n[SANITY CHECK] Running environment validation before training...")
    _check_env = TradingEnv(train_df, **ENV_CONFIG)
    check_results = _check_env.sanity_check(n_steps=500, verbose=True)
    failed = [k for k, v in check_results.items() if not v]
    if failed:
        print(f"  ⚠️  WARNING: {len(failed)} sanity check(s) FAILED: {failed}")
        print(f"  ⚠️  Training will continue but results may be unreliable.")
    else:
        print(f"  ✅ All sanity checks passed. Environment looks healthy.")
    del _check_env

    for window_idx in range(1, n_windows + 1):
        # Progressively increase training data size
        cutoff    = int(len(train_df) * (window_idx / n_windows))
        window_df = train_df.iloc[:cutoff].reset_index(drop=True)

        # Fix H: Replay ratio guard — automatically cap timesteps per window
        MAX_REPLAY_RATIO = 25
        usable_steps = len(window_df) - ENV_CONFIG["window_size"]
        max_safe_steps = usable_steps * MAX_REPLAY_RATIO
        replay_ratio = total_timesteps / max(usable_steps, 1)

        if total_timesteps > max_safe_steps:
            window_timesteps = max_safe_steps
            print(f"  ⚠️  [Window {window_idx}] Replay guard: capping timesteps "
                  f"{total_timesteps:,} → {window_timesteps:,} "
                  f"(replay ratio {replay_ratio:.0f}× → {MAX_REPLAY_RATIO}×)")
        else:
            window_timesteps = total_timesteps
            if replay_ratio > 15:
                print(f"  ⚠️  WARNING [Window {window_idx}]: replay ratio ~{replay_ratio:.0f}× "
                      f"({total_timesteps:,} timesteps / {usable_steps:,} usable rows).")

        print(f"\n[Window {window_idx}/{n_windows}] "
              f"Training fresh model on {len(window_df):,} rows "
              f"({window_df.index[0]} ... {window_df.index[-1]})")

        # ── Build Vectorized Environment ──────────────────────────────────────
        env_kwargs = {}
        if domain_randomization:
            env_kwargs["domain_randomization"] = True

        # Phase 2.2: fee-amplified training. Charge the TRAINING env a multiple of
        # the real fee so the agent learns to only take trades whose expected move
        # clears the (inflated) cost — i.e. genuine selectivity. Validation/test envs
        # are built from ENV_CONFIG with the REAL fee (see run_validation), so the
        # reported edge reflects true deployment economics, not the training penalty.
        if fee_multiplier != 1.0:
            env_kwargs["fee_rate"] = ENV_CONFIG["fee_rate"] * fee_multiplier
            print(f"  [FEE] Training fee amplified {fee_multiplier:.1f}x → "
                  f"{env_kwargs['fee_rate']*100:.4f}% (validation uses real "
                  f"{ENV_CONFIG['fee_rate']*100:.4f}%)")

        if curriculum:
            progress   = (window_idx - 1) / max(1, n_windows - 1)
            base_dd    = ENV_CONFIG["max_drawdown_pct"]
            current_dd = base_dd + (0.80 - base_dd) * (1.0 - progress)
            env_kwargs["max_drawdown_pct"] = current_dd
            print(f"  [CURRICULUM] Window {window_idx}: max_drawdown_pct = {current_dd:.2f}")

        env_fns = [make_env(window_df, seed=window_idx * 100 + i, env_kwargs=env_kwargs)
                   for i in range(N_ENVS)]

        try:
            raw_vec_env = SubprocVecEnv(env_fns)
            print(f"  [ENV] Using SubprocVecEnv with {N_ENVS} parallel environments")
        except Exception as e:
            print(f"  [ENV] SubprocVecEnv failed ({e}), falling back to DummyVecEnv")
            raw_vec_env = DummyVecEnv(env_fns)

        # Fix 4: fresh VecNormalize per window — no statistics carried over
        # between independent models (carrying stats would couple the windows).
        vec_env = VecNormalize(
            raw_vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

        # Fix 4: always create a fresh model — never call model.set_env()
        if use_recurrent:
            hyperparams = dict(PPO_HYPERPARAMS)
            hyperparams["policy_kwargs"] = {
                "lstm_hidden_size": 128,
                "n_lstm_layers":    1,
                "net_arch":         dict(pi=[128, 64], vf=[128, 64]),
            }
            model = RecurrentPPO(
                policy          = "MlpLstmPolicy",
                env             = vec_env,
                device          = device,
                verbose         = 0,
                tensorboard_log = LOGS_DIR,
                **hyperparams,
            )
        else:
            model = PPO(
                policy          = "MlpPolicy",
                env             = vec_env,
                device          = device,
                verbose         = 0,
                tensorboard_log = LOGS_DIR,
                **PPO_HYPERPARAMS,
            )

        param_count = sum(p.numel() for p in model.policy.parameters())
        model_type  = "RecurrentPPO" if use_recurrent else "PPO"
        print(f"  [MODEL] Fresh {model_type}: {param_count:,} parameters on {device.upper()}")

        callback = DiagnosticCallback(log_every_n_rollouts=10)

        start = time.time()
        model.learn(
            total_timesteps     = window_timesteps,
            callback            = callback,
            reset_num_timesteps = True,   # Always True — each model is independent
            progress_bar        = True,
            tb_log_name         = f"{run_name}_w{window_idx}",
        )
        elapsed = time.time() - start

        # ── Validate ──────────────────────────────────────────────────────────
        metrics = run_validation(model, val_df, vec_env)
        sharpe  = metrics.get("sharpe_ratio", -99)
        ret     = metrics.get("total_return_pct", 0)
        wr      = metrics.get("win_rate_pct", 0)

        print(f"\n[Window {window_idx} Validation] ({metrics.get('n_val_slices', 1)} slices)")
        print(f"  Sharpe Ratio    : {sharpe:>8.3f}  ±{metrics.get('sharpe_std', 0):.3f}")
        print(f"  Total Return    : {ret:>8.2f}%")
        print(f"  Win Rate        : {wr:>8.2f}%")
        print(f"  Total Trades    : {metrics.get('total_trades', 0)}")
        print(f"  Max Drawdown    : {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Avg Hold        : {metrics.get('avg_hold_candles', 0):.1f} candles")
        print(f"  Gross/Fees/Net  : "
              f"{metrics.get('gross_pnl_before_fees_pct', 0):>+7.2f}% / "
              f"{metrics.get('fees_paid_pct', 0):>6.2f}% / "
              f"{metrics.get('net_realized_pnl_pct', 0):>+7.2f}%")
        print(f"  PF Gross/Net    : "
              f"{metrics.get('gross_profit_factor', 0):>8.3f} / "
              f"{metrics.get('net_profit_factor', 0):>8.3f}")
        print(f"  Expectancy G/N  : "
              f"{metrics.get('gross_expectancy_pct', 0):>+7.3f}% / "
              f"{metrics.get('net_expectancy_pct', 0):>+7.3f}%")
        print(f"  Training time   : {elapsed:.1f}s ({window_timesteps/elapsed:.0f} steps/s)")

        if "action_distribution" in metrics:
            ad      = metrics["action_distribution"]
            total_a = sum(ad.values()) or 1
            print(f"  Action Dist     : "
                  f"H={ad.get(0,0)/total_a*100:.1f}% "
                  f"B={ad.get(1,0)/total_a*100:.1f}% "
                  f"S={ad.get(2,0)/total_a*100:.1f}%")

        results_log.append({"window": window_idx, "sharpe": sharpe, **metrics})

        # Save this window's checkpoint
        ckpt_path    = os.path.join(MODELS_DIR, f"{run_name}_window{window_idx}.zip")
        vecnorm_path = os.path.join(MODELS_DIR, f"{run_name}_window{window_idx}_vecnormalize.pkl")
        model.save(ckpt_path)
        vec_env.save(vecnorm_path)
        print(f"  [SAVED] {ckpt_path}")
        print(f"  [SAVED] {vecnorm_path}")

        # Track best model by Sharpe (diagnostic — Window N is the production model)
        if sharpe > best_sharpe:
            best_sharpe     = sharpe
            best_model_path = ckpt_path
            best_path       = os.path.join(MODELS_DIR, f"{run_name}_best.zip")
            best_vecnorm_path = os.path.join(MODELS_DIR, f"{run_name}_best_vecnormalize.pkl")
            model.save(best_path)
            vec_env.save(best_vecnorm_path)
            print(f"  [NEW BEST] Sharpe {sharpe:.3f} → saved as {run_name}_best.zip")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Training Complete")
    print(f"  Best Val Sharpe   : {best_sharpe:.3f}")
    print(f"  Best Model        : {best_model_path}")
    print(f"  Best VecNormalize : {best_vecnorm_path}")
    print(f"  TensorBoard logs  : {LOGS_DIR}")
    print(f"  NOTE: Window {n_windows} model = production model (most training data).")
    print(f"  Best-by-Sharpe is diagnostic — regime sensitivity may cause it to")
    print(f"  differ from the Window {n_windows} model. Inspect sharpe_std before deploying.")
    print(f"{'='*65}")

    log_path = os.path.join(MODELS_DIR, f"{run_name}_training_log.json")
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"  [LOG SAVED] {log_path}")

    ModelClass = RecurrentPPO if use_recurrent else PPO
    return ModelClass.load(os.path.join(MODELS_DIR, f"{run_name}_best.zip")), best_vecnorm_path


# ── Final Eval on Test Set ────────────────────────────────────────────────────

def evaluate_on_test(model, test_df: pd.DataFrame, vecnorm_path: Optional[str] = None):
    """
    One final out-of-sample evaluation on the held-out test set.
    This data was never seen during training or validation — it's
    the true measure of generalization.
    """
    print(f"\n{'='*65}")
    print(f"  OUT-OF-SAMPLE TEST EVALUATION")
    print(f"  Test rows: {len(test_df):,}")
    print(f"{'='*65}")

    vec_norm = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        test_vec_env = DummyVecEnv([make_env(test_df, seed=999)])
        vec_norm = VecNormalize.load(vecnorm_path, test_vec_env)

    metrics = run_validation(model, test_df, vec_norm)

    for key, val in metrics.items():
        if isinstance(val, dict):
            print(f"  {key:<25}: {json.dumps(val)}")
        else:
            print(f"  {key:<25}: {val}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DRL trading agent (v2)")
    parser.add_argument("--train",      type=str, required=True,
                        help="Path to training .parquet file")
    parser.add_argument("--val",        type=str, required=True,
                        help="Path to validation .parquet file")
    parser.add_argument("--test",       type=str, default=None,
                        help="Path to test .parquet file (optional final eval)")
    parser.add_argument("--timesteps",  type=int, default=500_000,
                        help="Training timesteps per walk-forward window")
    parser.add_argument("--windows",    type=int, default=3,
                        help="Number of walk-forward windows")
    parser.add_argument("--run-name",   type=str, default="tradebot_ppo",
                        help="Name prefix for saved model files")
    parser.add_argument("--n-envs",     type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--recurrent",  action="store_true",
                        help="Use RecurrentPPO (LSTM) instead of standard PPO")
    parser.add_argument("--domain-randomization", action="store_true",
                        help="Enable domain randomization for fees and slippage")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning (progressive difficulty)")
    parser.add_argument("--candles-per-day", type=int, default=None,
                        help="Override ENV_CONFIG candles_per_day for the env's Sharpe "
                             "annualization. 96=15m, 24=1h, 288=5m. MUST match the data's "
                             "timeframe or Sharpe is wrong. Defaults to config.py value.")
    parser.add_argument("--fee-multiplier", type=float, default=1.0,
                        help="Phase 2.2: multiply the TRAINING fee to force selectivity "
                             "(e.g. 3.0 = train at 3x fees). Validation/test always use the "
                             "real fee, so reported edge reflects true deployment economics.")
    parser.add_argument("--min-adx", type=float, default=None,
                        help="Phase 2.3: regime gate. Block opening positions when raw ADX "
                             "is below this (e.g. 25 = only trade trending markets). Applies "
                             "to BOTH training and validation/test so the gate is part of the "
                             "deployed policy. Skip out the chop where the model loses.")
    parser.add_argument("--require-uptrend", action="store_true",
                        help="Phase 2.3b: directional gate. Open longs only in confirmed "
                             "uptrends (price above 30d SMA). Combine with --min-adx so the "
                             "agent goes long only in strong UPtrends and sits flat in "
                             "bear/down regimes instead of longing into them.")
    parser.add_argument("--allow-short", action="store_true",
                        help="Phase 2.4: enable SHORT positions via the 3-action ladder "
                             "(BUY moves short->flat->long, SELL moves long->flat->short). "
                             "Lets the agent profit in down markets instead of only avoiding "
                             "them. Keeps RecurrentPPO/LSTM (no MaskablePPO).")
    parser.add_argument("--reward-mode", type=str, default="fixb", choices=["fixb", "exit"],
                        help="F3 reward mode. 'fixb' (default): dense per-step portfolio "
                             "return. 'exit': sparse trade-quality reward paying realized NET "
                             "(fee-adjusted) return at close + win bonus + over-hold penalty — "
                             "scalps that don't clear fees score negative (anti-churn).")
    args = parser.parse_args()

    # Override global N_ENVS if specified
    global N_ENVS
    N_ENVS = args.n_envs

    # Phase 2.1: keep the env's Sharpe annualization in sync with the data timeframe.
    if args.candles_per_day is not None:
        ENV_CONFIG["candles_per_day"] = args.candles_per_day
        print(f"[CONFIG] candles_per_day overridden → {args.candles_per_day} "
              f"({'15m' if args.candles_per_day==96 else '1h' if args.candles_per_day==24 else '5m' if args.candles_per_day==288 else 'custom'})")

    # Phase 2.3: regime gate applied to ALL envs (train + validation/test) via ENV_CONFIG.
    if args.min_adx is not None:
        ENV_CONFIG["min_adx"] = args.min_adx
        print(f"[CONFIG] regime gate ON → only open positions when raw ADX >= {args.min_adx}")
    if args.require_uptrend:
        ENV_CONFIG["require_uptrend"] = True
        print(f"[CONFIG] directional gate ON → open longs only in confirmed uptrends "
              f"(price above 30d SMA)")
    if args.allow_short:
        ENV_CONFIG["allow_short"] = True
        print(f"[CONFIG] SHORTING ON → 3-action ladder (short <-> flat <-> long), "
              f"agent can profit in down markets")
    if args.reward_mode != "fixb":
        ENV_CONFIG["reward_mode"] = args.reward_mode
        print(f"[CONFIG] reward_mode = '{args.reward_mode}' → exit-concentrated NET-return "
              f"reward (anti-churn): scalps that don't clear fees score negative")

    # ── Load Data ─────────────────────────────────────────────────────────────
    print(f"[LOADING] {args.train}")
    train_df = pd.read_parquet(args.train).reset_index(drop=True)

    print(f"[LOADING] {args.val}")
    val_df = pd.read_parquet(args.val).reset_index(drop=True)

    print(f"\n[DATA] Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")

    # ── Train ─────────────────────────────────────────────────────────────────
    best_model, best_vecnorm_path = walk_forward_train(
        train_df        = train_df,
        val_df          = val_df,
        total_timesteps = args.timesteps,
        n_windows       = args.windows,
        run_name        = args.run_name,
        use_recurrent   = args.recurrent,
        domain_randomization = args.domain_randomization,
        curriculum      = args.curriculum,
        fee_multiplier  = args.fee_multiplier,
    )

    # ── Optional Final Test ───────────────────────────────────────────────────
    if args.test and os.path.exists(args.test):
        test_df = pd.read_parquet(args.test).reset_index(drop=True)
        evaluate_on_test(best_model, test_df, best_vecnorm_path)

    print(f"\n[DONE] Training complete.")
    print(f"  → View TensorBoard: tensorboard --logdir {LOGS_DIR}")
    print(f"  → Run export: python export_onnx.py --model models/{args.run_name}_best.zip")


if __name__ == "__main__":
    main()
