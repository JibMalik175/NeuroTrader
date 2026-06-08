"""
diagnostic_callback.py
──────────────────────
Comprehensive Stable-Baselines3 training callback that surfaces the
diagnostics you actually need when debugging RL trading agents.

What it tracks (per rollout window):
  1. Action distribution   — % of HOLD / BUY / SELL with collapse alerts
  2. Policy entropy         — read from SB3's logger or estimated from actions
  3. Episode reward stats   — mean, min, max from Monitor wrapper
  4. Episode length stats   — mean episode length
  5. Reward component breakdown — aggregated from info["reward_components"]
  6. GPU memory usage       — torch.cuda.memory_allocated (if available)
  7. Training throughput    — steps per second

Usage:
    from scripts.diagnostic_callback import DiagnosticCallback

    callback = DiagnosticCallback(log_every_n_rollouts=10)
    model.learn(total_timesteps=500_000, callback=callback)

    # After training, inspect all tracked metrics:
    summary = callback.get_summary()
"""

import time
from collections import defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

ACTION_NAMES  = {0: "H", 1: "B", 2: "S"}
COLLAPSE_THRESHOLD = 0.90   # single action > 90% → policy collapse alert


# ── Callback ──────────────────────────────────────────────────────────────────

class DiagnosticCallback(BaseCallback):
    """
    Diagnostic training callback for RL trading agents.

    Logs detailed per-rollout diagnostics to the console in a compact,
    aligned format.  Detects policy collapse (one action dominates) and
    tracks reward component breakdowns when the environment provides them.

    Parameters
    ----------
    log_every_n_rollouts : int
        Print a diagnostic line every N rollout completions.  Lower values
        give more visibility but noisier output; 10 is a good default for
        n_steps=4096.
    verbose : int
        Verbosity level passed to ``BaseCallback``.
    """

    def __init__(self, log_every_n_rollouts: int = 10, verbose: int = 1):
        super().__init__(verbose)
        self.log_every = log_every_n_rollouts

        # ── Counters (reset every log interval) ──────────────────────────────
        self.rollout_count   = 0
        self.action_counts   = np.zeros(3, dtype=np.int64)   # H / B / S
        self.ep_rewards: list[float]  = []
        self.ep_lengths: list[int]    = []
        self.reward_components: dict[str, list[float]] = defaultdict(list)

        # ── Persistent history (never reset, used by get_summary) ────────────
        self._history_rewards:    list[float] = []
        self._history_lengths:    list[int]   = []
        self._history_entropy:    list[float] = []
        self._history_action_pct: list[dict]  = []
        self._history_gpu_mb:     list[float] = []
        self._history_throughput: list[float] = []
        self._collapse_count     = 0

        # ── Timing ───────────────────────────────────────────────────────────
        self._interval_start_time  = None
        self._interval_start_steps = 0

    # ── SB3 hooks ─────────────────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        """Initialize timing on the very first training start."""
        self._interval_start_time  = time.perf_counter()
        self._interval_start_steps = 0

    def _on_step(self) -> bool:
        """
        Called after every environment step.  Accumulates per-step
        metrics (actions, episode completions, reward components).
        """
        # ── Action tracking ──────────────────────────────────────────────────
        actions = self.locals.get("actions")
        if actions is not None:
            for a in np.asarray(actions).flatten():
                if 0 <= a <= 2:
                    self.action_counts[a] += 1

        # ── Episode info from Monitor wrapper ────────────────────────────────
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])

            # ── Reward component breakdown ───────────────────────────────────
            rc = info.get("reward_components")
            if rc and isinstance(rc, dict):
                for key, val in rc.items():
                    self.reward_components[key].append(float(val))

        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of every rollout collection.  Every
        ``log_every_n_rollouts`` rollouts, prints a formatted
        diagnostic line and resets interval counters.
        """
        self.rollout_count += 1

        if self.rollout_count % self.log_every != 0:
            return

        # ── Action distribution ──────────────────────────────────────────────
        total_actions = self.action_counts.sum()
        if total_actions > 0:
            pcts = self.action_counts / total_actions * 100.0
        else:
            pcts = np.zeros(3)

        action_str = (
            f"H={pcts[0]:4.1f}% B={pcts[1]:4.1f}% S={pcts[2]:4.1f}%"
        )

        # Policy collapse detection
        collapsed = any(p > COLLAPSE_THRESHOLD * 100 for p in pcts)
        health_icon = "⚠️" if collapsed else "✅"
        if collapsed:
            self._collapse_count += 1

        # ── Entropy ──────────────────────────────────────────────────────────
        entropy = self._read_entropy()
        # P2-6 fix: entropy_loss is negative; closer to 0 = less exploration
        if entropy is not None:
            entropy_str = f"{entropy:.3f}" + (" ⚠️collapse?" if entropy > -0.1 else "")
        else:
            entropy_str = "N/A"

        # ── Episode reward stats ─────────────────────────────────────────────
        if self.ep_rewards:
            mean_r = np.mean(self.ep_rewards)
            min_r  = np.min(self.ep_rewards)
            max_r  = np.max(self.ep_rewards)
            reward_str = f"{mean_r:>8.2f} (min: {min_r:.1f} max: {max_r:.1f})"
        else:
            mean_r = 0.0
            reward_str = "     N/A (no episodes completed)"

        # ── Episode length stats ─────────────────────────────────────────────
        if self.ep_lengths:
            mean_len = np.mean(self.ep_lengths)
            len_str  = f"{mean_len:>5.0f}"
        else:
            mean_len = 0
            len_str  = "  N/A"

        # ── GPU memory ───────────────────────────────────────────────────────
        gpu_mb = self._read_gpu_mb()
        gpu_str = f"{gpu_mb:.0f}MB" if gpu_mb is not None else "N/A"

        # ── Throughput ───────────────────────────────────────────────────────
        now          = time.perf_counter()
        elapsed      = now - self._interval_start_time if self._interval_start_time else 1.0
        steps_delta  = self.num_timesteps - self._interval_start_steps
        throughput   = steps_delta / max(elapsed, 1e-6)
        tp_str       = f"{throughput:,.0f} it/s"

        # ── Print ────────────────────────────────────────────────────────────
        line = (
            f"[Rollout {self.rollout_count:>4}] "
            f"Reward: {reward_str} | "
            f"EpLen: {len_str} | "
            f"Actions: {action_str} {health_icon} | "
            f"Entropy: {entropy_str} | "
            f"GPU: {gpu_str} | "
            f"{tp_str} | "
            f"Steps: {self.num_timesteps:>10,}"
        )
        print(f"\n{line}")

        # ── Reward component breakdown (if available) ────────────────────────
        if self.reward_components:
            parts = []
            for key, vals in sorted(self.reward_components.items()):
                parts.append(f"{key}={np.mean(vals):+.4f}")
            comp_str = " | ".join(parts)
            print(f"           Components: {comp_str}")

        # ── Persist to history ───────────────────────────────────────────────
        self._history_rewards.extend(self.ep_rewards)
        self._history_lengths.extend(self.ep_lengths)
        if entropy is not None:
            self._history_entropy.append(entropy)
        self._history_action_pct.append({
            "H": float(pcts[0]), "B": float(pcts[1]), "S": float(pcts[2]),
        })
        if gpu_mb is not None:
            self._history_gpu_mb.append(gpu_mb)
        self._history_throughput.append(throughput)

        # ── Reset interval counters ──────────────────────────────────────────
        self.action_counts[:]      = 0
        self.ep_rewards.clear()
        self.ep_lengths.clear()
        self.reward_components.clear()
        self._interval_start_time  = now
        self._interval_start_steps = self.num_timesteps

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _read_entropy(self) -> float | None:
        """
        Attempts to read the policy entropy from the SB3 logger.
        Falls back to ``None`` if the key is not yet populated.
        """
        try:
            name_to_value = self.model.logger.name_to_value
            # P2-6 fix: removed abs() — entropy_loss is negative in SB3 (it's the
            # negative entropy used in the PPO loss). Taking abs() masked the sign,
            # making a collapsing policy (entropy → 0) look like an exploring one.
            # Raw value: large negative = high exploration, near 0 = policy collapse.
            for key in ("train/entropy_loss", "train/entropy"):
                if key in name_to_value:
                    return float(name_to_value[key])
        except (AttributeError, TypeError):
            pass
        return None

    def _read_gpu_mb(self) -> float | None:
        """Returns current GPU memory allocated in MB, or None if no GPU."""
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """
        Returns a dict summarizing all diagnostics collected across the
        entire training run.  Useful for post-training analysis.

        Returns
        -------
        dict
            Keys include:
            - total_rollouts:       int
            - total_timesteps:      int
            - collapse_alerts:      int
            - reward_mean / reward_min / reward_max:  float
            - episode_length_mean:  float
            - entropy_mean / entropy_final: float | None
            - action_pct_mean:      dict  {"H": %, "B": %, "S": %}
            - gpu_mb_mean:          float | None
            - throughput_mean:      float
        """
        r = np.array(self._history_rewards) if self._history_rewards else np.array([0.0])
        l = np.array(self._history_lengths) if self._history_lengths else np.array([0])

        # Aggregate action percentages across all log intervals
        if self._history_action_pct:
            mean_pct = {
                k: float(np.mean([d[k] for d in self._history_action_pct]))
                for k in ("H", "B", "S")
            }
        else:
            mean_pct = {"H": 0.0, "B": 0.0, "S": 0.0}

        return {
            "total_rollouts":      self.rollout_count,
            "total_timesteps":     self.num_timesteps,
            "collapse_alerts":     self._collapse_count,
            "reward_mean":         float(r.mean()),
            "reward_min":          float(r.min()),
            "reward_max":          float(r.max()),
            "reward_std":          float(r.std()),
            "episode_length_mean": float(l.mean()),
            "entropy_mean":        float(np.mean(self._history_entropy)) if self._history_entropy else None,
            "entropy_final":       self._history_entropy[-1] if self._history_entropy else None,
            "action_pct_mean":     mean_pct,
            "gpu_mb_mean":         float(np.mean(self._history_gpu_mb)) if self._history_gpu_mb else None,
            "throughput_mean":     float(np.mean(self._history_throughput)) if self._history_throughput else 0.0,
        }
