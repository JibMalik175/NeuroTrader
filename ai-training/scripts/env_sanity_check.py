"""
env_sanity_check.py
───────────────────
Standalone diagnostic script that validates the TradingEnv is correctly
configured before you spend GPU hours on a training run.

Performs seven categories of checks:
  1. Random-policy episodes   — reward / length / action distribution
  2. Observation space        — NaN, Inf, shape, and bounds validation
  3. Reward distribution      — histogram, magnitude, and flatline checks
  4. Buy-and-hold baseline    — naive benchmark for comparison
  5. Episode completion       — early termination ratio
  6. Transition sanity        — position / balance state machine checks
  7. Summary verdict          — PASS / WARN / FAIL roll-up

Usage:
    python scripts/env_sanity_check.py --data data/BTC_USDT_1h_train.parquet
    python scripts/env_sanity_check.py --data data/BTC_USDT_1h_train.parquet --episodes 10
"""

import sys
import os
import argparse
import warnings

import numpy as np
import pandas as pd

# Add parent directory so we can import environments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from environments.trading_env import TradingEnv
# P1-2/P3-1 fix: import shared config instead of maintaining a local copy.
# candles_per_day was missing here → Sharpe annualization was wrong for non-15m data.
from scripts.config import ENV_CONFIG, CANDLES_PER_DAY

# Suppress pandas copy warnings during checks
warnings.filterwarnings("ignore", category=FutureWarning)


# ── Formatting Helpers ────────────────────────────────────────────────────────

SEPARATOR = "─" * 68
PASS_ICON = "✅"
WARN_ICON = "⚠️"
FAIL_ICON = "❌"

_warnings_collected: list[str] = []
_failures_collected: list[str] = []


def _header(title: str) -> None:
    """Print a section header."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def _ok(msg: str) -> None:
    print(f"  {PASS_ICON}  {msg}")


def _warn(msg: str) -> None:
    _warnings_collected.append(msg)
    print(f"  {WARN_ICON}  {msg}")


def _fail(msg: str) -> None:
    _failures_collected.append(msg)
    print(f"  {FAIL_ICON}  {msg}")


def _stat(label: str, value) -> None:
    print(f"  {label:<28}: {value}")


# ── Environment Config ───────────────────────────────────────────────────────

# Matches the config in train_agent.py so the sanity check reflects
# the exact same environment the agent will train on.
# ENV_CONFIG imported from scripts.config (P3-1 fix — single source of truth)


# ── 1. Random-Policy Episodes ────────────────────────────────────────────────

def check_random_episodes(df: pd.DataFrame, n_episodes: int = 5) -> None:
    """
    Run N episodes with a uniform random policy and report aggregate stats.
    This establishes a baseline: if the random agent consistently gets
    the same reward, the reward signal may be degenerate.
    """
    _header("1. Random-Policy Episodes")

    rewards_per_ep:  list[float] = []
    lengths_per_ep:  list[int]   = []
    action_totals    = np.zeros(3, dtype=np.int64)

    for ep in range(n_episodes):
        env = TradingEnv(df, **ENV_CONFIG)
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_len    = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_len    += 1
            action_totals[action] += 1
            done = terminated or truncated

        rewards_per_ep.append(ep_reward)
        lengths_per_ep.append(ep_len)

    rewards_arr = np.array(rewards_per_ep)
    lengths_arr = np.array(lengths_per_ep)

    _stat("Episodes run", n_episodes)
    _stat("Mean episode reward", f"{rewards_arr.mean():.2f}  (std: {rewards_arr.std():.2f})")
    _stat("Mean episode length", f"{lengths_arr.mean():.0f}  (std: {lengths_arr.std():.0f})")

    total = action_totals.sum()
    pcts  = action_totals / total * 100 if total > 0 else np.zeros(3)
    _stat("Action distribution",
          f"HOLD={pcts[0]:.1f}%  BUY={pcts[1]:.1f}%  SELL={pcts[2]:.1f}%")

    # All 3 actions should occur with a random policy
    zero_actions = [name for i, name in enumerate(["HOLD", "BUY", "SELL"]) if action_totals[i] == 0]
    if zero_actions:
        _fail(f"Actions never taken: {', '.join(zero_actions)} — action space may be broken")
    else:
        _ok("All 3 actions produced non-zero counts")


# ── 2. Observation Space ─────────────────────────────────────────────────────

def check_observation_space(df: pd.DataFrame) -> None:
    """
    Validates that observations are well-formed: correct shape, no NaN/Inf,
    and values within reasonable bounds.  Bad observations are the #1 cause
    of silent training failures in RL.
    """
    _header("2. Observation Space Validation")

    env = TradingEnv(df, **ENV_CONFIG)
    obs, _ = env.reset()
    expected_shape = env.observation_space.shape

    # Collect observations over a short episode
    all_obs: list[np.ndarray] = [obs]
    for _ in range(min(500, len(df) - ENV_CONFIG["window_size"] - 1)):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        all_obs.append(obs)
        if terminated or truncated:
            break

    obs_stack = np.array(all_obs)

    # Shape check
    if obs.shape == expected_shape:
        _ok(f"Observation shape matches: {expected_shape}")
    else:
        _fail(f"Shape mismatch: got {obs.shape}, expected {expected_shape}")

    # NaN / Inf check
    nan_count = np.isnan(obs_stack).sum()
    inf_count = np.isinf(obs_stack).sum()

    if nan_count == 0:
        _ok("No NaN values in observations")
    else:
        _fail(f"Found {nan_count:,} NaN values across {len(all_obs)} observations")

    if inf_count == 0:
        _ok("No Inf values in observations")
    else:
        _fail(f"Found {inf_count:,} Inf values across {len(all_obs)} observations")

    # Bounds check
    obs_max = np.abs(obs_stack).max()
    obs_mean = np.abs(obs_stack).mean()
    _stat("Max |obs| value", f"{obs_max:.4f}")
    _stat("Mean |obs| value", f"{obs_mean:.4f}")

    if obs_max > 100.0:
        _warn(f"Observation values exceed 100 (max={obs_max:.2f}) — "
              "consider normalization or VecNormalize")
    else:
        _ok("All observation values within [-100, 100]")


# ── 3. Reward Distribution ───────────────────────────────────────────────────

def check_reward_distribution(df: pd.DataFrame) -> None:
    """
    Collects all per-step rewards over one episode and analyzes
    the distribution.  Catches common bugs: broken scaling, flatline
    rewards, or extreme magnitude.
    """
    _header("3. Reward Distribution")

    env = TradingEnv(df, **ENV_CONFIG)
    obs, _ = env.reset()
    done = False
    all_rewards: list[float] = []

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        all_rewards.append(reward)
        done = terminated or truncated

    rewards = np.array(all_rewards)

    _stat("Total steps", len(rewards))
    _stat("Mean reward", f"{rewards.mean():.6f}")
    _stat("Std reward", f"{rewards.std():.6f}")
    _stat("Min reward", f"{rewards.min():.4f}")
    _stat("Max reward", f"{rewards.max():.4f}")
    _stat("Median reward", f"{np.median(rewards):.6f}")
    _stat("Non-zero rewards", f"{(rewards != 0).sum()} / {len(rewards)}  "
          f"({(rewards != 0).sum() / len(rewards) * 100:.1f}%)")

    # Text histogram
    print(f"\n  Reward Histogram:")
    _print_histogram(rewards, bins=12, width=40)

    # Magnitude check
    max_mag = np.abs(rewards).max()
    if max_mag > 10.0:
        _warn(f"Reward magnitude exceeds 10.0 (max |r|={max_mag:.2f}) — "
              "suggests broken scaling or extreme penalty")
    else:
        _ok(f"Reward magnitude reasonable (max |r|={max_mag:.4f})")

    # Flatline check
    if rewards.std() < 1e-10:
        _warn("Reward std ≈ 0 — rewards appear to be constant (flatline)")
    else:
        _ok(f"Reward variance present (std={rewards.std():.6f})")


def _print_histogram(data: np.ndarray, bins: int = 12, width: int = 40) -> None:
    """Render a compact text-based histogram to the console."""
    if len(data) == 0:
        print("    (no data)")
        return

    counts, edges = np.histogram(data, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1

    for i, count in enumerate(counts):
        lo, hi   = edges[i], edges[i + 1]
        bar_len  = int(count / max_count * width)
        bar      = "█" * bar_len
        pct      = count / len(data) * 100
        print(f"    [{lo:>9.3f}, {hi:>9.3f}) │{bar:<{width}} {count:>5} ({pct:4.1f}%)")


# ── 4. Buy-and-Hold Baseline ─────────────────────────────────────────────────

def check_buy_and_hold(df: pd.DataFrame) -> None:
    """
    Simulates a naive buy-at-open, hold-to-close strategy.
    The RL agent should ideally beat this baseline;
    if not, the learned policy may be worse than random.
    """
    _header("4. Buy-and-Hold Baseline")

    window = ENV_CONFIG["window_size"]
    if window >= len(df):
        _warn("Not enough data for buy-and-hold calculation")
        return

    first_price = float(df["close"].iloc[window])
    last_price  = float(df["close"].iloc[-1])

    # Account for fees (buy + sell)
    fee_rate    = ENV_CONFIG["fee_rate"]
    slippage    = ENV_CONFIG["slippage_pct"]
    buy_fill    = first_price * (1 + slippage)
    sell_fill   = last_price * (1 - slippage)
    net_return  = (sell_fill / buy_fill - 1) * 100
    # Subtract fees on initial balance (buy + sell)
    fee_drag    = (1 - fee_rate) * (1 - fee_rate)
    net_return  = (sell_fill / buy_fill * fee_drag - 1) * 100

    _stat("Entry price", f"{first_price:,.2f}")
    _stat("Exit price", f"{last_price:,.2f}")
    _stat("Gross return", f"{(last_price / first_price - 1) * 100:+.2f}%")
    _stat("Net return (w/ fees)", f"{net_return:+.2f}%")
    _stat("Holding period", f"{len(df) - window} candles")

    _ok(f"Buy-and-hold baseline: {net_return:+.2f}% — "
        "RL agent should aim to beat this")


# ── 5. Episode Completion Test ────────────────────────────────────────────────

def check_episode_completion(df: pd.DataFrame, n_episodes: int = 10) -> None:
    """
    Runs multiple episodes and categorizes how they ended:
    early termination (drawdown) vs. natural completion (reached end of data).
    If most episodes die early, the agent has too little training signal.
    """
    _header("5. Episode Completion Test")

    max_possible_steps = len(df) - ENV_CONFIG["window_size"]
    early_count = 0
    full_count  = 0
    lengths     = []

    for _ in range(n_episodes):
        env  = TradingEnv(df, **ENV_CONFIG)
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            done = terminated or truncated

        lengths.append(steps)
        # If the episode ran roughly to the end of data, it's "full"
        if steps >= max_possible_steps - 1:
            full_count += 1
        else:
            early_count += 1

    early_pct = early_count / n_episodes * 100

    _stat("Episodes run", n_episodes)
    _stat("Full completion", f"{full_count}  ({full_count / n_episodes * 100:.0f}%)")
    _stat("Early termination", f"{early_count}  ({early_pct:.0f}%)")
    _stat("Mean episode length", f"{np.mean(lengths):.0f} / {max_possible_steps} max steps")

    if early_pct > 80:
        _warn(f"{early_pct:.0f}% of episodes terminate early — "
              "drawdown threshold may be too tight or initial risk too high")
    else:
        _ok(f"Early termination rate ({early_pct:.0f}%) is acceptable")


# ── 6. Transition Sanity ─────────────────────────────────────────────────────

def check_transitions(df: pd.DataFrame) -> None:
    """
    Verifies the state-machine invariants of the trading environment:
      - BUY flips position_held from False → True
      - SELL flips position_held from True  → False
      - Balance changes on trade execution

    These are the most common sources of environment bugs.
    """
    _header("6. Transition Sanity Checks")

    env = TradingEnv(df, **ENV_CONFIG)
    obs, _ = env.reset()

    # ── BUY transition ───────────────────────────────────────────────────────
    assert not env.position_held, "Expected flat position after reset"
    balance_before = env.balance

    obs, reward, terminated, truncated, info = env.step(1)  # BUY

    if env.position_held:
        _ok("BUY changes position_held: False → True")
    else:
        _fail("BUY did NOT change position_held to True")

    if env.balance != balance_before:
        _ok(f"BUY changes balance: {balance_before:.2f} → {env.balance:.2f} "
            f"(fee deducted: {balance_before - env.balance:.2f})")
    else:
        _warn("Balance unchanged after BUY — fees may not be applied")

    # ── SELL transition ──────────────────────────────────────────────────────
    if env.position_held and not (terminated or truncated):
        balance_before_sell = env.balance

        obs, reward, terminated, truncated, info = env.step(2)  # SELL

        if not env.position_held:
            _ok("SELL changes position_held: True → False")
        else:
            _fail("SELL did NOT change position_held to False")

        if env.balance != balance_before_sell:
            _ok(f"SELL changes balance: {balance_before_sell:.2f} → {env.balance:.2f}")
        else:
            # Balance could stay the same if PnL exactly cancels fees, which
            # is astronomically unlikely but not impossible
            _warn("Balance unchanged after SELL — may be coincidental")
    else:
        _warn("Could not test SELL transition (episode ended after BUY)")

    # ── Invalid action penalties ─────────────────────────────────────────────
    if not (terminated or truncated):
        # Double SELL (no position) → should get a penalty
        obs, reward_sell_flat, _, _, info_sell_flat = env.step(2)  # SELL while flat
        if info_sell_flat.get("invalid_remap") and info_sell_flat.get("action") == 0 and reward_sell_flat == 0:
            _ok("Invalid SELL while flat remapped to HOLD with zero penalty")
        else:
            _warn("No penalty for SELL while flat — invalid actions not penalized")

        # BUY then BUY again → should get a penalty
        env.step(1)  # BUY to get into position
        if env.position_held:
            obs, reward_double_buy, _, _, info_double_buy = env.step(1)  # BUY again
            if info_double_buy.get("invalid_remap") and info_double_buy.get("action") == 0:
                _ok(f"Invalid double-BUY remapped to HOLD; reward={reward_double_buy:.5f}")
            else:
                _warn("No penalty for double-BUY — invalid actions not penalized")


# ── 7. Summary ───────────────────────────────────────────────────────────────

def print_summary() -> None:
    """Roll up all warnings and failures into a final verdict."""
    _header("Summary")

    n_warn = len(_warnings_collected)
    n_fail = len(_failures_collected)

    if n_fail > 0:
        print(f"\n  {FAIL_ICON}  RESULT: {n_fail} FAILURE(S), {n_warn} WARNING(S)\n")
        for msg in _failures_collected:
            print(f"    {FAIL_ICON}  {msg}")
    elif n_warn > 0:
        print(f"\n  {WARN_ICON}  RESULT: {n_warn} WARNING(S), 0 FAILURES\n")
    else:
        print(f"\n  {PASS_ICON}  ALL CHECKS PASSED — environment looks healthy!\n")

    for msg in _warnings_collected:
        print(f"    {WARN_ICON}  {msg}")

    if n_fail == 0 and n_warn == 0:
        print("    No issues detected. Safe to proceed with training.")

    print()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate TradingEnv before training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python scripts/env_sanity_check.py "
               "--data data/BTC_USDT_1h_train.parquet --episodes 10",
    )
    parser.add_argument(
        "--data", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "BTC_USDT_1h_train.parquet"),
        help="Path to a .parquet data file (default: ../data/BTC_USDT_1h_train.parquet)",
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of random-policy episodes to run (default: 5)",
    )
    parser.add_argument(
        "--timeframe", type=str, default="15m",
        choices=list(CANDLES_PER_DAY.keys()),
        help="Candle timeframe — sets correct Sharpe annualization (P1-2 fix, default: 15m)",
    )
    args = parser.parse_args()

    # P1-2 fix: inject candles_per_day into ENV_CONFIG based on --timeframe
    import copy
    ENV_CONFIG_LOCAL = copy.deepcopy(ENV_CONFIG)
    ENV_CONFIG_LOCAL["candles_per_day"] = CANDLES_PER_DAY[args.timeframe]
    # Patch module-level ENV_CONFIG so all check functions pick it up
    import scripts.env_sanity_check as _self
    _self.ENV_CONFIG = ENV_CONFIG_LOCAL

    # ── Load Data ────────────────────────────────────────────────────────────
    data_path = args.data
    if not os.path.exists(data_path):
        print(f"\n{FAIL_ICON}  Data file not found: {data_path}")
        print("   Use --data to specify the path to your .parquet file.")
        sys.exit(1)

    print(f"\n{'=' * 68}")
    print(f"  TradingEnv Sanity Check")
    print(f"  Data: {os.path.abspath(data_path)}")
    print(f"{'=' * 68}")

    df = pd.read_parquet(data_path).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows  |  {len(df.columns)} columns")

    # ── Run All Checks ───────────────────────────────────────────────────────
    check_random_episodes(df, n_episodes=args.episodes)
    check_observation_space(df)
    check_reward_distribution(df)
    check_buy_and_hold(df)
    check_episode_completion(df, n_episodes=10)
    check_transitions(df)
    print_summary()


if __name__ == "__main__":
    main()
