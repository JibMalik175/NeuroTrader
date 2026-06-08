"""
trading_env.py
──────────────
Custom Gymnasium (OpenAI Gym) trading environment for Deep Reinforcement Learning.

The agent interacts with simulated historical market data and learns to
maximize risk-adjusted returns. Every aspect of real-world trading is
simulated: exchange fees, slippage, position sizing, and drawdown penalties.

Observation space: sliding window of N candles × M features + 7 portfolio state
Action space:      Discrete(3)  →  0=HOLD, 1=BUY, 2=SELL

Key design decisions (v3 — post-overhaul):
  - Continuous mark-to-market reward (dense signal at every step)
  - Fractional position sizing (default 20% of balance per trade)
  - Graduated drawdown penalty (proportional, not terminal)
  - No hold penalty (holding winning positions is GOOD)
  - Tiny invalid-action penalty (prevents collapse to single action)
  - No reward_scaling (let VecNormalize handle normalization)
  - Reward decomposition in info dict for debugging

Fixes applied in v3 (overhaul):
  Fix 1 — Sharpe ratio: computed on step-level equity curve returns, not per-trade PnL.
           Annualization factor derived from candles_per_day param (default 96 for 15m),
           never hardcoded, so changing timeframe cannot silently break the metric.
  Fix 2 — Reward double-counting removed: SELL now gives only the incremental
           mark-to-market reward for its final step (same as HOLD), plus a tiny
           winner-exit bonus (+0.001) so the agent has a reason to close profitable trades.
           The cumulative trade reward is the natural sum of all step rewards.
  Fix 4 — position_cost_basis tracking: entry cash amount stored explicitly so the
           SELL fee calculation uses the correct denominator regardless of interim
           balance changes.
  Fix 7 — Corrected SELL fee math: fee charged on gross exit proceeds
           (position value at exit price), not on pre-PnL balance.
           balance update is now order-independent.
  Fix 8 — Hold-winner bonus removed: continuous mark-to-market is sufficient incentive.
           Bonus was causing fixed-duration holding behavior (duration farming).

Phase 1 fixes (carried forward):
  - _get_observation uses _active_features (not hardcoded FEATURE_COLS)
  - _get_info uses explicit None checks instead of "or"
  - sanity_check() for pre-training environment validation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional


# ── Constants ─────────────────────────────────────────────────────────────────

class Action(IntEnum):
    HOLD = 0
    BUY  = 1
    SELL = 2


@dataclass
class TradeRecord:
    entry_step:        int
    entry_price:       float
    position_cost:     float           # Cash allocated at entry (after entry fee)
    entry_fee:         float           = 0.0
    exit_step:         Optional[int]   = None
    exit_price:        Optional[float] = None
    pnl_pct:           Optional[float] = None
    held_steps:        Optional[int]   = None
    exit_fee:          float           = 0.0
    gross_pnl:         float           = 0.0
    net_pnl:           float           = 0.0
    fees_paid:         float           = 0.0


# ── Environment ───────────────────────────────────────────────────────────────

class TradingEnv(gym.Env):
    """
    Single-asset crypto trading environment (v3).

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame from feature_engineering.py.
        Must contain OHLCV columns + all feature columns.
    window_size : int
        Number of past candles the agent observes at each step.
    initial_balance : float
        Starting virtual capital (in USDT).
    fee_rate : float
        Taker fee per trade (Binance default: 0.001 = 0.1%).
    slippage_pct : float
        Simulated slippage on market orders (0.0005 = 0.05%).
    max_drawdown_pct : float
        Episode terminates early if drawdown exceeds this threshold.
    reward_scaling : float
        Global multiplier on reward signal. Set to 1.0 and let
        VecNormalize handle scaling. Kept for backward compatibility.
    position_fraction : float
        Fraction of balance committed per trade (0.0–1.0).
        Default 0.20 = 20% per trade (prevents single-trade blowups).
    candles_per_day : int
        Number of candles per calendar day. Used ONLY for Sharpe annualization.
        96 = 15m candles, 24 = 1h candles, 288 = 5m candles.
        Must be set to match your actual dataset timeframe or Sharpe is wrong.
    """

    metadata = {"render_modes": ["human"]}
    DEBUG_LOG_PATH = "debug-627897.log"
    DEBUG_SESSION_ID = "627897"

    FEATURE_COLS = [
        "log_return", "log_return_h", "log_return_l", "log_return_v",
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "candle_direction",
        "rsi",
        "macd", "macd_signal", "macd_hist",
        "bb_position", "bb_width",
        "atr_ratio",
        "ema_cross_short", "ema_cross_long",
        "volume_ratio",
        # Phase 2 features (v2)
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "adx",
        "obv_ratio",
    ]

    # v1 feature set for backward compatibility with old parquet files
    FEATURE_COLS_V1 = [
        "log_return", "log_return_h", "log_return_l", "log_return_v",
        "body_ratio", "upper_wick_ratio", "lower_wick_ratio", "candle_direction",
        "rsi",
        "macd", "macd_signal", "macd_hist",
        "bb_position", "bb_width",
        "atr_ratio",
        "ema_cross_short", "ema_cross_long",
        "volume_ratio",
    ]

    FEATURE_COLS_V2 = FEATURE_COLS

    FEATURE_COLS_V3 = FEATURE_COLS + [
        "4h_rsi", "4h_macd", "1d_rsi", "1d_macd"
    ]

    # Fix E: V4 features — macro regime context
    FEATURE_COLS_V4 = FEATURE_COLS_V3 + [
        "dist_from_high", "macro_trend_sma", "macro_volatility", "macro_obv_ratio"
    ]

    # Number of portfolio state features appended to the observation
    N_PORTFOLIO_FEATURES = 7

    def __init__(
        self,
        df:                 pd.DataFrame,
        window_size:        int   = 48,
        initial_balance:    float = 10_000.0,
        fee_rate:           float = 0.001,
        slippage_pct:       float = 0.0005,
        max_drawdown_pct:   float = 0.50,
        reward_scaling:     float = 1.0,
        position_fraction:  float = 0.20,
        domain_randomization: bool = False,
        candles_per_day:    int   = 96,      # Fix 1: explicit timeframe param for Sharpe
    ):
        super().__init__()

        self.df              = df.reset_index(drop=True)
        self.window_size     = window_size
        self.initial_balance = initial_balance
        self.fee_rate        = fee_rate
        self.slippage_pct    = slippage_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.reward_scaling  = reward_scaling
        self.position_fraction = np.clip(position_fraction, 0.01, 1.0)
        self.domain_randomization = domain_randomization
        # Fix 1: store candles_per_day for use in _compute_sharpe.
        # Never hardcode this — pass it from ENV_CONFIG to stay in sync with data.
        self.candles_per_day = candles_per_day

        # Validate required columns — auto-detect v1 vs v2 vs v3 vs v4 features
        if all(col in self.df.columns for col in self.FEATURE_COLS_V4):
            self._active_features = self.FEATURE_COLS_V4
        elif all(col in self.df.columns for col in self.FEATURE_COLS_V3):
            self._active_features = self.FEATURE_COLS_V3
        elif all(col in self.df.columns for col in self.FEATURE_COLS_V2):
            self._active_features = self.FEATURE_COLS_V2
        else:
            # Fall back to v1 features for old parquet files
            self._active_features = self.FEATURE_COLS_V1
            for col in self._active_features + ["close", "high", "low", "atr_ratio"]:
                assert col in self.df.columns, f"Missing column: {col}"

        n_features = len(self._active_features)

        # ── Spaces ────────────────────────────────────────────────────────────
        # Observation: [window_size × n_features] + [N_PORTFOLIO_FEATURES]
        #   Portfolio state: position_held, unrealized_pnl, drawdown,
        #                    steps_in_position, portfolio_return,
        #                    position_size, steps_since_last_trade
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * n_features + self.N_PORTFOLIO_FEATURES,),
            dtype=np.float32,
        )

        # Actions: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Pre-compute close prices as numpy array for fast access
        self._close_prices = self.df["close"].values.astype(np.float64)

        # ── PERF (Phase 0.1): precompute the feature matrix ONCE ──────────────
        # The old _get_observation did self.df[self._active_features].iloc[a:b]
        # .values + nan_to_num on EVERY step (~604us/call, 59% of step time).
        # Precomputing a contiguous float32 numpy matrix turns the per-step cost
        # into a pure slice (~1.6us/call) — a ~377x speedup on the hottest path.
        # nan_to_num here is equivalent to the per-slice cleaning the old code did
        # (the parquet is already dropna'd, so in practice there are no NaN/Inf).
        self._feature_matrix = np.nan_to_num(
            self.df[self._active_features].values.astype(np.float32),
            nan=0.0,
        )

        self._reset_state()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_state(self):
        self.current_step        = self.window_size
        self.balance             = self.initial_balance
        self.peak_balance        = self.initial_balance
        self.position_held       = False
        self.entry_price         = 0.0
        self.position_size       = 0.0      # Fraction of balance in position
        # Fix 7: track exact cash allocated at entry so SELL fee math is correct
        self.position_cost_basis = 0.0      # Cash actually put into position (post entry-fee)
        self.trade_history: list[TradeRecord] = []
        self._episode_pnl        = 0.0
        self._step_count         = 0
        self.max_drawdown_seen   = 0.0
        self._prev_portfolio     = self.initial_balance
        self._steps_in_position  = 0
        self._steps_since_trade  = 0
        # Fix 1: equity curve recorded every step for proper Sharpe calculation
        self._equity_curve: list[float] = [self.initial_balance]
        self._gross_pnl          = 0.0
        self._total_fees_paid    = 0.0
        self._entry_fees_paid    = 0.0
        self._exit_fees_paid     = 0.0

        # Domain Randomization for robustness against exchange variations
        if getattr(self, "domain_randomization", False):
            # Randomize fees by +/- 20%
            self.current_fee_rate = self.fee_rate * np.random.uniform(0.8, 1.2)
            # Randomize slippage by +/- 50%
            self.current_slippage = self.slippage_pct * np.random.uniform(0.5, 1.5)
        else:
            self.current_fee_rate = self.fee_rate
            self.current_slippage = self.slippage_pct

        # Diagnostic counters (reset per episode)
        self._raw_action_counts = {0: 0, 1: 0, 2: 0}
        self._action_counts   = {0: 0, 1: 0, 2: 0}
        self._reward_components = {
            # ── FIX-B: Active reward signal ───────────────────────────────────
            "portfolio_return":      0.0,   # The ONLY reward: (pv_now - pv_prev) / initial
            # ── Legacy components: tracked for diagnostics, NOT added to reward ─
            "mark_to_market":       0.0,
            "fee_cost":             0.0,
            "invalid_action":       0.0,
            "invalid_action_count": 0.0,
            "drawdown_penalty":     0.0,
            "terminal_penalty":     0.0,
            "winner_exit":          0.0,
            "early_exit_penalty":   0.0,
            "loss_duration_penalty": 0.0,
        }
        # Priority 2: per-step reward magnitude trackers
        self._reward_abs_sums = {
            "mtm": 0.0, "fee": 0.0, "invalid": 0.0,
            "drawdown": 0.0, "sell_total": 0.0,
        }
        self._reward_abs_counts = {
            "mtm": 0, "fee": 0, "invalid": 0,
            "drawdown": 0, "sell_total": 0,
        }
        # Priority 1: sell decision log (capped per episode)
        self._sell_log = []
        self._max_sell_logs = 50
        self._debug_flags = {
            "entry_logged": False,
            "exit_logged": False,
            "invalid_logged": False,
            "hold_logged": False,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        current_price = self._get_close_price(self.current_step)
        reward        = 0.0
        terminated    = False
        truncated     = False
        reward_tag    = "none"

        raw_action = Action(action)
        self._raw_action_counts[int(raw_action)] = self._raw_action_counts.get(int(raw_action), 0) + 1
        invalid_remap = False

        # Experiment A: remap invalid actions to HOLD instead of penalizing them.
        if raw_action == Action.BUY and self.position_held:
            action = Action.HOLD
            reward_tag = "invalid_to_hold"
            invalid_remap = True
            self._reward_components["invalid_action_count"] += 1
        elif raw_action == Action.SELL and not self.position_held:
            action = Action.HOLD
            reward_tag = "invalid_to_hold"
            invalid_remap = True
            self._reward_components["invalid_action_count"] += 1
        else:
            action = raw_action

        # ── Experiment E: Hard Stop-Loss ──────────────────────────────────────
        # Force a SELL if the position drops 2.5% below entry.
        # This replaces soft penalties and forces the agent to learn better entries
        # while mathematically guaranteeing we cannot hold giant losers.
        if action == Action.HOLD and self.position_held:
            unrealized = (current_price - self.entry_price) / (self.entry_price + 1e-8)
            if unrealized <= -0.025:
                action = Action.SELL
                reward_tag = "hard_stop_loss"
                self._reward_components["stop_loss_count"] = self._reward_components.get("stop_loss_count", 0) + 1

        # Track effective action after remapping/stop-loss
        self._action_counts[int(action)] = self._action_counts.get(int(action), 0) + 1

        # ── Execute Action ────────────────────────────────────────────────────

        if action == Action.BUY:
            if not self.position_held:
                # Apply slippage: we always get a slightly worse fill
                fill_price = current_price * (1 + self.current_slippage)
                # Entry fee deducted from balance before position is sized
                cash_allocated   = self.balance * self.position_fraction
                entry_fee        = cash_allocated * self.current_fee_rate
                self.balance    -= entry_fee
                self._total_fees_paid += entry_fee
                self._entry_fees_paid += entry_fee
                # Fix 7: record exact post-fee cash basis so SELL math is order-independent
                self.position_cost_basis = cash_allocated - entry_fee
                self.entry_price         = fill_price
                self.position_held       = True
                self.position_size       = self.position_fraction
                self._steps_in_position  = 0
                self._steps_since_trade  = 0
                self.trade_history.append(
                    TradeRecord(
                        entry_step=self.current_step,
                        entry_price=fill_price,
                        position_cost=self.position_cost_basis,
                        entry_fee=entry_fee,
                    )
                )
                # FIX-B: fee tracked diagnostically — portfolio return captures the
                # balance drop naturally (portfolio_value falls by entry_fee at BUY).
                fee_component = -(entry_fee / self.initial_balance)
                reward_tag    = "fee_cost"
                self._reward_components["fee_cost"] += fee_component
                reward = 0.0  # portfolio return (computed below) replaces this
            else:
                # NOTE: This branch is unreachable — invalid BUY-while-holding is
                # remapped to HOLD at the top of step() before we get here.
                # Kept as a safety fallback only.
                reward = 0.0
                reward_tag = "invalid_to_hold"

        elif action == Action.SELL:
            if self.position_held:
                fill_price = current_price * (1 - self.current_slippage)

                # ── Fix 7: Correct fee math ───────────────────────────────────
                # Fee is charged on gross exit proceeds (notional value at exit),
                # not on pre-PnL balance. This matches real exchange mechanics.
                pnl_pct        = (fill_price - self.entry_price) / self.entry_price
                gross_proceeds = self.position_cost_basis * (1 + pnl_pct)
                exit_fee       = gross_proceeds * self.current_fee_rate
                net_proceeds   = gross_proceeds - exit_fee
                rec            = self.trade_history[-1]
                gross_pnl      = gross_proceeds - self.position_cost_basis
                trade_fees     = rec.entry_fee + exit_fee
                net_pnl        = gross_pnl - trade_fees

                # Update balance: remove the allocated capital, add back net proceeds
                self.balance   = (self.balance - self.position_cost_basis) + net_proceeds
                self._gross_pnl       += gross_pnl
                self._total_fees_paid += exit_fee
                self._exit_fees_paid  += exit_fee

                # Log trade
                rec.exit_step   = self.current_step
                rec.exit_price  = fill_price
                rec.pnl_pct     = pnl_pct
                rec.held_steps  = self.current_step - rec.entry_step
                rec.exit_fee    = exit_fee
                rec.gross_pnl   = gross_pnl
                rec.net_pnl     = net_pnl
                rec.fees_paid   = trade_fees
                held_steps      = rec.held_steps

                # Fix A: store position_size BEFORE zeroing for reward calc
                old_position_size = self.position_size

                self.position_held       = False
                self.entry_price         = 0.0
                self.position_size       = 0.0
                self.position_cost_basis = 0.0
                self._steps_since_trade  = 0

                # FIX-B: All SELL components tracked diagnostically — NOT added to reward.
                # Portfolio return (computed below) naturally captures:
                #   • Final-step price move (MTM)
                #   • Exit fee (balance drops by exit_fee)
                #   • Net PnL from the trade
                # No need for separate fee_signal, early_exit_penalty, or winner_exit.
                prev_price       = self._get_close_price(self.current_step - 1)
                price_change_pct = (fill_price - prev_price) / (prev_price + 1e-8)
                sell_mtm         = price_change_pct * old_position_size
                reward_tag       = "mark_to_market"
                self._reward_components["mark_to_market"] += sell_mtm

                fee_signal = -(exit_fee / self.initial_balance)
                self._reward_components["fee_cost"] += fee_signal

                # Tracked for diagnostics — not applied
                sell_early_pen = 0.0
                if held_steps < 5:
                    sell_early_pen = -0.0003
                    self._reward_components["early_exit_penalty"] += sell_early_pen

                sell_winner = 0.0
                if pnl_pct > 0 and held_steps >= 5:
                    MIN_HOLD   = 5
                    MAX_HOLD   = 20
                    hold_scale = min(1.0, (held_steps - MIN_HOLD) / (MAX_HOLD - MIN_HOLD))
                    sell_winner = 0.001 * hold_scale
                    self._reward_components["winner_exit"] += sell_winner

                reward = 0.0  # portfolio return (computed below) replaces all SELL rewards

                # Priority 1: Log every SELL decision
                # FIX-B: total_sell_reward is now net_pnl/initial_balance (portfolio impact)
                trade_portfolio_return = net_pnl / self.initial_balance
                self._reward_abs_sums["sell_total"] += abs(trade_portfolio_return)
                self._reward_abs_counts["sell_total"] += 1
                if len(self._sell_log) < self._max_sell_logs:
                    self._sell_log.append({
                        "step": int(self.current_step),
                        "entry_price": float(rec.entry_price),
                        "fill_price": float(fill_price),
                        "pnl_pct": float(pnl_pct),
                        "held_steps": int(held_steps),
                        "sell_mtm": float(sell_mtm),
                        "fee_signal": float(fee_signal),
                        "early_pen": float(sell_early_pen),
                        "winner_bonus": float(sell_winner),
                        "total_sell_reward": float(trade_portfolio_return),  # FIX-B: net impact
                        "gross_pnl": float(gross_pnl),
                        "fees_paid": float(trade_fees),
                        "net_pnl": float(net_pnl),
                        "was_winner": bool(pnl_pct > 0),
                    })

            else:
                # NOTE: This branch is unreachable — invalid SELL-while-flat is
                # remapped to HOLD at the top of step() before we get here.
                # Kept as a safety fallback only.
                reward = 0.0
                reward_tag = "invalid_to_hold"

        elif action == Action.HOLD:
            if self.position_held:
                # FIX-B: MTM tracked diagnostically — NOT added to reward.
                # Portfolio return (computed below) captures price movement naturally.
                prev_price       = self._get_close_price(self.current_step - 1)
                price_change_pct = (current_price - prev_price) / (prev_price + 1e-8)
                mtm_component    = price_change_pct * self.position_size
                reward_tag       = "mark_to_market"
                self._reward_components["mark_to_market"] += mtm_component
                self._steps_in_position += 1

                # Priority 2: track MTM step magnitude
                self._reward_abs_sums["mtm"] += abs(mtm_component)
                self._reward_abs_counts["mtm"] += 1

                # ── Experiment D removed ──────────────────────────────────────
                # The continuous proportional penalty failed because the agent
                # preferred to absorb the slow bleed rather than take a realized
                # -5% loss. Replaced by a hard -2.5% stop-loss above.

                # region agent log
                if (not self._debug_flags["hold_logged"]) and self._steps_in_position >= 20:
                    self._debug_log(
                        run_id="fix-b",
                        hypothesis_id="H3",
                        location="trading_env.py:HOLD_MTM",
                        message="Long hold mark-to-market snapshot",
                        data={
                            "step": int(self.current_step),
                            "steps_in_position": int(self._steps_in_position),
                            "price_change_pct": float(price_change_pct),
                            "position_size": float(self.position_size),
                            "mtm_component_diagnostic": float(mtm_component),
                        },
                    )
                    self._debug_flags["hold_logged"] = True
                # endregion

                # reward carries any accumulated penalty forward to the portfolio return += below
            else:
                # No penalty for flat HOLD — being cautious is fine
                reward = 0.0
                self._steps_since_trade += 1

        if invalid_remap and not self._debug_flags["invalid_logged"]:
            self._debug_log(
                run_id="experiment-a",
                hypothesis_id="A",
                location="trading_env.py:INVALID_REMAP",
                message="First invalid action remapped to HOLD",
                data={
                    "step": int(self.current_step),
                    "raw_action": int(raw_action),
                    "effective_action": int(action),
                    "invalid_action_count": float(self._reward_components["invalid_action_count"]),
                    "invalid_action_reward": float(self._reward_components["invalid_action"]),
                },
            )
            self._debug_flags["invalid_logged"] = True

        # ── Update Peak & Drawdown ────────────────────────────────────────────
        portfolio_value = self._get_portfolio_value(current_price)
        self.peak_balance = max(self.peak_balance, portfolio_value)
        drawdown = (self.peak_balance - portfolio_value) / (self.peak_balance + 1e-8)
        self.max_drawdown_seen = max(self.max_drawdown_seen, drawdown)

        # Fix 1: append to equity curve every step for Sharpe calculation
        self._equity_curve.append(portfolio_value)

        # ── FIX-B: Drawdown tracked diagnostically — NOT penalized in reward ────
        # The portfolio return already decreases when portfolio drops, providing
        # a natural signal. Adding a separate drawdown penalty double-counts and
        # was the ROOT CAUSE of the 26.3:1 noise:signal ratio.
        # Keeping the tracking for comparison against prior experiments.
        if drawdown > 0.25:
            dd_penalty = (drawdown - 0.25) * 0.002
            # FIX-B: tracked but NOT applied
            self._reward_components["drawdown_penalty"] -= dd_penalty
            self._reward_abs_sums["drawdown"] += abs(dd_penalty)
            self._reward_abs_counts["drawdown"] += 1

        # Terminate on max drawdown breach — termination IS the signal, no extra penalty
        if drawdown > self.max_drawdown_pct:
            # FIX-B: no terminal reward penalty — episode ending carries the consequence
            self._reward_components["terminal_penalty"] -= 0.0
            terminated = True

        # Terminate at end of data
        self.current_step += 1
        self._step_count  += 1
        if self.current_step >= len(self.df):
            terminated = True

        # ── FIX-B + Exp D: Portfolio Return + proportional loss-duration penalty ─
        # reward starts at 0.0 (BUY/SELL/flat HOLD) or -loss_dur_pen (HOLD in loss).
        # We ADD portfolio return so both signals combine cleanly:
        #   Net reward = portfolio_return + (-loss_dur_pen if applicable)
        # This keeps portfolio return as the dominant signal while adding the
        # proportional loss urgency on top (10-20% of MTM magnitude at 5-10% loss).
        portfolio_return = (portfolio_value - self._prev_portfolio) / self.initial_balance
        reward += portfolio_return
        self._reward_components["portfolio_return"] += portfolio_return

        self._prev_portfolio = portfolio_value

        obs  = self._get_observation()
        info = self._get_info(portfolio_value=portfolio_value, drawdown=drawdown,
                              include_economics=terminated)

        info["reward_raw"]        = reward
        info["reward_tag"]        = reward_tag
        info["raw_action"]        = int(raw_action)
        info["invalid_remap"]     = bool(invalid_remap)
        info["action"]            = int(action)
        # PERF (Phase 0.5): these 3 dict copies fed ONLY the DiagnosticCallback,
        # which aggregates per rollout. Emitting them only at episode end removes
        # ~3 dict copies per step per env (the env is no longer the training
        # bottleneck — the LSTM + this per-step info plumbing now are). The callback
        # already skips steps where reward_components is absent, and validation reads
        # metrics from get_episode_metrics() directly, so this is behavior-safe.
        if terminated:
            info["raw_action_counts"] = dict(self._raw_action_counts)
            info["action_counts"]     = dict(self._action_counts)
            info["reward_components"] = dict(self._reward_components)
        # region agent log
        if terminated:
            component_totals = {k: float(v) for k, v in self._reward_components.items()}
            component_for_total = {k: v for k, v in component_totals.items() if not k.endswith("_count")}
            total_reward_components = float(sum(component_for_total.values()))
            component_pct = {}
            denom = abs(total_reward_components)
            for k, v in component_for_total.items():
                component_pct[k] = float((v / denom) * 100.0) if denom > 1e-12 else 0.0
            # Priority 2: compute mean absolute magnitudes
            mean_abs = {}
            for k in self._reward_abs_sums:
                n = self._reward_abs_counts[k]
                mean_abs[k] = float(self._reward_abs_sums[k] / n) if n > 0 else 0.0

            # Priority 1+3: sell summary stats
            # Experiment C additions: Case 1 vs Case 2 diagnostic metrics
            sell_summary = {}
            if self._sell_log:
                sell_rewards = [s["total_sell_reward"] for s in self._sell_log]
                sell_pnls   = [s["pnl_pct"] for s in self._sell_log]
                sell_gross  = [s["gross_pnl"] for s in self._sell_log]
                sell_fees   = [s["fees_paid"] for s in self._sell_log]
                sell_net    = [s["net_pnl"] for s in self._sell_log]
                winners     = [s for s in self._sell_log if s["was_winner"]]
                losers      = [s for s in self._sell_log if not s["was_winner"]]

                winner_pnls  = [s["pnl_pct"] for s in winners]
                loser_pnls   = [s["pnl_pct"] for s in losers]
                winner_holds = [s["held_steps"] for s in winners]
                loser_holds  = [s["held_steps"] for s in losers]
                winner_gross = [s["gross_pnl"] for s in winners]
                loser_gross  = [s["gross_pnl"] for s in losers]

                sell_summary = {
                    # ── Core counts ──────────────────────────────────────────
                    "n_sells":   len(self._sell_log),
                    "n_winners": len(winners),
                    "n_losers":  len(losers),
                    # ── Reward quality ───────────────────────────────────────
                    "mean_sell_reward":   float(np.mean(sell_rewards)),
                    "mean_winner_reward": float(np.mean([s["total_sell_reward"] for s in winners])) if winners else 0.0,
                    "mean_loser_reward":  float(np.mean([s["total_sell_reward"] for s in losers]))  if losers  else 0.0,
                    # ── PnL quality (Case 1 vs 2 diagnostics) ───────────────
                    "mean_sell_pnl_pct":    float(np.mean(sell_pnls)),
                    "mean_winner_pnl_pct":  float(np.mean(winner_pnls))            if winner_pnls else 0.0,
                    "mean_loser_pnl_pct":   float(np.mean(loser_pnls))             if loser_pnls  else 0.0,
                    "median_winner_pnl_pct": float(np.median(winner_pnls))         if winner_pnls else 0.0,
                    "median_loser_pnl_pct":  float(np.median(loser_pnls))          if loser_pnls  else 0.0,
                    "largest_winner_pnl_pct": float(max(winner_pnls))              if winner_pnls else 0.0,
                    "largest_loser_pnl_pct":  float(min(loser_pnls))               if loser_pnls  else 0.0,
                    "largest_winner_gross":   float(max(winner_gross))              if winner_gross else 0.0,
                    "largest_loser_gross":    float(min(loser_gross))               if loser_gross  else 0.0,
                    # ── Hold duration by outcome ──────────────────────────────
                    "mean_hold_duration":    float(np.mean([s["held_steps"] for s in self._sell_log])),
                    "mean_hold_winner":      float(np.mean(winner_holds))           if winner_holds else 0.0,
                    "mean_hold_loser":       float(np.mean(loser_holds))            if loser_holds  else 0.0,
                    "median_hold_winner":    float(np.median(winner_holds))         if winner_holds else 0.0,
                    "median_hold_loser":     float(np.median(loser_holds))          if loser_holds  else 0.0,
                    "max_hold_winner":       int(max(winner_holds))                 if winner_holds else 0,
                    "max_hold_loser":        int(max(loser_holds))                  if loser_holds  else 0,
                    # ── Totals ───────────────────────────────────────────────
                    "gross_pnl_before_fees": float(np.sum(sell_gross)),
                    "fees_paid":             float(np.sum(sell_fees)),
                    "net_pnl":               float(np.sum(sell_net)),
                    # ── Sample trades (expanded for Case 1 vs 2 analysis) ────
                    "first_20_sells": self._sell_log[:20],
                }

            trade_economics = self._get_trade_economics()
            self._debug_log(
                run_id="experiment-a",
                hypothesis_id="H4",
                location="trading_env.py:EPISODE_END",
                message="Episode reward/action summary",
                data={
                    "episode_steps": int(self._step_count),
                    "raw_actions": {str(k): int(v) for k, v in self._raw_action_counts.items()},
                    "actions": {str(k): int(v) for k, v in self._action_counts.items()},
                    "reward_components": component_totals,
                    "total_reward_components": total_reward_components,
                    "reward_component_pct_of_abs_total": component_pct,
                    "mean_abs_reward_per_step": mean_abs,
                    "sell_summary": sell_summary,
                    "trade_economics": trade_economics,
                    "total_trades": int(len(self.trade_history)),
                    "portfolio_value": float(portfolio_value),
                    "drawdown": float(drawdown),
                },
            )
        # endregion

        return obs, reward * self.reward_scaling, terminated, truncated, info

    # ── Observation Builder ───────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the observation vector:
          - Flattened window of past candle features
          - 7 portfolio state features appended at the end:
            1. position_held (0 or 1)
            2. unrealized_pnl (fraction)
            3. drawdown (fraction)
            4. steps_in_position (normalized by window_size)
            5. portfolio_return (fraction from initial)
            6. position_size (fraction)
            7. steps_since_last_trade (normalized by window_size)
        """
        window_start = max(0, self.current_step - self.window_size)
        window_end   = min(len(self.df), self.current_step)

        # PERF (Phase 0.1): pure numpy slice from the precomputed matrix.
        # Equivalent output to the old pandas path, ~377x faster.
        features = self._feature_matrix[window_start:window_end].flatten()

        expected_len = self.window_size * len(self._active_features)
        if len(features) < expected_len:
            features = np.pad(features, (expected_len - len(features), 0))

        current_price   = self._get_close_price(min(self.current_step, len(self.df) - 1))
        portfolio_value = self._get_portfolio_value(current_price)

        pos_held     = float(self.position_held)
        unrealized   = (current_price - self.entry_price) / (self.entry_price + 1e-8) \
                       if self.position_held else 0.0
        drawdown     = (self.peak_balance - portfolio_value) / (self.peak_balance + 1e-8)
        steps_in_pos = self._steps_in_position / self.window_size if self.position_held else 0.0
        port_return  = (portfolio_value / self.initial_balance) - 1.0
        pos_size     = self.position_size
        steps_since  = min(self._steps_since_trade / self.window_size, 5.0)

        portfolio_state = np.array(
            [pos_held, unrealized, drawdown, steps_in_pos, port_return, pos_size, steps_since],
            dtype=np.float32,
        )

        return np.concatenate([features, portfolio_state])

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_close_price(self, step: int) -> float:
        idx = min(max(0, step), len(self._close_prices) - 1)
        return float(self._close_prices[idx])

    def _get_portfolio_value(self, current_price: float) -> float:
        """Total portfolio value accounting for any open fractional position."""
        if not self.position_held:
            return self.balance
        unrealized_pnl_pct = (current_price - self.entry_price) / (self.entry_price + 1e-8)
        return self.balance + self.position_cost_basis * unrealized_pnl_pct

    # region agent log
    def _debug_log(self, run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
        payload = {
            "sessionId": self.DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(pd.Timestamp.utcnow().timestamp() * 1000),
        }
        try:
            with open(self.DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except Exception:
            pass
    # endregion

    def _get_info(self, portfolio_value: float = None, drawdown: float = None,
                  include_economics: bool = False) -> dict:
        cp = self._get_close_price(min(self.current_step, len(self.df) - 1))
        pv = portfolio_value if portfolio_value is not None else self._get_portfolio_value(cp)
        dd = drawdown if drawdown is not None else 0.0
        info = {
            "step":             self.current_step,
            "portfolio_value":  pv,
            "balance":          self.balance,
            "position_held":    self.position_held,
            "drawdown":         dd,
            "total_trades":     len(self.trade_history),
            "total_return_pct": (pv / self.initial_balance - 1) * 100,
        }
        # PERF (Phase 0.2): _get_trade_economics() rebuilds 6+ Python lists over
        # the entire trade history. It was called EVERY step but nothing reads it
        # during training (SB3 reads info["episode"]; the callback reads
        # reward_components). Compute it only when explicitly requested (episode end).
        if include_economics:
            info["trade_economics"] = self._get_trade_economics()
        return info

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        cp  = self._get_close_price(min(self.current_step, len(self.df) - 1))
        pv  = self._get_portfolio_value(cp)
        dd  = (self.peak_balance - pv) / self.peak_balance * 100
        ret = (pv / self.initial_balance - 1) * 100
        pos = "IN TRADE" if self.position_held else "FLAT"
        print(
            f"Step {self.current_step:>5} | Price: {cp:>10,.2f} | "
            f"Portfolio: {pv:>10,.2f} | Return: {ret:>+7.2f}% | "
            f"Drawdown: {dd:>5.2f}% | Position: {pos} | "
            f"Trades: {len(self.trade_history)}"
        )

    # ── Evaluation Metrics ────────────────────────────────────────────────────

    def get_episode_metrics(self) -> dict:
        """
        Call after an episode ends to get comprehensive performance stats.
        Returns the key quantitative finance metrics you'd see in a backtest report.
        """
        closed_trades = [t for t in self.trade_history if t.pnl_pct is not None]
        pnls   = [t.pnl_pct for t in closed_trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        econ = self._get_trade_economics(closed_trades)

        final_price  = self._get_close_price(len(self.df) - 1)
        final_value  = self._get_portfolio_value(final_price)
        total_return = (final_value / self.initial_balance - 1) * 100

        # Fix 1: Sharpe computed from step-level equity curve
        sharpe = self._compute_sharpe_from_equity()

        return {
            "total_return_pct":     round(total_return, 3),
            "total_trades":         len(closed_trades),
            "win_rate_pct":         round(len(wins) / len(closed_trades) * 100, 2) if closed_trades else 0.0,
            "avg_win_pct":          round(np.mean(wins) * 100, 4) if wins else 0.0,
            "avg_loss_pct":         round(np.mean(losses) * 100, 4) if losses else 0.0,
            "profit_factor":        round(abs(sum(wins)) / abs(sum(losses)), 3) if losses else (float("inf") if wins else 0.0),
            "gross_pnl_before_fees_pct": econ["gross_pnl_before_fees_pct"],
            "fees_paid_pct":        econ["fees_paid_pct"],
            "net_realized_pnl_pct": econ["net_realized_pnl_pct"],
            "gross_profit_factor":  econ["gross_profit_factor"],
            "net_profit_factor":    econ["net_profit_factor"],
            "gross_expectancy_pct": econ["gross_expectancy_pct"],
            "net_expectancy_pct":   econ["net_expectancy_pct"],
            "sharpe_ratio":         round(sharpe, 3),
            "max_drawdown_pct":     round(self.max_drawdown_seen * 100, 3),
            "avg_hold_candles":     round(np.mean([t.held_steps for t in closed_trades]), 1) if closed_trades else 0.0,
            "raw_action_distribution": dict(self._raw_action_counts),
            "action_distribution":  dict(self._action_counts),
            "reward_components":    dict(self._reward_components),
        }

    def _get_trade_economics(self, closed_trades: list[TradeRecord] = None) -> dict:
        """Gross/net diagnostics that separate trade edge from fee drag."""
        trades = closed_trades
        if trades is None:
            trades = [t for t in self.trade_history if t.pnl_pct is not None]

        gross_pnls = [t.gross_pnl for t in trades]
        net_pnls = [t.net_pnl for t in trades]
        fees = [t.fees_paid for t in trades]
        gross_wins = [p for p in gross_pnls if p > 0]
        gross_losses = [p for p in gross_pnls if p <= 0]
        net_wins = [p for p in net_pnls if p > 0]
        net_losses = [p for p in net_pnls if p <= 0]
        gross_returns = [t.pnl_pct for t in trades]
        net_returns = [
            t.net_pnl / (t.position_cost + 1e-8)
            for t in trades
            if t.position_cost > 0
        ]

        gross_total = float(sum(gross_pnls))
        fee_total = float(sum(fees))
        net_total = float(sum(net_pnls))
        entry_fee_total = float(sum(t.entry_fee for t in trades))
        exit_fee_total = float(sum(t.exit_fee for t in trades))

        return {
            "gross_pnl_before_fees": round(gross_total, 6),
            "fees_paid": round(fee_total, 6),
            "net_realized_pnl": round(net_total, 6),
            "gross_pnl_before_fees_pct": round(gross_total / self.initial_balance * 100, 4),
            "fees_paid_pct": round(fee_total / self.initial_balance * 100, 4),
            "net_realized_pnl_pct": round(net_total / self.initial_balance * 100, 4),
            "entry_fees_paid": round(entry_fee_total, 6),
            "exit_fees_paid": round(exit_fee_total, 6),
            "gross_profit_factor": round(abs(sum(gross_wins)) / abs(sum(gross_losses)), 3)
                                   if gross_losses else (float("inf") if gross_wins else 0.0),
            "net_profit_factor": round(abs(sum(net_wins)) / abs(sum(net_losses)), 3)
                                 if net_losses else (float("inf") if net_wins else 0.0),
            "gross_expectancy_pct": round(float(np.mean(gross_returns)) * 100, 4)
                                    if gross_returns else 0.0,
            "net_expectancy_pct": round(float(np.mean(net_returns)) * 100, 4)
                                  if net_returns else 0.0,
        }

    def sanity_check(self, n_steps: int = 500, verbose: bool = True) -> dict:
        """
        Pre-training environment sanity checker.

        Runs a short random-policy episode and validates:
          1. No NaN/Inf in observations
          2. Reward magnitude in reasonable range (no 100× scaling)
          3. All 3 actions appear (env is not degenerate)
          4. Episodes don't terminate immediately
          5. Balance / portfolio value remain positive
          6. Obs shape matches declared observation_space

        Returns a dict with pass/fail results for each check.
        """
        results: dict[str, bool] = {}
        rewards, obs_maxs = [], []
        action_seen = set()
        early_terminations = 0
        neg_portfolio = 0
        shape_ok = True

        obs, _ = self.reset()
        for _ in range(n_steps):
            if obs.shape != self.observation_space.shape:
                shape_ok = False
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                obs_maxs.append(np.inf)
            else:
                obs_maxs.append(np.abs(obs).max())

            action = self.action_space.sample()
            action_seen.add(int(action))
            obs, reward, terminated, truncated, info = self.step(action)
            rewards.append(reward)

            pv = info.get("portfolio_value", self.initial_balance)
            if pv <= 0:
                neg_portfolio += 1

            if terminated or truncated:
                early_terminations += 1
                obs, _ = self.reset()

        results["obs_no_nan_inf"]      = not any(np.isinf(v) for v in obs_maxs)
        results["obs_shape_correct"]   = shape_ok
        results["reward_scale_sane"]   = max(abs(r) for r in rewards) < 10.0
        results["all_actions_sampled"] = len(action_seen) == 3
        results["episodes_run_full"]   = early_terminations < n_steps * 0.5
        results["portfolio_positive"]  = neg_portfolio == 0

        if verbose:
            print(f"\n{'-'*55}")
            print(f"  Environment Sanity Check ({n_steps} random steps)")
            print(f"{'-'*55}")
            for check, passed in results.items():
                icon = "OK" if passed else "FAIL"
                print(f"  [{icon}]  {check}")
            print(f"\n  Reward range : [{min(rewards):.5f}, {max(rewards):.5f}]")
            print(f"  Obs max abs  : {max(obs_maxs):.3f}")
            print(f"  Early terms  : {early_terminations} / {n_steps} steps")
            print(f"  Actions seen : {sorted(action_seen)}")
            print(f"{'-'*55}\n")

        self.reset()
        return results

    # ── Sharpe — Fix 1 ────────────────────────────────────────────────────────

    def _compute_sharpe_from_equity(self, risk_free: float = 0.0) -> float:
        """
        Annualized Sharpe Ratio computed from the step-level equity curve.

        Why this is correct (vs the old per-trade PnL approach):
          - Uses a uniform time series (one value per candle) so the metric
            is comparable across agents with different trade frequencies.
          - Annualization factor is derived from self.candles_per_day, which
            is an explicit constructor parameter — changing timeframe requires
            updating one config value, not hunting for hardcoded constants.
          - Standard formula: E[r] / std[r] * sqrt(periods_per_year)

        Annualization constants for common timeframes:
          15m → candles_per_day=96  → sqrt(96*365)  ≈ 187
          1h  → candles_per_day=24  → sqrt(24*365)  ≈  94
          5m  → candles_per_day=288 → sqrt(288*365) ≈ 324
        """
        if len(self._equity_curve) < 3:
            return 0.0

        equity       = np.array(self._equity_curve, dtype=np.float64)
        step_returns = np.diff(equity) / (equity[:-1] + 1e-10)

        mean = step_returns.mean() - risk_free
        std  = step_returns.std()
        if std < 1e-10:
            return 0.0

        periods_per_year = self.candles_per_day * 365
        return float(mean / std * np.sqrt(periods_per_year))

    def _compute_sharpe(self, pnls: list[float], risk_free: float = 0.0) -> float:
        """
        Legacy per-trade Sharpe kept for backward compatibility with any
        external code that calls it directly. New internal code uses
        _compute_sharpe_from_equity() instead.

        WARNING: This method has known flaws (irregular trade spacing,
        wrong annualization for variable trade frequencies). Do not use
        this as a model selection criterion. Use _compute_sharpe_from_equity.
        """
        if len(pnls) < 2:
            return 0.0
        arr  = np.array(pnls)
        mean = arr.mean() - risk_free
        std  = arr.std()
        if std == 0:
            return 0.0
        # Use candles_per_day for consistency — still not truly correct for
        # per-trade Sharpe, but at least the constant matches the timeframe.
        return float(mean / std * np.sqrt(self.candles_per_day * 365))
