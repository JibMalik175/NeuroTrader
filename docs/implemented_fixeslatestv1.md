# Implemented Fixes — Latest v1

**Files changed:** `trading_env.py` (v3), `feature_engineering.py`, `train_agent.py`  
**Fixes:** 8 of 9 from the overhaul plan (Fix 9 was partially pre-existing; completed here)

---

## Fix 1 — Sharpe Ratio: Equity Curve + Parameterized Annualization

**Files:** `trading_env.py`  
**Root cause:** `_compute_sharpe` took a list of per-trade PnL percentages and multiplied by `sqrt(252 * 24)` — a constant for hourly returns applied to irregularly-spaced trade returns. Two agents with identical P&L but different trade frequencies produced incomparable Sharpe values. This was the primary model selection metric, so the wrong checkpoint was potentially being saved every run.

**What changed:**
- Added `self._equity_curve: list[float]` to `_reset_state()`, initialised with `[initial_balance]`.
- Every `step()` call appends the current `portfolio_value` to `_equity_curve`.
- New method `_compute_sharpe_from_equity()` computes step-to-step percentage returns from the equity curve and annualizes with `sqrt(candles_per_day * 365)`.
- `get_episode_metrics()` now calls `_compute_sharpe_from_equity()` instead of the old method.
- New constructor parameter `candles_per_day: int = 96` (96 = 15m, 24 = 1h, 288 = 5m). This flows from `ENV_CONFIG` in `train_agent.py` so a single config change keeps feature engineering, environment, and Sharpe in sync.
- Old `_compute_sharpe(pnls)` kept as a deprecated stub with a docstring warning. External code that calls it won't break.

**Annualization reference:**
| Timeframe | candles_per_day | sqrt factor |
|-----------|----------------|-------------|
| 5m        | 288            | ≈ 324       |
| 15m       | 96             | ≈ 187       |
| 1h        | 24             | ≈ 94        |
| 1d        | 1              | ≈ 19        |

---

## Fix 2 — Reward Double-Counting Removed

**File:** `trading_env.py`  
**Root cause:** The agent received continuous mark-to-market rewards during every HOLD step, then on SELL received a second reward based on the full cumulative PnL since entry — paying twice for the same price movement. This also created a duration-farming incentive via the hold-winner bonus (removed in Fix 8).

**What changed:**
- The `SELL` branch now computes the **incremental** mark-to-market reward for the final step only (identical formula to HOLD: `price_change_pct * position_size`), rather than calling `_compute_reward(pnl_pct, held_steps)`.
- The cumulative reward for a trade is now the natural sum of all step rewards — no double payment.
- `_compute_reward()` method removed entirely.
- A tiny winner-exit bonus (`+0.001` if `pnl_pct > 0`) is added on SELL to give the agent a non-zero incentive to close profitable trades. Without this, SELL and HOLD produce identical rewards on the final step, and the agent has no reason to exit. The bonus is small enough that it cannot be farmed via duration tricks.
- `reward_components` dict updated: removed `"pnl"` key, added `"winner_exit"` key.

---

## Fix 3 — Macro Feature Windows Parameterized

**File:** `feature_engineering.py`  
**Root cause:** `add_macro_features` hardcoded `candles_per_day = 96` (15m candles) internally. When the bot ran on 1h data (24 candles/day), `window_30d = 2880` silently became a 120-day lookback instead of 30 days. This destroyed ~4 months of training data due to `NaN` warmup and computed meaningless "macro" features over 4× the intended horizon.

**What changed:**
- `add_macro_features(df, candles_per_day=96)` now accepts `candles_per_day` as an explicit parameter with a detailed docstring explaining valid values.
- `build_features(df, version, candles_per_day=96)` accepts and passes through `candles_per_day`.
- CLI `feature_engineering.py` exposes `--candles-per-day` argument (default 96).
- The unused `window_90d` variable was also removed (it was computed but never used).

**Usage example:**
```bash
# 1h data — previously would silently compute 120-day windows
python feature_engineering.py --input btc_1h.parquet --candles-per-day 24

# 15m data — correct default
python feature_engineering.py --input btc_15m.parquet --candles-per-day 96
```

---

## Fix 4 — Independent Walk-Forward Models

**File:** `train_agent.py`  
**Root cause:** The walk-forward loop trained a single model progressively across all windows via `model.set_env(vec_env)`. Window 3's validation metric compared a model with 3× the training budget against Window 1's model — an unfair comparison that biased checkpoint selection toward later windows. VecNormalize statistics were also carried forward, coupling the normalizer state between what should be independent experiments.

**What changed:**
- Every window now calls `PPO(...)` or `RecurrentPPO(...)` from scratch — `model.set_env()` is gone.
- Every window gets a fresh `VecNormalize()` — no statistics carried over.
- `reset_num_timesteps=True` is always set (was conditional on `window_idx == 1`).
- The summary print now notes that Window N (full data) is the production model, and that the best-by-Sharpe is diagnostic information about regime sensitivity, not necessarily the deployment choice.
- `sharpe_std` from Fix 6 helps quantify how regime-sensitive each window's policy is.

---

## Fix 5 — Dead Import Removed

**File:** `feature_engineering.py`  
**Root cause:** `from sklearn.preprocessing import RobustScaler` was imported at module level but never called anywhere in the file. Either it was removed during a refactor and the import left behind, or the scaling step was accidentally omitted.

**What changed:**
- Import replaced with a comment: `# RobustScaler removed — scaling is handled by VecNormalize in train_agent.py`
- No functional change; eliminates a misleading import that implied scaling was happening in the feature pipeline when it wasn't.

---

## Fix 6 — Validation Across Multiple Independent Market Periods

**File:** `train_agent.py`  
**Root cause:** `run_validation` ran exactly one deterministic episode on the full `val_df`. A single episode on a deterministic dataset always produces the same result. With only ~2,598 validation rows generating 20–50 trades, the Sharpe estimate had enormous sampling variance. Running multiple stochastic episodes on the same data doesn't help — it adds noise from random action selection, not from different market conditions.

**What changed:**
- `val_df` is split into 3 non-overlapping chronological slices (thirds).
- One deterministic episode is run on each slice independently.
- Reported metrics are the **mean** across all slices that produced trades.
- `sharpe_std` is reported as a new key — the standard deviation of Sharpe across slices. A high `sharpe_std` means the policy is regime-sensitive and unreliable.
- `n_val_slices` is reported so the caller knows how many slices produced valid trades.
- If `val_df` is too small to split (< 300 rows per slice, configurable via `MIN_ROWS_PER_SLICE`), falls back to a single full episode with a printed warning.
- Training print now shows `Sharpe ± std` format.

---

## Fix 7 — Correct SELL Fee Calculation

**File:** `trading_env.py`  
**Root cause:** The old SELL fee was charged on `self.balance * self.position_size` (pre-PnL balance). This slightly under-charged winning trades (fee on smaller balance) and over-charged losing trades. More critically, the balance update was order-dependent: adding PnL first then subtracting fee on the pre-PnL amount created a systematic accounting inconsistency that compounded over many trades.

**What changed:**

New fee math:
```python
gross_proceeds = position_cost_basis * (1 + pnl_pct)   # Actual exit value
exit_fee       = gross_proceeds * fee_rate               # Fee on notional exit value
net_proceeds   = gross_proceeds - exit_fee
balance        = (balance - position_cost_basis) + net_proceeds
```

- Added `self.position_cost_basis: float` to track the exact cash allocated at entry (post entry-fee), reset to `0.0` on SELL and in `_reset_state()`.
- `TradeRecord` dataclass gains a `position_cost: float` field for audit trail.
- `_get_portfolio_value()` updated to use `position_cost_basis` for mark-to-market: `balance + position_cost_basis * unrealized_pnl_pct` (was `balance * (1 + position_size * unrealized_pnl_pct)` which had the same order-dependency problem).
- The balance update is now fully order-independent: remove the cost basis, add back net proceeds.

---

## Fix 8 — Hold-Winner Bonus Removed

**File:** `trading_env.py`  
**Root cause:** The multiplier `hold_bonus = 1.0 + 0.01 * min(held_steps, 48)` rewarded the agent proportionally to how long it held a winning trade, up to a cap at 48 candles. This created a fixed-duration farming exploit: the optimal policy was to hold every profitable trade for exactly 48 candles regardless of market signals, then exit. This is the opposite of adaptive exit timing.

**What changed:**
- `_compute_reward(pnl_pct, held_steps)` method removed entirely (also removing the asymmetric 1.3× loss penalty).
- The SELL branch now uses the same incremental mark-to-market formula as HOLD (see Fix 2).
- Continuous mark-to-market rewards during HOLD provide sufficient incentive to hold winning trades — no separate bonus needed.
- The tiny winner-exit bonus (`+0.001`) from Fix 2 provides a clean signal to close profitable positions without duration bias.

---

## Fix 9 — MTF Look-Ahead Bias (Completed)

**File:** `feature_engineering.py`  
**Root cause:** After resampling to 4h/1d candles and computing RSI/MACD, the features were joined back to the base timeframe without a temporal shift. The last base candle of each higher-timeframe block could see the higher-TF indicator value computed using its own close price — a candle seeing its contribution to a higher-TF bar before that bar closes.

**What changed:**
- `.shift(1)` applied to the resampled feature columns **before** joining to the base dataframe:
  ```python
  shifted = resampled[[f'{prefix}_rsi', f'{prefix}_macd']].shift(1)
  df_temp = df_temp.join(shifted, how='left').ffill()
  ```
- This ensures every base candle at time T sees only the higher-TF bar that **completed before** T, which is exactly what a live trader has access to.
- Docstring updated with a clear before/after explanation of the fix.
- Note: The first higher-TF bar's features will be NaN (no prior completed bar), which is handled by the existing `fillna(0)` in the copy-back step.

---

## Config Changes Summary

| Config key | Old value | New value | Reason |
|---|---|---|---|
| `ENV_CONFIG["candles_per_day"]` | *(absent)* | `96` | Fix 1/3: Sharpe and macro windows |
| `add_macro_features(candles_per_day)` | hardcoded `96` | explicit param | Fix 3 |
| `build_features(candles_per_day)` | *(absent)* | `96` (passthrough) | Fix 3 |

---

## Breaking Changes

These changes affect any code that depends on the old interfaces:

1. **`TradingEnv.__init__`** gains `candles_per_day=96` parameter. Existing instantiation without it defaults to 96 (15m). If you're on 1h data, pass `candles_per_day=24` or add it to `ENV_CONFIG`.

2. **`TradeRecord`** gains `position_cost: float` as a required field. Any code constructing `TradeRecord` directly must add this argument.

3. **`get_episode_metrics()`** no longer has a `"pnl"` key in `reward_components`. Has `"winner_exit"` instead. Any dashboard/logging code reading `reward_components["pnl"]` will get a `KeyError`.

4. **`run_validation()`** now returns additional keys: `sharpe_std`, `n_val_slices`. Existing code consuming the metrics dict will still work (extra keys are ignored). Code checking for exactly known keys may need updating.

5. **`add_macro_features()`** now requires `candles_per_day` to be passed explicitly for non-15m data. The default is still `96`, so existing 15m pipelines are unaffected.

6. **`build_features()`** gained `candles_per_day` parameter. Existing callers without it default to `96`.

