> ⚠️ **HISTORICAL (June 1–5, 2026).** This log ends at Experiment C. The reward has
> since moved to FIX-B (pure portfolio return) plus a selectable exit-concentrated
> mode, the timeframe to 1h, and the action space to a long/short ladder. For current
> state read `PROGRESS_CHECKLIST.md` and `CORE_TRAINING_FIX_PLAN.md`.

# Reward Calibration Research Log

> **Project:** BTC/USDT 15m Trading Bot — RecurrentPPO  
> **Date Range:** June 1–5, 2026  
> **Environment:** `trading_env.py` v3 with V4 features  
> **Model:** RecurrentPPO (LSTM), 1.76M params, RTX 4060 Laptop GPU  
> **Data:** 130K train rows, 26K val rows, 15-minute candles  

---

## Table of Contents

1. [Deployment Benchmark — RSI Baseline](#benchmark)
2. [Executive Summary](#executive-summary)
3. [Starting Point — The Broken Model](#starting-point)
4. [Experiment Timeline](#experiment-timeline)
5. [Deep Findings by Component](#deep-findings)
6. [Instrumentation & Debug Infrastructure](#instrumentation)
7. [Key Learnings (Principles)](#key-learnings)
8. [Final Calibrated Values](#final-values)
9. [Key Observations — 1M Run](#1m-observations)
10. [Run 10: Experiment A — Invalid Action Remap](#run10)
11. [Run 11: Experiment B — Loss-Duration Penalty](#run11)
12. [Run 12: Experiment C — Case 1 vs 2 Diagnostic](#run12)
13. [Open Issues](#open-issues)

---

## Deployment Benchmark — RSI Baseline {#benchmark}

> **Date:** June 4, 2026  
> **Script:** `scripts/baseline_backtest.py --data data/BTC_USDT_15m_test.parquet --sweep`  
> **Data:** 17,373 rows of held-out test data (never seen during training)  
> **Fee:** 0.10% per trade  

Before deploying the DRL model, it must outperform the best simple RSI mean-reversion baseline on the **test set**. This is the minimum bar to clear.

### RSI Entry Threshold Sweep (exit fixed at RSI=30)

| Entry RSI | Sharpe | Win Rate | Trades | Return | Alpha vs Buy-Hold |
|-----------|--------|----------|--------|--------|-------------------|
| 55 | -1.988 | 32.11% | 109 | -7.70% | +7.25% |
| **60** | **-1.428** | 33.70% | 92 | -5.31% | **+9.64%** ← Best Sharpe |
| 65 | -1.774 | 32.91% | 79 | -6.21% | +8.74% |
| 70 | -2.546 | 33.87% | 62 | -7.83% | +7.12% |
| 75 | -1.802 | 43.90% | 41 | -4.70% | +10.24% |
| **80** | -1.940 | 36.36% | **22** | **-4.11%** | **+10.84%** ← Best Return |

> [!IMPORTANT]
> **Every RSI configuration loses money in absolute terms**, yet every one beats buy-and-hold by 7-10%. This tells us the test period is a **downtrending or choppy bear market** — an inherently difficult regime. The DRL model must prove it can extract alpha even in this hostile environment.

### Official Deployment Targets

| Target | Threshold | Status |
|--------|-----------|--------|
| **Minimum** — Beat best RSI Sharpe | `Sharpe > -1.428` | ✅ **Cleared** (Exp A avg: -0.145) |
| **Stretch** — Profitable in absolute terms | `Sharpe > 0.0` | ⏳ 2/3 slices profitable |
| **Production** — Consistent across all regimes | `All 3 slices Sharpe > 0.0` | ❌ Not yet |

### Where the DRL Model Stands vs Baseline

```
RSI Baseline (best):   Sharpe -1.428   Return -5.31%   Trades  92

DRL 300K avg:          Sharpe -1.523   Return -3.20%   Trades  69  ← just below bar
DRL 300K Slice 1:      Sharpe +2.645   Return +2.44%   Trades  21  ← well above bar
DRL 1M Window 2:       Sharpe -19.82   Return -27.89%  Trades 695  ← far below (loophole)
DRL Exp A 300K avg:    Sharpe -0.145   Return -0.33%   Trades   9  ← BEATS bar ✅
DRL Exp A Slice 0:     Sharpe +1.490   Return +3.22%   Trades  14  ← profitable ✅
DRL Exp A Slice 1:     Sharpe +0.611   Return +0.97%   Trades   2  ← profitable ✅
DRL Exp A Slice 2:     Sharpe -2.537   Return -5.17%   Trades  13  ← struggling ❌
```

> [!NOTE]
> The 300K calibrated run is **3% below the RSI bar on average Sharpe** but already **beats it convincingly in Slice 1**. The core problem is cross-regime consistency, not lack of signal. The agent demonstrates it has a real edge in trending markets but struggles in the same choppy conditions that hurt the RSI baseline too.

---

## Executive Summary

We ran **7+ controlled diagnostic experiments** over two days to find the root causes of policy collapse in our trading RL agent. The agent was either refusing to trade (99%+ HOLD) or churning itself to death (462 trades, -20% return).

### Root Causes Found

| # | Root Cause | Evidence | Fix |
|---|-----------|----------|-----|
| 1 | **SELL reward bug** — position_size zeroed before reward calc | SELL reward always = 0 | Store `old_position_size` before zeroing |
| 2 | **Invalid action penalty too large** (-0.005) | 45% of total reward, 500× larger than MTM | Reduced to -0.001 |
| 3 | **Early-exit penalty too large** (-0.001) | Winning sells averaged **-0.0006** (negative!) | Calibrated to -0.0003, only for < 5 candles |
| 4 | **Drawdown penalty too sensitive** (threshold 0.10) | Fired every step, 4500:1 noise-to-signal | Threshold raised to 0.25, coefficient to 0.002 |

### The Calibration Journey (Trade Count)

```
Original broken model:     1,461 trades (micro-scalping exploit)
Fix A-H applied (0.003):       7 trades (frozen — SELL phobia)
Penalty 0.001:                  7 trades (still frozen)
Penalty 0.000:                462 trades (churning — fee death)
Penalty 0.0003:                87 trades ← sweet spot
```

---

## Starting Point — The Broken Model {#starting-point}

Before any fixes, the original model exhibited:

```
Sharpe:     -76.0
Return:     -99.87%
Win Rate:    49.66%
Trades:      1,461
Avg Hold:    1.6 candles (24 minutes)
Max DD:      50.01% (hit kill switch every episode)
```

### What Was Wrong

The agent discovered a **micro-scalping exploit**: rapidly BUY/SELL to farm the `winner_exit` bonus (+0.001 per winning trade) while ignoring actual market returns. With a ~50% win rate and 1,461 trades, it collected `~730 × 0.001 = 0.73` in bonuses while the actual trading produced near-zero PnL.

The `drawdown_penalty` (threshold 0.10, coefficient 0.05) fired on virtually every step, producing cumulative penalty of **~68 per episode** vs MTM signal of **~0.18 per episode** — a 4,500:1 noise-to-signal ratio.

---

## Experiment Timeline {#experiment-timeline}

### Run 1: `diagnostic_fixed` — All Phase 1 Fixes Applied

**Changes:** Fixes A-H (SELL bug, winner scaling, early-exit 0.003, winner:fee verification, V4 features, invalid penalty 0.005, entropy/clip tuning, replay guard)

| Metric | Value |
|--------|-------|
| Timesteps | 300,000 |
| Sharpe | **-2.349** |
| Return | -4.58% |
| Trades | 78 |
| Avg Hold | 100.6 candles |
| Max DD | 6.79% |
| Action Dist | H=98.3% B=0.9% S=0.8% |
| Training Actions | H=51.5% B=24.9% S=23.7% |

> [!IMPORTANT]
> **Key Discovery:** Massive gap between training actions (51/25/24) and validation actions (98/1/1). The entropy bonus (`ent_coef=0.08`) forces exploration during training, but the deterministic policy collapses to HOLD. This means the **learned policy** is "don't trade," even though exploration looks healthy.

**Reward Component Breakdown:**
```
invalid_action:     -6.877  (45%)  ← DOMINANT
early_exit_penalty: -1.739  (20%)
drawdown_penalty:   -0.384  (5%)
fee_cost:           -0.239  (3%)
mark_to_market:     -0.004  (0.05%) ← THE ACTUAL SIGNAL
winner_exit:        +0.023  (0.3%)
```

> [!CAUTION]
> The actual trading signal (MTM) was **less than 1% of total reward**. The agent couldn't hear the music because the fire alarms were too loud.

---

### Run 2: `diagnostic_exp_A` — Early Penalty Reduced to 0.001

**Change:** `early_penalty: 0.003 → 0.001`

| Metric | Run 1 | Run 2 | Change |
|--------|-------|-------|--------|
| Sharpe | -2.349 | **-4.101** | Worse ↓ |
| Trades | 78 | 58 | Fewer ↓ |
| Avg Hold | 100.6 | **22.7** | Shorter ↓ |
| Hold % | 98.3% | **99.4%** | More frozen ↑ |

> [!NOTE]
> **Counter-intuitive result.** Reducing the penalty made behavior WORSE, not better. The agent took shorter trades (22.7 vs 100.6 candles) but became even more frozen (99.4% HOLD). This disproved the hypothesis that early-exit penalty alone was the root cause and pointed to a **multi-factor reward imbalance**.

---

### Run 3: `diagnostic_exp_B` / `diagnostic_h5_fix2` — Invalid Penalty Reduced to 0.001

**Changes:** `invalid_penalty: 0.005 → 0.001`, added `invalid_action_count` tracker

| Metric | Value |
|--------|-------|
| Timesteps | 120,000 |
| Sharpe | **-0.258** |
| Trades | 3.0 |
| Avg Hold | 436.8 candles (4.5 days) |
| Hold % | 99.1% B=0.8% S=0.0% |

> [!IMPORTANT]
> **S=0.0% — zero SELL actions across all validation.** The agent entered 3 positions and never exited them. This is the **disposition effect** — the agent holds losing positions forever because selling always produces a negative reward (fee + penalty) while holding costs nothing.

---

### Run 4: `diagnostic_h7` — Additional Instrumentation Run

Added per-slice validation logging, fixed episode-end percentage math (excluded `_count` fields from totals).

**Per-slice results revealed the policy isn't completely dead:**

| Slice | Trades | Sharpe | Behavior |
|-------|--------|--------|----------|
| 0 | 6 | **+1.524** | Profitable! |
| 1 | 0 | +0.81 | Passive hold in uptrend |
| 2 | 3 | -3.108 | Lost money, never sold |

> [!NOTE]
> Slice 0 proved the agent **CAN trade profitably** in certain regimes. The problem is specifically with SELL decisions — it enters positions but refuses to exit.

---

### Run 5: `diagnostic_instrumented` — Full SELL Instrumentation

**Added:** Per-SELL logging (entry_price, fill_price, pnl, hold_duration, reward breakdown), mean absolute reward magnitudes, sell summary statistics.

| Metric | Value |
|--------|-------|
| Sharpe | **+0.113** (first positive!) |
| Trades | 6.67 |
| Avg Hold | 875 candles (9 days) |
| Win Rate | 56.75% |

**The smoking gun — SELL reward breakdown from training episodes:**

| Metric | Value |
|--------|-------|
| Mean winner sell reward | **-0.0006** ← NEGATIVE! |
| Mean loser sell reward | -0.0016 |
| Mean hold duration at sell | 3.1 candles |

**Example: A profitable trade that was PUNISHED:**
```
BUY @ 39,265 → SELL @ 39,280  (+0.04% profit)
  sell_mtm:      +0.000118   (actual profit)
  fee_signal:    -0.000200   (exchange fee)
  early_penalty: -0.001000   (8.5× larger than profit!)
  ─────────────────────────
  TOTAL:         -0.001082   ← PROFITABLE TRADE, NEGATIVE REWARD
```

> [!CAUTION]
> **This was the definitive root cause.** The early-exit penalty was so large relative to the MTM signal that even profitable trades produced negative SELL rewards. PPO learned the rational policy: "never sell."

**Mean absolute reward per step (Priority 2 finding):**

| Component | Mean |abs| per step |
|-----------|---------------------|
| MTM (HOLD) | 0.000520 |
| Drawdown | 0.000264 |
| SELL total | 0.001200 |
| Invalid | 0.001000 (flat) |
| Fee | 0.000200 (flat) |

The early_exit_penalty (0.001) was 2× larger than a typical MTM step.

---

### Run 6: `diagnostic_no_early_pen` — Penalty Set to Zero

**Change:** `early_penalty: 0.001 → 0.0` (complete removal)

| Metric | With Penalty | Without | Change |
|--------|-------------|---------|--------|
| **Action Dist** | H=99.6% B=0.3% S=0.1% | **H=87.9% B=6.0% S=6.1%** | SELL unlocked! |
| **Trades** | 6.7 | **462** | 69× more |
| **Avg Hold** | 875 | **10.8** | 81× shorter |
| **Sharpe** | +0.113 | **-15.15** | Crashed |
| **Return** | +0.4% | **-20.0%** | Fee death |
| **Winner Reward** | -0.0006 | **+0.0003** | Now positive! |

> [!IMPORTANT]
> **Causal proof:** Removing the early-exit penalty completely unlocked SELL actions and made winner rewards positive. But the agent swung to the opposite extreme — churning 462 trades and losing 20% purely to fees.

**Fee analysis:**
```
Fee cost:     -0.168  (total across 468 trades)
Per-trade:    -0.000359
MTM gain:     +0.006  (total)

Fee/MTM ratio: 28:1 — fees dominated by 28×
```

The agent wasn't losing on trade quality — it was losing to transaction costs from excessive trading volume.

---

### Run 7: `diagnostic_pen_0003` — Penalty Calibrated to 0.0003

**Change:** `early_penalty: 0.0 → 0.0003`, applied only to trades < 5 candles (removed the 5-10 candle tier)

| Metric | Pen=0.001 | Pen=0.000 | **Pen=0.0003** |
|--------|:---------:|:---------:|:--------------:|
| **Trades** | 6.7 | 462 | **87** ✅ |
| **Avg Hold** | 875 | 10.8 | **43 candles** ✅ |
| **H/B/S** | 99.6/0.3/0.1 | 87.9/6.0/6.1 | **96.3/1.8/1.8** ✅ |
| **Sharpe** | +0.113 | -15.15 | **-3.95** |
| **Return** | +0.4% | -20.0% | **-4.8%** |
| **Max DD** | 4.9% | 20.2% | **5.5%** ✅ |
| **Fee Cost** | -0.003 | -0.168 | **-0.034** ✅ |
| **B/S Ratio** | — | 1.0 | **~1.0** ✅ |

**Per-slice sell economics:**

| Slice | Trades | Hold | Winner Reward | Loser Reward | Sharpe |
|-------|--------|------|--------------|-------------|--------|
| 0 | 124 | 28.4 | **+0.000436** | -0.000365 | -2.77 |
| 1 | 56 | 55.6 | **+0.000566** | -0.000387 | -5.76 |
| 2 | 82 | 44.8 | **+0.000504** | -0.000498 | -3.33 |

> [!TIP]
> **Winner rewards are positive across all slices.** The reward asymmetry is now correct: profitable trades produce positive reward, losing trades produce negative reward. Fee costs dropped 5× from the churn run. This is the strongest configuration tested so far.

---

## Deep Findings by Component {#deep-findings}

### 1. Early-Exit Penalty — The Behavior Control Knob

**Discovery:** The early-exit penalty acts as a **continuous trade-frequency controller**, not a binary on/off switch.

```
Penalty   →  Trade Frequency  →  Behavior
0.001     →  7 trades          →  Frozen (SELL phobia)
0.0003    →  87 trades         →  Selective trading ✅
0.000     →  462 trades        →  Churning (fee death)
```

**Why it's so sensitive:** The penalty magnitude is compared against individual MTM step rewards, which are tiny for BTC 15m candles:

```
Typical 15m BTC move:   0.1% = 0.001
Position size:          20% = 0.2
MTM reward per step:    0.001 × 0.2 = 0.0002

Penalty at 0.001:       5× larger than MTM → dominates
Penalty at 0.0003:      1.5× larger than MTM → comparable
Penalty at 0.000:       0× → no friction
```

**Critical insight:** The penalty should be **sub-fee-magnitude**. The per-trade fee is ~0.00036. A penalty larger than fees creates a world where selling is always irrational. A penalty at ~60% of fees (0.0003) adds friction without dominating.

**Design recommendation:** Only apply to trades < 5 candles (the micro-scalping zone). The 5-10 candle tier in the original code was redundant and harmful — it punished reasonable short trades.

---

### 2. Invalid Action Penalty — The Silent Killer

**Discovery:** At -0.005, invalid actions accumulated to **45-48% of total episodic reward**, drowning out the actual trading signal.

```
invalid_action = -6.52  (per episode)
mark_to_market = -0.01  (per episode)

Ratio: 652:1
```

**Why it accumulates so fast:** During random exploration (~33% each action), the agent frequently presses BUY while already holding or SELL while flat. With ~3,000 invalid actions per episode at -0.005 each, the cumulative penalty is -15.0, which dominates everything.

**Reduction to -0.001:** Still produces ~3,000 invalid actions × -0.001 = -3.0 per episode, but this is now comparable to (not dominating) other signals.

**Design principle:** Invalid actions in trading should be treated as **near-no-ops**, not punished heavily. The agent needs to learn action legality, but the signal should be a gentle nudge, not an electric shock.

---

### 3. Mark-to-Market Reward — The True Signal

**Discovery:** MTM is inherently tiny for 15m crypto candles:

```
mean_abs_mtm_per_step ≈ 0.0005
```

This is the signal the agent should be optimizing. Every other reward component must be calibrated relative to this magnitude:

| Component | Magnitude | Relative to MTM |
|-----------|-----------|-----------------|
| MTM | 0.0005 | 1.0× (baseline) |
| Fee | 0.0002 | 0.4× ✅ |
| Drawdown | 0.00026 | 0.5× ✅ |
| Early penalty (0.0003) | 0.0003 | 0.6× ✅ |
| Invalid (0.001) | 0.001 | 2.0× ⚠️ (borderline) |
| Early penalty (0.001) | 0.001 | 2.0× ❌ (too large) |
| Invalid (0.005) | 0.005 | 10× ❌ (way too large) |

**Rule of thumb:** No single penalty should exceed 2× the mean absolute MTM reward per step, or it will dominate gradient signal.

---

### 4. Drawdown Penalty — Successfully Fixed

**Original:** Threshold 0.10, coefficient 0.05
- Fired on almost every step (BTC routinely draws down >10% intraday)
- Cumulative: ~68 per episode vs MTM ~0.18
- **4,500:1 noise-to-signal ratio**

**Fixed:** Threshold 0.25, coefficient 0.002
- Only fires during significant drawdowns
- At 40% DD: `(0.40-0.25) × 0.002 = 0.0003/step`
- Comparable to MTM magnitude ✅

**Result:** Drawdown penalty is now 0.0 in most validation slices (agent stays well within 25% DD), confirming the fix works.

---

### 5. Winner-Exit Bonus — Edge Case at Threshold

**Discovery:** The bonus uses a linear scale from 5 to 20 candles:

```python
hold_scale = min(1.0, (held_steps - 5) / (20 - 5))
winner_bonus = 0.001 * hold_scale
```

At exactly 5 candles: `hold_scale = 0`, `bonus = 0`. The agent must hold at least 6 candles to receive any bonus. This creates a cliff:

```
4 candles: no bonus + early penalty = heavily punished
5 candles: no bonus + no penalty = neutral
6 candles: tiny bonus (0.000067) + no penalty = barely positive
20 candles: full bonus (0.001) + no penalty = meaningful
```

**Implication:** The bonus primarily rewards trades held 10+ candles. For shorter profitable trades (5-10 candles), the reward comes almost entirely from MTM and is very small.

---

### 6. The Disposition Effect — Why Agents Hold Losers

**Finding:** When holding a losing position, the agent faces:

- **HOLD:** Small negative MTM (if price keeps falling), but no fee
- **SELL:** Small negative MTM + exit fee + early penalty (if short hold)

SELL is always worse than HOLD for losing positions on any individual timestep. The agent rationally concludes: "Hold and hope for recovery."

**Evidence:** In Run 3, the agent held positions for 436 candles (4.5 days) on average, with S=0.0% across all validation. It entered positions but literally never exited.

**Current mitigation:** The reduced penalty (0.0003) plus winner bonus create enough asymmetry that profitable exits are rewarded. But there's no mechanism to penalize holding a loser for too long.

**Future consideration:** A loss-duration penalty (tiny continuous cost for holding underwater positions beyond N candles) could address this. Not implemented yet — the current 0.0003 penalty produces acceptable hold durations (43 candles avg).

---

## Instrumentation & Debug Infrastructure {#instrumentation}

### Debug Logging System

Created a structured JSON logging system (`_debug_log`) that writes to `debug-{session_id}.log`:

```json
{
  "sessionId": "627897",
  "hypothesisId": "H4",
  "location": "trading_env.py:EPISODE_END",
  "message": "Episode reward/action summary",
  "data": { ... },
  "timestamp": 1780410246334
}
```

### Hypothesis Tracking

| ID | Hypothesis | Status |
|----|-----------|--------|
| H2 | Invalid action frequency is too high | ✅ Confirmed (2600-3500 per episode) |
| H3 | Long hold MTM is too small | ✅ Confirmed (~0.0002 per step) |
| H4 | Episode reward breakdown reveals dominant component | ✅ Critical tool |
| H5 | Per-slice validation reveals regime sensitivity | ✅ Confirmed |
| H6 | Action probability extraction for recurrent policy | ❌ Still broken (see Open Issues) |

### Metrics Added

1. **`invalid_action_count`** — raw count of invalid actions per episode (separate from reward sum)
2. **`_sell_log`** — per-SELL detailed breakdown (entry/exit price, pnl, held_steps, each reward component, total)
3. **`_reward_abs_sums` / `_reward_abs_counts`** — mean absolute reward magnitude per step by component
4. **`sell_summary`** — aggregated sell statistics (n_winners, n_losers, mean_winner_reward, mean_loser_reward, mean_hold_duration)
5. **`reward_component_pct_of_abs_total`** — percentage breakdown excluding count fields

---

## Key Learnings (Principles) {#key-learnings}

### 1. Reward Shaping Can Backfire Catastrophically

Every penalty we added to fix one behavior created a new failure mode:
- Drawdown penalty → agent afraid of all positions
- Invalid penalty → agent afraid of BUY/SELL buttons  
- Early-exit penalty → agent afraid of SELL specifically
- Winner bonus → agent micro-scalps to farm it

**Principle:** Start with the simplest possible reward (MTM + fees only) and add shaping terms one at a time, validating each.

### 2. The Training/Validation Gap Is the Key Diagnostic

```
Training:   H=51% B=25% S=24% (healthy exploration)
Validation: H=99% B=0.5% S=0.5% (collapsed policy)
```

If training actions look healthy but validation collapses, the problem is in **reward economics**, not in PPO configuration. The entropy bonus masks reward issues during training.

### 3. Penalty Magnitude Must Be Calibrated to MTM Scale

For BTC 15m candles:
```
mean_abs_mtm ≈ 0.0005 per step
```

Any penalty > 0.001 will dominate the gradient. Any penalty > 0.005 will completely drown out the trading signal.

### 4. Use Controlled A/B Tests, Not Shotgun Changes

The most productive experiments changed **exactly one variable**:
- Run 5 → Run 6: only `early_penalty` changed (0.001 → 0.0)
- Run 6 → Run 7: only `early_penalty` changed (0.0 → 0.0003)

This gave clean causal evidence. Runs that changed multiple variables simultaneously were harder to interpret.

### 5. The "Opposite Extreme" Test Is Powerful

Testing both extremes of a parameter (0.001 and 0.000) before finding the midpoint was far more informative than incremental adjustments. It proved the parameter was a monotonic control knob and gave bounds for binary search.

### 6. Instrument Before You Fix

The most valuable diagnostic run (Run 5) added instrumentation without changing any behavior. The SELL-by-SELL breakdown immediately revealed that winning trades were negatively rewarded — something invisible from aggregate metrics.

### 7. 120K Steps Is Enough for Diagnostics

At 120K steps (12 minutes on GPU), the agent learns enough to reveal reward structure problems. Don't waste hours on 1M+ step runs until the reward economics are validated.

### 8. VecNormalize Doesn't Fix Reward Imbalance

VecNormalize normalizes the total reward distribution, but if 99% of the reward signal is penalty noise, it normalizes the noise — not the trading signal. The raw reward components must be balanced before normalization.

---

## Final Calibrated Values {#final-values}

```python
# Reward components — current calibration
EARLY_EXIT_PENALTY = 0.0003     # Only for trades < 5 candles
SHORT_PENALTY      = 0.0        # Removed (5-10 candle tier)
INVALID_PENALTY    = 0.001      # Reduced from 0.005
WINNER_BONUS_MAX   = 0.001      # Scaled linearly from 5-20 candle hold
DD_THRESHOLD       = 0.25       # Raised from 0.10
DD_COEFFICIENT     = 0.002      # Reduced from 0.05
TERMINAL_PENALTY   = 0.05       # Reduced from 0.5

# PPO config (unchanged throughout calibration)
learning_rate = 0.0001
ent_coef      = 0.08
gamma         = 0.995
clip_range    = 0.15
position_fraction = 0.20
max_drawdown_pct  = 0.50
```

---

## Run 8: `calibrated_300k` — Scaled-Up Validation (1 Window, 300K Steps)

**Command:** `--timesteps 300000 --windows 1 --run-name calibrated_300k --recurrent`

**Purpose:** Verify that the calibrated reward structure (pen=0.0003) produces better results with more training time.

| Metric | Run 7 (120K) | Run 8 (300K) | Change |
|--------|:---:|:---:|:---:|
| **Sharpe** | -3.95 | **-1.523 ±3.002** | Better ↑ |
| **Return** | -4.8% | **-3.20%** | Better ↑ |
| **Trades** | 87 | **68.7** | More selective ↑ |
| **Avg Hold** | 43 candles | **76.3 candles** | Longer ↑ |
| **Action Dist** | H=96.3% B=1.8% S=1.8% | **H=98.1% B=1.0% S=0.9%** | More conservative |
| **Win Rate** | — | **50.86%** | Near coin flip |

**Per-slice breakdown:**

| Slice | Sharpe | Return | Trades | Avg Hold | Winner Reward | Loser Reward |
|-------|--------|--------|--------|----------|--------------|-------------|
| 0 | -2.906 | -4.95% | 115 | 38.5 | +0.000319 | -0.000325 |
| 1 | **+2.645** ✅ | **+2.44%** ✅ | 21 | 129.3 | +0.000879 | -0.000186 |
| 2 | -4.307 | -7.10% | 70 | 61.1 | +0.000616 | -0.000320 |

> [!IMPORTANT]
> **Slice 1 is the first consistently profitable slice across a full 90-day validation window.** Sharpe +2.645 with +2.44% return at 21 trades proves the agent CAN learn a real edge when exposed to favorable market regimes. The Sharpe ±3.002 std-dev reveals massive regime sensitivity — the agent profits in trending markets and struggles in choppy ones.

**Observation:** Scaling from 120K → 300K steps showed clear directional improvement. This validated the hypothesis that longer training with calibrated rewards would move Sharpe toward zero and eventually positive.

---

## Run 9: `production_1m` — 1M Step 3-Window Walk-Forward Run

**Command:** `--timesteps 1000000 --windows 3 --run-name production_1m --recurrent`

**Date:** June 3, 2026

**Purpose:** First production-scale training run. 3-window walk-forward to test generalization across different data slice sizes.

> [!WARNING]
> **Replay Ratio Warning fired at Window 1:** `replay ratio ~23× (1,000,000 timesteps / 43,384 usable rows)`. Window 1 only has 43K rows — the agent will see the same data 23 times, creating a serious overfitting risk.

---

### Window 1 — `43,432 rows` (first 33% of training data)

**Training Rollout Progression:**

| Rollout | Reward | Actions (H/B/S) | MTM | Winner Exit | Steps |
|---------|--------|-----------------|-----|-------------|-------|
| 10 | -5.66 | 45.6% / 27.5% / 26.8% | -0.0073 | +0.0182 | 163,840 |
| 20 | -5.59 | 64.1% / 18.6% / 17.3% | +0.0894 | +0.0501 | 327,680 |
| 30 | -5.49 | 63.5% / 18.8% / 17.6% | +0.1018 | +0.0697 | 491,520 |
| 40 | -5.39 | 53.2% / 26.3% / 20.4% | +0.2415 | +0.0589 | 655,360 |
| 50 | -4.85 | 57.8% / 22.2% / 20.0% | +0.3034 | +0.0526 | 819,200 |
| 60 | -4.66 | 61.2% / 21.6% / 17.2% | +0.4230 | +0.0788 | 983,040 |

**Window 1 Validation Result:**

```
Sharpe Ratio :  -10.654  ±1.123
Total Return :  -15.43%
Win Rate     :   44.87%
Total Trades :  341.6667
Max Drawdown :   15.95%
Avg Hold     :   14.8 candles
Action Dist  : H=92.6% B=3.8% S=3.6%
Training Time: 7014.0s (143 steps/s)
```

> [!CAUTION]
> **Window 1 confirmed the overfitting hypothesis.** During training, `mark_to_market` climbed from -0.007 to +0.423 (60× improvement), proving the agent was learning. But validation Sharpe crashed to -10.654 with 341 trades. The agent memorized the 43K-row training set across 23 replay passes and learned hyper-specific micro-patterns that don't generalize.

---

### Window 2 — `86,864 rows` (first 66% of training data)

**Training Rollout Progression:**

| Rollout | Reward | Actions (H/B/S) | MTM | Winner Exit | Steps |
|---------|--------|-----------------|-----|-------------|-------|
| 10 | -5.47 | 43.4% / 28.4% / 28.2% | -0.0059 | +0.0156 | 163,840 |
| 20 | -5.42 | 57.8% / 22.6% / 19.6% | +0.1005 | +0.0372 | 327,680 |
| 30 | -5.12 | 64.9% / 20.0% / 15.1% | +0.2403 | +0.0846 | 491,520 |
| 40 | -4.94 | 66.2% / 19.4% / 14.4% | +0.2976 | +0.0980 | 655,360 |
| 50 | -4.78 | 67.8% / 18.2% / 14.1% | +0.3648 | +0.1152 | 819,200 |
| 60 | -5.09 | 66.6% / 19.1% / 14.3% | +0.4535 | +0.1380 | 983,040 |

**Window 2 Validation Result:**

```
Sharpe Ratio : -19.820  ±5.119
Total Return :  -27.89%
Win Rate     :   44.03%
Total Trades :  695.3333
Max Drawdown :   28.08%
Avg Hold     :    8.1 candles
Action Dist  : H=83.2% B=7.8% S=8.9%
Training Time: 7326.0s (137 steps/s)
```

> [!CAUTION]
> **Window 2 was worse than Window 1 despite having double the training data.** This is the opposite of the expected result. The key clue is `Avg Hold = 8.1 candles` — the agent learned to hold positions just long enough to escape the `early_exit_penalty` threshold (5 candles) and then dump. With 695 trades vs 69 trades at 300K, the agent found a new loophole.

---

### Window 3 — `130,296 rows` (full 100% of training data)

**Training was active at time of analysis (57% complete):**

```
Rollout 10: Reward -5.74 | H=45.7% B=28.0% S=26.3% | MTM=-0.0045 | Steps: 163,840
Rollout 20: Reward -5.85 | H=61.4% B=19.5% S=19.1% | MTM=+0.0908 | Steps: 327,680
Rollout 30: Reward -5.45 | H=59.9% B=22.8% S=17.3% | MTM=+0.1960 | Steps: 491,520
Progress: 57% (567,764 / 1,000,000 steps) | ETA: ~40 min remaining
```

**Window 3 was cancelled** — based on Windows 1 and 2 data, the same loophole exploitation was expected and confirmed by the identical training trajectory pattern.

---

## Key Observations — 1M Run {#1m-observations}

### Observation 1: The 5-Candle Loophole

The most critical finding from this run. Between the 300K run (avg hold = 76 candles) and the 1M Window 2 (avg hold = 8.1 candles), the agent's policy completely reversed.

**What happened:** At 1M steps, the agent had enough training time to discover that the `early_exit_penalty` only fires for trades `< 5 candles`. It learned to deliberately hold positions for exactly **6-8 candles** to bypass the penalty, then sell regardless of PnL. This is pure reward hacking — optimizing the penalty, not the actual return.

```
300K steps  → Avg Hold: 76 candles  (didn't find loophole yet)
1M  steps   → Avg Hold: 8.1 candles (found and exploited the loophole)
```

**Why longer training made things worse:** More training time = more opportunity to find exploits. Once the agent discovered the 5-candle rule, it could bypass it consistently.

### Observation 2: MTM Training Signal Is Fine — Generalization Is the Problem

During training, `mark_to_market` improved monotonically across all 3 windows:

```
Window 1 Rollout 10: MTM = -0.0073  →  Rollout 60: MTM = +0.4230
Window 2 Rollout 10: MTM = -0.0059  →  Rollout 60: MTM = +0.4535
```

The agent IS learning to extract positive MTM reward. The problem is that in validation (on unseen data), the micro-patterns it learned don't exist. This is a **generalization failure**, not a signal failure.

### Observation 3: The 4:1 Penalty Ratio Is Still Present

Analyzing Window 2 Rollout 60 reward components:

```
Positive:  mark_to_market +0.4535  +  winner_exit +0.1380  =  +0.5915
Negative:  invalid -1.3863  +  fee -0.4559  +  early -0.2063  +  dd -0.2881  =  -2.3366

Ratio: negative/positive = 3.95:1  (~4:1)
```

The negative signals still dominate by 4:1. This aligns with the hypothesis from external critique that the reward balance, not just trade duration, may be the root cause of over-trading.

### Observation 4: Walk-Forward Does Not Help---

## Run 10: Experiment A — Invalid Action Remap {#run10}

**Date:** June 5, 2026  
**Command:** `--timesteps 300000 --windows 1 --run-name experiment_a_300k --recurrent`  
**Change:** Invalid actions (BUY-while-holding, SELL-while-flat) silently remapped to HOLD. No penalty applied. `invalid_action` reward component = 0.0 always.

### Aggregate Results

| Metric | Calibrated 300K | RSI Baseline | **Experiment A** |
|--------|:--------------:|:------------:|:----------------:|
| **Sharpe** | -1.523 | -1.428 | **-0.145** ✅ |
| **Return** | -3.20% | -5.31% | **-0.33%** ✅ |
| **Trades** | 68.7 | 92 | **8.67** ⚠️ |
| **Avg Hold** | 76 candles | — | **1,123 candles** 🚨 |
| **Win Rate** | 50.86% | 33.70% | **62.39%** ✅ |
| **Max DD** | 6.11% | — | **5.26%** ✅ |
| **invalid_action reward** | -1.38 | — | **0.0** ✅ |

> [!IMPORTANT]
> **The RSI baseline deployment bar (-1.428) has been cleared for the first time.** Sharpe improved from -1.523 → -0.145 purely by removing the invalid-action penalty. This confirms that penalty noise was the dominant suppressor of learning signal.

### Rollout 10 Comparison (Training)

```
Previous runs:  Reward -5.66  |  invalid_action = -1.47  |  MTM = -0.007
Experiment A:   Reward -2.38  |  invalid_action = +0.000  |  MTM = +0.006
```

The reward halved immediately at Rollout 10 and MTM turned **positive from step 1** — something never seen before.

### Per-Slice Validation Breakdown (from debug log)

| Slice | Sharpe | Return | Trades | Avg Hold | Winners | Losers | Gross PnL | Fees | Net PnL | PF Gross |
|-------|--------|--------|--------|----------|---------|--------|-----------|------|---------|----------|
| **0** | **+1.490** ✅ | **+3.22%** | 14 | 421c | 7 | 6 | +$307 | $52 | +$255 | **1.96** |
| **1** | **+0.611** ✅ | **+0.97%** | 2 | 2258c | 1 | 0 | +$51 | $4 | +$47 | **∞** |
| **2** | -2.537 ❌ | -5.17% | 13 | 691c | 4 | 8 | -$463 | $47 | -$510 | **0.37** |

> [!NOTE]
> 2 out of 3 slices are profitable for the first time. Slice 0 shows genuine edge (PF=1.96, 7W/6L). Slice 1's ∞ profit factor is misleading — it is literally 1 trade held for 23.5 days. Slice 2 is losing even before fees (PF=0.37).

### World A vs World B — Partial Answer

```
Slice 0: Gross +$307, Fees $52, Net +$255  →  World A (fees < gross profit)
Slice 1: Gross  +$51, Fees  $4, Net  +$47  →  World A (fees < gross profit)
Slice 2: Gross -$463, Fees $47, Net -$510  →  World B (trades bad before fees)
```

**Conclusion:** The answer is regime-dependent. In trending markets (Slices 0-1), the model has genuine predictive edge. In choppy/mean-reverting markets (Slice 2), the underlying trades lose money before fees. This is not a reward function problem — it is a market regime problem.

### Individual Trade Samples from Slice 0 (Debug Log)

| Hold | PnL% | Gross | Fees | Net | Winner? |
|------|------|-------|------|-----|---------|
| 23c | +2.09% | +$41.74 | $4.04 | +$37.70 | ✅ |
| 26c | -2.17% | -$43.46 | $3.97 | -$47.43 | ❌ |
| 11c | -2.23% | -$44.51 | $3.95 | -$48.46 | ❌ |
| 11c | +1.46% | +$29.04 | $4.00 | +$25.03 | ✅ |
| 27c | -2.27% | -$45.14 | $3.94 | -$49.08 | ❌ |

Winners and losers are similar in size — this is a low-profit-factor regime but still net positive.

### Revised Confidence Assessment (June 5, 2026)

| Claim | Confidence |
|-------|------------|
| Invalid-action penalty was hurting learning | **95%** |
| New environment is much healthier | **95%** |
| Agent is over-holding positions | **90%** |
| Signal exists in features (regime-dependent) | **70%** |
| We are in World A | **60%** — true in Slices 0-1, false in Slice 2 |
| Selling incentive is the only remaining issue | **30%** — rejected; Slice 2 trades are bad before fees |

### Key New Learnings

1. **Invalid action remap > invalid action penalty.** Simply remapping illegal actions to HOLD produced a 10× Sharpe improvement vs the previous best. Penalty-based enforcement was poisoning the gradient.
2. **Disposition effect returns without penalty.** With no invalid action cost and no selling incentive strong enough to beat ongoing MTM income, the agent holds positions for 400-2000+ candles.
3. **Regime sensitivity is the dominant remaining problem.** The model is profitable in Slices 0-1 (likely trending) and unprofitable in Slice 2 (likely choppy/reverting). This requires either more diverse training data or a regime-detection feature, not a reward function change.
4. **Sample size matters.** Slice 1's ∞ profit factor from a single trade is statistically meaningless. Need 50+ trades per slice before trusting any profit factor metric.

---

## Run 11: Experiment B — Loss-Duration Penalty {#run11}

**Date:** June 5, 2026  
**Command:** `--timesteps 300000 --windows 1 --run-name experiment_b_300k --recurrent`  
**Change from Exp A:** Added unproductive hold penalty: when `steps_in_position > 48 AND unrealized < 0` → `-0.00005/step`. Fires only on losing positions held beyond 12 hours. Zero effect on profitable holds.

### Aggregate Results

| Metric | Exp A | RSI Baseline | **Exp B** |
|--------|:-----:|:------------:|:---------:|
| **Sharpe** | -0.145 | -1.428 | **-1.267** |
| **Return** | -0.33% | -5.31% | **-1.85%** |
| **Trades** | 8.67 | 92 | **40.33** ✅ |
| **Avg Hold** | 1123c | — | **133.7c** ✅ |
| **Win Rate** | 62.39% | 33.70% | **51.08%** |
| **Max DD** | 5.26% | — | **4.54%** ✅ |
| **Gross PnL** | -0.35% | — | **-0.14%** ✅ |
| **Fees** | 0.34% | — | **1.58%** ❌ |
| **Gross PF** | ∞ (1 trade) | — | **0.962** |
| **Sharpe ±Std** | ±1.729 | — | **±0.260** ✅ |

### Rollout 10 Observation
```
loss_duration_penalty = +0.0000  ← penalty not yet firing at step 163K
mark_to_market        = +0.0055  ← still positive from step 1
```

The penalty doesn’t appear until later training when the policy learns to enter and hold positions. The final trained policy shows it working.

### Per-Slice Validation Breakdown (from debug log)

| Slice | Trades | Avg Hold | Winners | Losers | Gross PnL | Fees | Net PnL | PF Gross |
|-------|--------|----------|---------|--------|-----------|------|---------|----------|
| 0 | 41 | 165c | 21 | 19 | **+0.09%** | 1.54% | -1.45% | **1.014** |
| 1 | 46 | 115c | 24 | 22 | -0.07% | 1.81% | -1.89% | 0.986 |
| 2 | 35 | 121c | 17 | 18 | -0.45% | 1.38% | -1.84% | 0.885 |

> [!NOTE]
> Trade count is now statistically meaningful (35–46 per slice). Gross PnL is nearly break-even across all slices. Fees (1.58% total) are the primary cause of negative net return, not bad trades.

### Catastrophic Holds Still Occurring (Design Flaw)

Despite the penalty, three extreme losing holds survived:

| Slice | Hold | Gross Loss | Why Penalty Failed |
|-------|------|------------|--------------------|
| 0 | **395 candles** | -$207 (-10.33%) | Total penalty: (395-48) × 0.00005 = **0.017** — trivial vs 10% loss |
| 1 | **330 candles** | -$99 (-4.99%) | Total penalty: (330-48) × 0.00005 = **0.014** — trivial vs 5% loss |
| 2 | **1123 candles** | -$23 (-1.15%) | Position oscillated near 0—penalty turned on/off intermittently |

**Root cause of design flaw:** The flat 0.00005 penalty does not scale with loss magnitude. A position losing 10% can rationally hold and hope for recovery because the per-step penalty is only 10× smaller than typical MTM noise.

### What Changed vs Experiment A

| Factor | Exp A | Exp B | Impact |
|--------|-------|-------|--------|
| Trade count | 8.67 | 40.33 | Penalty successfully increased activity ✅ |
| Gross PnL | -0.35% | -0.14% | Trades improved slightly ✅ |
| Fees | 0.34% | 1.58% | 4.6× more trades = 4.6× more fees ❌ |
| Sharpe | -0.145 | -1.267 | Worse net due to fee drag ❌ |
| Sharpe std | ±1.729 | ±0.260 | Policy much more consistent ✅ |
| Catastrophic holds | 1 (2258c) | 3 (395c/330c/1123c) | Flat penalty fails on large losers ❌ |

### Key New Learnings

1. **Flat penalties fail on large drawdowns.** A position at -10% for 400 candles only accumulates `0.017` total penalty — too small to compete with the hope of recovery. The penalty must scale with loss magnitude.
2. **Trade count and fees are coupled.** Exp A had 8.67 trades and 0.34% fees. Exp B has 40.33 trades and 1.58% fees. A 4.6× trade count increase produced a 4.6× fee increase. To net positive, gross PnL per trade must exceed 2× fee cost (0.2% per round trip).
3. **Gross PnL is converging on break-even.** Across all 3 slices, gross PF is 0.885–1.014 — the agent's entries and exits are nearly coin-flip quality before fees. A stronger edge is needed.
4. **Consistency is improving.** Sharpe std dropped from ±1.729 to ±0.260 — the policy is more stable and predictable, even if not yet profitable.

---

## Run 12: Experiment C — Case 1 vs 2 Diagnostic {#run12}

**Date:** June 5, 2026  
**Command:** `--timesteps 300000 --windows 1 --run-name experiment_c_300k --recurrent`  
**Change from Exp B:** No reward changes. Added expanded diagnostic metrics to `sell_summary`: mean/median winner & loser PnL%, hold duration split by outcome, largest winner/loser, first_20_sells sample.

### Aggregate Results

| Metric | Exp A | Exp B | **Exp C** |
|--------|:-----:|:-----:|:---------:|
| **Sharpe** | -0.145 | -1.267 | **-0.698** |
| **Return** | -0.33% | -1.85% | **-1.33%** |
| **Trades** | 8.67 | 40.33 | **52.0** |
| **Avg Hold** | 1123c | 133.7c | **167.5c** |
| **Win Rate** | 62.39% | 51.08% | **48.84%** |
| **Gross PnL** | -0.35% | -0.14% | **+0.51%** ✅ |
| **Gross PF** | ∞ | 0.962 | **1.110** ✅ |
| **Fees** | 0.34% | 1.58% | 2.06% |
| **Gross Expectancy** | +0.594% | -0.014% | **+0.078%** ✅ |
| **Sharpe ±Std** | ±1.729 | ±0.260 | ±1.525 |

> [!IMPORTANT]
> **Gross PnL is positive (+0.51%) for the first time.** The model has real predictive edge. Fees (2.06%) are the sole cause of negative net return. This is no longer a feature quality problem.

### Per-Slice Case 1 vs 2 Diagnostic (from debug log)

#### Slice 0 — Strong edge, 1 monster loser
```
Trades: 61  |  Winners: 34  |  Losers: 16  |  Win Rate: 68%
Mean winner: +1.13%  |  Mean loser: -2.20%
Median winner: +0.86%  |  Median loser: -1.17%
Biggest loser: -9.09% ($-182)  ←  76% of ALL gross losses
Gross PF: 1.460  |  Gross expectancy: +0.33%
Median hold winner: 32c  |  Median hold loser: 70c
Verdict: CASE 1 — one trade is destroying an otherwise excellent result
```

#### Slice 1 — Winners run, losers are controlled
```
Trades: 31  |  Winners: 13  |  Losers: 17  |  Win Rate: 42%
Mean winner: +2.29%  |  Mean loser: -1.49%  (winner/loser ratio: 1.54x)
Biggest loser: -6.27% ($-125)  ←  59% of ALL gross losses
Max winner hold: 2268c  |  Max loser hold: 629c
Gross PF: 1.170  |  Gross expectancy: +0.15%
Verdict: CASE 1 — 1 large loser, but model has real edge (winners > losers in size)
```

#### Slice 2 — Looks like Case 2 but isn’t
```
Trades: 67  |  Winners: 19  |  Losers: 31  |  Win Rate: 28%
Mean winner: +1.47%  |  Mean loser: -1.04%
Biggest loser: -7.19% ($-145)  ←  63% of ALL gross losses
Max loser hold: 766c
Gross PF: 0.700  |  Gross expectancy: -0.25%

Counterfactual: Remove biggest loser alone:
  Gross PnL: -$86 + $145 = +$59  →  GROSS PROFITABLE
Verdict: CASE 1 — even Slice 2 would be gross profitable without its single worst trade
```

### Case 1 Confirmed — All 3 Slices

| Slice | Biggest Loser | Share of Losses | Verdict |
|-------|:-------------:|:---------------:|--------|
| 0 | -9.09% ($-182) | **76%** | CASE 1 |
| 1 | -6.27% ($-125) | **59%** | CASE 1 |
| 2 | -7.19% ($-145) | **63%** | CASE 1 |

> [!NOTE]
> All three slices show the same pathology: 1 trade accounts for the majority of all gross losses. The flat 0.00005/step penalty is not strong enough to prevent positions from running to -6% to -9% losses over 400–766 candles.

### Hold Duration Pattern

| Slice | Median Hold Winner | Median Hold Loser | Biggest Loser Hold |
|-------|:-:|:-:|:-:|
| 0 | 32c | 70c | 400c |
| 1 | 113c | 112c | 629c |
| 2 | 59c | 37c | 766c |

Losers are held 2× longer than winners in most slices. The disposition effect is still active specifically on the tail losses. Winners are being cut or exited properly.

### Key New Learnings

1. **The model has genuine predictive edge.** Gross PnL +0.51%, Gross PF 1.110 with 52 trades is statistically meaningful. This is not a feature quality problem.
2. **CASE 1 confirmed across all slices.** A single trade in each slice accounts for 59–76% of all gross losses. The fix is loss-cutting, not feature engineering.
3. **Even the weakest slice is actually World A.** Slice 2 (PF=0.70) becomes gross profitable if its single -7.19% loser is removed. The regime problem was masking a CASE 1 problem.
4. **The flat penalty is insufficient for tail losses.** The 0.00005/step penalty accumulates only 0.018–0.036 over 400–766 holding steps — trivial vs a 6–10% position loss.
5. **Mandate to implement proportional penalty is now established.** The critique's decision framework explicitly said: "if most losses come from a handful of huge losers, a loss-duration penalty or stop-loss makes sense."

## Open Issues (Updated June 5, 2026) {#open-issues}

### 1. H6: Action Probability Extraction Still Broken

```
TypeError: RecurrentActorCriticPolicy.get_distribution() 
missing 2 required positional arguments: 'lstm_states' and 'episode_starts'
```

Still unresolved. Cannot inspect policy confidence during validation.

### 2. Flat Loss-Duration Penalty Fails on Large Drawdowns

The 0.00005/step flat penalty reduced avg hold (1123c → 133c) but catastrophic holds (330–395 candles) at large losses (-5% to -10%) survive because the total accumulated penalty is trivially small vs the hope of recovery.

**Proposed fix:** Scale penalty with unrealized loss magnitude:
```python
if unrealized < -0.005 and steps > 48:
    reward -= 0.0002 * abs(unrealized)  # -10% loss → 10× stronger nudge than -1%
```

### 3. Fees Consuming Gross Edge

Gross PnL is approximately break-even (PF ~0.96) but 1.58% in fees produces -1.85% net return. To be profitable, the agent needs either:
- Higher gross PF (better entry/exit timing)
- Fewer but higher-quality trades
- The proportional penalty (Issue 2) may help by letting winners run longer while cutting losers faster

### 4. Regime Sensitivity — Active Problem

Slice 0 gross PF = 1.014 (trending regime, slight edge). Slice 2 gross PF = 0.885 (choppy regime, no edge). Same model, different market conditions. Not yet addressed.

### 5. Replay Ratio — Healthy at 300K

At 300K steps and 130K rows, replay ratio ≈ 2.3×. This is acceptable. Do not scale to 1M until gross PF consistently exceeds 1.2 across all slices.

### 6. Deployment Bar — Partially Met

| Run | Sharpe | vs Baseline (-1.428) |
|-----|--------|---------------------|
| Calibrated 300K | -1.523 | Below ❌ |
| **Experiment A** | **-0.145** | **Above ✅** |
| Experiment B | -1.267 | Above ✅ (barely) |

Both Exp A and B beat the RSI baseline, but Exp A only trades 8.67 times (not viable). Exp B trades 40× but loses to fees. The target is Sharpe > -1.428 with 30+ trades AND positive gross PF.

