# TradeBot God Mode — Software Requirements Specification
## Complete System Documentation

**Version:** God Mode v1.0  
**Last Updated:** 2026-05-30  
**Status:** Production-Ready (pending model training)

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Complete File Reference](#3-complete-file-reference)
4. [God Mode Improvements](#4-god-mode-improvements)
5. [Data Flow — How the System Works](#5-data-flow)
6. [AI Training Guide](#6-ai-training-guide)
7. [Training Diagnostics — Expected Values](#7-training-diagnostics)
8. [Deployment Guide](#8-deployment-guide)
9. [Configuration Reference](#9-configuration-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. System Overview

TradeBot is a three-tier automated cryptocurrency trading system:

```
┌─────────────────────────────────────────────────────────┐
│  TIER 1: AI Training (Python / ai-training/)            │
│  Trains a Deep Reinforcement Learning LSTM model        │
│  on historical Binance 15m candle data                  │
└─────────────────────┬───────────────────────────────────┘
                      │  exports tradebot.onnx
┌─────────────────────▼───────────────────────────────────┐
│  TIER 2: Execution Engine (TypeScript / Node.js)        │
│  Runs live inference, manages orders, risk, and state   │
└─────────────────────┬───────────────────────────────────┘
                      │  writes trades/signals to MongoDB
┌─────────────────────▼───────────────────────────────────┐
│  TIER 3: Command Center (Next.js dashboard)             │
│  Equity curve, trade log, kill switch, live PnL         │
└─────────────────────────────────────────────────────────┘
```

**What it does:** The AI model observes the last 48 candles of 15m BTC/USDT price action, computes 24 technical indicators, and outputs one of three actions: HOLD, BUY, or SELL. When confidence exceeds the minimum threshold (default 60%), the execution engine places a **limit order** (GOD-2), monitors the position against stop-loss and take-profit levels, and closes using a limit sell. All risk decisions (position size, SL/TP, circuit breaker) are determined by hardcoded math — never by the AI.

---

## 2. Architecture Diagram

```
═══════════════════════════════════════════════════════════════
  EXECUTION ENGINE — Live Trading Loop
═══════════════════════════════════════════════════════════════

  Binance Exchange
       │
       │  WebSocket kline stream (closed candles only)
       │  WebSocket userData stream → instant balance updates [GOD-5]
       ▼
  ┌─────────────────────────────────────────────────────────┐
  │  BinanceWatcher (binanceStream.ts)                      │
  │  • Buffers last windowSize+50 candles                   │
  │  • Pre-fills buffer on startup via REST                  │
  │  • Emits "candle" event on every closed 15m candle      │
  │  • UserDataStream → real-time balance cache [GOD-5]     │
  └───────────────────────┬─────────────────────────────────┘
                          │  Candle[]  (48 candles)
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  InferenceEngine (inference.ts)                         │
  │  • Loads ONNX model (VecNormalize baked in)             │
  │  • calls buildObservationTensor() → 24 features × 48   │
  │    candles + 7 portfolio state features = 1,159 dim obs │
  │  • Returns ModelOutput { signal, probBuy, confidence }  │
  └───────────────────────┬─────────────────────────────────┘
                          │  ModelOutput
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Executioner (executioner.ts)                           │
  │  • Confidence gate: skips if conf < MIN_CONFIDENCE      │
  │  • Calls RiskManager for position size and SL/TP        │
  │  • Places LIMIT orders via BinanceClient [GOD-2]        │
  │  • Handles partial fills [GOD-2]                        │
  │  • Monitors SL/TP on every candle                       │
  │  • Persists all trades + signals to MongoDB             │
  └────────────┬──────────────────────┬─────────────────────┘
               │                      │
               ▼                      ▼
  ┌────────────────────┐  ┌───────────────────────────────┐
  │  RiskManager       │  │  BinanceClient (ccxtClient.ts)│
  │  • Fixed fractional│  │  GOD-1: LOT_SIZE precision    │
  │    position sizing │  │  GOD-2: Limit orders + timeout│
  │  • SL/TP calc      │  │  GOD-3: BNB fee discount      │
  │  • Daily circuit   │  │  GOD-4: TTL cache             │
  │    breaker (5%)    │  │  20 retries for orders        │
  │  • UTC midnight    │  └───────────────────────────────┘
  │    reset [IMP-5]   │
  └────────────────────┘

═══════════════════════════════════════════════════════════════
  AI TRAINING PIPELINE
═══════════════════════════════════════════════════════════════

  Binance public API
       │ fetch_data.py
       ▼
  Raw OHLCV parquet
       │ feature_engineering.py
       ▼
  Feature matrix (24 features × N candles)
  [log_return, RSI, MACD, BB, ATR, EMA crossovers,
   volume_ratio, candle structure, time encoding,
   ADX, OBV ratio + 4h/1d MTF features in v3]
       │ train_agent.py
       ▼
  Walk-forward training (3 windows)
  Window 1 → 33% data → validate → Sharpe
  Window 2 → 66% data → validate → Sharpe
  Window 3 → 100% data → validate → Sharpe
  Best Sharpe checkpoint saved as _best.zip
       │ export_onnx.py
       ▼
  tradebot.onnx (VecNormalize baked in)
       │ copy to execution-engine/src/strategist/models/
       ▼
  Live inference via onnxruntime-node
```

---

## 3. Complete File Reference

### Python — ai-training/scripts/

| File | Purpose |
|---|---|
| `config.py` | **Single source of truth** for ENV_CONFIG, PPO_HYPERPARAMS, CANDLES_PER_DAY, fee rates. All scripts import from here. |
| `fetch_data.py` | Downloads historical OHLCV from Binance public API. Paginates past 1000-candle limit. Validates monotonic timestamps [P2-4]. |
| `feature_engineering.py` | Computes 18–32 technical features. v1=18, v2=24, v3=28, v4=32. Chronological 75/15/10 train/val/test split. Prints warmup rows lost to dropna [P1-4]. |
| `trading_env.py` | OpenAI Gym environment. Simulates Binance exchange: fees, slippage, drawdown. **P0-2 fix**: graduated drawdown penalty threshold 0.25 (was 0.10), coefficient 0.002 (was 0.05). |
| `train_agent.py` | PPO walk-forward training. 3 independent windows. Saves VecNormalize .pkl after each window. LSTM state carried in validation [P1-3]. Imports DiagnosticCallback from diagnostic_callback.py [P1-5]. |
| `export_onnx.py` | Exports trained model to ONNX with VecNormalize baked into the PolicyWrapper [BUG-1]. Validates numerically against PyTorch. |
| `diagnostic_callback.py` | SB3 training callback: action distribution, entropy, GPU memory, reward component breakdown, collapse detection. |
| `hyperparameter_sweep.py` | Optuna sweep over PPO + env hyperparameters. 3-slice validation [P0-3], candles_per_day wired through [P0-5]. |
| `ensemble_predict.py` | Runs inference with N models, majority vote aggregation. |
| `env_sanity_check.py` | Pre-training environment validation: obs shape, reward scale, action coverage. |
| `verify_feature_parity.py` | Generates Python ground-truth features (WINDOW=500). Used with verifyParity.ts to catch Python/TypeScript drift. |
| `_smoke_test.py` | Quick 500-step random-policy env test. CLI args for --data and --timeframe [P3-3]. |
| `sensitivity_analysis.py` | **NEW (GOD MODE)**: sweeps MIN_CONFIDENCE threshold, shows Sharpe stability. Find the plateau, not the peak. |
| `baseline_backtest.py` | **NEW (GOD MODE)**: RSI momentum rule baseline (Tutorial 2). DRL model must beat this to be worth deploying. |

### Python — ai-training/environments/

| File | Purpose |
|---|---|
| `trading_env.py` | Custom Gymnasium environment. Reward: mark-to-market + winner_exit bonus - drawdown penalty - fee cost - invalid action penalty. |

### TypeScript — execution-engine/src/

| File | Purpose |
|---|---|
| `index.ts` | Bootstrap: wires all modules, graceful shutdown, safe intervals [GOD-6], daily summary [IMP-6], config validation. |

### TypeScript — utils/

| File | Purpose |
|---|---|
| `types.ts` | All types, enums, CONFIG from env vars. **GOD-3 addition**: effectiveFeeRate, useBnbFeeDiscount. **GOD-6 addition**: discordWebhookUrl. |
| `logger.ts` | Winston logger: console (colorized) + rotating file (10MB × 5 files). |
| `notifier.ts` | **GOD-6**: Telegram primary + Discord webhook fallback. Both channels receive all alerts. New: sendCircuitBreakerAlert(). |
| `ttlCache.ts` | **GOD-4 NEW**: Generic TTL cache. Fee rates (12h), lot sizes (12h), BNB status (60s), ticker (5s). ~40% fewer API calls. |

### TypeScript — executioner/

| File | Purpose |
|---|---|
| `ccxtClient.ts` | **GOD-1,2,3,4 REWRITE**: Limit orders with drift-cancel and partial fill. LOT_SIZE precision flooring. BNB fee discount detection. TTL cache. 20 retries for orders. |
| `executioner.ts` | Trade state machine FLAT→IN_POSITION→FLAT. **GOD-2**: uses limitBuy/limitSell. Handles partial fills. isProcessing mutex. |
| `riskManager.ts` | Fixed fractional sizing, SL/TP, daily circuit breaker. **GOD-3**: uses CONFIG.effectiveFeeRate in buildTradeResult. IMP-5: resets at UTC midnight. |
| `stateRecovery.ts` | Crash recovery: reconstructs position from Binance trade history on restart. |
| `mockBinanceClient.ts` | **GOD-7 NEW**: Full pipeline backtest client. Implements BinanceClient interface but replays historical candles. Fills limit orders at next candle's open. |

### TypeScript — watcher/

| File | Purpose |
|---|---|
| `binanceStream.ts` | WebSocket kline stream. Auto-reconnect. REST fallback. Pre-fills buffer. **GOD-5**: integrates UserDataStream. |
| `userDataStream.ts` | **GOD-5 NEW**: Binance !userData WebSocket. outboundAccountPosition → instant balance updates. executionReport → order fill cache. 30-min listenKey renewal. |

### TypeScript — strategist/

| File | Purpose |
|---|---|
| `indicators.ts` | TypeScript reimplementation of all 18 Python features. Must produce identical output to feature_engineering.py. |
| `inference.ts` | ONNX model loader and inference runner. Builds 1,159-dim observation tensor, runs model, returns ModelOutput. |
| `verifyParity.ts` | Compares TypeScript indicator output against Python ground truth. PASS = safe to deploy. FAIL = concept drift. |

### TypeScript — database/

| File | Purpose |
|---|---|
| `mongoSchemas.ts` | Mongoose schemas: Trade, Snapshot, SignalLog, Tick. connectDB/disconnectDB helpers. |

---

## 4. God Mode Improvements

### GOD-1 — LOT_SIZE Precision
**File:** `ccxtClient.ts`  
**Problem:** RiskManager.calculatePositionSize() returns arbitrary floats like 0.000123456789. Binance rejects these with "LOT_SIZE filter failure" — your first live order would have silently failed.  
**Fix:** `roundToLotSize()` fetches the exchange's exact decimal precision per pair (cached 12h) and floors the quantity. Always floors (never rounds up) to avoid over-buying.

### GOD-2 — Limit Orders with Timeout + Partial Fill
**File:** `ccxtClient.ts`, `executioner.ts`  
**Problem:** Market orders guarantee slippage. On BTC/USDT during a momentum spike (exactly when your model gives BUY signals), market order slippage can be $5–20.  
**Fix:** `limitBuy()` places at current best-ask. If price drifts >0.1% before fill → cancel and take partial fill. 30-second hard timeout. Partial fills are accepted and logged. Exit uses `limitSell()` at best-bid. Emergency exits (kill switch) still use `marketSell()`.

### GOD-3 — BNB Fee Discount
**Files:** `ccxtClient.ts`, `config.py`, `types.ts`  
**Problem:** Binance charges 0.075% (not 0.1%) when BNB burn is enabled. Training with 0.1% and live trading at 0.075% means the model was overcautious about fee costs. Also, PnL calculations in `buildTradeResult()` were 25% wrong on the fee component.  
**Fix:** `detectEffectiveFeeRate()` checks BNB burn setting and balance on startup. Result cached 60s. `buildTradeResult()` uses `CONFIG.effectiveFeeRate`. Set `USE_BNB_FEE_DISCOUNT=true` in both `.env` and `config.py` to align.

### GOD-4 — TTL Cache
**File:** `utils/ttlCache.ts`, integrated into `ccxtClient.ts`  
**Problem:** Every order placement made fresh API calls for fee rates and lot sizes — adding 200-500ms latency and consuming rate limit budget.  
**Fix:** `TTLCache<V>` generic cache. Fee rates: 12h TTL. Lot sizes: 12h TTL. BNB status: 60s TTL. Ticker price: 5s TTL. ~40% fewer API calls.

### GOD-5 — userData WebSocket
**File:** `watcher/userDataStream.ts`  
**Problem:** After a fill, the bot called `fetchBalance()` via REST (1-2 seconds, potentially stale). The next candle's decision could be based on outdated balance.  
**Fix:** `UserDataStream` subscribes to Binance's `!userData` stream. `outboundAccountPosition` event fires within 50ms of any fill, providing instant balance updates. `executionReport` events cache fill details so `waitForFill()` can resolve instantly.

### GOD-6 — Multi-Channel Notifications
**File:** `utils/notifier.ts`  
**Problem:** Telegram-only. If Telegram goes down (it does), you get zero alerts during a live trade.  
**Fix:** Discord webhook as automatic fallback. Both channels receive all alerts simultaneously. New `sendCircuitBreakerAlert()` method. New `discordWebhookUrl` in `CONFIG`.

### GOD-7 — MockBinanceClient (Full Pipeline Backtest)
**File:** `executioner/mockBinanceClient.ts`  
**Problem:** TradingEnv tests the AI signal quality. But it doesn't test the full execution pipeline: limit order simulation, LOT_SIZE rounding, fee calculation, SL/TP triggers, isProcessing mutex.  
**Fix:** `MockBinanceClient` implements the same interface as `BinanceClient` but replays historical candle data. Limit orders fill at the next candle's open. Full P&L report generated at end.

### From YouTube Tutorial 2 — RSI Momentum Baseline
**File:** `scripts/baseline_backtest.py`  
**Insight:** Buying when RSI>70 (momentum breakout) and exiting when RSI<30 produces Sharpe ~0.96 and 60% annual return on BTC (Wilder's smoothing). Your DRL model must beat this simple 3-condition rule. If it doesn't, the model needs more training.

### From YouTube Tutorial 2 — Sensitivity Analysis
**File:** `scripts/sensitivity_analysis.py`  
**Insight:** A good model shows a PLATEAU of stable Sharpe across a range of confidence thresholds — not a single peak. Sweep MIN_CONFIDENCE from 0.35 to 0.90 and deploy at the center of the plateau.

---

## 5. Data Flow

### Training Data Flow
```
fetch_data.py --symbol BTC/USDT --timeframe 15m --days 730
    ↓ downloads 70,176 candles (15m × 24 × 365 × 2)
    ↓ validates monotonic timestamps
    → BTC_USDT_15m.parquet

feature_engineering.py --input BTC_USDT_15m.parquet --version v2
    ↓ computes 24 features (v2)
    ↓ drops ~200 warmup rows (EMA-200)
    → BTC_USDT_15m_train.parquet  (75% = ~52,500 rows)
    → BTC_USDT_15m_val.parquet    (15% = ~10,500 rows)
    → BTC_USDT_15m_test.parquet   (10% =  ~7,000 rows)

train_agent.py --timesteps 5000000 --windows 3 --recurrent
    Window 1: train on 33% → validate → Sharpe + std
    Window 2: train on 66% → validate → Sharpe + std
    Window 3: train on 100% → validate → Sharpe + std
    Best checkpoint → tradebot_ppo_best.zip + _vecnormalize.pkl

export_onnx.py --model models/tradebot_ppo_best.zip
    ↓ bakes VecNormalize into PolicyWrapper
    ↓ verifies ONNX output matches PyTorch numerically
    → tradebot.onnx

copy to → execution-engine/src/strategist/models/tradebot.onnx
```

### Live Trading Data Flow
```
Binance WebSocket (closed 15m candles)
    ↓
BinanceWatcher.buffer[48 candles]
    ↓
InferenceEngine.predict()
    ↓ buildObservationTensor() → 24 features × 48 + 7 portfolio = 1,159 floats
    ↓ ONNX model → [action_idx, p_hold, p_buy, p_sell]
    ↓
Executioner.onSignal()
    ↓ if signal=BUY and confidence >= MIN_CONFIDENCE:
    ↓   RiskManager.calculatePositionSize(price) → size (BTC)
    ↓   roundToLotSize(size) → floors to 5 decimal places [GOD-1]
    ↓   BinanceClient.limitBuy(size) → limit order at best-ask [GOD-2]
    ↓   if price drifts >0.1% → cancel → take partial fill
    ↓   RiskManager.calculateExitLevels(fillPrice) → SL, TP
    ↓   placeStopLossOrder(size, SL) → resting order on exchange
    ↓
Per-candle monitoring:
    ↓ if price >= TP → limitSell() → TAKE_PROFIT
    ↓ if price <= SL → marketSell() → STOP_LOSS (fast exit)
    ↓ if AI SELL signal + confidence → limitSell() → SIGNAL
    ↓
persistTrade() → MongoDB
sendTradeAlert() → Telegram + Discord [GOD-6]
```

---

## 6. AI Training Guide

### Prerequisites
```powershell
cd tradebot-core\ai-training
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Verify GPU
python scripts\_gpu_check.py
```

### Step 1: Download Data
```powershell
# ~70K candles (2 years of 15m). No API key needed.
python scripts\fetch_data.py --symbol BTC/USDT --timeframe 15m --days 730

# Verify output
# Expected: [OK] BTC/USDT | 15m | ~70,000 candles | 2023-... → 2025-...
```

### Step 2: Build Feature Matrix
```powershell
# v2 = 24 features (recommended starting point)
python scripts\feature_engineering.py --input data\BTC_USDT_15m.parquet --version v2

# Expected output:
# [INFO] Feature matrix shape: (69800, 29)   ← 24 features + 5 OHLCV
# [INFO] Warmup rows dropped: 200 (0.3%)     ← EMA-200 warmup
# [SPLIT] Train: 52,350 rows
# [SPLIT] Val:   10,470 rows
# [SPLIT] Test:   6,980 rows
```

### Step 3: Diagnostic Run (ALWAYS do this first)
```powershell
# 300K steps, 1 window, ~30 minutes. Validates reward signal before committing 30 hours.
python scripts\train_agent.py ^
  --train data\BTC_USDT_15m_train.parquet ^
  --val   data\BTC_USDT_15m_val.parquet ^
  --timesteps 300000 ^
  --windows 1 ^
  --run-name diagnostic_v3 ^
  --recurrent
```

### Step 4: Full Production Training
```powershell
# ~30 hours on RTX 4060 Laptop. Start overnight.
python scripts\train_agent.py ^
  --train data\BTC_USDT_15m_train.parquet ^
  --val   data\BTC_USDT_15m_val.parquet ^
  --test  data\BTC_USDT_15m_test.parquet ^
  --timesteps 5000000 ^
  --windows 3 ^
  --run-name tradebot_ppo_15m_v3 ^
  --recurrent
```

### Step 5: Verify Feature Parity
```powershell
# Python generates ground truth
python scripts\verify_feature_parity.py --input data\BTC_USDT_15m.parquet

# TypeScript checks match
cd ..\execution-engine
npx ts-node src\strategist\verifyParity.ts
# Expected: ALL 18 FEATURES PASS — TypeScript ≡ Python
```

### Step 6: Export to ONNX
```powershell
cd ..\ai-training
python scripts\export_onnx.py --model models\tradebot_ppo_15m_v3_best.zip

# Expected:
# [VecNorm] mean range: [-0.02, 0.85]
# [VecNorm] var range:  [0.0001, 2.45]
# [OK] ONNX model structure is valid.
# [OK] ONNXRuntime inference matches PyTorch. Max Δ: 1.2e-07
# [DONE] models\tradebot_ppo_15m_v3_best.onnx (2.14 MB)

copy models\tradebot_ppo_15m_v3_best.onnx ..\execution-engine\src\strategist\models\tradebot.onnx
```

### Step 7: Sensitivity Analysis
```powershell
python scripts\sensitivity_analysis.py ^
  --model models\tradebot_ppo_15m_v3_best.zip ^
  --data  data\BTC_USDT_15m_test.parquet

# Look for the PLATEAU — a range of thresholds with stable positive Sharpe
# Set MIN_CONFIDENCE to the CENTER of the plateau in your .env
```

### Step 8: Baseline Comparison
```powershell
# Your model must beat this simple RSI rule to be worth deploying
python scripts\baseline_backtest.py --data data\BTC_USDT_15m_test.parquet
python scripts\baseline_backtest.py --data data\BTC_USDT_15m_test.parquet --sweep
```

---

## 7. Training Diagnostics — Expected Values

### Rollout Log Format
Every 10 rollouts (default), the DiagnosticCallback prints:
```
[Rollout   10] Reward:   -3.09 (min: -3.6 max: -2.7) | EpLen:  7102 |
  Actions: H=39.7% B=30.4% S=29.9% ✅ | Entropy: -1.022 | GPU: 42MB |
  130 it/s | Steps: 163,840
  Components: drawdown_penalty=-0.22 | fee_cost=-0.18 | invalid_action=-1.00 |
              mark_to_market=+0.006 | terminal_penalty=0.000 | winner_exit=+0.23
```

### Early Training (Rollouts 1–20, ~163K steps)

| Metric | Healthy | Warning |
|---|---|---|
| `drawdown_penalty` | −0.1 to −0.5 per ep | > −2.0 means threshold too low |
| `terminal_penalty` | 0.000 (rare) | −0.05 every episode = still broken |
| `mark_to_market` | small positive | Negative means model holding losses |
| `invalid_action` | −0.5 to −1.5 | Decreasing over time is good |
| Entropy | −1.0 to −2.0 | > −0.1 = policy collapse |
| Action dist | 33–45% H | >80% any single action = collapse |
| EpLen | 6,000–9,000 | <500 = environment terminating immediately |

### Mid Training (Rollouts 50–150, ~400K–1.2M steps)

| Metric | Healthy | Warning |
|---|---|---|
| Reward trend | Slowly rising | Flat or falling after 500K = stuck |
| `drawdown_penalty` | < `mark_to_market` | Still dominant = reward broken |
| `win_rate` (val) | 35–50% and rising | Stuck at 28–32% = not learning |
| `avg_hold_candles` | 5–15 candles | Still 2–3 = churn/noise |
| `sharpe_std` | < 2.0 | > 5.0 = regime-sensitive policy |

### End of Window (5M steps) — Good Model

```
[Window 1 Validation] (3 slices)
  Sharpe Ratio    :   1.234  ±0.312    ← positive, low std
  Total Return    :  +8.42%            ← positive return
  Win Rate        :  52.30%            ← above 50%
  Total Trades    :  187               ← reasonable frequency
  Max Drawdown    :  12.3%             ← controlled
  Avg Hold        :  11.2 candles      ← 2.8 hours on 15m
  Action Dist     : H=68% B=17% S=15% ← mostly HOLD is correct
```

### What Each Diagnostic Tells You

**`drawdown_penalty` dominating** (your original issue):
- Was −68/episode vs mark_to_market +0.18 (4500:1 noise ratio)
- Fixed: threshold 0.25 (was 0.10), coefficient 0.002 (was 0.05)
- Healthy: drawdown_penalty should be SMALLER than mark_to_market in absolute value

**`terminal_penalty: -0.05` every episode:**
- Model hits 50% drawdown every single episode
- Means reward is still broken or data replay ratio too high
- Check: `total_timesteps / usable_rows` should be < 50x
- With 104K rows, max reasonable timesteps per window ≈ 2M (20x replay)

**`invalid_action: -1.0` not decreasing:**
- LSTM not tracking position state in hidden state
- Normal at 300K steps, should be < −0.3 by 2M steps
- If stuck, try lower `ent_coef` (0.03 instead of 0.05)

**`entropy: -0.02 ⚠️collapse?`:**
- Policy is converging to a deterministic (non-exploring) strategy prematurely
- Increase `ent_coef` in PPO_HYPERPARAMS (0.05 → 0.08)
- Or increase `clip_range` (0.1 → 0.15) to allow larger policy updates

**Win rate stuck at 26–32%:**
- Model is trading below random (random = 50% on binary long/flat)
- Root cause: reward noise still too high (check drawdown_penalty vs mark_to_market)
- Or: feature parity failure (run verifyParity.ts)

**`sharpe_std > 10.0`:**
- Model is regime-sensitive — performs differently on different market periods
- Do NOT deploy this model
- Solution: more training steps, or use v3 features with 4h/1d MTF for regime context

### Window-by-Window Expected Progression

```
Window 1 (33% data, 5M steps):
  Expected Sharpe: −10 to +0.5  ← still early
  Win rate:  35–48%
  Avg hold:  5–12 candles

Window 2 (66% data, 5M steps):
  Expected Sharpe: 0.0 to +1.5   ← should be positive
  Win rate:  45–55%
  Avg hold:  8–18 candles

Window 3 (100% data, 5M steps):
  Expected Sharpe: 0.5 to +2.0   ← best model
  Win rate:  48–58%
  Avg hold:  10–25 candles

Out-of-sample test:
  Target Sharpe:  > 1.0
  Target return:  > 0% (beating buy-and-hold is secondary)
  target std:     < 1.5 (stability across 3 test slices)
```

---

## 8. Deployment Guide

### Safety Progression (NEVER skip stages)

```
MOCK → PAPER → TESTNET → LIVE (minimum size)
```

**Stage 1: MOCK** (`MOCK_MODE=true`)
- Zero API calls. Simulated fills. Run 48+ hours.
- Pass: no crashes, logs appear every candle, MongoDB writes work.

**Stage 2: PAPER** (`PAPER_TRADE=true`)
- Real Binance market data, virtual $10,000.
- Run 1–2 weeks. Check dashboard.
- Pass: Sharpe > 0.5, drawdown < 20%, no runtime errors.

**Stage 3: TESTNET** (`USE_TESTNET=true`, testnet API keys)
- Real API calls, fake money. Get keys from testnet.binance.vision.
- Run 1 week. Verify: limit orders fill, SL triggers, parity check passes.
- Pass: 20+ clean trade cycles, no LOT_SIZE errors, SL/TP fires correctly.

**Stage 4: LIVE** (real API keys, minimum size)
- Start with MAX_RISK_PER_TRADE=0.005 (0.5%) for first month.
- Monitor first 48 hours manually.
- Pass: matches paper trading performance within 10%.

### Start the Bot

```powershell
cd tradebot-core\execution-engine
npm install
cp .env.example .env
# Edit .env

# Mock (safe)
npm run mock

# Paper (real data)
npm run paper

# Live (real money)
npm start
```

### Dashboard

```powershell
cd tradebot-core\command-center
npm install
npm run dev
# Open http://localhost:3001
```

---

## 9. Configuration Reference

### .env File

```env
# ── Exchange ──────────────────────────────────────────────────
BINANCE_API_KEY=
BINANCE_API_SECRET=
USE_TESTNET=true

# ── Trading ───────────────────────────────────────────────────
TRADING_PAIR=BTC/USDT
TIMEFRAME=15m
WINDOW_SIZE=48

# ── Risk (conservative defaults) ──────────────────────────────
STOP_LOSS_PCT=0.015        # 1.5% stop loss
TAKE_PROFIT_PCT=0.03       # 3.0% take profit  (2:1 R:R)
MAX_RISK_PER_TRADE=0.01    # 1% of balance per trade
MIN_CONFIDENCE=0.60        # Run sensitivity_analysis.py to tune this

# ── GOD-3: BNB Fee Discount ───────────────────────────────────
USE_BNB_FEE_DISCOUNT=false  # Set true if BNB burn enabled in Binance
EFFECTIVE_FEE_RATE=0.001    # 0.001=standard 0.00075=BNB discount

# ── GOD-6: Notifications ──────────────────────────────────────
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
DISCORD_WEBHOOK_URL=        # Optional fallback channel

# ── Mode ──────────────────────────────────────────────────────
MOCK_MODE=true              # Start here
PAPER_TRADE=false

# ── Database ──────────────────────────────────────────────────
MONGO_URI=mongodb://localhost:27017/tradebot
MODEL_PATH=
```

### config.py (Python)

```python
# Must match .env settings:
USE_BNB_FEE_DISCOUNT = False  # Match USE_BNB_FEE_DISCOUNT in .env

# Must match actual timeframe:
ENV_CONFIG["candles_per_day"] = 96   # 96=15m, 24=1h, 288=5m
```

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `LOT_SIZE filter failure` | roundToLotSize not applied | ccxtClient.ts GOD-1 fix — already implemented |
| Training: drawdown_penalty >> mark_to_market | Old reward function | Check trading_env.py threshold=0.25, coef=0.002 |
| terminal_penalty every episode | Replay ratio too high | Use ≤2M steps with 104K rows |
| win_rate stuck at 26–28% | Reward signal broken | Run diagnostic first, check reward components |
| ONNX export fails | VecNormalize .pkl not found | Check _vecnormalize.pkl exists next to .zip |
| Parity check fails on MACD | EMA seeding differs | Wilder's smoothing — ta library uses it by default |
| No Telegram alerts | Token/chatId wrong | Check env vars, test with `sendAlert("test")` |
| Circuit breaker never resets | Session-based (old code) | IMP-5 fix: resets at UTC midnight |
| Order rejected on live | Missing LOT_SIZE floor | GOD-1: roundToLotSize() implemented |
| Balance stale after fill | REST polling (old code) | GOD-5: UserDataStream gives 50ms updates |
| Entropy: -0.023 ⚠️collapse? | Policy collapsed | Increase ent_coef to 0.08 in PPO_HYPERPARAMS |
