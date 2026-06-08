# TradeBot — Review, Completed Work & Recommendations

> **Author:** Antigravity AI Assistant  
> **Date:** 2026-05-22  
> **Scope:** Full codebase audit + project assembly + forward-looking recommendations

---

## v2 Overhaul Status (2026-05-23)

### Phase 1 & 2 Completed
The training pipeline underwent a major v2 overhaul to fix the "death spiral" policy collapse and upgrade the observation space:
- **Reward Function Stabilized:** Switched to continuous mark-to-market rewards, fractional position sizing (20%), and a graduated drawdown penalty. Reward magnitude reduced by ~400,000×, stabilizing PPO gradients.
- **PPO Hyperparameters:** Tuned for financial data (lr=1e-4, ent_coef=0.05, Tanh activation). Added DiagnosticCallback for early collapse detection.
- **Observation Space Expanded:** Dimension increased from 867 to 1159. Added 6 new Phase 2 features (time encoding, ADX, OBV) and upgraded to 24 core features. Added `RecurrentPPO` (LSTM) support.
- **New Files Added:** `env_sanity_check.py`, `diagnostic_callback.py`, `hyperparameter_sweep.py`.

### GPU Setup (RTX 4060)
To ensure PyTorch uses the RTX 4060 GPU instead of falling back to CPU, install the CUDA 12.4 build:
`pip install torch --index-url https://download.pytorch.org/whl/cu124`

### Remaining Work (Phases 3-4)
- Run the Optuna hyperparameter sweep (`hyperparameter_sweep.py`).
- Implement Curriculum Learning and Domain Randomization (Phase 3).
- Explore Ensemble models and advanced techniques (Phase 4).

---

## Table of Contents

1. [What Was Done](#1-what-was-done)
2. [Critical Bugs — Must Fix Before Any Trading](#2-critical-bugs--must-fix-before-any-trading)
3. [High-Priority Fixes](#3-high-priority-fixes)
4. [Medium-Priority Improvements](#4-medium-priority-improvements)
5. [Low-Priority Cleanups](#5-low-priority-cleanups)
6. [Architecture & Feature Recommendations](#6-architecture--feature-recommendations)
7. [Security Recommendations](#7-security-recommendations)
8. [Performance Recommendations](#8-performance-recommendations)
9. [Testing Recommendations](#9-testing-recommendations)
10. [Dashboard Recommendations](#10-dashboard-recommendations)
11. [AI/ML Training Recommendations](#11-aiml-training-recommendations)
12. [Deployment Recommendations](#12-deployment-recommendations)

---

## 1. What Was Done

### 1.1 Full Project Review

Every source file across all three directories (`files/`, `files2/`, `files3/`) was read in its entirety — **22 files, ~5,500+ lines of code** — and analyzed for:

- Architecture and data flow correctness
- Cross-file type/import consistency
- Bug identification (logic, race conditions, data loss)
- Security concerns
- Performance bottlenecks
- Missing functionality

### 1.2 Project Assembly

The flat file dumps were reorganized into the documented `tradebot-core/` directory structure:

```
Source                          → Destination
─────────────────────────────────────────────────────────────────────
files2/trading_env.py           → ai-training/environments/trading_env.py
files2/fetch_data.py            → ai-training/scripts/fetch_data.py
files2/feature_engineering.py   → ai-training/scripts/feature_engineering.py
files2/train_agent.py           → ai-training/scripts/train_agent.py
files2/export_onnx.py           → ai-training/scripts/export_onnx.py
files3/verify_feature_parity.py → ai-training/scripts/verify_feature_parity.py
files2/requirements.txt         → ai-training/requirements.txt
files2/README.md                → ai-training/README.md

files/binanceStream.ts          → execution-engine/src/watcher/binanceStream.ts
files/indicators.ts             → execution-engine/src/strategist/indicators.ts
files/inference.ts              → execution-engine/src/strategist/inference.ts
files3/verifyParity.ts          → execution-engine/src/strategist/verifyParity.ts
files3/ccxtClient.ts            → execution-engine/src/executioner/ccxtClient.ts      ← UPDATED version
files3/executioner.ts           → execution-engine/src/executioner/executioner.ts      ← UPDATED version
files/riskManager.ts            → execution-engine/src/executioner/riskManager.ts
files3/stateRecovery.ts         → execution-engine/src/executioner/stateRecovery.ts
files/notifier.ts               → execution-engine/src/utils/notifier.ts
files/index.ts                  → execution-engine/src/index.ts

files/page.tsx                  → command-center/src/app/page.tsx
files/README.md                 → README.md (root)
```

**Key decision:** Where `files/` and `files3/` had overlapping files (`ccxtClient.ts`, `executioner.ts`), the `files3/` versions were used because they include three important fixes:
1. Crash recovery via `stateRecovery.ts`
2. Actual fill price tracking via `getActualFillPrice()`
3. Feature parity verification tooling

### 1.3 Missing Files Created

10 files that were referenced in imports/documentation but didn't exist were created from scratch:

| # | File | Purpose |
|---|---|---|
| 1 | `execution-engine/src/utils/types.ts` | All shared types, interfaces, enums, and CONFIG object |
| 2 | `execution-engine/src/utils/logger.ts` | Winston logger (console + rotating file) |
| 3 | `execution-engine/src/database/mongoSchemas.ts` | Mongoose schemas: Trade, Snapshot, SignalLog, Tick |
| 4 | `execution-engine/.env.example` | Environment variable template with documentation |
| 5 | `execution-engine/package.json` | Node.js project config with all dependencies |
| 6 | `execution-engine/tsconfig.json` | TypeScript compiler config |
| 7 | `command-center/src/app/api/stats/route.ts` | Dashboard stats API endpoint |
| 8 | `command-center/src/app/api/kill/route.ts` | Kill switch API endpoint |
| 9 | `command-center/package.json` | Next.js project config |
| 10 | `.gitignore` | Git ignore rules for all three components |

All imports across every source file were cross-verified — zero missing types.

---

## 2. Critical Bugs — Must Fix Before Any Trading

### 🔴 BUG-1: VecNormalize Not Baked into ONNX Export

**File:** `ai-training/scripts/export_onnx.py`  
**Impact:** Model produces garbage predictions in production — effectively random trading

**Problem:**  
During training, `train_agent.py` wraps the environment in `VecNormalize`, which learns running mean/std statistics and normalizes all observations before they reach the PPO model. The ONNX export in `export_onnx.py` wraps only the raw policy network — the normalization layer is stripped out.

When the Node.js execution engine sends raw features to the ONNX model, the model sees inputs on a completely different scale than what it was trained on.

**Fix Options:**

Option A — Bake normalization into the ONNX wrapper:
```python
class PolicyWrapper(nn.Module):
    def __init__(self, policy, obs_mean, obs_var, clip_obs=10.0, epsilon=1e-8):
        super().__init__()
        self.policy = policy
        self.register_buffer('obs_mean', torch.tensor(obs_mean, dtype=torch.float32))
        self.register_buffer('obs_var', torch.tensor(obs_var, dtype=torch.float32))
        self.clip_obs = clip_obs
        self.epsilon = epsilon

    def forward(self, obs):
        # Replicate VecNormalize transform
        obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        # ... rest of forward pass
```

Option B — Export normalization stats as JSON and normalize in TypeScript:
```python
# In export_onnx.py, after loading the model:
vec_norm = VecNormalize.load("models/vecnormalize.pkl", venv)
stats = {
    "mean": vec_norm.obs_rms.mean.tolist(),
    "var": vec_norm.obs_rms.var.tolist(),
    "clip": vec_norm.clip_obs,
    "epsilon": vec_norm.epsilon
}
with open("models/normalization_stats.json", "w") as f:
    json.dump(stats, f)
```

Then in `inference.ts`, apply normalization before feeding the tensor.

**Recommendation:** Option A is cleaner — self-contained ONNX model with no external dependencies.

---

### 🔴 BUG-2: PAPER Mode Falls Through to LIVE API Calls

**File:** `execution-engine/src/executioner/ccxtClient.ts`  
**Impact:** PAPER mode places real orders on Binance

**Problem:**  
`marketBuy()` and `marketSell()` only check `CONFIG.mockMode`. When `MOCK_MODE=false` and `PAPER_TRADE=true`, the code falls through to the live API branch and places real orders.

**Current code (simplified):**
```typescript
async marketBuy(size: number): Promise<PlacedOrder> {
    if (CONFIG.mockMode) {
        return this.mockOrder(...)  // Only MOCK gets simulated
    }
    // PAPER falls through here → REAL API CALLS
    return this.withRetry(() => this.exchange.createMarketBuyOrder(...))
}
```

**Fix:**
```typescript
async marketBuy(size: number): Promise<PlacedOrder> {
    if (CONFIG.mockMode || CONFIG.paperTrade) {
        return this.mockOrder(...)  // Both MOCK and PAPER get simulated
    }
    return this.withRetry(() => this.exchange.createMarketBuyOrder(...))
}
```

Apply the same fix to `marketSell()`, `getUsdtBalance()`, `placeStopLossOrder()`.

---

### 🔴 BUG-3: persistTrade() Reads Nulled Position

**File:** `execution-engine/src/executioner/executioner.ts`  
**Impact:** Stop-loss and take-profit values always saved as `0` in database

**Problem:**  
In `closePosition()`, `this.position` is set to `null` before `persistTrade()` is called. Since `persistTrade()` reads `this.position?.stopLoss` and `this.position?.takeProfit`, these are always `undefined` → `0`.

**Fix:**
```typescript
async closePosition(exitPrice: number, exitReason: string) {
    // ... calculate PnL ...

    // Capture values BEFORE nulling
    const closedPosition = { ...this.position };
    this.position = null;

    await this.persistTrade(tradeResult, closedPosition);
    // ...
}

private async persistTrade(result: TradeResult, position: Position) {
    // Use position.stopLoss and position.takeProfit directly
}
```

---

## 3. High-Priority Fixes

### 🟠 FIX-1: Partial Candles Processed as Closed

**File:** `execution-engine/src/watcher/binanceStream.ts`  
**Impact:** Model may trade on incomplete candle data

The WebSocket `watchOHLCV` handler passes all candles to `onClosedCandle()` without checking if they're actually closed. CCXT Pro returns both partial (live-updating) and completed candles.

**Fix:** Add a closed-candle filter, or track the last known candle timestamp and only emit when a new timestamp appears (indicating the previous candle closed).

---

### 🟠 FIX-2: Sharpe Ratio Uses Wrong Annualization Factor

**File:** `ai-training/environments/trading_env.py`  
**Impact:** Validation metrics are inflated ~5× for hourly data, leading to false confidence

**Current:** `sharpe = mean / std * sqrt(252)` — assumes daily bars  
**Correct for 1h:** `sharpe = mean / std * sqrt(252 * 24)` — 6,048 trading periods/year

**Fix:** Accept timeframe as a parameter and compute the correct annualization:
```python
ANNUALIZATION = {
    '1m': 252 * 24 * 60,
    '5m': 252 * 24 * 12,
    '15m': 252 * 24 * 4,
    '1h': 252 * 24,
    '4h': 252 * 6,
    '1d': 252,
}
```

---

### 🟠 FIX-3: VecNormalize Stats Lost Between Walk-Forward Windows

**File:** `ai-training/scripts/train_agent.py`  
**Impact:** Training instability — model sees differently-scaled inputs per window

Each walk-forward window creates a fresh `VecNormalize`. The running mean/variance statistics from Window 1 are discarded when Window 2 starts.

**Fix:** Save and reload VecNormalize between windows:
```python
# After Window 1 training:
env.save("models/vecnormalize_w1.pkl")

# Before Window 2 training:
env = VecNormalize.load("models/vecnormalize_w1.pkl", new_env)
```

---

### 🟠 FIX-4: REST Polling Interval Can Be NaN

**File:** `execution-engine/src/watcher/binanceStream.ts`  
**Impact:** Infinite loop without delay if timeframe isn't in `TF_MS` map

`TF_MS[CONFIG.timeframe]` returns `undefined` for unmapped timeframes → `undefined / 4` = `NaN` → `Math.min(NaN, 15_000)` = `NaN`.

**Fix:** Add a fallback: `const interval = Math.min((TF_MS[CONFIG.timeframe] ?? 60_000) / 4, 15_000)`

---

## 4. Medium-Priority Improvements

### 🟡 IMP-1: Add Duplicate Entry Protection

**File:** `execution-engine/src/executioner/executioner.ts`  
**Risk:** Two rapid candle events could trigger two BUY orders before `this.position` is set

**Fix:** Add an `isProcessing` mutex:
```typescript
private isProcessing = false;

async onSignal(output: ModelOutput, currentPrice: number) {
    if (this.isProcessing) return;
    this.isProcessing = true;
    try {
        // ... existing logic
    } finally {
        this.isProcessing = false;
    }
}
```

---

### 🟡 IMP-2: Fix normalizeOrder Always Setting FILLED Status

**File:** `execution-engine/src/executioner/ccxtClient.ts`  
**Risk:** Could mislead the system if an order isn't actually filled

**Fix:** Map CCXT order status to the `OrderStatus` enum properly:
```typescript
function mapStatus(ccxtStatus: string): OrderStatus {
    switch (ccxtStatus) {
        case 'closed': return OrderStatus.FILLED;
        case 'canceled': return OrderStatus.CANCELED;
        default: return OrderStatus.PENDING;
    }
}
```

---

### 🟡 IMP-3: Fix log_return_v Extreme Value at i=0

**File:** `execution-engine/src/strategist/indicators.ts`  
**Risk:** First candle produces `log(volume / 1e-8)` which could be enormous

When `i=0`, `volumes[i-1]` is `undefined`, falling back to `1e-8`. If the current volume is e.g. 1000, this produces `log(1000 / 0.00000001) ≈ 25.3` — an extreme outlier.

**Fix:** Set `logRV = 0` for the first candle, or start the loop at `i=1`.

---

### 🟡 IMP-4: MACD Signal Alignment is Fragile

**File:** `execution-engine/src/strategist/indicators.ts`  
**Risk:** Re-alignment of MACD signal to original array assumes 1:1 mapping of non-NaN positions

**Fix:** Compute MACD signal in-place using the same indexing as the raw MACD array, rather than filtering/re-aligning.

---

### 🟡 IMP-5: Circuit Breaker is Session-Based, Not Daily

**File:** `execution-engine/src/executioner/riskManager.ts`  
**Risk:** If the bot runs 24/7, the "daily loss" circuit breaker never resets

**Fix:** Add a daily reset scheduler:
```typescript
private scheduleReset() {
    const msUntilMidnight = /* compute next UTC midnight */;
    setTimeout(() => {
        this.dailyLossTripped = false;
        this.sessionStartBalance = this.currentBalance;
        this.scheduleReset();
    }, msUntilMidnight);
}
```

---

### 🟡 IMP-6: Wire Up sendDailySummary()

**File:** `execution-engine/src/utils/notifier.ts` / `execution-engine/src/index.ts`  
**Issue:** `sendDailySummary()` is fully implemented but never called

**Fix:** Add a 24-hour interval in `index.ts`:
```typescript
setInterval(async () => {
    const summary = risk.getSessionSummary();
    await notifier.sendDailySummary(summary);
}, 24 * 60 * 60 * 1000);
```

---

### 🟡 IMP-7: Increase State Recovery Trade Lookback

**File:** `execution-engine/src/executioner/stateRecovery.ts`  
**Risk:** Only fetches 50 recent trades — may not cover full position with many partial fills

**Fix:** Increase to 200, or paginate backwards until the net position is fully reconstructed.

---

## 5. Low-Priority Cleanups

| # | Issue | File | Fix |
|---|---|---|---|
| 1 | Empty `catch {}` blocks swallow errors | Multiple files | Add `logger.debug()` in each catch |
| 2 | Hardcoded signal enum `2` instead of `Signal.SELL` | `index.ts` L182 | Use `Signal.SELL` |
| 3 | `RobustScaler` imported but unused | `feature_engineering.py` | Remove import |
| 4 | `torchvision` in requirements.txt but unused | `requirements.txt` | Remove (~800MB savings) |
| 5 | `requests` in requirements.txt but unused | `requirements.txt` | Remove |
| 6 | `python-dotenv` in requirements.txt but unused | `requirements.txt` | Remove |
| 7 | `MODELS_DIR` defined but unused | `export_onnx.py` | Remove |
| 8 | Dead variables: `recoveredBuyPrice`, `recoveredBuySize` | `stateRecovery.ts` | Remove |
| 9 | Deprecated `datetime.utcfromtimestamp()` | `fetch_data.py` | Use `datetime.fromtimestamp(ts, tz=timezone.utc)` |
| 10 | Docstring says `environment.py` but file is `trading_env.py` | `trading_env.py` | Fix docstring |
| 11 | ONNX model loaded twice in export flow | `export_onnx.py` | Pass loaded model to `export_to_onnx()` |
| 12 | Telegram uses legacy Markdown parse mode | `notifier.ts` | Switch to `MarkdownV2` or `HTML` |
| 13 | `uncaughtException` handler doesn't await shutdown | `index.ts` | Use `process.exitCode = 1` + async cleanup |
| 14 | `unhandledRejection` only logs, doesn't shutdown | `index.ts` | Add graceful shutdown call |
| 15 | Parity file path uses fragile `__dirname` + `../../..` | `verifyParity.ts` | Accept path via CLI arg or env var |

---

## 6. Architecture & Feature Recommendations

### 🏗️ ARCH-1: Add a Trailing Stop-Loss

**Current:** Fixed SL at entry price × (1 - stopLossPct). In a strong uptrend, large gains can be given back before TP is hit.

**Recommendation:** Implement a trailing stop that locks in profits:
```typescript
// In riskManager.ts
updateTrailingStop(position: Position, currentPrice: number): number {
    const trailDistance = currentPrice * CONFIG.stopLossPct;
    const newStop = currentPrice - trailDistance;
    return Math.max(position.stopLoss, newStop); // Only moves UP
}
```

Call this on every candle while in position. Cancel and replace the exchange-side SL order when the trailing stop moves.

---

### 🏗️ ARCH-2: Add Multi-Pair Support

**Current:** Single pair hardcoded in CONFIG.

**Recommendation:** Refactor to support multiple pairs running in parallel:
- One `BinanceWatcher` per pair
- Shared `InferenceEngine` (load one model per pair, or one universal model)
- Separate `Executioner` instances per pair with independent position state
- Portfolio-level risk management across all pairs

---

### 🏗️ ARCH-3: Add a Backtesting Module

**Current:** Model is evaluated only during training. No way to backtest the full pipeline (TypeScript indicators + ONNX inference + risk management) on historical data.

**Recommendation:** Create `execution-engine/src/backtest.ts` that:
1. Loads historical candles from parquet (via a Python helper) or MongoDB
2. Replays them through the same `onSignal()` pipeline
3. Produces a performance report: equity curve, Sharpe, max drawdown, trade list

This closes the gap between "training metrics" and "live performance expectations."

---

### 🏗️ ARCH-4: Add Position Sizing Modes

**Current training:** 100% balance per trade (extremely aggressive).  
**Current live:** Risk-based sizing via `riskManager.calculatePositionSize()`.

**Recommendation:** Add configurable position sizing in the training environment too:
- Fixed fraction (e.g., 10% of balance per trade)
- Kelly criterion
- Volatility-adjusted (ATR-based)

Training with realistic position sizing will produce a model that better matches live conditions.

---

### 🏗️ ARCH-5: Add a Config Validation Layer

**Current:** CONFIG reads env vars with defaults, but invalid combinations aren't caught (e.g., `MOCK_MODE=false` + no API keys).

**Recommendation:** Add a `validateConfig()` function called at startup:
```typescript
function validateConfig(config: typeof CONFIG): void {
    if (!config.mockMode && !config.paperTrade && !config.apiKey) {
        throw new Error("LIVE mode requires BINANCE_API_KEY");
    }
    if (config.stopLossPct >= config.takeProfitPct) {
        logger.warn("SL >= TP: risk/reward ratio < 1.0");
    }
    if (config.observationWindow < 200) {
        logger.warn("Window < 200: EMA-200 will have no warmup data");
    }
}
```

---

## 7. Security Recommendations

### 🔒 SEC-1: Encrypt API Keys at Rest

**Current:** API keys stored in plaintext `.env` file.

**Recommendation:** Use OS keychain or encrypted env files:
- **macOS/Linux:** Use `keytar` or OS keyring
- **Windows:** Use Windows Credential Manager
- **VPS:** Use `dotenvx` or `sops` for encrypted `.env` files

---

### 🔒 SEC-2: Add API Key Permission Validation at Startup

Call `exchange.fetchBalance()` at boot and verify the key works, then check that withdrawal permissions are NOT enabled (Binance returns this in the API key info endpoint).

---

### 🔒 SEC-3: Rate-Limit Telegram Messages

**Current:** No rate limiting — rapid trading could exceed Telegram's 30 msg/s limit.

**Fix:** Add a message queue with throttling (1 message per second max).

---

### 🔒 SEC-4: Add HMAC Signature to Kill Switch API

**Current:** `/api/kill` has no authentication — anyone who can reach the endpoint can kill the bot.

**Fix:** Require a shared secret HMAC signature on kill switch requests.

---

## 8. Performance Recommendations

### ⚡ PERF-1: Use ONNX Runtime WASM for Faster Cold Start

`onnxruntime-node` has a large native binary. For smaller models, `onnxruntime-web` (WASM) may cold-start faster and avoids native compilation issues.

---

### ⚡ PERF-2: Pre-compute Indicator Buffers Incrementally

**Current:** `buildObservationTensor()` recomputes all 18 indicators from scratch on every candle for the full window (48 × 200+ candles of history).

**Recommendation:** Maintain rolling indicator buffers that update incrementally:
- EMA: `newEMA = α × price + (1-α) × prevEMA` — O(1) per candle
- RSI: Maintain running gain/loss averages
- ATR: Maintain rolling TR buffer

This reduces per-candle computation from O(window × history) to O(1).

---

### ⚡ PERF-3: Connection Pooling for MongoDB

**Current:** Each component creates its own MongoDB connection.

**Recommendation:** Use connection pooling with `maxPoolSize` configured:
```typescript
mongoose.connect(MONGO_URI, { maxPoolSize: 5 });
```

---

## 9. Testing Recommendations

### 🧪 TEST-1: Unit Tests for Indicators

Create `__tests__/indicators.test.ts` that validates each of the 18 features against known reference values. This catches regressions when modifying indicator code.

---

### 🧪 TEST-2: Unit Tests for Risk Manager

Test edge cases:
- Position sizing with very small balances
- Circuit breaker trigger and reset
- SL/TP calculation accuracy
- Minimum notional enforcement

---

### 🧪 TEST-3: Integration Test for Full Signal Pipeline

Mock a sequence of candles, feed them through `indicators → inference → executioner` and verify:
- Correct entry/exit behavior
- PnL calculation accuracy
- MongoDB writes contain expected data

---

### 🧪 TEST-4: Add ONNX Export Verification

Compare PyTorch model output vs ONNX Runtime output for the same input tensor. The current `verify_onnx()` function runs ONNX inference but doesn't compare against PyTorch — it only checks that ONNX doesn't crash.

---

### 🧪 TEST-5: Automated Feature Parity in CI

Run `verify_feature_parity.py` → `verifyParity.ts` as part of CI/CD pipeline. Any commit that changes `indicators.ts` or `feature_engineering.py` must pass parity.

---

## 10. Dashboard Recommendations

### 📊 DASH-1: Add Mobile Responsiveness

**Current:** Fixed grid layout (`1fr 280px`) breaks on mobile.

**Fix:** Use CSS media queries or a responsive grid:
```css
@media (max-width: 768px) {
    .grid { grid-template-columns: 1fr; }
}
```

---

### 📊 DASH-2: Extract Inline Styles to CSS Modules

**Current:** All 403 lines of `page.tsx` use inline styles, making it hard to maintain.

**Fix:** Extract to `page.module.css` or use a design system.

---

### 📊 DASH-3: Add React Error Boundary

**Current:** Any React error crashes the entire dashboard.

**Fix:** Wrap the dashboard in an error boundary that shows a fallback UI.

---

### 📊 DASH-4: Add More Dashboard Panels

Recommended additions:
- **Open Position Monitor:** Live PnL, entry price, current price, SL/TP levels with visual bar
- **Model Confidence Histogram:** Distribution of model confidence scores over time
- **Feature Heatmap:** Visual of current indicator values feeding the model
- **Drawdown Chart:** Session drawdown over time with circuit breaker threshold line
- **System Health:** Memory usage, uptime, WebSocket status, last candle age

---

### 📊 DASH-5: WebSocket for Real-Time Updates

**Current:** Polls `/api/stats` every 10 seconds.

**Fix:** Use Server-Sent Events (SSE) or WebSocket for push-based updates — lower latency, less server load.

---

## 11. AI/ML Training Recommendations

### 🤖 ML-1: Use Multiple Validation Episodes

**Current:** `run_validation()` runs a single deterministic episode. Metrics can be noisy.

**Fix:** Run 3-5 episodes and average the metrics for more stable validation.

---

### 🤖 ML-2: Add Fractional Position Sizing to Training

**Current:** Training environment uses 100% balance per trade.

**Fix:** Match the live risk management by using fractional sizing (e.g., 1-5% risk per trade) in the Gymnasium environment. This trains the model under realistic conditions.

---

### 🤖 ML-3: Fix Max Drawdown Calculation

**File:** `trading_env.py`

**Current:** Uses `self.balance` which doesn't include unrealized PnL.

**Fix:** Use `self._get_portfolio_value()` for accurate drawdown tracking.

---

### 🤖 ML-4: Add Regime Detection

**Current:** Same model for all market conditions.

**Recommendation:** Train separate models or add regime features:
- Trending vs. ranging (ADX indicator)
- High vs. low volatility (ATR percentile)
- Bull vs. bear (200-EMA position)

Route to the appropriate model based on detected regime.

---

### 🤖 ML-5: Implement Walk-Forward with Purging and Embargo

**Current:** Standard walk-forward split.

**Recommendation:** Add a "purge" gap between train and validation sets to prevent data leakage from overlapping indicator windows. Standard in quantitative finance.

---

### 🤖 ML-6: Save VecNormalize Statistics

**Current:** VecNormalize stats are not saved alongside the model checkpoint.

**Fix:** Always `env.save()` after training and `VecNormalize.load()` during export — critical for BUG-1 fix.

---

## 12. Deployment Recommendations

### 🚀 DEP-1: Add Docker Containerization

Create `Dockerfile` and `docker-compose.yml` for reproducible deployment:
```yaml
services:
  execution-engine:
    build: ./execution-engine
    env_file: .env
    depends_on: [mongodb]
    restart: always
  command-center:
    build: ./command-center
    ports: ["3001:3001"]
    depends_on: [mongodb]
  mongodb:
    image: mongo:7
    volumes: [mongo-data:/data/db]
```

---

### 🚀 DEP-2: Add Health Check Endpoint

Create `/api/health` in the execution engine that returns:
- Bot status (running/stopped/error)
- Last candle timestamp
- Memory usage
- WebSocket connection status
- MongoDB connection status

Use this for monitoring with UptimeRobot, Grafana, or similar.

---

### 🚀 DEP-3: Add Structured Logging for Log Aggregation

**Current:** Human-readable Winston logs.

**Recommendation:** Add a JSON transport for structured logging compatible with ELK stack, Datadog, or CloudWatch.

---

### 🚀 DEP-4: Add Automated Model Deployment Pipeline

Create a script that:
1. Fetches latest data
2. Retrains the model
3. Runs parity verification
4. Compares new model metrics against current production model
5. Only deploys if new model is strictly better
6. Sends Telegram notification with comparison report

---

### 🚀 DEP-5: Add Database Backup Schedule

MongoDB data (trades, snapshots, signals) is valuable for analysis.

**Fix:** Schedule daily `mongodump` to cloud storage (S3, GCS, or just a separate disk).

---

## Priority Matrix

| Priority | Item | Status | Impact |
|---|---|---|---|
| ✅ P0 | BUG-1: VecNormalize in ONNX | **Done** | Model correctness |
| ✅ P0 | BUG-2: PAPER mode real orders | **Done** | Safety critical |
| ✅ P0 | BUG-3: persistTrade null position | **Done** | Data integrity |
| ✅ P1 | FIX-1: Partial candles | **Done** | Signal quality |
| ✅ P1 | FIX-2: Sharpe annualization | **Done** | Training quality |
| ✅ P1 | FIX-3: VecNormalize between windows | **Done** | Training quality |
| ✅ P1 | FIX-4: REST polling NaN | **Done** | Reliability |
| ✅ P2 | IMP-1: Duplicate entry guard | **Done** | Safety |
| ✅ P2 | IMP-2: Order status mapping | **Done** | Correctness |
| ✅ P2 | IMP-5: Daily circuit breaker reset | **Done** | Safety |
| 🟢 P3 | ARCH-5: Config validation | **Done** | Reliability |
| 🟡 P2 | IMP-3: logRV extreme value | Pending | Feature quality |
| 🟡 P2 | IMP-6: Wire daily summary | Pending | Monitoring |
| 🟢 P3 | ARCH-1: Trailing stop | Pending | Profitability |
| 🟢 P3 | ARCH-3: Backtesting module | Pending | Confidence |
| 🟢 P3 | TEST-1-5: Test suite | Pending | Reliability |
| 🟢 P3 | DASH-1-5: Dashboard improvements | Pending | Usability |
| 🔵 P4 | ARCH-2: Multi-pair | Pending | Scale |
| 🔵 P4 | ML-4: Regime detection | Pending | Profitability |
| 🔵 P4 | DEP-1: Docker | Pending | Deployment |
| 🔵 P4 | All low-priority cleanups | Pending | Code quality |

---

*End of review. All P0 and P1 items have been successfully addressed. You are clear for continued development and simulated testing.*
