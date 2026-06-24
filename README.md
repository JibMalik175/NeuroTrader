# NueroTrader — Deep RL Crypto Trading System
(NEURO TRADER IS A PERSONAL PRODUCT , PLEASE RESPECT OWNERSHIP)

End-to-end BTC/USDT trading system: a custom Gymnasium environment and
RecurrentPPO (LSTM) agent trained in Python, exported to ONNX, and served by a
TypeScript execution engine with layered risk protections and a Next.js
monitoring dashboard.

```
tradebot-core/
├── ai-training/        ← Python: data pipeline, custom env, DRL training, ONNX export
├── execution-engine/   ← Node.js/TypeScript: live trading engine (ccxt, Binance)
└── command-center/     ← Next.js: dashboard, PnL charts, kill switch
```

---

## Highlights

**AI / training**
- Custom `TradingEnv` (Gymnasium): long/short via a 3-action ladder
  (BUY moves short→flat→long, SELL moves long→flat→short), direction-aware
  fees/slippage/stop-loss, 1,543-dim observation (32 market features × 48-candle
  window + 7 portfolio-state features)
- RecurrentPPO (LSTM, 1.76M params) — chosen over plain PPO by A/B test
  (PPO's policy collapsed: gross profit factor 0.058 vs 1.006)
- Selectable reward: dense portfolio-return (`fixb`) or sparse exit-concentrated
  net-return reward that scores fee-losing scalps negative (anti-churn)
- Behavior gates, all opt-in and A/B-tested: ADX regime gate, trend-directional
  gate, regime router (longs in uptrends / shorts in downtrends), entry cooldown,
  out-of-distribution candle gate (z-score vs train distribution)
- Walk-forward training with fresh per-window models and VecNormalize stats
  (no cross-window leakage), best-checkpoint selection by Sortino during
  training with an overfit report (final-vs-best), 3-slice validation protocol
- Performance engineering: precomputed numpy feature matrix took `env.step`
  from 982 → 52,800 steps/s (the LSTM is now the bottleneck, not the env)
- Tooling: `compare_runs.py` (A/B tables), `fee_sensitivity.py` (profitability
  vs execution-cost scenarios), Optuna sweeps with selectable
  Sharpe/Sortino/Calmar objective, throughput benchmark harness

**Execution engine**
- Layered protections in the risk manager, ported from Freqtrade's design:
  cooldown after stop-loss, stop-loss-guard (halt after N stops in a window),
  peak-drawdown halt, low-profit cooloff, daily circuit breaker with UTC reset,
  kill switch
- Post-only (maker) order mode: rests orders on the passive side of the book so
  fills pay maker fees with zero spread-crossing cost
- Limit orders with drift-cancel and partial-fill handling, LOT_SIZE flooring,
  BNB fee-discount detection, crash recovery that reconstructs an orphaned
  position from exchange state, WebSocket user-data stream with REST fallback
- Strict TypeScript (`tsc --noEmit` clean)

---

## Honest results (methodology over hype)

Every change ships with an A/B run against baselines on out-of-sample data;
the full experiment ladder (25+ runs) with verdicts lives in
[docs/CORE_TRAINING_FIX_PLAN.md](docs/CORE_TRAINING_FIX_PLAN.md).

Key finding: judged at spot taker fees (0.20% round-trip), the model's gross
edge (+0.13–0.21%/trade) never cleared costs. But the strategy requires
shorting, so deployment targets USDT-M futures — where post-only maker fees are
0.04% round-trip. Re-evaluated at deployment economics, the two best
checkpoints are net-profitable on **both** the validation period **and** a
−34% bear-market test set the model never saw during selection:

| Checkpoint | Validation (net PF / net %) | Bear test (net PF / net %) |
|---|---|---|
| regime router @ 0.04% RT | 1.657 / +1.21% | 1.147 / +0.39% |
| maker-fee retrain @ 0.04% RT | 1.441 / +1.05% | 1.285 / +0.36% |

Caveats are documented alongside the results (thin per-slice trade counts;
checkpoint selection pressure). **Status: validating via paper trading before
any real capital.** Current rule: $0 at risk until the paper-trading bar is met.

---

## Setup

### Prerequisites
- Python 3.10+, Node.js 18+, MongoDB (local or Atlas free tier)
- Binance account (API keys only needed for live mode)

### Train the model

```bash
cd ai-training
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 4 years of BTC/USDT 1h candles (no API key needed)
python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 1460

# Feature matrix + chronological train/val/test split
python scripts/feature_engineering.py --input data/BTC_USDT_1h.parquet

# Train (current best config: all-weather regime router)
python scripts/train_agent.py ^
  --train data/BTC_USDT_1h_train.parquet ^
  --val   data/BTC_USDT_1h_val.parquet ^
  --test  data/BTC_USDT_1h_test.parquet ^
  --timesteps 300000 --windows 1 --recurrent --n-envs 4 ^
  --candles-per-day 24 --allow-short --reward-mode exit ^
  --cooldown 12 --regime-router --eval-every 100000 ^
  --run-name my_run

# Compare against previous runs
python scripts/compare_runs.py my_run p2_8_regime_router

# Where does it cross net-profitable as execution costs improve?
python scripts/fee_sensitivity.py --model models/my_run_window1_besttrain.zip ^
  --vecnorm models/my_run_window1_besttrain_vecnormalize.pkl ^
  --val data/BTC_USDT_1h_val.parquet --test data/BTC_USDT_1h_test.parquet

# Export for the execution engine (bakes normalization into the graph)
python scripts/export_onnx.py --model models/my_run_window1_besttrain.zip
```

### Run the execution engine

```bash
cd execution-engine
npm install
copy .env.example .env   # set MONGO_URI; Telegram/Discord optional

# ALWAYS in this order:
npm run mock    # 1. zero API calls, simulated fills
npm run paper   # 2. real market data, virtual $10k, no real orders
npm start       # 3. LIVE — only after paper trading clears the bar
                #    (start with USE_TESTNET=true)
```

Useful `.env` flags: `USE_MAKER_ORDERS=true` (post-only execution),
`USE_BNB_FEE_DISCOUNT=true`, `EFFECTIVE_FEE_RATE=0.0002`.

### Dashboard

```bash
cd command-center
npm install && npm run dev   # http://localhost:3001
```

---

## Data flow

```
Binance WebSocket (REST fallback)
      │
      ▼
  [Watcher]      closed-candle filtering, 200-candle indicator warmup
      ▼
  [Strategist]   32-feature × 48-candle observation → ONNX inference
      ▼            → BUY / SELL / HOLD + confidence
  [Executioner]  protection gate (cooldown / stoploss-guard / drawdown /
      │          low-profit) → position sizing → limit or post-only order
      │          → SL/TP placement → per-tick monitoring → outcome recorded
      ▼          back into the protections
  [MongoDB]      trades, signals, snapshots
      ▼
  [Dashboard]    equity curve, stats, kill switch
```

---

## Safety rules

| Rule | Why |
|---|---|
| Never enable API withdrawal permission | Prevents fund theft even if keys leak |
| MOCK → PAPER → Testnet → Live, in order | Each stage catches bugs before real money |
| `MAX_RISK_PER_TRADE` stays at 1–2% | One bad streak can't blow the account |
| Deploy F6 *best checkpoints*, never final models | Final models overfit in 3 of 3 measured runs |
| Judge models net-of-fees on out-of-sample slices | Gross edge and validation-only wins are mirages |
| `.env` never committed | API keys stay out of git |

---

## Documentation

| Doc | What's in it |
|---|---|
| [docs/PROGRESS_CHECKLIST.md](docs/PROGRESS_CHECKLIST.md) | Current state, feature status, resume guide |
| [docs/CORE_TRAINING_FIX_PLAN.md](docs/CORE_TRAINING_FIX_PLAN.md) | Full experiment ladder with verdicts |
| [docs/TRADEBOT_RUNNING_GUIDE.md](docs/TRADEBOT_RUNNING_GUIDE.md) | Operational runbook |
