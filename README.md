# TradeBot Core — Full Project

Automated Binance trading bot with a custom-trained Deep Reinforcement
Learning model, Node.js execution engine, and Next.js command center.

```
tradebot-core/
├── ai-training/        ← Python: data pipeline, DRL training, ONNX export
├── execution-engine/   ← Node.js/TypeScript: live trading engine
└── command-center/     ← Next.js: dashboard, PnL charts, kill switch
```

---

## Full Setup (Run Once)

### Prerequisites
- Python 3.10+
- Node.js 18+
- MongoDB (local or Atlas free tier)
- Binance account (no API keys needed until Phase 3)

---

## Implemented Safety & Correctness Fixes

- PAPER mode now simulates balances and market orders; only LIVE mode can place Binance orders.
- Training saves matching `*_vecnormalize.pkl` files, and ONNX export bakes those normalization stats into the model graph.
- Walk-forward training reloads VecNormalize stats between windows instead of resetting observation scaling each window.
- Validation/test predictions use the same VecNormalize observation transform as training.
- Hourly Sharpe annualization and max drawdown metrics now use mark-to-market portfolio value.
- The watcher filters out partial candles, keeps at least 200 candles for indicator warmup, and has a safe REST polling fallback interval.
- Execution has duplicate-entry protection, correct CCXT order status mapping, SL/TP persistence from the closed position snapshot, daily circuit-breaker reset, enum-based SELL signals, config validation, and awaited graceful shutdown on fatal errors.

---

## Phase 2 — Train the AI Model

```bash
cd ai-training
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Step 1: Download 2 years of BTC/USDT 1h candles (no API key needed)
python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 730

# Step 2: Build feature matrix + train/val/test splits
python scripts/feature_engineering.py --input data/BTC_USDT_1h.parquet

# Step 3: Train DRL agent (start with 200k steps to test, scale up to 2M+)
python scripts/train_agent.py `
  --train data/BTC_USDT_1h_train.parquet `
  --val   data/BTC_USDT_1h_val.parquet `
  --test  data/BTC_USDT_1h_test.parquet `
  --timesteps 500000 --windows 3 --run-name tradebot_ppo

# Step 4: Export to ONNX
# Automatically loads models/tradebot_ppo_best_vecnormalize.pkl
# and bakes observation normalization into the ONNX graph.
python scripts/export_onnx.py --model models/tradebot_ppo_best.zip

# Step 5: Copy model into execution engine
copy models/tradebot_ppo_best.onnx ../execution-engine/src/strategist/models/tradebot.onnx
```

---

## Phase 3 — Run the Execution Engine

```bash
cd execution-engine

# Install dependencies
npm install

# Set up your environment
copy .env.example .env
# Edit .env — set MONGO_URI, and optionally Telegram keys

# ── Development Progression (ALWAYS follow this order) ──

# 1. MOCK mode — zero API calls, just logs
npm run mock

# 2. PAPER mode — real market data, virtual $10,000, no real orders
npm run paper

# 3. LIVE mode — real money (only after paper trading is profitable)
#    First: get Binance API keys, set in .env, set USE_TESTNET=true initially
npm start
```

### Getting Binance API Keys (for Phase 3)
1. Log into Binance → Profile → API Management
2. Create new API key → System Generated
3. Enable: ✅ Enable Reading, ✅ Enable Spot & Margin Trading
4. Disable: ❌ Enable Withdrawals (NEVER enable this)
5. Set IP restriction to your VPS/machine IP
6. Copy API Key and Secret into `.env`

---

## Phase 4 — Launch the Dashboard

```bash
cd command-center
npm install

# Add to .env.local:
# MONGO_URI=mongodb://localhost:27017/tradebot
# NEXT_PUBLIC_PAIR=BTC/USDT
# NEXT_PUBLIC_RISK=2
# NEXT_PUBLIC_SL=1.5
# NEXT_PUBLIC_TP=3

npm run dev
# Open http://localhost:3001
```

---

## Safety Rules (Read Before Going Live)

| Rule | Why |
|---|---|
| Never enable API Withdrawal permission | Prevents fund theft even if keys leak |
| Always start with MOCK → PAPER → Testnet → Live | Each stage catches bugs before real money |
| Keep `MAX_RISK_PER_TRADE` at 1-2% | One bad streak can't blow the account |
| Monitor the first 48 hours of live trading | Model may behave differently on live data |
| Set `USE_TESTNET=true` for your first real API test | Binance Testnet uses fake money but real API behavior |
| Back up your `.env` file securely (not in Git) | Never commit API keys |

---

## Data Flow

```
Binance WebSocket
      │
      ▼
  [Watcher]  — buffers closed candles with 200-candle indicator warmup
      │
      ▼
  [Strategist]  — builds 18-feature observation tensor
      │           runs ONNX model inference
      │           outputs: BUY / SELL / HOLD + confidence
      │
      ▼
  [Executioner]  — validates config and checks confidence threshold
      │            calculates position size (fixed fractional)
      │            places market order via CCXT
      │            sets stop-loss + take-profit levels
      │            monitors position on every tick
      │
      ▼
  [MongoDB]  ←─── trades, signals, snapshots
      │
      ▼
  [Dashboard]  — equity curve, stats, kill switch
```

---

## Retraining Schedule

```bash
# Run every 4 weeks to keep the model fresh
cd ai-training
python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 730
python scripts/feature_engineering.py --input data/BTC_USDT_1h.parquet
python scripts/train_agent.py --train data/BTC_USDT_1h_train.parquet `
  --val data/BTC_USDT_1h_val.parquet --timesteps 1000000
# Saves both the model zip and matching *_vecnormalize.pkl stats
python scripts/export_onnx.py --model models/tradebot_ppo_best.zip
copy models/tradebot_ppo_best.onnx ../execution-engine/src/strategist/models/tradebot.onnx
```

---

## Deployment on a VPS (Optional but Recommended)

For 24/7 uptime, deploy the execution engine to a cheap VPS
(DigitalOcean $6/month, or Hetzner). Choosing a server in Frankfurt
or Tokyo minimizes latency to Binance's matching engines.

```bash
# On your VPS
npm install -g pm2
cd execution-engine
npm run build
pm2 start dist/index.js --name tradebot
pm2 save && pm2 startup
```
