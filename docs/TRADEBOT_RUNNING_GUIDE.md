# TradeBot — Complete Running Guide

> Read this fully before touching a terminal. Every section builds on the last.
> Following the order here is the difference between a working bot and a blown account.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Phase 1 — Environment Setup](#3-phase-1--environment-setup)
4. [Phase 2 — Train the AI Model](#4-phase-2--train-the-ai-model)
5. [Phase 3 — Execution Engine](#5-phase-3--execution-engine)
6. [Phase 4 — Command Center Dashboard](#6-phase-4--command-center-dashboard)
7. [Safety Progression: Mock → Paper → Live](#7-safety-progression-mock--paper--live)
8. [Getting Binance API Keys](#8-getting-binance-api-keys)
9. [Feature Parity Verification](#9-feature-parity-verification)
10. [VPS Deployment (24/7 Uptime)](#10-vps-deployment-247-uptime)
11. [Retraining Schedule](#11-retraining-schedule)
12. [Telegram Alerts Setup](#12-telegram-alerts-setup)
13. [Troubleshooting](#13-troubleshooting)
14. [Emergency Procedures](#14-emergency-procedures)
15. [v2 Training Architecture](#15-v2-training-architecture)

---

## 1. Prerequisites

Install these before anything else.

### Python 3.10+
```bash
# Check version
python --version   # needs 3.10 or above

# Ubuntu/Debian
sudo apt update && sudo apt install python3.10 python3.10-venv python3-pip -y

# macOS (via Homebrew)
brew install python@3.10

# Windows — download installer from python.org
# ✅ Tick "Add Python to PATH" during install
```

### Node.js 18+
```bash
# Check version
node --version   # needs 18 or above

# Ubuntu/Debian — use NVM (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install 18
nvm use 18

# macOS
brew install node@18

# Windows — download from nodejs.org
```

### MongoDB
```bash
# Ubuntu/Debian
sudo apt install mongodb -y
sudo systemctl start mongodb
sudo systemctl enable mongodb   # auto-start on reboot

# macOS
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community

# Alternative — MongoDB Atlas (free cloud tier, no local install)
# Sign up at mongodb.com/atlas → free M0 cluster → get connection string
# Use that string as MONGO_URI in your .env
```

### Verify all three are working
```bash
python --version      # Python 3.10.x
node --version        # v18.x.x
mongod --version      # db version v6.x.x  (or Atlas connection works)
```

---

## 2. Project Structure

```
tradebot-core/
│
├── ai-training/                  ← Python: offline model training
│   ├── data/                     ← Downloaded OHLCV parquet files
│   ├── environments/
│   │   └── trading_env.py        ← OpenAI Gym DRL environment
│   ├── models/                   ← Saved checkpoints + .onnx exports
│   ├── scripts/
│   │   ├── fetch_data.py         ← Step 1: download candles from Binance
│   │   ├── feature_engineering.py← Step 2: compute 24 features + split
│   │   ├── train_agent.py        ← Step 3: PPO walk-forward training
│   │   ├── export_onnx.py        ← Step 4: export to .onnx
│   │   ├── verify_feature_parity.py ← Step 5: parity check (run before live)
│   │   ├── diagnostic_callback.py← v2: training diagnostics (action dist, reward stats)
│   │   ├── env_sanity_check.py   ← v2: environment smoke test before training
│   │   ├── hyperparameter_sweep.py← v2: Optuna-based hyperparameter optimization
│   │   └── ensemble_predict.py   ← v2: multi-model ensemble inference
│   └── requirements.txt
│
├── execution-engine/             ← Node.js/TypeScript: live trading
│   ├── src/
│   │   ├── watcher/
│   │   │   └── binanceStream.ts  ← WebSocket candle stream
│   │   ├── strategist/
│   │   │   ├── indicators.ts     ← TypeScript feature computation
│   │   │   ├── inference.ts      ← ONNX model runner
│   │   │   └── verifyParity.ts   ← Parity checker (run before live)
│   │   ├── executioner/
│   │   │   ├── executioner.ts    ← Trade state machine
│   │   │   ├── ccxtClient.ts     ← Binance REST API wrapper
│   │   │   ├── riskManager.ts    ← Position sizing, SL/TP, kill switch
│   │   │   └── stateRecovery.ts  ← Crash recovery
│   │   ├── database/
│   │   │   └── mongoSchemas.ts   ← MongoDB models
│   │   ├── utils/
│   │   │   ├── types.ts          ← Shared types + CONFIG
│   │   │   ├── logger.ts         ← Winston logger
│   │   │   └── notifier.ts       ← Telegram alerts
│   │   └── index.ts              ← Main bootstrap
│   ├── .env.example
│   ├── package.json
│   └── tsconfig.json
│
├── command-center/               ← Next.js dashboard
│   ├── src/app/
│   │   ├── page.tsx              ← Main dashboard UI
│   │   ├── api/stats/route.ts    ← Stats API
│   │   └── api/kill/route.ts     ← Kill switch API
│   └── package.json
│
├── .gitignore
└── README.md
```

---

## 3. Phase 1 — Environment Setup

### 3.1 Clone / place your project folder

Your `tradebot-core/` folder should be on your machine. If you downloaded
the files from Claude, place them all inside a folder called `tradebot-core`.

### 3.2 Create your .env file

```bash
cd tradebot-core/execution-engine
copy .env.example .env
```

Open `.env` in any text editor. For now, leave API keys empty — we won't
need them until Phase 3 live trading. Set these values:

```env
# Leave blank for now (needed only for live trading)
BINANCE_API_KEY=
BINANCE_API_SECRET=

# Keep testnet ON until you're ready for real money
USE_TESTNET=true

# Your trading pair
TRADING_PAIR=BTC/USDT
CANDLE_TIMEFRAME=1h

# Must match Python training (default 48)
OBSERVATION_WINDOW=48

# Risk settings — start conservative
MAX_RISK_PER_TRADE=0.01      # 1% of portfolio per trade
STOP_LOSS_PCT=0.015          # 1.5% stop loss
TAKE_PROFIT_PCT=0.03         # 3% take profit

# START HERE — never touch real money first
MOCK_MODE=true
PAPER_TRADE=false

# MongoDB — use localhost or your Atlas connection string
MONGO_URI=mongodb://localhost:27017/tradebot

# Optional — add later (see Section 12)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

### 3.3 Set up Python virtual environment

```bash
cd tradebot-core/ai-training
python -m venv venv

# Activate (do this every time you work in this folder)
venv\Scripts\activate

# Install all Python dependencies
pip install -r requirements.txt

# This takes 3–5 minutes — it installs PyTorch, stable-baselines3, etc.
```

### 3.4 Install Node.js dependencies

```bash
# Execution engine
cd tradebot-core/execution-engine
npm install

# Command center dashboard
cd tradebot-core/command-center
npm install
```

---

## 4. Phase 2 — Train the AI Model

All commands below run from inside `ai-training/` with the venv activated.

```bash
cd tradebot-core/ai-training
venv\Scripts\activate
```

### Step 1 — Download historical candles

No API keys needed. Binance's historical data endpoint is public.

```bash
# 2 years of BTC/USDT hourly candles (~17,520 candles)
python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 730

# Optional: also fetch ETH for multi-pair training later
python scripts/fetch_data.py --symbol ETH/USDT --timeframe 1h --days 730
```

Expected output:
```
============================================================
  Symbol    : BTC/USDT
  Timeframe : 1h
  Days      : 730
============================================================
Fetching BTC/USDT [1h]: 100%|████████| 17520/17520 [02:14]
[OK] BTC/USDT | 1h | 17,520 candles | 2023-... → 2025-...
[SAVED] data/BTC_USDT_1h.parquet (1.23 MB)
```

### Step 2 — Build the feature matrix

Computes 24 technical indicators (v2 adds hour_sin, hour_cos, day_sin, day_cos, adx, obv_ratio) and splits into train/val/test sets.

```bash
python scripts/feature_engineering.py --input data/BTC_USDT_1h.parquet
```

Expected output:
```
[INFO] Feature matrix shape: (17320, 29)
[INFO] Feature columns (24): ['log_return', 'log_return_h', ..., 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'adx', 'obv_ratio']
[SPLIT] Train :  12,990 rows  (2023-01-... → 2024-08-...)
[SPLIT] Val   :   2,598 rows  (2024-08-... → 2025-01-...)
[SPLIT] Test  :   1,732 rows  (2025-01-... → 2025-...)
```

This creates:
- `data/BTC_USDT_1h_train.parquet`
- `data/BTC_USDT_1h_val.parquet`
- `data/BTC_USDT_1h_test.parquet`

### Step 3 — Train the DRL agent

#### 3a. Run the environment sanity check (do this first)

Before training, verify the environment produces valid observations and rewards:

```bash
python scripts/env_sanity_check.py --data data/BTC_USDT_1h_train.parquet
```

Expected output:
```
[SANITY] Observation shape: (871,)
[SANITY] Reward range: [-0.05, 0.12]  (should be small, ~±0.5)
[SANITY] Actions sampled: {0: 334, 1: 328, 2: 338}  (roughly uniform)
[SANITY] ✅ Environment is healthy — proceed to training
```

If rewards are enormous (>100) or observations have NaN, do NOT proceed — debug first.

#### 3b. Train with PPO (v2 defaults)

**First run: use 200,000 steps** to verify everything works (takes ~10 min).
If training completes without errors, scale up to 1–2 million steps for
a production-quality model (takes 1–3 hours depending on your hardware).

> **Note:** v2 observation dimension is **1159** (was 867). This accounts for 6 new Phase 2 features (24 total) + 7 portfolio features over a 48-candle window.

```bash
# Quick test run first
python scripts/train_agent.py `
  --train data/BTC_USDT_1h_train.parquet `
  --val   data/BTC_USDT_1h_val.parquet `
  --test  data/BTC_USDT_1h_test.parquet `
  --timesteps 200000 `
  --windows 3 `
  --n-envs 4 `
  --run-name tradebot_v2
```

Watch the output — you'll see validation metrics after each window:
```
[Window 1/3] Training on 4,330 rows...
[DiagnosticCallback] Step 2048 | mean_reward: 0.0023 | actions: {HOLD: 41%, BUY: 30%, SELL: 29%}
[Window 1 Validation]
  Sharpe Ratio    :    1.234
  Total Return    :   +8.42%
  Win Rate        :   54.20%
  Total Trades    :      187
  Max Drawdown    :    12.3%

[NEW BEST] Sharpe 1.234 → saved as tradebot_v2_best.zip
```

**What good metrics look like:**
- Sharpe Ratio > 1.0 ✅ (above 1.5 is excellent)
- Win Rate > 50% ✅
- Max Drawdown < 20% ✅
- Profit Factor > 1.3 ✅
- Action distribution: no single action >70% ✅

If metrics look poor after 200k steps, train for longer:
```bash
# Production run — 1 million steps per window
python scripts/train_agent.py `
  --train data/BTC_USDT_1h_train.parquet `
  --val   data/BTC_USDT_1h_val.parquet `
  --test  data/BTC_USDT_1h_test.parquet `
  --timesteps 1000000 `
  --windows 3 `
  --n-envs 4 `
  --run-name tradebot_v2
```

#### 3c. Monitor training with TensorBoard

v2 logs detailed training metrics to TensorBoard:

```bash
tensorboard --logdir logs/
```

Open **http://localhost:6006** and monitor:
- `rollout/ep_rew_mean` — should trend upward
- `diagnostics/action_distribution` — should NOT collapse to >90% HOLD
- `diagnostics/mean_reward` — should be small but positive (0.001–0.01)
- `train/entropy_loss` — should decrease gradually, not crash to zero

#### 3d. RecurrentPPO (experimental)

For sequence-aware training using LSTM-based policy:

```bash
python scripts/train_agent.py `
  --train data/BTC_USDT_1h_train.parquet `
  --val   data/BTC_USDT_1h_val.parquet `
  --test  data/BTC_USDT_1h_test.parquet `
  --timesteps 500000 `
  --windows 3 `
  --n-envs 4 `
  --recurrent `
  --run-name tradebot_v2_lstm
```

> **Note:** RecurrentPPO requires `sb3-contrib`. Install with: `pip install sb3-contrib`

#### 3e. Hyperparameter sweep (Optuna)

To automatically find optimal hyperparameters:

```bash
python scripts/hyperparameter_sweep.py `
  --train data/BTC_USDT_1h_train.parquet `
  --val   data/BTC_USDT_1h_val.parquet `
  --n-trials 50 `
  --n-envs 4
```

This runs an Optuna study that optimizes learning rate, entropy coefficient,
gamma, clip range, and network architecture. Results are saved to
`models/optuna_study.db`.

### Step 4 — Export to ONNX

Converts the best checkpoint into the format Node.js can load.
*Note: The exporter automatically loads the matching `*_vecnormalize.pkl` stats and bakes observation normalization into the ONNX graph.*

```bash
python scripts/export_onnx.py --model models/tradebot_ppo_best.zip
```

Expected output:
```
[LOAD] Loading SB3 model: models/tradebot_ppo_best.zip
[INFO] Detected observation dimension: 867
[EXPORT] Writing ONNX model to: models/tradebot_ppo_best.onnx
[VERIFY] ONNX model structure is valid.
[VERIFY] Running inference with ONNXRuntime...
[OK] ONNXRuntime inference successful.
     Action      : 0 (HOLD)
     P(HOLD)     : 0.612341
     P(BUY)      : 0.213456
     P(SELL)     : 0.174203
[DONE] models/tradebot_ppo_best.onnx (2.14 MB)
       Copy this file to: execution-engine/src/strategist/models/
```

### Step 5 — Copy the model to the execution engine

```bash
# Create the models directory if it doesn't exist
mkdir tradebot-core/execution-engine/src/strategist/models

# Copy the exported model
copy tradebot-core/ai-training/models/tradebot_ppo_best.onnx `
   tradebot-core/execution-engine/src/strategist/models/tradebot.onnx
```

---

## 5. Phase 3 — Execution Engine

### 5.1 Verify feature parity BEFORE running (critical)

This confirms that TypeScript computes identical features to Python.
A mismatch means the live bot feeds garbage to the model.

```bash
# Step A: generate Python reference data
cd tradebot-core/ai-training
venv\Scripts\activate
python scripts/verify_feature_parity.py --input data/BTC_USDT_1h.parquet

# Step B: run TypeScript checker
cd tradebot-core/execution-engine
npx ts-node src/strategist/verifyParity.ts
```

Expected output:
```
============================================================
  FEATURE PARITY CHECK — Python vs TypeScript
  Candles: 250 | Tolerance: ±0.001
============================================================
  Feature                Python         TypeScript          Delta   Status
  ------------------------------------------------------------------------
  log_return        0.00234521       0.00234521     0.00000000   ✅ PASS
  rsi              -0.12345678      -0.12345601     0.00000077   ✅ PASS
  macd              0.00045231       0.00045198     0.00000033   ✅ PASS
  ...
============================================================
  ✅ ALL 24 FEATURES PASS — TypeScript ≡ Python
     Safe to deploy. Model will see identical inputs in production.
============================================================
```

**If any feature FAILS** — do not proceed to live trading. Compare the
formula in `indicators.ts` against `feature_engineering.py` for that feature.

### 5.2 Run the bot

Always follow the safety progression in order. See Section 7 for details.

```bash
cd tradebot-core/execution-engine

# Mode 1: Mock (no API needed, logs only)
npm run mock

# Mode 2: Paper (real market data, virtual balance)
npm run paper

# Mode 3: Live (real money — only after paper trading is profitable)
npm start
```

---

## 6. Phase 4 — Command Center Dashboard

```bash
cd tradebot-core/command-center

# Create environment file
Set-Content -Path .env.local -Value @"
MONGO_URI=mongodb://localhost:27017/tradebot
NEXT_PUBLIC_PAIR=BTC/USDT
NEXT_PUBLIC_RISK=1
NEXT_PUBLIC_SL=1.5
NEXT_PUBLIC_TP=3
"@

# Start the dashboard
npm run dev
```

Open your browser at **http://localhost:3001**

The dashboard shows:
- Live balance and session PnL
- Equity curve (updates every 10 seconds)
- Win rate, profit factor, avg win/loss
- Trade history table with exit reasons
- 🔴 Kill Switch (requires two clicks to confirm)

**The dashboard reads from MongoDB.** The execution engine must be running
and writing trades to the database for the dashboard to show live data.

---

## 7. Safety Progression: Mock → Paper → Live

**Never skip stages. Each one catches different bugs.**

### Stage 1: Mock Mode ✅ Start here

```bash
# In .env: MOCK_MODE=true
npm run mock
```

What it does:
- Zero API calls to Binance
- Logs every decision to console and file
- Simulates fills using real market prices
- Runs for 48+ hours to verify no memory leaks, crashes, or logic errors

**Pass criteria:** Bot runs for 48 hours without crashing. Signals appear at
expected intervals. Log file grows normally. MongoDB gets written to.

### Stage 2: Paper Trading

```bash
# In .env: MOCK_MODE=false, PAPER_TRADE=true
npm run paper
```

What it does:
- Connects to Binance WebSocket (real market data)
- Tracks a virtual $10,000 balance
- **Simulates market orders and fills using real prices** (no real API orders placed)
- Tests the full signal → decision → position lifecycle

**Pass criteria:** Run for at least 1–2 weeks. Review the trade history
in the dashboard. Sharpe Ratio > 1.0, win rate > 50%, no drawdown > 15%.

### Stage 3: Testnet

```bash
# In .env: USE_TESTNET=true, MOCK_MODE=false, PAPER_TRADE=false
# Add your TESTNET API keys (from testnet.binance.vision — NOT real Binance)
npm start
```

What it does:
- Real API calls to Binance Testnet (fake money, real exchange mechanics)
- Tests order placement, fill confirmation, stop-loss placement
- Surfaces any rate-limit, auth, or API errors before real money is involved

Get testnet keys at: **https://testnet.binance.vision**

**Pass criteria:** Successfully places and closes at least 20 trades without
API errors. Fill prices recorded correctly. Stop-loss orders appear in Binance UI.

### Stage 4: Live — Minimum Size

```bash
# In .env: USE_TESTNET=false, real API keys set
# Keep MAX_RISK_PER_TRADE=0.01 (1%) to limit exposure
npm start
```

Start with the MINIMUM possible trade size (around $10–15 USDT).
The goal at this stage is to verify:
- Real fills are processed correctly
- SL/TP levels trigger at the right prices
- Telegram alerts arrive on your phone
- Dashboard updates in real time

Only increase `MAX_RISK_PER_TRADE` after 2+ weeks of live operation
where performance matches your paper trading results.

---

## 8. Getting Binance API Keys

Required for Stage 3 (Testnet) and Stage 4 (Live).

### For Testnet (fake money)
1. Go to **https://testnet.binance.vision**
2. Click "Log In with GitHub"
3. Go to API Management → Generate HMAC_SHA256 key
4. Copy both keys into `.env` with `USE_TESTNET=true`

### For Live Trading (real money)
1. Log into **https://www.binance.com**
2. Profile icon → API Management
3. Click "Create API" → System Generated
4. Label it something like `tradebot-spot`
5. Complete identity verification if prompted
6. **Set permissions EXACTLY as follows:**

```
✅ Enable Reading           ← required
✅ Enable Spot & Margin Trading  ← required
❌ Enable Futures           ← leave OFF
❌ Enable Margin            ← leave OFF
❌ Enable Withdrawals       ← NEVER enable this
```

7. **Restrict access to your IP address.** On the same page, find
   "IP Access Restrictions" → enter your machine or VPS IP.
   This prevents anyone else from using the keys even if they steal them.

8. Copy API Key and Secret into `.env`:
```env
BINANCE_API_KEY=your_actual_key_here
BINANCE_API_SECRET=your_actual_secret_here
USE_TESTNET=false
```

9. Test the connection (MOCK mode still won't use the keys — switch to PAPER):
```bash
npm run paper
# Look for: [Boot] Executioner ready | Mode: PAPER
# And:      [Client] USDT Balance: 10000.00
```

---

## 9. Feature Parity Verification

Run this check every time you:
- Update `indicators.ts`
- Update `feature_engineering.py`
- Retrain the model on new data
- Deploy to a new machine

```bash
# From ai-training/ (venv activated)
python scripts/verify_feature_parity.py --input data/BTC_USDT_1h.parquet

# From execution-engine/
npx ts-node src/strategist/verifyParity.ts
```

If a feature fails, the output tells you exactly which one and the delta:
```
  macd              0.00045231       0.00089123     0.00043892   ❌ FAIL
```

Fix the matching formula in `indicators.ts` for that indicator, then re-run
until all 24 features pass.

---

## 10. VPS Deployment (24/7 Uptime)

The bot needs to run continuously. Your laptop going to sleep will miss signals.
A cheap VPS ($5–10/month) solves this.

### Recommended providers
- **Hetzner** (Frankfurt or Helsinki) — €4/month, best value
- **DigitalOcean** (Frankfurt) — $6/month, great docs
- **Vultr** (Amsterdam) — $6/month

Choose a server geographically close to Binance's matching engine
(Frankfurt or Tokyo minimize WebSocket latency).

### Setup on VPS

```bash
# 1. SSH into your VPS
ssh root@your-vps-ip

# 2. Install Node.js
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc && nvm install 18

# 3. Install MongoDB
sudo apt install mongodb -y && sudo systemctl start mongodb

# 4. Install PM2 (process manager — keeps bot running after SSH disconnect)
npm install -g pm2

# 5. Transfer your project to the VPS (from your local machine)
scp -r tradebot-core root@your-vps-ip:/home/tradebot-core

# 6. On the VPS — install dependencies and build
cd /home/tradebot-core/execution-engine
npm install
npm run build   # compiles TypeScript → dist/

# 7. Create your .env file on the VPS
nano .env
# Paste your production .env contents

# 8. Start with PM2
pm2 start dist/index.js --name tradebot
pm2 save           # save process list
pm2 startup        # auto-start on server reboot
```

### Useful PM2 commands
```bash
pm2 status              # see if bot is running
pm2 logs tradebot       # live log stream
pm2 logs tradebot --lines 200  # last 200 lines
pm2 restart tradebot    # restart bot
pm2 stop tradebot       # stop bot (does NOT close positions)
pm2 delete tradebot     # remove from PM2 entirely
```

### Access the dashboard remotely

The command-center runs on port 3001. To access it from your browser:

```bash
# Option A: SSH tunnel (secure, no ports exposed)
# On your local machine:
ssh -L 3001:localhost:3001 root@your-vps-ip

# Then open: http://localhost:3001

# Option B: Open port in firewall (less secure)
sudo ufw allow 3001
# Then open: http://your-vps-ip:3001
```

---

## 11. Retraining Schedule

Model performance decays as market conditions change (ranging → trending,
bull → bear, etc.). Retrain every 4 weeks or after any major market
regime shift.

### Full retrain workflow
```bash
cd tradebot-core/ai-training
venv\Scripts\activate

# 1. Download latest data (overwrites old parquet file)
python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 730

# 2. Rebuild features
python scripts/feature_engineering.py --input data/BTC_USDT_1h.parquet

# 3. Retrain (use more timesteps for production)
python scripts/train_agent.py `
  --train data/BTC_USDT_1h_train.parquet `
  --val   data/BTC_USDT_1h_val.parquet `
  --test  data/BTC_USDT_1h_test.parquet `
  --timesteps 1000000 `
  --windows 3 `
  --run-name tradebot_ppo_$(Get-Date -Format 'yyyyMM')   # date-stamp the run

# 4. Export
# Saves both the model zip and matching *_vecnormalize.pkl stats
python scripts/export_onnx.py --model models/tradebot_ppo_$(Get-Date -Format 'yyyyMM')_best.zip

# 5. Verify parity
python scripts/verify_feature_parity.py --input data/BTC_USDT_1h.parquet
cd ../execution-engine
npx ts-node src/strategist/verifyParity.ts

# 6. Deploy new model
copy ../ai-training/models/tradebot_ppo_$(Get-Date -Format 'yyyyMM')_best.onnx `
   src/strategist/models/tradebot.onnx

# 7. Restart the bot (closes and re-opens from flat)
pm2 restart tradebot
```

**Important:** Only swap the model when the bot is flat (no open position).
Watch the dashboard to confirm no position is active before restarting.

---

## 12. Telegram Alerts Setup

Get notified on your phone for every trade, circuit breaker, and emergency stop.

### Create a Telegram bot
1. Open Telegram → search for **@BotFather**
2. Send `/newbot`
3. Choose a name (e.g. `My TradeBot`)
4. Choose a username (e.g. `mytradebot_btc_bot`)
5. BotFather sends you a token — copy it

### Get your Chat ID
1. Search for **@userinfobot** on Telegram
2. Send any message — it replies with your Chat ID (a number)

### Add to .env
```env
TELEGRAM_BOT_TOKEN=7123456789:AAHdqTcvCH1vGWJxfSeofSs0K38M...
TELEGRAM_CHAT_ID=123456789
```

### Test it
```bash
# Restart the bot — it sends a startup message
pm2 restart tradebot
```

You should receive on Telegram:
```
🤖 TradeBot started
Pair: BTC/USDT | Mode: PAPER
```

---

## 13. Troubleshooting

### "ONNX model not found"
```
Error: ONNX model not found at: ./src/strategist/models/tradebot.onnx
```
Solution: You haven't copied the model yet.
```bash
copy ai-training/models/tradebot_ppo_best.onnx `
   execution-engine/src/strategist/models/tradebot.onnx
```

### "Need at least 200 candles, got X"
The watcher buffer hasn't filled yet. This happens at startup.
The bot will log this and skip inference until the buffer is full.
Normal behaviour — wait for the next candle.

### "Authentication failed — check API keys"
Your `BINANCE_API_KEY` or `BINANCE_API_SECRET` is wrong, or the key
has been deleted from Binance. Re-generate keys and update `.env`.

### "Rate limited — waiting 60000ms"
You're hitting Binance's request limit. The bot handles this automatically
with a 60-second wait. If it happens frequently, increase `CANDLE_TIMEFRAME`
from `1h` to `4h` to reduce API call frequency.

### MongoDB connection errors
```
[Boot] MongoDB unavailable — continuing without DB
```
Start MongoDB: `sudo systemctl start mongodb`
Or check your Atlas connection string in `MONGO_URI`.

### Dashboard shows "Cannot connect to MongoDB"
Both the execution engine and command center must point to the same `MONGO_URI`.
Check `.env` in `execution-engine/` and `.env.local` in `command-center/` match.

### Feature parity check fails
Compare the failing indicator formula between `indicators.ts` and
`feature_engineering.py`. The most common mismatch is EMA seeding —
Python's `ta` library uses Wilder's smoothing; make sure TypeScript matches.

### Bot enters the same trade twice after restart
This is the crash recovery triggering incorrectly. Check `stateRecovery.ts`
logs — look for "[Recovery]" entries. If it's falsely detecting a position,
check that your `TRADING_PAIR` in `.env` exactly matches what's on Binance
(e.g. `BTC/USDT` not `BTCUSDT`).

### "Policy collapse detected"
If the action distribution shows >90% HOLD, the agent has learned that trading is too risky. This usually means your `fee_rate`, `slippage_pct`, or `max_drawdown_pct` penalties are too harsh. Try relaxing them.

### "CUDA not available"
If PyTorch is using the CPU and training is slow, you likely have the wrong PyTorch build. For RTX 4060, install the CUDA 12.4 build:
`pip install torch --index-url https://download.pytorch.org/whl/cu124`

### "SubprocVecEnv fails"
If you see a multiprocessing error during training startup, the script will automatically fall back to `DummyVecEnv` (single process). This is normal on some Windows setups.

---

## 14. Emergency Procedures

### Kill the bot immediately (dashboard)
1. Open **http://localhost:3001** (or your VPS address)
2. Click 🔴 **KILL SWITCH** in the bottom right
3. Click again to confirm
4. The bot cancels all open orders and closes any position at market price
5. You receive a Telegram alert confirming the stop

### Kill the bot from terminal
```bash
# Graceful — closes position cleanly
pm2 stop tradebot

# If PM2 is unresponsive
pkill -f "node dist/index.js"
```

### Manually close a position on Binance
If the bot is completely unresponsive:
1. Log into **binance.com**
2. Spot Wallet → find your BTC balance
3. Click Sell → Market → 100% → Sell BTC
4. Cancel any open orders: Orders → Open Orders → Cancel All

### API keys compromised
If you suspect your API keys were leaked:
1. Immediately go to **binance.com → API Management**
2. Delete the compromised key
3. Generate a new key pair
4. Update `.env` with the new keys
5. Restart the bot

Since "Withdrawals" was disabled on the key, funds cannot be transferred out
even with a compromised key — but cancel it immediately anyway.

### What to do in a flash crash
The circuit breaker in `riskManager.ts` automatically halts trading if the
session drawdown exceeds 5% (`MAX_DAILY_LOSS_PCT`). The circuit breaker will automatically reset at UTC midnight. You will receive a
Telegram alert. The bot will stop entering new trades but does NOT
automatically close existing positions — monitor and close manually if needed.

---

## 15. v2 Training Architecture

### The "Death Spiral" Fix
In v1, the agent frequently learned to output `HOLD` 100% of the time. This was caused by asymmetric reward penalties that were too harsh and continuous drawdown penalties that made trading mathematically disadvantageous. 

In v2, the reward function uses **continuous mark-to-market rewards**, fractional position sizing (20% per trade instead of 100%), and a graduated drawdown penalty. This provides a dense reward signal at every step and teaches risk management without causing a policy collapse.

### DiagnosticCallback & TensorBoard
The `DiagnosticCallback` monitors the action distribution directly. If the agent outputs any single action >85% of the time, it warns you. Watch these metrics in TensorBoard (`tensorboard --logdir logs/`):
- `diagnostics/action_distribution`: Should remain balanced (e.g., 30/30/40).
- `rollout/ep_rew_mean`: Should trend upward.
- `diagnostics/mean_reward`: Should be small but positive.

### Healthy vs Unhealthy Training
- **Healthy:** Sharpe ratio > 1.0, steady win rate > 50%, action distribution is balanced, drawdown < 20%.
- **Unhealthy:** Agent buys/sells <2% of the time (low diversity), Sharpe ratio < 0, training throughput drops unexpectedly.

---

## Quick Reference Card

```
TRAIN THE MODEL:
  cd ai-training && venv\Scripts\activate
  python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 730
  python scripts/feature_engineering.py --input data/BTC_USDT_1h.parquet
  python scripts/train_agent.py --train data/BTC_USDT_1h_train.parquet `
    --val data/BTC_USDT_1h_val.parquet --timesteps 500000
  python scripts/export_onnx.py --model models/tradebot_ppo_best.zip
  copy models/tradebot_ppo_best.onnx ../execution-engine/src/strategist/models/tradebot.onnx

VERIFY PARITY:
  python scripts/verify_feature_parity.py --input data/BTC_USDT_1h.parquet
  cd ../execution-engine && npx ts-node src/strategist/verifyParity.ts

RUN THE BOT:
  cd execution-engine
  npm run mock      ← Stage 1: no API calls
  npm run paper     ← Stage 2: real data, virtual money
  npm start         ← Stage 3/4: testnet or live

DASHBOARD:
  cd command-center && npm run dev
  → http://localhost:3001

VPS (production):
  pm2 start dist/index.js --name tradebot
  pm2 logs tradebot
  pm2 restart tradebot
```
