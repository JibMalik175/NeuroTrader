# AI Training Pipeline — `ai-training/`

This is the **Python Quantitative Lab** for the TradeBot project.  
Everything here runs offline. The output is a single `.onnx` file that gets
dropped into the Node.js execution engine for live trading.

---

## Setup

```bash
cd ai-training
python -m venv venv
venv\Scripts\activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Workflow (Run in Order)

### Step 1 — Fetch Historical Data
Downloads OHLCV candlestick data from Binance's public API.  
No API keys required.

```bash
# 2 years of BTC/USDT 1-hour candles
python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 730

# 1 year of ETH/USDT 15-minute candles
python scripts/fetch_data.py --symbol ETH/USDT --timeframe 15m --days 365
```

Output: `data/BTC_USDT_1h.parquet`

---

### Step 2 — Feature Engineering
Transforms raw OHLCV into a normalized feature matrix with 18 features:
RSI, MACD, Bollinger Bands, ATR, EMA crossovers, candle structure ratios,
and log-returns. Also creates chronological train/val/test splits.

```bash
python scripts/feature_engineering.py --input data/BTC_USDT_1h.parquet
```

Output:
- `data/BTC_USDT_1h_features.parquet`
- `data/BTC_USDT_1h_train.parquet`  (75%)
- `data/BTC_USDT_1h_val.parquet`    (15%)
- `data/BTC_USDT_1h_test.parquet`   (10%)

---

### Step 3 — Train the DRL Agent
Trains a PPO agent with walk-forward validation.
The agent learns BUY/SELL/HOLD decisions via the Adaptive Risk Control reward.

```bash
python scripts/train_agent.py `
  --train data/BTC_USDT_1h_train.parquet `
  --val   data/BTC_USDT_1h_val.parquet `
  --test  data/BTC_USDT_1h_test.parquet `
  --timesteps 1000000 `
  --windows 3 `
  --run-name tradebot_ppo
```

Output:
- `models/tradebot_ppo_window1.zip`
- `models/tradebot_ppo_window2.zip`
- `models/tradebot_ppo_window3.zip`
- `models/tradebot_ppo_best.zip`       ← best by validation Sharpe
- `models/tradebot_ppo_window*_vecnormalize.pkl`
- `models/tradebot_ppo_best_vecnormalize.pkl` ← matching observation normalization stats
- `models/tradebot_ppo_training_log.json`

**Tip:** Start with `--timesteps 200000` to do a fast sanity-check run before committing to a full million-step training session.

---

### Step 4 — Export to ONNX
Converts the best checkpoint into a format that Node.js can load.
The exporter loads the matching `*_vecnormalize.pkl` file by default and
bakes observation normalization into the ONNX graph, so the execution engine
can send raw feature tensors directly.

```bash
python scripts/export_onnx.py --model models/tradebot_ppo_best.zip
```

Output: `models/tradebot_ppo_best.onnx`

If the normalization stats live somewhere else:
```bash
python scripts/export_onnx.py `
  --model models/tradebot_ppo_best.zip `
  --vecnormalize models/tradebot_ppo_best_vecnormalize.pkl
```

**Copy this file to the execution engine:**
```bash
copy models/tradebot_ppo_best.onnx ../execution-engine/src/strategist/models/tradebot.onnx
```

---

## Directory Structure

```
ai-training/
├── data/               ← Parquet files (gitignored — can be large)
├── environments/
│   ├── __init__.py
│   └── trading_env.py  ← Custom Gymnasium env with Adaptive Risk reward
├── models/             ← Saved checkpoints + ONNX exports (gitignored)
├── scripts/
│   ├── fetch_data.py
│   ├── feature_engineering.py
│   ├── train_agent.py
│   └── export_onnx.py
└── requirements.txt
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| PPO over DQN | More stable gradient updates, built-in entropy bonus prevents premature convergence |
| Walk-forward validation | Prevents data leakage — never trains on future data |
| Persistent VecNormalize | Carries observation scaling across walk-forward windows and into ONNX export |
| Adaptive Risk reward | Scales loss penalties with ATR; teaches capital preservation during volatile markets |
| Parquet storage | 10-100× smaller than CSV, much faster I/O for large datasets |
| ONNX export | Framework-agnostic and self-normalizing — runs in Node.js without Python at runtime |
| Log-returns over prices | Stationary signal, better gradient flow in neural nets |

---

## Retraining Schedule

Once deployed, retrain the model periodically on fresh data:

```bash
# Pull latest 730 days
python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 730

# Re-engineer features
python scripts/feature_engineering.py --input data/BTC_USDT_1h.parquet

# Retrain (continue from best checkpoint or fresh)
python scripts/train_agent.py --train data/BTC_USDT_1h_train.parquet `
                               --val   data/BTC_USDT_1h_val.parquet

# Re-export
python scripts/export_onnx.py --model models/tradebot_ppo_best.zip
```

Recommended: **every 4 weeks** or after any major market regime change (bear→bull, etc.)
