# TradeBot — Improvement Progress Checklist

**Single source of truth for resuming work.** Last updated: 2026-06-10.
Legend: ✅ done · 🔄 in progress · ⬜ not started · ❌ tried, didn't help (kept gated/reverted)

---

## How to resume (read this first)
- Project root: `F:\all projects\trading bot\tradebot-core` (git repo here)
- Training: `cd ai-training` → `venv\Scripts\activate.bat` → `python scripts\train_agent.py ...`
- Active data: **1h** timeframe — `data/BTC_USDT_1h_{train,val,test}.parquet` (4yr, 26k train rows)
- Compare runs: `python scripts\compare_runs.py <run1> <run2> ...`
- Full plan + changelog: `docs/CORE_TRAINING_FIX_PLAN.md`
- Reference bot being mined: Freqtrade at `C:\Users\Jibran Malik\Downloads\freqtrade-develop` (READ-ONLY)
- Git workflow: Claude commits granularly; **user pushes** (`git push`). Daily activity = natural pacing.

---

## Codebase quality pass (2026-06-11) ✅ DONE
- [x] First-ever clean `tsc --noEmit`: fixed ccxt v4 named-import types + undefined guards
       (latent NaN risk in stateRecovery trade math); committed package-lock.json + @types/ws
- [x] Removed 233 lines of per-session debug instrumentation (debug-627897 probes) from
       trading_env + train_agent; verified by full-episode smoke run
- [x] `.gitattributes` (LF normalization, binary protection) — no more CRLF warning spam
- [x] README rewritten: accurate architecture (1,543-dim obs, fresh-per-window VecNormalize,
       shorting/router/protections/maker), honest-results section, current commands
- [x] Stale docs (reward_calibration_findings, GODMODE_SRS, training_diagnosis) bannered HISTORICAL
- [x] requirements.txt: added missing sb3-contrib + optuna (imported but undeclared!), pytest
- [x] **Test suite**: `ai-training/tests/test_trading_env.py` — 20 invariant tests on synthetic
       data (fee identity, ladder, gates, stops, scaler, reward modes, metrics contract)
- [x] GitHub Actions CI: env tests (torch-free, fast) + strict type-check on every push

## Phase 0 — Speed ✅ DONE
- [x] Precompute numpy feature matrix in env (`_get_observation` 604µs→1.6µs, 377×)
- [x] Lazy `_get_info` (trade economics only at episode end) — env.step 982→52,823/s
- [x] Per-step info dicts gated to terminal (Phase 0.5)
- [x] Algo A/B: **RecurrentPPO kept** (PPO collapsed: gross PF 0.058 vs 1.006)
- [x] Benchmark tool `scripts/_bench_speed.py`
- Result: real training 113→~150-540 it/s depending on config; env no longer the bottleneck (LSTM is)

## Phase 1 — Stop the wrong lever ✅ DONE (cleanup partial)
- [x] **position_fraction 0.02 → 0.15** (PROVEN: gross PF flat ~0.63 across 0.02-0.20 → pf can't change edge)
- [x] `compare_runs.py` A/B table tool
- [ ] ⬜ (optional) quarantine dead reward-shaping code behind a flag
- [ ] ⬜ (optional) reconcile stale docs (reward_calibration_findings, GODMODE_SRS) with current FIX-B reward

## Phase 2 — Break the fee ceiling 🔄 IN PROGRESS
- [x] **2.1 Timeframe 15m → 1h** ✅ — gross PF 1.006→1.15, gross expectancy +0.063%/trade (KEEP)
- [x] **2.2 Fee-amplified training** (`--fee-multiplier`) ❌ — 3× gave first net-profit on VAL (net PF 1.27) but overfit (test collapsed). Mechanism works, overshoots. (flag kept)
- [x] **2.3 Regime gate** (`--min-adx`) ✅ best VAL yet (gross PF 1.63, std 0.40) but loses on bear test
- [x] **2.3b Directional gate** (`--require-uptrend`) ⚠️ cuts bear loss but over-restricts (8 trades)
- [x] **2.4 Shorting** (`--allow-short`, 3-action ladder) ❌ raw shorting CHURNS (130 trades, fees 3.83%, net −3.45%). Capability built+gated; needs the reward rework to be useful.

### KEY REFRAME (don't lose this)
The **2026 test set is a −34% bear crash** (buy&hold −34%). A long-only bot structurally
can't profit there; it lost only ~1% = beat buy&hold by 32pts by sitting out. The model
is NOT broken — we were judging it on an impossible bar. Core wall everywhere: **gross
edge ~0.06%/trade < round-trip fee 0.20%/trade.**

---

## Freqtrade-inspired feature backlog (the current work)
Source analysis in `docs/CORE_TRAINING_FIX_PLAN.md` + memory. Adopt ideas, keep our LSTM.

- [x] **F1 Direction-aware position engine** (position_dir, generalized PnL) — shorting foundation
- [x] **F2 Shorting via 3-action ladder** (env engine `_open/_close_position` + `--allow-short`)
- [x] **F3 Exit-concentrated reward** (`--reward-mode exit`) — ✅ VALIDATED by runs: tamed the
       shorting churn (130→41-59 trades), enabled the p2_6/p2_8 results. Keep.
- [x] **F4 Fee-in-price model** — ✅ DONE-BY-EQUIVALENCE (skipped). Our env already charges fee on
       entry+exit notional (`cost_basis=cash×(1−fee)`, `net=gross×(1−fee)`) = mathematically identical
       to Freqtrade's fee-in-price. Implementing it would be a no-op refactor. Verified, not needed.
- [x] **F5 MinMaxScaler(-1,1) feature norm fit on train** — ✅ DONE, opt-in `--feature-scaling minmax`:
       env scales features to [-1,1] by TRAIN min/max (val/test use train stats), VecNormalize
       norm_obs auto-disabled. Unit-verified (range, inverse parity, default path untouched).
       Not yet A/B'd in a training run. (Optional PCA idea: skipped — LSTM handles 1543-dim fine.)
- [x] **F6 EvalCallback best-model-during-training** ✅ — `--eval-every N` checkpoints best model
       by Sortino during training + overfit report. Opt-in (0=off). Verified by smoke test.
- [x] **F7 Protections** — ✅ FULLY WIRED: env cooldown (`--cooldown N`); riskManager.ts
       `canOpenPosition()` + `recordTradeOutcome()`; executioner.ts gates every entry (blocked
       entries logged with PROTECTION skip reason) and feeds every close back. ⬜ only remaining:
       `npx tsc --noEmit` after `npm install` (no node_modules on this machine).
- [x] (was deferred) re-test shorting AFTER F3 reward — done (p2_6_disc_short, p2_7_400k)
- [x] (was deferred) directional SHORT gate — done as **regime router** (p2_8, see below)

### NEW finds from deeper Freqtrade sweep (2026-06-09) — not yet evaluated
- [ ] ⬜ **F8 Transformer option** — Freqtrade has `PyTorchTransformerRegressor`. A transformer
       captures longer-range dependencies than our LSTM (relates to "long-term memory" goal).
       Note: it's a SUPERVISED predictor, not RL — would be an alternate/ensemble approach, bigger lift.
- [x] **F9 Better metrics/objectives** — ✅ FULLY DONE: Sortino + Calmar in env metrics, run_validation,
       training_log, compare_runs, BestEvalCallback selection, AND Optuna `--objective` flag
       (default sortino_ratio).
- [x] **F10 Outlier / novelty detection** — ✅ DONE: mean per-feature |z| vs TRAIN distribution
       (`--outlier-threshold`, feature_ref plumbed so val/test score against train stats). Gated
       opt-in; not yet A/B'd in a training run.

### Freqtrade deep-dive PASS 2 (2026-06-11) — unmined areas, prioritized for our path
- [x] **G1 Lookahead-bias audit** (optimize/analysis/lookahead.py idea) — ✅ `scripts/audit_features.py`.
       VERDICT: all 32 v4 features CLEAN (8 cut points). Our backtests are honest.
- [x] **G2 Warmup/recursive-bias audit** (recursive.py idea) — ✅ same script. FOUND + FIXED a
       pre-paper-trading bug: live watcher buffered 200-300 candles but features need **2000**
       for training parity (<500 can't compute at all). Watcher now paginates prefill to
       `WARMUP_CANDLES` (default 2000). Also hardened add_time_encoding (positional fallback → error).
- [x] **G3 Funding fees** — ✅ `scripts/funding_cost_analysis.py` measured REAL rates over our
       val/test ranges: worst expected drag ~0.008%/trade vs +0.15-0.18% edge (~5% haircut), and
       shorts RECEIVE positive funding (90% of val period). NOT an edge-killer; no model change.
- [x] **G5 Order chasing** — ✅ unfilled post-only orders re-place at the fresh best bid/ask up
       to 6 attempts (~3 min) instead of one 30s window. Partial fills end the chase. Taker
       path unchanged. (`placeLimitWithChase` in ccxtClient.)
- [x] **G4 Trailing stop** — ✅ opt-in `USE_TRAILING_STOP` (default OFF — would fight the model's
       learned exits): profit > 2% offset ratchets an engine-side stop 1% behind price; exchange
       SL order stays as catastrophic backstop. Paper-trading safety net.
- [x] **G6 profit_drawdown objective** — ✅ `--objective profit_drawdown` in the sweep
       (return − maxDD, 1:1), default remains sortino.
- [x] **G7 Operations runbook** — ✅ `docs/OPERATIONS_RUNBOOK.md`: 2-week sliding-window retrain
       gated by fee sweep, besttrain-only promotion rule, daily health checks, kill criteria,
       4-week paper-trading gate.
- [x] **G8 Leverage guard** — ✅ `LEVERAGE` env (default 1); canOpenPosition() hard-blocks ≠1x
       until liquidation handling exists.

## OctoBot deep-dive (2026-06-11) — second open-source bot, monorepo at
## `C:\Users\Jibran Malik\Downloads\OctoBot-master\OctoBot-master` (READ-ONLY)
OctoBot's identity vs Freqtrade: NON-PREDICTIVE strategy modes (grid, staggered orders,
market making, DCA, index) + heterogeneous signal matrix (TA/social/RT evaluators) + new
LLM agent-team layer (ai_trading_mode, packages/agents). Backlog, prioritized for us:

- [ ] ⬜ **H4 Two-model agreement gate** (their evaluator-matrix idea, applied to our ensemble) —
       trade only when p2_8 AND p2_9 besttrain checkpoints agree on direction; expect fewer,
       higher-conviction trades. Pure evaluation with existing artifacts + fee sweep. DO FIRST.
- [x] ❌ **H1 Flat-regime grid overlay** — REJECTED by evidence (`scripts/grid_backtest.py`).
       32-config scan (levels 3/5 × spacing 1-2 ATR × threshold × hysteresis): ALL val configs
       net-negative (best −1.30%); breakout liquidation bleed (−4 to −24%) always exceeds
       ping-pong earnings (+3 to +18%) even with slow-enter/fast-exit hysteresis. BTC 1h "flat"
       regimes still trend intrabar. Structural, not tunable — don't revisit without a genuinely
       mean-reverting instrument. Harness kept for future pairs.
- [ ] ⬜ **H3 Alternative-data features V5** (their social evaluators, made rigorous) — funding
       rate + open interest histories are backfillable from Binance futures API → features
       (funding level/momentum, OI change) → A/B retrain. Their Google-Trends/Reddit signals are
       NOT cleanly backfillable — skip those.
- [ ] ⬜ **H2 Order-book-aware maker placement** (simple_market_making mode: min/max spread,
       book distribution) — place inside the spread at a configurable offset instead of AT best
       bid/ask; reduces adverse selection on our post-only fills. Refine MAKER-1/G5.
- [ ] ⬜ **H5 LLM analyst, advisory-only** (ai_trading_mode agent teams) — daily report agent
       reading trades/metrics; NEVER in the trade loop. Optional/fun.
- Skipping deliberately: web/mobile/Telegram UI (command-center exists), profiles/cloud/copy-
  trading, index/basket modes (single pair), their backtest engine, DSL mode, arbitrage
  (multi-exchange), strategy_optimizer (we have Optuna).

### NOT relevant to us (deliberately skipping — don't chase these)
Exchange abstraction, pairlist managers, Telegram/RPC, the full backtesting engine (we have
env-eval + MockBinanceClient), live-bot orchestration (freqtradebot.py), leverage/liquidation
(only if we go futures), gradient-boosted supervised models (LightGBM/XGBoost = different approach).

---

## Deploy bar (unchanged)
gross PF > 1.2 AND net PF > 1.0 across all 3 validation slices, ≥30 trades/slice, beating
buy-and-hold per regime. Then: ONNX export → feature parity → paper trade (MOCK→PAPER→TESTNET→LIVE).

## Honest status (2026-06-10 session 2 — THE FEE REFRAME PAID OFF)
The fee sweep (`scripts/fee_sensitivity.py`) reframed everything: the router needs shorts ⇒
deployment = USDT-M futures ⇒ maker fee 0.04% RT, 5× cheaper than the 0.20% we judged at.
At deployment economics, TWO independent runs' F6 checkpoints are net-profitable on BOTH
validation AND the −34% bear test:
  - p2_8 besttrain: VAL net PF 1.657/+1.21% | TEST 1.147/+0.39% (45 val trades)
  - p2_9 besttrain: VAL net PF 1.441/+1.05% | TEST 1.285/+0.36% (53 val trades)
Replicated shape, test untouched by checkpoint selection, maker rows conservative (slippage
still taker). Caveats: ~15-18 trades/slice (deploy bar wants ≥30); final models always
overfit (use F6 checkpoints only). MAKER-1 (post-only orders) implemented in ccxtClient.
**Path to money: per-slice deploy-bar check → ONNX export + parity → futures-testnet paper
trading with USE_MAKER_ORDERS=true. Risk $0 until paper trading confirms.**
