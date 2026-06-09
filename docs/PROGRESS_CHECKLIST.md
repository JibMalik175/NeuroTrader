# TradeBot — Improvement Progress Checklist

**Single source of truth for resuming work.** Last updated: 2026-06-09.
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
- [ ] 🔄 **F3 Exit-concentrated + duration-shaped reward** ← **NEXT, NON-OPTIONAL.** Freqtrade-style:
       reward realized PnL at EXIT (`pnl × factor`) + win bonus + penalize churn/over-holding
       (max_trade_duration). This is what stops the shorting churn and the overtrading-into-fees.
       Add as a SELECTABLE reward mode (A/B vs current FIX-B portfolio_return).
- [ ] ⬜ **F4 Fee-in-price model** (`add_entry_fee`/`add_exit_fee`) — PnL inherently net of fees
- [ ] ⬜ **F5 MinMaxScaler(-1,1) feature norm fit on train** — replaces VecNormalize-obs; kills the
       ONNX-baking headache; optional PCA for the 1543-dim obs
- [ ] ⬜ **F6 EvalCallback best-model-during-training** (save best checkpoint by val during learn)
- [ ] ⬜ **F7 Execution-engine protections** (TS): cooldown_period, stoploss_guard,
       max_drawdown halt, low_profit — port to executioner
- [ ] ⬜ (deferred) re-test shorting AFTER F3 reward lands
- [ ] ⬜ (deferred) directional SHORT gate (short only in downtrends) once shorting is disciplined

---

## Deploy bar (unchanged)
gross PF > 1.2 AND net PF > 1.0 across all 3 validation slices, ≥30 trades/slice, beating
buy-and-hold per regime. Then: ONNX export → feature parity → paper trade (MOCK→PAPER→TESTNET→LIVE).

## Honest status
Real but thin edge (gross PF ~1.1-1.6) that fees eat out-of-sample. Long-only capped; shorting
churns without discipline. F3 (better reward) is the most promising lever left. Realistic odds
of a deployable small edge: ~1-in-3 to 1-in-4. Risk $0 until paper trading proves it.
