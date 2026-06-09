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

### NOT relevant to us (deliberately skipping — don't chase these)
Exchange abstraction, pairlist managers, Telegram/RPC, the full backtesting engine (we have
env-eval + MockBinanceClient), live-bot orchestration (freqtradebot.py), leverage/liquidation
(only if we go futures), gradient-boosted supervised models (LightGBM/XGBoost = different approach).

---

## Deploy bar (unchanged)
gross PF > 1.2 AND net PF > 1.0 across all 3 validation slices, ≥30 trades/slice, beating
buy-and-hold per regime. Then: ONNX export → feature parity → paper trade (MOCK→PAPER→TESTNET→LIVE).

## Honest status (post-capstone, 2026-06-10)
**p2_8 regime router = best model of the entire ladder** (val Sharpe −0.94, gross PF 1.386,
gExp +0.132%/trade ≈ 2× the plain-1h edge) — and it STILL sits under the 0.20% fee bar
(net PF 0.879 val / 0.60 test). 20+ runs across every lever show gross edge asymptoting
below fees. Reward/gate tuning is EXHAUSTED. Remaining honest paths: (1) cut the fee bar —
BNB discount + maker/limit entries would make +0.132% net-positive (execution fix, not model
fix); (2) F8 transformer or 2017+ data (new information); (3) paper-trade the router to prove
the pipeline with $0 at risk. Risk $0 until paper trading proves an edge.
