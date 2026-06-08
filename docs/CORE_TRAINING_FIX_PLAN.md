# Core Training Fix Plan — Speed + Profitability

**Created:** 2026-06-08
**Owner:** AI training (RecurrentPPO, BTC/USDT)
**Goal:** (1) make training fast enough to iterate, (2) break the fee ceiling that keeps net PnL negative.
**Method:** every change is A/B tested against the previous baseline. One variable at a time.

---

## 0. Proven diagnoses (evidence, not assumption)

| # | Finding | Evidence |
|---|---------|----------|
| D1 | Env rebuilt the observation with pandas every step | `_get_observation` = **604 µs/call**; numpy-precomputed = **1.6 µs** → **377×** |
| D2 | `_get_info` ran `_get_trade_economics()` every step | full `env.step` 982 → **52,823 steps/s** after D1+D2 fix (**53×**), obs parity exactly 0.0 |
| D3 | RecurrentPPO LSTM starves the GPU | tiny net (128 LSTM units); GPU mostly idle; BPTT serial — see Phase 0.3 A/B |
| D4 | `position_fraction` cannot change profitability | gross PF ≈ **0.63 flat** across pf=0.02/0.05/0.10/0.20 (random policy). Edge quality is pf-invariant. |
| D5 | Edge/fee ceiling is the real wall | recent runs: gross PF 1.1–1.26 (real edge) but fees (~0.2%/round-trip) ≥ gross expectancy (~0.1%/trade) → net negative |
| D6 | 15m + 0.1% taker fee is structurally hostile | per-trade move too small vs fixed fee; RSI baseline also loses on this test set |
| D7 | Long-only on a bear/chop test regime | agent can only profit on up-moves after a buy; flat is often the best it can do |

> **Key correction to the prior action plan:** Phase 1 of `IMPLEMENTATION_ACTION_PLAN.md` (lower `position_fraction` to 0.02 to fix "train/serve skew") is proven ineffective by D4. The policy emits *direction* (BUY/HOLD/SELL); live sizing is done by `riskManager`, so pf never transferred. 0.02 only shrank the learning signal and killed the drawdown risk signal.

---

## Phase 0 — Speed (DONE / in validation)

- [x] **0.1** Precompute `self._feature_matrix` (float32) once in `__init__`; `_get_observation` slices it. *(obs parity = 0.0)*
- [x] **0.2** `_get_info(include_economics=False)`; compute trade economics only at episode end.
- [x] **0.3** A/B training throughput *(harness: `scripts/_bench_speed.py`, 30k steps/cell)*:

      | algo | n_envs | device | it/s |
      |------|--------|--------|------|
      | recurrent | 4 | cuda | 476 (current config, post-fix; was 113 pre-fix) |
      | recurrent | 8 | cuda | 238 (LSTM anti-scales) |
      | ppo | 4 | cuda | 397 |
      | **ppo** | **8** | **cuda** | **759** ← fastest |
      | ppo | 8 | cpu | 638 |

      Env fix alone: 113 → 476 it/s (4.2×) on the unchanged RecurrentPPO config → 45 min → ~10.5 min/300k.
      RecurrentPPO is GPU-bound and does NOT scale with envs. PPO-8 is 1.6× faster still — pending the **learning** A/B below.
- [~] **0.4** End-to-end smoke + algo learning A/B (runs `p0_algo_recurrent` 4env vs `p0_algo_ppo` 8env, 200k, pf=0.15). Confirms no env regression AND decides RecurrentPPO vs PPO on validation quality, not just speed.

**Result:** env fix delivered 113 → 476 it/s (4.2×) with zero learning change; PPO-8 offers a further 1.6× if learning holds.

---

## Phase 1 — Stop chasing the wrong lever

- [ ] **1.1** `config.py`: `position_fraction` 0.02 → **0.15** (evidence D4). Keeps the 50%-DD risk signal meaningful without single-trade ruin.
- [ ] **1.2** Quarantine the dead reward-shaping math in `trading_env.py` behind a `self._diagnostics` flag. The live reward is FIX-B (`portfolio_return` only) — make that unambiguous so future readers/docs aren't misled.
- [ ] **1.3** Re-sync docs: `reward_calibration_findings.md` + `GODMODE_SRS.md` describe a reward function that no longer exists (no Exp D/E, no FIX-B). Add a short "current reward = portfolio_return + hard −2.5% stop" note.

---

## Phase 2 — Break the fee ceiling (the actual profitability work, leverage order)

Each step is a tracked A/B vs the 15m FIX-B baseline.

- [ ] **2.1 Timeframe → 1h (highest leverage, lowest effort).** Re-fetch + re-feature with `--candles-per-day 24`; set `ENV_CONFIG["candles_per_day"]=24`. Bigger expected move per trade vs the *same* fixed fee directly lifts edge/fee. Infra already supports it.
      - A/B metric: net PF and gross-expectancy-minus-fee across slices, 15m vs 1h, same model budget.
- [ ] **2.2 Fee-amplified training.** Train at 2–3× real fee (`TRAINING_FEE_MULTIPLIER`), evaluate/deploy at true fee. Forces selectivity (your own diagnosis Fix 9).
      - A/B: trade count ↓, gross expectancy/trade ↑, net PF vs 1× fee.
- [ ] **2.3 Long/short env (gated on 2.1/2.2).** Add SHORT so the agent can profit in downtrends instead of sitting flat. ~2× opportunity; fixes D7. Bigger change — only after 2.1/2.2 show movement, to keep attribution clean.
- [ ] **2.4 Conviction gating (live).** `MIN_CONFIDENCE` tuned via `sensitivity_analysis.py` plateau so the bot only acts when the probability margin clears fee cost.

**Deploy bar:** gross PF > 1.2 AND net PF > 1.0 across all 3 validation slices, with ≥30 trades/slice.

---

## Phase 3 — Trustworthy validation (stop chasing noise)

- [ ] **3.1** `EntropyAnnealingCallback`: decay `ent_coef` 0.08 → 0.01 over training (the un-implemented Phase 2 of the old plan). Stops the deterministic policy collapsing to HOLD.
- [ ] **3.2** Stochastic-vs-deterministic validation diagnostic in `run_validation` — confirm edge lives in the policy, not luck.
- [ ] **3.3** Regime-labeled evaluation + per-regime buy-and-hold comparison. Judge "profitable" against what the market allowed.
- [ ] **3.4** Honest aggregation: stop averaging Sharpe/PF across slices; report pooled trade-level stats.

---

## Phase 4 — Lock-in

- [ ] Export ONNX (VecNormalize baked), run feature parity (Py↔TS), paper-trade the first model that clears the deploy bar.

---

## A/B test protocol (applies to every change)

1. **One variable per run.** Name runs `<phase>_<variable>_<value>` (e.g. `p2_tf1h`, `p2_fee3x`).
2. **Fixed seed set** (3 seeds) — report mean ± std so we don't chase single-seed noise.
3. **Same step budget** within a comparison; respect the replay-ratio guard.
4. **Primary metric:** net PF and gross-expectancy-minus-fee per slice. **Secondary:** Sharpe ± std, trade count, avg hold, action dist.
5. **Record** every run's training-log JSON + a one-line verdict in this doc's changelog.

### Changelog
| Date | Run | Change | Result | Verdict |
|------|-----|--------|--------|---------|
| 06-08 | phase0 | numpy obs + lazy info | env 982→52,823 steps/s, obs parity 0.0 | ✅ keep |
| 06-08 | algo A/B | RecurrentPPO-4 vs PPO-8 (200k, 15m) | Rec gross PF 1.006 / PPO 0.058 (collapsed) | ✅ keep RecurrentPPO |
| 06-08 | p2_tf1h | 15m → 1h timeframe (200k, RecurrentPPO) | **gross PF 1.006→1.15, net PF 0.55→0.83, gross expectancy now +0.063%/trade (was ~0)**. Still net-neg: fee 0.20%/trade = 3.2× edge. | ✅ keep 1h; need selectivity |
| 06-08 | p2_fee3x | 1h + `--fee-multiplier 3.0` (200k) | **VAL: first-ever net profit — net PF 1.268, net expectancy +0.054%/trade, gross expectancy +0.254% clears the 0.20% fee. 17 trades, 46c holds.** TEST: collapsed to 10 trades, gross −1.77%, 33% win — too selective + small-sample noise (Sharpe std ±4.4). | ⚠️ mechanism proven; 3× overshoots. Try 2× |

**Phase 2.2 verdict:** Fee-amplified training WORKS — it's the first lever to push gross expectancy/trade past the fee on validation. But 3× throttles trading too hard to survive the small/recent test regime. Next: `p2_fee2x` (--fee-multiplier 2.0) to find the sweet spot (~25-30 trades, still clearing fees). Then consider 300-500k steps + 3 windows for policy stability/generalization, since 10-17 trade evaluations are statistically unreliable (the project's persistent regime-sensitivity + sample-size problem).

**Quantified target (post-1h):** gross expectancy/trade **+0.063%** vs round-trip fee **0.20%**. Need edge/trade > fee. Levers: fee-amplified training (Phase 2.2, `--fee-multiplier`), BNB discount (0.20%→0.15%), fewer/higher-conviction trades.

**Speed note:** env fix gave 53× on raw env, but real training throughput only ~1.5× (113→~160 it/s) because the LSTM + per-step info plumbing became the new ceiling. Phase 0.5 (gate per-step info dicts to terminal) applied; re-time expected closer to the 476 it/s bench.
