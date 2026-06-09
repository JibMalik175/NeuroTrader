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

| 06-09 | p2_fee2x | 1h + `--fee-multiplier 2.0` (200k) | VAL: net PF 0.816, gExp +0.057% (below fee bar), 29 trades. TEST: gross PF 0.421, gExp −0.60%, net −1.75%, 15 trades. | ❌ overfits |

**Phase 2.2 FINAL VERDICT — fee amplification OVERFITS (abandon this lever).**
Across 1×/2×/3×, VALIDATION improves monotonically (net PF 0.75→0.82→1.27) while
TEST degrades monotonically (gross PF 1.15→0.42→noise; net −0.74→−1.75→−2.07).
That divergence = textbook overfitting. The "net-profitable validation" at 3× was a
mirage. **Plain 1h (no fee amp) is the best generalizer** — the only config with a
positive gross edge out-of-sample (test gross PF 1.15, gExp +0.063%).

**Re-framed core problem (proven):** the gross edge is real but ~3× too small to clear
fees out-of-sample (+0.06%/trade vs 0.20% round-trip), AND it's regime-sensitive
(val/test disagree). No fee/reward knob closes this. Remaining real levers:
  1. **Lower the fee to reality:** BNB discount → 0.15% round-trip (account-dependent). Train/eval at the true deployment fee.
  2. **Strengthen/stabilize the edge:** longer training (400-500k) on plain 1h; more/longer data (Binance BTC goes back to 2017, we only use 2022+).
  3. **Regime-gating:** the edge lives in trending regimes (consistent across project history). Only trade when ADX/volatility says the model is in its competence zone.
Fee-multiplier sweeps and reward shaping are EXHAUSTED — stop tuning them.

| 06-09 | p2_3_regime25 | 1h + `--min-adx 25` regime gate (200k) | VAL: **best & most stable yet** — gross PF 1.626 (highest), net PF 0.975, gExp +0.185%, **54 trades**, **Sharpe std 0.40** (lowest). TEST: gross PF 0.847, net −1.70%. | ✅ keep gating; make it DIRECTIONAL |

**KEY REFRAME (06-09) — the test set is a −34% bear crash.** Measured: TEST(2026)
buy&hold = **−34.0%** (maxDD 38.8%) vs VAL buy&hold −10.5%. We were judging a
**long-only** bot on an unwinnable −34% market. On it, the regime model lost only
**−1.7%** (98% HOLD) — i.e. it **beat buy-and-hold by ~32 pts** by sitting out the
crash. The model is NOT broken; the test bar was impossible for long-only.

**Flaw in ADX-only gate:** ADX = trend STRENGTH, not direction. In the 2026 bear,
most ADX>25 bars were strong DOWNtrends, and the gate let the agent open LONGS into
them (source of the −1.7%). Fix = **directional gate**: long only when ADX strong AND
price uptrend (macro_trend_sma>0 or ema_cross_long>0) → sits flat in bears.

**Two clear paths from here:**
  1. **Directional regime gate (Phase 2.3b, cheap):** ADX strong AND uptrend. Keeps the
     long-only model FLAT (≈0% loss) in bear/down regimes, long only in confirmed uptrends.
     Expected to turn the 2026 −1.7% toward ~0 and lift validation net PF past 1.0.
  2. **Add shorting (Phase 2.4, bigger):** long/short env so the bot can PROFIT in bear
     markets like the 2026 test. Now strongly justified — a long-only bot leaves all the
     −34% downside on the table. The real unlock for all-weather profitability.
Also: judge models vs buy-and-hold per regime, not absolute return on a bear test.

| 06-09 | p2_6_exit_long | 1h + `--reward-mode exit` (long-only, 200k) | VAL improved (net PF 0.75→0.96, gExp +0.14%) but TEST worse (net −2.32%). Exit reward sharpens trade quality but overfits val; long-only still can't win the bear test. | ⚠️ exit reward helps val, doesn't fix long-only bear |
| 06-09 | **p2_6_disc_short** | 1h + `--allow-short --reward-mode exit --cooldown 12` (200k) | **FIRST net-profitable OUT-OF-SAMPLE result: TEST net +0.15%, net PF 1.176, gross expectancy +0.274% (clears fee), 41 trades.** Churn tamed (raw 92→41 test trades). BUT VAL negative (net PF 0.59) + huge variance (test Sharpe std 5.37). | 🎯 MILESTONE but a BEAR SPECIALIST, not all-weather |

**MILESTONE (06-09) — disciplined shorting cracked the bear test.** shorting + exit-reward
+ cooldown = first model to make money OOS on the 2026 −34% bear (net PF 1.176). Validates
today's whole direction: the anti-churn features worked (92→41 trades), gross edge finally
beat fees. BUT: it's a BEAR SPECIALIST — wins the harsh test bear, LOSES the milder val
period (val/test inverted), and high variance (std 5.37) means the win is concentrated in the
most-bearish slice. Mirror image of the long-only model (bull specialist). NEITHER is robust.

**NEXT problem = CONSISTENCY/ROBUSTNESS, not edge.** Options: (a) more training/data so the
policy converges (200k is low); (b) walk-forward across more windows; (c) ensemble or
regime-router between the long-biased and short-biased specialists; (d) evaluate on a BULL
test slice too (current test is a bear, which flatters the short-biased model).

| 06-10 | p2_7_disc_short_400k | disc_short scaled 200k→400k + `--eval-every 100000` | **Milestone DISSOLVED.** Test Sharpe std 5.37→0.71 (variance fixed by training) BUT test net +0.15%→**−1.79%**, net PF 1.18→0.59. F6 flagged final model OVERFIT (best @300k > final); best checkpoint val Sortino still −1.34 (neg). | ❌ the +0.15% was NOISE, not edge |

**HARD CONCLUSION (06-10): no robust edge.** The 200k "milestone" was high-variance noise that
landed lucky on one bear slice. Scaling to 400k cut variance (5.37→0.71) and revealed the true
policy: consistently net-negative (PF 0.59). Rules out "just train longer." F6 proved its worth
(caught the overfit). We have rigorously shown thin/no robust OOS edge across: pos-sizing, fee
shaping, reward shaping, 15m→1h, regime gates, shorting, exit-reward, cooldown, fee-amplification,
and scale-up. Consistent with the ~1-in-3/1-in-4 odds. Next: genuinely different lever (F8
transformer / F10 outlier / different data) OR reframe goal (regime-router / beat-buy&hold).

| 06-09 | p2_4_longshort | 1h + `--allow-short` (200k, RecurrentPPO ladder) | WORSE: VAL net −3.45% (130 trades, 11c holds, fees **3.83%**), TEST net −3.35% (92 trades, gross PF 0.93). | ❌ raw shorting CHURNS — fee death, amplified 2× by the doubled action space |

**Phase 2.4 verdict — shorting alone is NOT the unlock.** Doubling the action space
doubled the trade frequency → the overtrading/fee-death problem (our oldest enemy)
got 2× worse, not better. The agent churns long↔short (11-candle holds, 130 trades,
3.83% fees) instead of profiting from bears. Shorting is a needed *capability* but
needs *discipline*. => Next feature (Freqtrade exit-concentrated reward + DURATION
shaping that penalizes churn/over-holding) is now NON-OPTIONAL — it's the mechanism
that makes shorting (and trading generally) selective enough to clear fees. Shorting
stays gated behind --allow-short; re-test it AFTER the new reward lands.

| 06-09 | p2_3b_dir25 | 1h + ADX>=25 + `--require-uptrend` (200k) | VAL: over-restricted — gross PF 0.858, net PF 0.692, only **8.7 trades**. TEST: net **−0.31%** (smallest bear loss) but only 7.3 trades. | ⚠️ directional gate works for bear protection but over-restricts long-only |

**LONG-ONLY FRONTIER MAPPED (06-09) — the tension is structural:**
  - ADX-only gate (regime25): best edge expression (54 trades, gross PF 1.63) but NO bear protection (test −1.70%).
  - ADX+uptrend gate (dir25): bear protection works (test −1.70% → −0.31%) but edge can't express (8.7 trades, gross PF 0.86).
  - These pull in opposite directions WITHIN long-only. More filtering ⇒ safer in bears but too few trades to profit.

**Conclusion — long-only is fundamentally capped.** Best achievable is ~breakeven-
after-fees in normal/up markets and flat-to-small-loss in bears. A long-only bot
CANNOT profit in a bear; it can only avoid it. The 2026 test (−34%) is structurally
unwinnable for long-only. We've now ruled out: position sizing, fee shaping, reward
shaping, timeframe (15m), and the long-only gate frontier.

**=> Phase 2.4 — ADD SHORTING is the decisive next lever (now strongly justified):**
the model already reads direction (profits in uptrends, avoids downtrends) but can't
ACT on the down-side. Shorting ~doubles opportunity and makes the strategy all-weather
— it would PROFIT in bear regimes like the 2026 test instead of just dodging them.
Build: env short-side logic (enter/exit short, inverse PnL, short stops, fees), action
space, then mirror in the execution engine. Bigger change; highest expected payoff.

**Quantified target (post-1h):** gross expectancy/trade **+0.063%** vs round-trip fee **0.20%**. Need edge/trade > fee. Levers: fee-amplified training (Phase 2.2, `--fee-multiplier`), BNB discount (0.20%→0.15%), fewer/higher-conviction trades.

**Speed note:** env fix gave 53× on raw env, but real training throughput only ~1.5× (113→~160 it/s) because the LSTM + per-step info plumbing became the new ceiling. Phase 0.5 (gate per-step info dicts to terminal) applied; re-time expected closer to the 476 it/s bench.
