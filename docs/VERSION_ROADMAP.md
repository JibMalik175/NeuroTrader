# Version Roadmap

## v1.0 — "Patient Trend-Follower" (current, tagged on GitHub)

**What it is.** A RecurrentPPO (LSTM) regime-router that trades BTC/USDT on 1h
candles: long in confirmed uptrends, short in downtrends, flat (~95% of the
time) in chop. Full stack — training pipeline, ONNX export with parity gate,
paper-trading engine (live data, simulated fills), decision journal, dashboard,
risk protections, account-coexistence safety, operations runbook.

**Honest status.** Slightly net-profitable on validation AND a −34% bear test
*at futures-maker fees* (p2_9: val net PF 1.44 / test 1.29), replicated across
two independent runs. NOT yet live-proven; the strict per-slice deploy bar
(net PF > 1 on every validation slice) is NOT met — both candidates share a
weak regime (VAL#1, TEST#2). Trades ~once per 4–5 days.

**Why it's v1.** It's a complete, coherent, working system with a defensible —
if modest — edge, and it's the baseline every future version must BEAT on the
same evidence. Frozen as a reference point.

---

## v2 — "Day Trader" (in development, branch `day-trader`, NOT pushed)

**Goal.** Test whether a higher-frequency intraday style can clear fees and
beat v1 — driven by the hypothesis that more frequent, high-probability setups
compound small daily gains. Target **1–5 trades/day** (vs v1's ~1 per 5 days).

**Design levers to try.**
- Shorter timeframe: 15m (have it) or fetch 5m.
- Reward tuned for frequent intraday round-trips; loosen/replace the regime gate.
- Optional: an intraday "flatten" bias (reduce overnight exposure) — noting
  crypto is 24/7, so this is a risk-window heuristic, not a true market close.

**The bar it must clear (identical gauntlet to v1 — no moved goalposts).**
1. `per_slice_check.py`: net PF > 1.0 on every validation slice at maker fees.
2. `fee_sensitivity.py`: net-profitable on val AND test at 0.04% RT.
3. Trade pace 1–5/day actually realized, not just targeted.
4. **Must beat p2_9's net profit factor**, or it does not ship.

**Falsifiable prediction (logged up front, 2026-06-12).** The day-trader will
trade 5–10× more but post a LOWER net PF than p2_9, because per-trade edge
shrinks on shorter horizons faster than higher frequency adds value (15m was
already measured worse than 1h). If the evidence refutes this, v2 wins and
ships. If it confirms it, v1 stands and we documented why.

---

## Versioning / git policy

- `main` = the shipped, GitHub-pushed line. Currently v1; only updated when a
  new version BEATS the incumbent on the gauntlet above.
- `v1.0.0` git tag = permanent snapshot of v1 (+ optional GitHub Release).
- `day-trader` branch = all v2 work, committed locally, **not pushed** until the
  verdict. If v2 wins → merge to main + push + tag v2.0.0. If v2 loses → main
  and the v1 release are untouched; the branch stays as a documented experiment.
- Decision is made on EVIDENCE (the gauntlet), not preference.
