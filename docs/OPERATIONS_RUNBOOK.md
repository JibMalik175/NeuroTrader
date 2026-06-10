# Operations Runbook — running TradeBot in production

**G7 (Freqtrade FreqAI `live_retrain_hours` idea):** a deployed model decays as the
market drifts away from its training distribution. Freqtrade solves this with
scheduled sliding-window retraining; this runbook adapts that discipline to our
stack. It is the checklist for *operating* the bot — training research lives in
`CORE_TRAINING_FIX_PLAN.md`, feature status in `PROGRESS_CHECKLIST.md`.

---

## Deployment artifacts (what actually runs)

- Model: a **besttrain (F6) checkpoint** — NEVER a final-step model (final
  overfit 3 runs straight: p2_7, p2_8, p2_9).
- Current candidates: `p2_8_regime_router_window1_besttrain.zip` (best val) and
  `p2_9_makerfee_window1_besttrain.zip` (best test), each with its matching
  `*_vecnormalize.pkl`.
- Engine config: `USE_MAKER_ORDERS=true`, `LEVERAGE=1` (G8 blocks anything else),
  `WARMUP_CANDLES=2000` (G2 — do not lower), `EFFECTIVE_FEE_RATE` set to the
  real account fee tier.

## Retraining cadence (sliding window)

| When | What | Why |
|------|------|-----|
| **Every 2 weeks** (or after any 2-week period the bot sat >90% flat) | Re-fetch 1h data (`fetch_data.py`), rebuild splits, retrain the deployment config (`--allow-short --reward-mode exit --cooldown 12 --regime-router --eval-every 100000`, 300k steps), take the **besttrain** checkpoint | 2 weeks ≈ 336 new 1h candles; beyond that the newest regime is invisible to the model |
| After every retrain | `fee_sensitivity.py` on the new checkpoint — require: net PF > 1 on val AND test at the deployment fee | the go/no-go gate; if it fails, KEEP the old model |
| After every retrain | `audit_features.py` on the refreshed dataset | data gaps/exchange changes can introduce silent NaN/warmup shifts |
| **Monthly** | `funding_cost_analysis.py` | funding regimes drift; re-verify it stays a minor haircut |
| **Quarterly** | full walk-forward re-validation (3 windows) + review this runbook | catch slow decay the 2-week cycle can't see |

Promotion rule: a new checkpoint replaces the live one ONLY if it beats it on
the fee sweep's deployment row on BOTH splits. Ties → keep incumbent (fewer
model swaps = fewer unknowns).

## Live health checks (daily glance)

1. **Protections log** — how often is `canOpenPosition()` blocking? A spike in
   `stoploss guard` / `low-profit` blocks = the edge is degrading; consider
   early retrain.
2. **Fill quality** — G5 chase logs: if entries regularly exhaust 6 chase
   attempts, the maker assumption is weakening in fast markets (acceptable:
   missed entries cost nothing; investigate only if >30% of signals miss).
3. **Skip reasons** in the signal log (`PROTECTION:` entries) — should be rare
   and explainable.
4. **Funding payments** (futures) — sanity-check against the monthly analysis.

## Kill criteria (stop trading, investigate)

- Live net PnL more than ~2× worse than the backtest's worst validation slice
  over a comparable trade count.
- Peak drawdown ≥ 15% (the riskManager halts automatically — do not restart
  without a written diagnosis).
- Any feature-parity or warmup warning in the engine logs.
- Exchange behavior change (fee tier, tick size, post-only semantics).

## The paper-trading gate (before ANY real money)

≥ 4 weeks on futures testnet with `USE_MAKER_ORDERS=true`, then compare against
the backtest expectation: trade count within ~2×, net expectancy sign matches,
maker fill rate ≥ 70%. Only then graduate to minimum-size live.
