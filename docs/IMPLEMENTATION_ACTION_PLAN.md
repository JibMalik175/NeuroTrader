# Implementation Action Plan — Reward Calibration Fixes

**Created:** 2026-06-08  
**Status:** In Progress  
**Goal:** Fix train/serve position sizing skew, close train/val entropy gap, verify gross edge

---

## Phase 1: Fix Train/Serve Position Sizing Skew (HIGHEST LEVERAGE)

### Problem
- Training uses `position_fraction=0.20` (20% per trade)
- Live uses `MAX_RISK_PER_TRADE=0.01-0.02` (1-2% risk per trade)
- MTM signal scales linearly with position size → **10-20× signal mismatch**

### Changes Required
- [ ] `config.py`: `position_fraction: 0.20 → 0.02`
- [ ] `trading_env.py`: Scale all penalties proportionally (÷10)
  - [ ] `early_exit_penalty: 0.0003 → 0.00003`
  - [ ] `loss_duration_penalty: 0.00005 → 0.000005`
  - [ ] `invalid_action` penalty already 0 (remapped) — keep
  - [ ] `winner_exit` bonus: keep at 0.001 (doesn't scale with position)
  - [ ] `drawdown_penalty` coefficient: 0.002 → 0.0002 (threshold stays 0.25)
  - [ ] `terminal_penalty`: 0.05 → 0.005

### Verification
- [ ] Run `env_sanity_check.py` — reward range should be ~±0.05
- [ ] Run 120K diagnostic — check action distribution, MTM magnitude

---

## Phase 2: Entropy Annealing + Clip Range

### Problem
- Training: entropy forces exploration (H=40%, B=30%, S=30%)
- Validation: deterministic policy collapses (H=99%)
- Policy never learns to act confidently without noise

### Changes Required
- [ ] `config.py`: `clip_range: 0.15 → 0.25`
- [ ] `train_agent.py`: Add `EntropyAnnealingCallback` class
  - [ ] Decay `ent_coef` from 0.08 → 0.01 over training
  - [ ] Apply to both PPO and RecurrentPPO
- [ ] `train_agent.py`: Add callback to callback list in `walk_forward_train`

### Verification
- [ ] TensorBoard: `train/entropy_loss` should decay smoothly
- [ ] DiagnosticCallback: action distribution should shift toward deterministic balance

---

## Phase 3: Stochastic Validation Diagnostic

### Problem
- Need to confirm edge exists in policy distribution
- Current validation uses `deterministic=True` only

### Changes Required
- [ ] `train_agent.py`: Modify `run_validation()` to run both deterministic and stochastic
- [ ] Compare Sharpe, trade count, action distribution between modes
- [ ] Log both to debug output

### Verification
- [ ] If stochastic > 0 Sharpe but deterministic ≈ HOLD collapse → edge exists, policy too risk-averse
- [ ] If both negative → feature/architecture issue

---

## Phase 4: Full 300K+ Calibrated Run

### Prerequisites
- [ ] Phase 1 sanity check passes
- [ ] Phase 2 entropy annealing working
- [ ] Phase 3 confirms edge exists

### Execution
- [ ] Run `train_agent.py --timesteps 300000 --windows 3 --run-name calibrated_v5`
- [ ] Monitor: Sharpe progression, action distribution, reward components
- [ ] Export ONNX, verify parity, deploy to paper trading

---

## Progress Log

| Date | Phase | Status | Notes |
|------|-------|--------|-------|
| 2026-06-08 | 1 | 🟡 In Progress | Starting config.py changes |
|  | 2 | ⏳ Pending | |
|  | 3 | ⏳ Pending | |
|  | 4 | ⏳ Pending | |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Penalties too small after scaling → churning | Early-exit penalty at 0.00003 still creates friction; monitor trade count |
| Entropy annealing too fast → premature convergence | Decay over full training (linear), not per-window |
| Position sizing too small → numerical precision issues | `roundToLotSize` floors to exchange precision; min notional $5.5 enforced |

---

## Files to Modify

1. `ai-training/scripts/config.py` — Position fraction, clip_range, penalty constants
2. `ai-training/environments/trading_env.py` — Penalty values in step()
3. `ai-training/scripts/train_agent.py` — EntropyAnnealingCallback, validation modification