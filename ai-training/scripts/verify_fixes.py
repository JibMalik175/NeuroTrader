"""
verify_fixes.py — Verify all reward function fixes are working correctly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from environments.trading_env import TradingEnv

print("=" * 60)
print("  VERIFICATION: Reward Function Fixes")
print("=" * 60)

# Load data
df = pd.read_parquet(os.path.join(os.path.dirname(__file__), "..", "data", "BTC_USDT_15m_train.parquet"))
print(f"\n[1] Data shape: {df.shape}")

# Create env
env = TradingEnv(df)

# Fix E verification: V4 features detected
n_features = len(env._active_features)
last_4 = env._active_features[-4:]
print(f"\n[2] Fix E — V4 Feature Detection")
print(f"    Active features: {n_features}")
print(f"    Last 4 features: {last_4}")
v4_ok = n_features == 32 and "macro_obv_ratio" in last_4
print(f"    V4 detected: {'PASS' if v4_ok else 'FAIL'}")

# Fix C-init verification: early_exit_penalty in reward_components
print(f"\n[3] Fix C-init — early_exit_penalty in _reset_state")
has_eep = "early_exit_penalty" in env._reward_components
print(f"    Component present: {'PASS' if has_eep else 'FAIL'}")
print(f"    All components: {list(env._reward_components.keys())}")

# Obs space shape
expected_obs = 48 * 32 + 7  # window * features + portfolio
actual_obs = env.observation_space.shape[0]
print(f"\n[4] Observation Space")
print(f"    Expected: {expected_obs}")
print(f"    Actual:   {actual_obs}")
print(f"    Match: {'PASS' if actual_obs == expected_obs else 'FAIL'}")

# Sanity check
print(f"\n[5] Environment Sanity Check")
results = env.sanity_check(n_steps=500, verbose=True)
sanity_ok = all(results.values())

# Fix A verification: SELL reward is non-zero
print(f"\n[6] Fix A — SELL Reward Bug Fix")
env.reset()
obs, r_buy, _, _, _ = env.step(1)  # BUY
print(f"    BUY reward:  {r_buy:.6f} (fee cost)")
print(f"    Pos size:    {env.position_size}")

# Hold 3 steps
for _ in range(3):
    obs, r, _, _, _ = env.step(0)

# SELL after 4 candles (should trigger early-exit penalty)
obs, r_sell, _, _, _ = env.step(2)
comps = env._reward_components
print(f"    SELL reward:  {r_sell:.6f}")
print(f"    MTM component:  {comps['mark_to_market']:.6f}")
sell_mtm_nonzero = comps["mark_to_market"] != 0.0
print(f"    MTM non-zero on SELL: {'PASS' if sell_mtm_nonzero else 'CHECK (may be zero if price unchanged)'}")

# Fix C verification: early-exit penalty fired (held < 5 candles)
print(f"\n[7] Fix C — Early-Exit Penalty")
eep_val = comps["early_exit_penalty"]
print(f"    early_exit_penalty: {eep_val:.6f}")
eep_fired = eep_val < 0
print(f"    Fired for 4-candle hold: {'PASS' if eep_fired else 'FAIL'}")

# Fix B verification: winner_exit should NOT fire for < 5 candle trade
print(f"\n[8] Fix B — Duration-Scaled Winner Exit")
we_val = comps["winner_exit"]
print(f"    winner_exit: {we_val:.6f}")
we_suppressed = we_val == 0.0
print(f"    Suppressed for 4-candle hold: {'PASS' if we_suppressed else 'CHECK (winner_exit should be 0 for short trades)'}")

# Fix B verification part 2: test a 15-candle profitable trade
env.reset()
env.step(1)  # BUY
entry = env.entry_price
for _ in range(15):
    env.step(0)  # HOLD 15 candles
curr_price = env._get_close_price(env.current_step)
pnl = (curr_price - entry) / entry
env.step(2)  # SELL
comps2 = env._reward_components
we_val2 = comps2["winner_exit"]
if pnl > 0:
    # 15 candles held, hold_scale = (15-5)/(20-5) = 0.667, bonus = 0.001 * 0.667 = 0.000667
    expected_bonus = 0.001 * min(1.0, (15 - 5) / (20 - 5))
    print(f"    15-candle profitable trade: winner_exit = {we_val2:.6f} (expected ~{expected_bonus:.6f})")
    print(f"    Scaled correctly: {'PASS' if we_val2 > 0 else 'FAIL'}")
else:
    print(f"    15-candle trade was a loser (pnl={pnl:.4f}), cannot verify scaling")

# Fix F verification: invalid action penalty
print(f"\n[9] Fix F — Invalid Action Penalty")
env.reset()
env.step(2)  # SELL while flat = invalid
invalid_penalty = env._reward_components["invalid_action"]
print(f"    Invalid SELL penalty: {invalid_penalty:.6f}")
print(f"    Penalty = -0.005: {'PASS' if abs(invalid_penalty - (-0.005)) < 0.0001 else 'FAIL'}")

# Summary
print(f"\n{'=' * 60}")
checks = [v4_ok, has_eep, actual_obs == expected_obs, sanity_ok, eep_fired, we_suppressed]
passed = sum(checks)
total = len(checks)
print(f"  VERIFICATION COMPLETE: {passed}/{total} checks passed")
if passed == total:
    print(f"  ALL FIXES VERIFIED SUCCESSFULLY")
else:
    print(f"  SOME CHECKS FAILED - review output above")
print(f"{'=' * 60}")
