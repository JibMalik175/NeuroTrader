"""Quick verification that Experiment D penalty fires and stacks with portfolio return."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from environments.trading_env import TradingEnv
from scripts.config import ENV_CONFIG

df = pd.read_parquet("data/BTC_USDT_15m_train.parquet")
env = TradingEnv(df, **ENV_CONFIG)

# Simulate a long underwater hold: 10% below entry, 50 steps in position
env.reset()
env.position_held = True
env.entry_price = env._get_close_price(env.current_step) * 1.10  # 10% underwater
env._steps_in_position = 50  # > 48 threshold

obs, reward, term, trunc, info = env.step(0)  # HOLD

pen = info["reward_components"].get("loss_duration_penalty", 0)
pr  = info["reward_components"].get("portfolio_return", 0)
raw = info["reward_raw"]

print(f"loss_duration_penalty : {pen:.7f}")
print(f"portfolio_return      : {pr:.7f}")
print(f"reward_raw (total)    : {raw:.7f}")
print(f"Expected penalty      : {-0.0002 * 0.10:.7f}  (0.0002 x 10% loss)")
print()
print(f"Penalty fires (< 0)   : {pen < 0}")
print(f"reward_raw < pr (pen subtracted): {raw < pr}")
print(f"reward_raw == pr + pen: {abs(raw - (pr + pen)) < 1e-9}")
print()

r = env.sanity_check(n_steps=300)
print("Sanity check:", r)
