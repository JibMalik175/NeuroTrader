"""
config.py — Single source of truth for all training scripts.

GOD-3: USE_BNB_FEE_DISCOUNT and EFFECTIVE_FEE_RATE added.
       Must match your .env USE_BNB_FEE_DISCOUNT setting.
       Misaligned fee rates mean the model was trained under different
       cost assumptions than it lives under — affects PnL predictions.
"""
import torch

CANDLES_PER_DAY: dict[str, int] = {
    "1m": 1440, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1,
}

# ── GOD-3: Fee Configuration ──────────────────────────────────────────────────
# Set True if Binance account has BNB with "Use BNB to pay fees" enabled.
# 0.1% → 0.075% = 25% fee reduction. Match this with .env USE_BNB_FEE_DISCOUNT.
USE_BNB_FEE_DISCOUNT = False
STANDARD_FEE_RATE    = 0.001    # 0.1%  standard Binance taker
BNB_FEE_RATE         = 0.00075  # 0.075% with BNB burn enabled
EFFECTIVE_FEE_RATE   = BNB_FEE_RATE if USE_BNB_FEE_DISCOUNT else STANDARD_FEE_RATE

ENV_CONFIG: dict = {
    "window_size":       48,
    "initial_balance":   10_000.0,
    "fee_rate":          EFFECTIVE_FEE_RATE,  # GOD-3: BNB-adjusted
    "slippage_pct":      0.0002,              # GOD-2: reduced — limit orders have minimal slippage
    "max_drawdown_pct":  0.50,
    "reward_scaling":    1.0,
    "position_fraction": 0.15,   # Phase 1.1 (2026-06-08): reverted 0.02 → 0.15.
                                  # PROVEN (D4): gross profit factor is flat (~0.63)
                                  # across pf 0.02..0.20 — position size cannot change
                                  # trade edge. The 0.02 "train/serve skew" fix was a
                                  # dead end (policy emits direction; live sizing is
                                  # done by riskManager). 0.15 keeps the 50%-drawdown
                                  # risk signal meaningful without single-trade ruin.
    "candles_per_day":   96,                  # 15m default; override per dataset
}

PPO_HYPERPARAMS: dict = {
    "learning_rate":  1e-4,
    "n_steps":        2048,   # PERF: was 4096→1024→2048. 1024 caused Reward:N/A on most rollouts
                              # because episodes run 18-45K steps. 2048 = faster than 4096 BPTT
                              # while still giving episode completions within ~9 rollouts.
    "batch_size":     512,    # PERF: was 256. Larger batches = better GPU utilization, fewer updates.
    "n_epochs":       5,
    "gamma":          0.995,
    "gae_lambda":     0.95,
    "clip_range":     0.25,      # Loosened to allow the policy to escape the HOLD basin
    "ent_coef":       0.08,      # High starting entropy (will be annealed to 0.01)
    "vf_coef":        0.5,
    "max_grad_norm":  0.5,
    "policy_kwargs": {
        "net_arch":      dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        "activation_fn": torch.nn.Tanh,
    },
}
