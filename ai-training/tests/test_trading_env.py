"""
Invariant tests for TradingEnv — the contract every training run relies on.

All tests use synthetic dataframes (no parquet needed) with the v1 feature set
and a controllable close-price path, so the suite runs on a fresh clone/CI.

Conventions used throughout:
- window_size=10, so episodes start at current_step=10 and the action taken on
  step k executes against close[k].
- position_fraction defaults to 0.15 (config), but each env here passes it
  explicitly so config changes don't silently alter test expectations.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environments.trading_env import TradingEnv, TradeRecord  # noqa: E402

WINDOW = 10
N_FEATURES_V1 = 18  # FEATURE_COLS_V1


def make_df(close=None, n=300, feature_value=0.01, ema_cross_long=0.5):
    """Synthetic v1-feature dataframe with a controllable close path."""
    if close is None:
        close = np.full(n, 100.0)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)
    data = {c: np.full(n, feature_value) for c in TradingEnv.FEATURE_COLS_V1}
    # ema_cross_long doubles as the regime-router trend signal (v1 fallback)
    data["ema_cross_long"] = np.full(n, float(ema_cross_long))
    data["close"] = close
    data["high"]  = close * 1.001
    data["low"]   = close * 0.999
    return pd.DataFrame(data)


def make_env(df, **kwargs):
    kwargs.setdefault("window_size", WINDOW)
    kwargs.setdefault("position_fraction", 0.15)
    return TradingEnv(df, **kwargs)


# ── Observation contract ──────────────────────────────────────────────────────

def test_observation_shape_and_finiteness():
    env = make_env(make_df())
    obs, _ = env.reset(seed=0)
    assert obs.shape == (WINDOW * N_FEATURES_V1 + TradingEnv.N_PORTFOLIO_FEATURES,)
    assert np.isfinite(obs).all()


def test_reset_is_deterministic():
    env = make_env(make_df())
    obs1, _ = env.reset(seed=0)
    env.step(1)
    obs2, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs1, obs2)


# ── Fee / PnL math ────────────────────────────────────────────────────────────

def test_long_roundtrip_accounting_identity():
    """After a full round trip starting flat, balance == initial + net_pnl,
    and fees_paid == entry_fee + exit_fee."""
    env = make_env(make_df())
    env.reset(seed=0)
    env.step(1)  # BUY  @ close=100
    env.step(2)  # SELL @ close=100
    assert len(env.trade_history) == 1
    rec: TradeRecord = env.trade_history[0]
    assert rec.exit_step is not None
    assert rec.fees_paid == pytest.approx(rec.entry_fee + rec.exit_fee)
    assert rec.net_pnl == pytest.approx(rec.gross_pnl - rec.fees_paid)
    assert env.balance == pytest.approx(10_000.0 + rec.net_pnl)


def test_flat_price_roundtrip_loses_fees_and_slippage():
    """Price never moves → the trade must lose exactly the friction costs."""
    env = make_env(make_df())
    env.reset(seed=0)
    env.step(1)
    env.step(2)
    rec = env.trade_history[0]
    assert rec.net_pnl < 0
    # gross loss is the 2×slippage drag; net adds both fees on top
    assert rec.gross_pnl < 0
    assert abs(rec.net_pnl) > abs(rec.gross_pnl)


def test_short_profits_when_price_falls():
    close = np.full(300, 100.0)
    close[12:] = 90.0
    env = make_env(make_df(close), allow_short=True)
    env.reset(seed=0)
    env.step(2)  # SELL from flat → open short @100
    env.step(0)  # HOLD
    env.step(1)  # BUY → cover @90
    rec = env.trade_history[0]
    assert rec.direction == -1
    assert rec.net_pnl > 0
    assert env.balance > 10_000.0


def test_long_loses_when_price_falls_and_short_mirrors_it():
    """Long and short on the same falling path must have opposite-sign gross PnL."""
    close = np.full(300, 100.0)
    close[12:] = 95.0
    df = make_df(close)

    env_l = make_env(df)
    env_l.reset(seed=0)
    env_l.step(1); env_l.step(0); env_l.step(2)

    env_s = make_env(df, allow_short=True)
    env_s.reset(seed=0)
    env_s.step(2); env_s.step(0); env_s.step(1)

    assert env_l.trade_history[0].gross_pnl < 0 < env_s.trade_history[0].gross_pnl


# ── Action ladder ─────────────────────────────────────────────────────────────

def test_ladder_transitions():
    env = make_env(make_df(), allow_short=True)
    env.reset(seed=0)
    assert env.position_dir == 0
    env.step(2); assert env.position_dir == -1   # flat  → short
    env.step(1); assert env.position_dir == 0    # short → cover
    env.step(1); assert env.position_dir == 1    # flat  → long
    env.step(2); assert env.position_dir == 0    # long  → close
    assert len(env.trade_history) == 2


def test_invalid_actions_remap_to_hold():
    env = make_env(make_df())  # long-only
    env.reset(seed=0)
    env.step(2)  # SELL while flat → invalid
    assert env.position_dir == 0
    env.step(1)  # BUY → long
    env.step(1)  # BUY while long → invalid
    assert env.position_dir == 1
    assert len(env.trade_history) == 1
    assert env._reward_components["invalid_action_count"] == 2


def test_long_only_can_never_go_short():
    env = make_env(make_df())
    env.reset(seed=0)
    for _ in range(20):
        env.step(2)
        assert env.position_dir == 0


# ── Risk controls ─────────────────────────────────────────────────────────────

def test_hard_stop_forces_exit():
    close = np.full(300, 100.0)
    close[12:] = 96.0  # −4% < −2.5% stop
    env = make_env(make_df(close))
    env.reset(seed=0)
    env.step(1)  # BUY @100
    env.step(0)  # HOLD @100
    env.step(0)  # HOLD @96 → forced stop
    assert env.position_dir == 0
    assert env._reward_components["stop_loss_count"] == 1
    assert env.trade_history[0].exit_step is not None


def test_hard_stop_covers_shorts_too():
    close = np.full(300, 100.0)
    close[12:] = 104.0  # +4% against a short
    env = make_env(make_df(close), allow_short=True)
    env.reset(seed=0)
    env.step(2)  # short @100
    env.step(0)
    env.step(0)  # @104 → forced cover
    assert env.position_dir == 0
    assert env._reward_components["stop_loss_count"] == 1


def test_drawdown_breach_terminates_episode():
    close = np.full(300, 100.0)
    close[12:] = 80.0
    env = make_env(make_df(close), position_fraction=1.0, max_drawdown_pct=0.05)
    env.reset(seed=0)
    env.step(1)  # all-in long @100
    terminated = False
    for _ in range(5):
        _, _, terminated, _, _ = env.step(0)
        if terminated:
            break
    assert terminated


# ── Behavior gates (all opt-in) ───────────────────────────────────────────────

def test_cooldown_blocks_reentry():
    env = make_env(make_df(), cooldown_candles=5)
    env.reset(seed=0)
    env.step(1)  # open  @10
    env.step(2)  # close @11 → cooldown until 16
    env.step(1)  # @12 blocked
    assert env.position_dir == 0
    assert env._reward_components["cooldown_count"] == 1
    for _ in range(3):  # steps 13,14,15 blocked too
        env.step(1)
    assert env.position_dir == 0
    env.step(1)  # @16 → allowed again
    assert env.position_dir == 1
    assert len(env.trade_history) == 2


def test_regime_router_routes_direction():
    # downtrend everywhere → longs blocked, shorts allowed
    df = make_df(ema_cross_long=-0.5)
    env = make_env(df, allow_short=True, regime_router=True)
    env.reset(seed=0)
    env.step(1)  # BUY in a downtrend → routed to HOLD
    assert env.position_dir == 0
    assert env._reward_components["regime_routed_count"] == 1
    env.step(2)  # SELL in a downtrend → short opens
    assert env.position_dir == -1


def test_regime_router_never_blocks_exits():
    df = make_df(ema_cross_long=-0.5)
    env = make_env(df, allow_short=True, regime_router=True)
    env.reset(seed=0)
    env.step(2)            # short
    env.step(1)            # BUY = cover; routing only gates NEW entries
    assert env.position_dir == 0
    assert env.trade_history[0].exit_step is not None


def test_outlier_gate_blocks_ood_candles():
    df = make_df()
    spike_cols = list(TradingEnv.FEATURE_COLS_V1)
    df.loc[WINDOW, spike_cols] = 50.0       # step 10 is wildly out-of-distribution
    env = make_env(df, outlier_threshold=3.0)
    env.reset(seed=0)
    env.step(1)  # entry attempt on the spike candle → gated
    assert env.position_dir == 0
    assert env._reward_components["outlier_gated_count"] == 1
    env.step(1)  # next candle is normal → entry allowed
    assert env.position_dir == 1


# ── Feature scaling (F5) ──────────────────────────────────────────────────────

def test_minmax_scaler_bounds_and_default_parity():
    df = make_df(close=100.0 + np.cumsum(np.random.default_rng(0).normal(0, 1, 300)))
    base = make_env(df)
    fm = base._feature_matrix
    scaler = (fm.min(axis=0), fm.max(axis=0))

    scaled = make_env(df, feature_scaler=scaler)
    assert scaled._feature_matrix.min() >= -1.0 - 1e-6
    assert scaled._feature_matrix.max() <= 1.0 + 1e-6

    again = make_env(df)  # no scaler → bit-identical to before
    np.testing.assert_array_equal(again._feature_matrix, fm)


# ── Reward modes ──────────────────────────────────────────────────────────────

def test_exit_reward_scores_fee_losing_scalp_negative():
    env = make_env(make_df(), reward_mode="exit")
    env.reset(seed=0)
    env.step(1)                       # open
    _, r, _, _, _ = env.step(2)       # close a flat-price scalp → net loss
    assert r < 0


def test_fixb_reward_is_portfolio_return():
    env = make_env(make_df())
    env.reset(seed=0)
    _, r, _, _, _ = env.step(0)       # flat HOLD on flat prices
    assert r == pytest.approx(0.0)


# ── Episode metrics ───────────────────────────────────────────────────────────

def test_episode_metrics_contract():
    close = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.5, 400))
    env = make_env(make_df(close=np.maximum(close, 10.0)), allow_short=True,
                   candles_per_day=24)
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(int(rng.integers(0, 3)))
        done = term or trunc
    m = env.get_episode_metrics()
    for key in ("sharpe_ratio", "sortino_ratio", "calmar_ratio",
                "gross_profit_factor", "net_profit_factor",
                "gross_expectancy_pct", "net_expectancy_pct", "total_trades"):
        assert key in m, f"missing metric: {key}"
        assert np.isfinite(m[key]), f"non-finite metric: {key}"
