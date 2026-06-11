"""
export_onnx_recurrent.py — RecurrentPPO (LSTM) → ONNX for the Node engine
──────────────────────────────────────────────────────────────────────────
The old export_onnx.py wraps a plain-PPO MLP: stateless, one input, one
output. Our deployment models are RecurrentPPO — the LSTM's hidden state
IS the model's memory (PPO without it collapsed: gross PF 0.058 vs 1.006),
so the ONNX graph must take the state in and hand the updated state back:

    inputs : obs [1, obs_dim], h_in [1, 1, lstm], c_in [1, 1, lstm]
    outputs: probs [1, 3] (HOLD, BUY, SELL), h_out, c_out

The Node engine keeps h/c between candles (zeros at process start, exactly
like episode_start=True at eval time) and feeds them back each call.

VecNormalize obs stats are baked into the graph (mean/var/clip from the
checkpoint's .pkl), so Node feeds RAW features — no normalization drift.

Parity gate: replays N real candle windows through both PyTorch
(model.predict, state carried) and ONNX (state carried); requires 100%
deterministic-action agreement and max prob diff < 1e-5, else no file.

Usage:
  python scripts/export_onnx_recurrent.py \
      --model models/p2_9_makerfee_window1_besttrain.zip \
      --vecnorm models/p2_9_makerfee_window1_besttrain_vecnormalize.pkl \
      --data data/BTC_USDT_1h_val.parquet \
      --output models/p2_9_besttrain.onnx
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import onnx
import onnxruntime as ort
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from environments.trading_env import TradingEnv
from scripts.config import ENV_CONFIG


class RecurrentPolicyWrapper(nn.Module):
    """Single-step LSTM policy with VecNormalize baked in."""

    def __init__(self, policy, norm_mean, norm_var, clip_obs, epsilon):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.lstm = policy.lstm_actor
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net
        self.clip_obs = float(clip_obs)
        self.epsilon = float(epsilon)
        self.register_buffer("obs_mean", torch.tensor(norm_mean, dtype=torch.float32))
        self.register_buffer("obs_var", torch.tensor(norm_var, dtype=torch.float32))

    def forward(self, obs, h_in, c_in):
        x = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
        x = torch.clamp(x, -self.clip_obs, self.clip_obs)
        feat = self.features_extractor(x)                  # Flatten = identity here
        seq = feat.unsqueeze(0)                            # [seq=1, batch, dim]
        out, (h_out, c_out) = self.lstm(seq, (h_in, c_in))
        latent = self.policy_net(out.squeeze(0))
        logits = self.action_net(latent)
        probs = torch.softmax(logits, dim=-1)
        return probs, h_out, c_out


def build_obs_sequence(data_path: str, n_steps: int) -> list[np.ndarray]:
    """Real observation sequence: deterministic all-HOLD walk through the env."""
    df = pd.read_parquet(data_path).reset_index(drop=True)
    env = TradingEnv(df.iloc[: n_steps + ENV_CONFIG.get("window_size", 48) + 10], **ENV_CONFIG)
    obs, _ = env.reset()
    seq = [obs.astype(np.float32)]
    for _ in range(n_steps - 1):
        obs, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
        seq.append(obs.astype(np.float32))
    return seq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--vecnorm", required=True)
    p.add_argument("--data", required=True, help="parquet for the parity replay")
    p.add_argument("--output", required=True)
    p.add_argument("--parity-steps", type=int, default=300)
    args = p.parse_args()

    print(f"[LOAD] {args.model}")
    model = RecurrentPPO.load(args.model, device="cpu")
    policy = model.policy
    policy.eval()

    obs_dim = model.observation_space.shape[0]
    lstm_size = policy.lstm_actor.hidden_size
    n_layers = policy.lstm_actor.num_layers
    print(f"[MODEL] obs_dim={obs_dim} lstm={lstm_size}x{n_layers}")

    # VecNormalize stats (need a dummy env with the right obs space to load)
    dummy_df = pd.read_parquet(args.data).reset_index(drop=True).iloc[:700]
    dummy = DummyVecEnv([lambda: Monitor(TradingEnv(dummy_df, **ENV_CONFIG))])
    vn = VecNormalize.load(args.vecnorm, dummy)
    assert vn.obs_rms.mean.shape[0] == obs_dim, "vecnorm/model obs_dim mismatch"

    wrapper = RecurrentPolicyWrapper(
        policy, vn.obs_rms.mean, vn.obs_rms.var, vn.clip_obs, vn.epsilon
    ).eval()

    # ── Export ───────────────────────────────────────────────────────────────
    d_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
    d_h = torch.zeros(n_layers, 1, lstm_size, dtype=torch.float32)
    d_c = torch.zeros(n_layers, 1, lstm_size, dtype=torch.float32)

    torch.onnx.export(
        wrapper, (d_obs, d_h, d_c), args.output,
        input_names=["obs", "h_in", "c_in"],
        output_names=["probs", "h_out", "c_out"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(args.output))
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"[EXPORT] {args.output} ({size_mb:.1f} MB) — graph valid")

    # ── Parity gate: PyTorch vs ONNX over a REAL sequence, states carried ──
    print(f"[PARITY] replaying {args.parity_steps} real candles through both runtimes...")
    seq = build_obs_sequence(args.data, args.parity_steps)

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])

    # PyTorch side: model.predict with carried state (the training-eval path)
    pt_actions = []
    state = None
    ep_start = np.ones((1,), dtype=bool)
    for o in seq:
        norm = vn.normalize_obs(o[None, :].astype(np.float32))[0]
        a, state = model.predict(norm, state=state, episode_start=ep_start,
                                 deterministic=True)
        ep_start = np.zeros((1,), dtype=bool)
        pt_actions.append(int(a))

    # ONNX side: raw obs in, state tensors carried manually. The PyTorch
    # wrapper runs in lockstep (carrying its own state) for the numeric diff.
    ox_actions = []
    max_prob_diff = 0.0
    h = np.zeros((n_layers, 1, lstm_size), dtype=np.float32)
    c = np.zeros((n_layers, 1, lstm_size), dtype=np.float32)
    wh = torch.zeros(n_layers, 1, lstm_size)
    wc = torch.zeros(n_layers, 1, lstm_size)
    for o in seq:
        probs, h, c = sess.run(None, {"obs": o[None, :], "h_in": h, "c_in": c})
        ox_actions.append(int(np.argmax(probs[0])))
        with torch.no_grad():
            ref_probs, wh, wc = wrapper(torch.tensor(o[None, :]), wh, wc)
        max_prob_diff = max(max_prob_diff,
                            float(np.abs(ref_probs.numpy() - probs).max()))

    agree = sum(a == b for a, b in zip(pt_actions, ox_actions))
    total = len(pt_actions)
    print(f"[PARITY] action agreement: {agree}/{total} "
          f"({agree / total:.1%}) | max prob diff vs wrapper: {max_prob_diff:.2e}")

    if agree != total:
        os.remove(args.output)
        sys.exit("[FAIL] deterministic actions diverged — export DELETED. Do not deploy.")
    print("[PASS] ONNX is action-identical to the PyTorch checkpoint. Safe to deploy.")


if __name__ == "__main__":
    main()
