"""
export_onnx.py
──────────────
Converts a trained Stable-Baselines3 PPO model into ONNX format
so it can be loaded by onnxruntime-node inside the Node.js execution engine.

The export process:
  1. Load the .zip checkpoint (SB3 format)
  2. Extract the policy network (actor/policy_net) as a raw PyTorch module
  3. Trace it with a dummy observation tensor
  4. Export to ONNX with dynamic batch dimension
  5. Verify the ONNX model produces identical outputs to PyTorch

Usage:
    python export_onnx.py --model ../models/tradebot_ppo_best.zip
    python export_onnx.py --model ../models/tradebot_ppo_best.zip --output ../models/tradebot.onnx
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

from stable_baselines3 import PPO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ── Policy Wrapper ────────────────────────────────────────────────────────────

class PolicyWrapper(nn.Module):
    """
    Wraps the SB3 PPO actor network into a clean PyTorch module.

    SB3's internal policy outputs logits for all actions.
    We apply softmax to get probabilities, then argmax for the
    deterministic action — matching model.predict(deterministic=True).

    The Node.js runtime will call this and read the output as:
      [action_index, prob_hold, prob_buy, prob_sell]
    where action_index = argmax of probabilities.
    """

    def __init__(self, sb3_policy, normalization_stats: dict):
        super().__init__()
        self.mlp_extractor  = sb3_policy.mlp_extractor
        self.action_net     = sb3_policy.action_net
        self.clip_obs       = float(normalization_stats["clip_obs"])
        self.epsilon        = float(normalization_stats["epsilon"])
        self.register_buffer(
            "obs_mean",
            torch.tensor(normalization_stats["mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "obs_var",
            torch.tensor(normalization_stats["var"], dtype=torch.float32),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Input:  obs  [batch, obs_dim]  float32
        Output: out  [batch, 4]        float32
                  → [action_idx, p_hold, p_buy, p_sell]
        """
        obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
        obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)

        # Extract latent features through shared MLP layers
        latent_pi, _ = self.mlp_extractor(obs)

        # Get action logits from the policy head
        action_logits = self.action_net(latent_pi)          # [batch, 3]

        # Convert to probabilities
        probs = torch.softmax(action_logits, dim=-1)        # [batch, 3]

        # Deterministic action: argmax
        action_idx = torch.argmax(probs, dim=-1, keepdim=True).float()  # [batch, 1]

        # Concatenate: [action, p_hold, p_buy, p_sell]
        out = torch.cat([action_idx, probs], dim=-1)        # [batch, 4]
        return out


# ── Export ────────────────────────────────────────────────────────────────────

def load_vecnormalize_stats(vecnormalize_path: str) -> dict:
    if not os.path.exists(vecnormalize_path):
        raise FileNotFoundError(
            f"VecNormalize stats not found: {vecnormalize_path}\n"
            "Train with train_agent.py first; it saves *_vecnormalize.pkl next to the model."
        )

    with open(vecnormalize_path, "rb") as f:
        vec_norm = pickle.load(f)

    # P2-2 fix: validate expected attributes exist before accessing them.
    # Direct pickle.load is fragile — if SB3's VecNormalize internal format
    # changes between versions, attribute access silently returns wrong data.
    required = [("obs_rms", "mean"), ("obs_rms", "var")]
    for parent_attr, child_attr in required:
        parent = getattr(vec_norm, parent_attr, None)
        if parent is None or not hasattr(parent, child_attr):
            raise ValueError(
                f"VecNormalize pickle is missing '{parent_attr}.{child_attr}'. "
                f"This file may be from an incompatible SB3 version or corrupted. "
                f"Re-train to generate a fresh .pkl."
            )

    mean = vec_norm.obs_rms.mean.astype(np.float32)
    var  = vec_norm.obs_rms.var.astype(np.float32)

    if np.any(np.isnan(mean)) or np.any(np.isnan(var)):
        raise ValueError("VecNormalize stats contain NaN — pkl is corrupted, re-train.")
    if np.all(var < 1e-8):
        raise ValueError("VecNormalize var is all near-zero — normalization stats not populated. "
                         "Did training complete at least one full rollout?")

    print(f"[VecNorm] mean range : [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"[VecNorm] var  range : [{var.min():.6f}, {var.max():.4f}]")

    return {
        "mean":     mean,
        "var":      var,
        "clip_obs": float(vec_norm.clip_obs),
        "epsilon":  float(getattr(vec_norm, "epsilon", 1e-8)),
    }


def export_to_onnx(
    model_path: str,
    output_path: str,
    obs_dim: int,
    normalization_stats: dict,
) -> tuple[str, PolicyWrapper]:
    """
    Loads a trained SB3 PPO model and exports its actor network to ONNX.

    Parameters
    ----------
    model_path  : Path to the .zip SB3 checkpoint
    output_path : Destination .onnx file path
    obs_dim     : Observation vector dimension (must match training env)

    Returns
    -------
    Path to the saved .onnx file
    """
    print(f"\n[LOAD] Loading SB3 model: {model_path}")
    model = PPO.load(model_path, device="cpu")
    policy = model.policy.to("cpu")
    policy.eval()

    # Wrap in our clean exporter module
    wrapper = PolicyWrapper(policy, normalization_stats)
    wrapper.eval()

    # Dummy input matching the observation space
    dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32)

    print(f"[INFO] Observation dimension: {obs_dim}")
    print(f"[INFO] Running dummy forward pass...")

    with torch.no_grad():
        dummy_out = wrapper(dummy_obs)
    print(f"[INFO] Dummy output: {dummy_out.numpy()}")
    print(f"        → Action: {int(dummy_out[0, 0].item())} | "
          f"P(HOLD): {dummy_out[0,1]:.4f} | "
          f"P(BUY):  {dummy_out[0,2]:.4f} | "
          f"P(SELL): {dummy_out[0,3]:.4f}")

    # ── ONNX Export ───────────────────────────────────────────────────────────
    print(f"\n[EXPORT] Writing ONNX model to: {output_path}")
    torch.onnx.export(
        model           = wrapper,
        args            = dummy_obs,
        f               = output_path,
        export_params   = True,
        opset_version   = 17,
        do_constant_folding = True,
        input_names     = ["observation"],
        output_names    = ["action_and_probs"],
        dynamic_axes    = {
            "observation":       {0: "batch_size"},
            "action_and_probs":  {0: "batch_size"},
        },
        verbose         = False,
    )

    return output_path, wrapper


# ── Verification ──────────────────────────────────────────────────────────────

def verify_onnx(onnx_path: str, wrapper: PolicyWrapper, obs_dim: int) -> None:
    """
    Runs ONNX model checker + compares output against original PyTorch model.
    This ensures the export is numerically identical to what was trained.
    """
    print(f"\n[VERIFY] Checking ONNX model validity...")
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print(f"[OK] ONNX model structure is valid.")

    print(f"[VERIFY] Running inference with ONNXRuntime...")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    dummy_obs = np.random.randn(1, obs_dim).astype(np.float32)
    outputs   = session.run(None, {"observation": dummy_obs})
    result    = outputs[0][0]

    with torch.no_grad():
        torch_result = wrapper(torch.from_numpy(dummy_obs)).numpy()[0]
    np.testing.assert_allclose(result, torch_result, rtol=1e-4, atol=1e-5)

    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    action_idx = int(result[0])

    print(f"[OK] ONNXRuntime inference successful and matches PyTorch wrapper.")
    print(f"     Action      : {action_idx} ({action_map.get(action_idx, '?')})")
    print(f"     P(HOLD)     : {result[1]:.6f}")
    print(f"     P(BUY)      : {result[2]:.6f}")
    print(f"     P(SELL)     : {result[3]:.6f}")

    # Print model I/O spec (useful when writing the Node.js inference code)
    print(f"\n[MODEL SPEC] Input/Output for onnxruntime-node:")
    for inp in session.get_inputs():
        print(f"  INPUT  name='{inp.name}' shape={inp.shape} type={inp.type}")
    for out in session.get_outputs():
        print(f"  OUTPUT name='{out.name}' shape={out.shape} type={out.type}")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export SB3 PPO model to ONNX")
    parser.add_argument("--model",  type=str, required=True,
                        help="Path to trained .zip SB3 model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .onnx path (default: same dir as model)")
    parser.add_argument("--obs-dim", type=int, default=None,
                        help="Observation dimension. Auto-detected from model if not set.")
    parser.add_argument("--vecnormalize", type=str, default=None,
                        help="Path to VecNormalize .pkl stats (default: model path with _vecnormalize.pkl)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    # Auto-detect obs_dim from the loaded model
    model = PPO.load(args.model, device="cpu")
    obs_dim = model.observation_space.shape[0]
    if args.obs_dim:
        obs_dim = args.obs_dim
    print(f"[INFO] Detected observation dimension: {obs_dim}")

    # Default output path
    output_path = args.output or args.model.replace(".zip", ".onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vecnormalize_path = args.vecnormalize or args.model.replace(".zip", "_vecnormalize.pkl")
    normalization_stats = load_vecnormalize_stats(vecnormalize_path)
    print(f"[INFO] Loaded VecNormalize stats: {vecnormalize_path}")

    _, wrapper = export_to_onnx(args.model, output_path, obs_dim, normalization_stats)
    verify_onnx(output_path, wrapper, obs_dim)

    size_mb = os.path.getsize(output_path) / 1_048_576
    print(f"\n[DONE] ONNX model saved: {output_path} ({size_mb:.2f} MB)")
    print(f"       Copy this file to: execution-engine/src/strategist/models/")


if __name__ == "__main__":
    main()
