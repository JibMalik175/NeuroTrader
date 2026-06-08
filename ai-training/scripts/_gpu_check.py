import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("CUDA version: N/A (CPU-only PyTorch installed)")
    print("GPU: None")
    print("")
    print("To install CUDA-enabled PyTorch, run:")
    print("  pip uninstall torch torchvision torchaudio -y")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")

# Also verify SB3 import
from stable_baselines3 import PPO
print(f"\nSB3 PPO: OK")
print(f"Device for training: {'cuda' if torch.cuda.is_available() else 'cpu'}")
