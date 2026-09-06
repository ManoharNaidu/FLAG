#!/usr/bin/env bash
set -euo pipefail

# Run this from the repository root on a Vast.ai CUDA image.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python -m pip install torch-geometric
python -m pip install -r requirements.txt
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"cuda device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Choose an NVIDIA Vast.ai instance with a working driver.")
PY
