# FLAG

This project combines:
- PyTorch
- PyTorch Geometric
- Sentence Transformers
- Hugging Face Transformers / PEFT
- graph-based node classification and LLM-assisted text generation

The repo is intended for GPU-first execution, but it also contains CPU-oriented setup instructions below.

## Recommended environment

- Python: 3.11.x
- OS: Windows 10/11
- GPU: NVIDIA CUDA-capable card, e.g. Quadro RTX 4000

## 1) Create a virtual environment

From the project root:

```powershell
cd "D:\Manohar\Uni\Major Project\Codes\FLAG"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

## 2) Install dependencies

### CPU-only setup

```powershell
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### GPU setup

For an NVIDIA GPU and a compatible driver:

```powershell
python -m pip uninstall -y torch torch-scatter
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

If needed, install the PyG wheel explicitly:

```powershell
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
python -m pip install torch-geometric
```

If you are on CPU, change `+cu124` to `+cpu` in the PyG wheel URL.

## 3) Verify installation

```powershell
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"
python -c "import torch_geometric, torch_scatter; print('pyg ok')"
```

Expected on GPU systems:
- `torch.cuda.is_available()` is `True`
- `torch.version.cuda` is not `None`

## 4) Dataset layout

The repo expects dataset files at the project root in folders such as:

```text
FLAG/
  Instagram/
    instagram.pt
  Reddit/
    reddit.pt
```

The scripts also expect generated split files such as:
- `Instagram/train.pt`
- `Instagram/val.pt`
- `Instagram/test.pt`
- `Reddit/train.pt`
- `Reddit/val.pt`
- `Reddit/test.pt`

If you only have the raw graph object (`instagram.pt` or `reddit.pt`), run the preprocessing script first to create the split files.

## 5) Preprocessing / embedding step

```powershell
python encode.py --path "Instagram/"
```

or

```powershell
python encode.py --path "Reddit/"
```

This script creates or updates the expected training/validation/test artifacts and adds embeddings using Sentence Transformers.

## 6) Model access for chat scripts

The chat scripts use Hugging Face models. They are not meant for arbitrary private or gated repos without access.

Use a public model such as:

```python
model_name = "microsoft/Phi-3.5-mini-instruct"
```

If you want to use a gated model such as Gemma, you must first gain access on Hugging Face and authenticate with `huggingface-cli login` or a valid token.

## 7) Run the chat pipeline

```powershell
python chat.py
```

or for Instagram:

```powershell
python chat1.py
```

## 8) Important notes

- The repo is GPU-first and uses direct CUDA calls in several scripts.
- On CPU-only systems, model loading and graph operations may fail unless you add device-aware checks.
- `torch-scatter` is a known compatibility-sensitive dependency and should use the PyG wheel matching your PyTorch version.
- Python 3.11 is the recommended version for this project.
