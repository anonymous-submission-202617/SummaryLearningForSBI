# Environment

Reproducible environment for the SBI benchmark repository. The tested
environment is macOS (Apple Silicon, PyTorch MPS) with Python 3.13.

## Quick start

```bash
conda env create -f environment.yml
conda activate sbi-summary
```

or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- On NVIDIA GPUs, install the matching CUDA build of PyTorch explicitly.
- `torch` may need a platform-specific wheel on some architectures; if the
  pins fail, install the latest compatible versions of the listed packages.
