#!/bin/bash
# ============================================================================
# GPU cloud environment setup script
# Base image: Ubuntu 22.04, Python 3.13, PyTorch 2.8.0, CUDA 12.8, cuDNN 8,
#             NVCC, VNC (preinstalled)
# Creates a dedicated conda environment "summary" to avoid Python 3.13
# compatibility issues.
# ============================================================================

set -e  # exit immediately on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

echo ""
echo -e "${BLUE}=========================================="
echo "  Summary experiment environment setup script"
echo "  Conda env: summary (Python 3.12)"
echo "==========================================${NC}"
echo ""

# ----------------------------------------------------------------------------
# Step 0: make sure conda is available
# ----------------------------------------------------------------------------
echo -e "${YELLOW}[0/7] Checking conda availability...${NC}"

# Try common conda locations (different cloud images install conda in
# different places).
CONDA_FOUND=false
for possible_conda in "conda" \
                       "/opt/conda/bin/conda" \
                       "$HOME/miniconda3/bin/conda" \
                       "$HOME/anaconda3/bin/conda" \
                       "/usr/local/miniconda3/bin/conda"; do
    if command -v "$possible_conda" &> /dev/null 2>&1; then
        CONDA_CMD="$possible_conda"
        CONDA_FOUND=true
        break
    fi
done

if [ "$CONDA_FOUND" = true ]; then
    echo -e "  ${GREEN}conda found: $CONDA_CMD${NC}"
else
    echo -e "  ${YELLOW}conda not found; installing Miniconda...${NC}"
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    wget -q "$MINICONDA_URL" -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    CONDA_CMD="$HOME/miniconda3/bin/conda"
    echo -e "  ${GREEN}Miniconda installed${NC}"
fi

# Make conda usable in the current script (shell hook).
eval "$("$CONDA_CMD" shell.bash hook)"

# Key fix: always run `conda init bash`, whether conda pre-existed or was just
# installed; otherwise conda is not loaded automatically in new shells.
"$CONDA_CMD" init bash
source ~/.bashrc 2>/dev/null || true

echo "  conda version: $(conda --version 2>/dev/null || echo 'unknown')"

# ----------------------------------------------------------------------------
# Step 0.5: reset conda mirror configuration (key fix!)
# ----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[0.5] Resetting conda mirror configuration...${NC}"

# Remove all condarc files (including backups).
rm -f ~/.condarc ~/.condarc.bak.*

# Re-initialize the configuration and force the official defaults channel.
conda config --remove-key channels 2>/dev/null || true
conda config --remove-key custom_channels 2>/dev/null || true
conda config --remove-key default_channels 2>/dev/null || true
conda config --add channels defaults
conda config --set show_channel_urls yes

# Verify.
if grep -q "tuna" ~/.condarc 2>/dev/null; then
    echo -e "  ${RED}Warning: ~/.condarc still contains a non-default mirror; please check manually${NC}"
    cat ~/.condarc
else
    echo -e "  ${GREEN}conda channels reset to defaults${NC}"
fi

# Clean the cache.
conda clean -a -y

# ----------------------------------------------------------------------------
# Step 1: create the conda environment
# ----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[1/7] Creating conda environment 'summary' (Python 3.12)...${NC}"

# Remove an existing environment first (idempotent).
if conda env list 2>/dev/null | grep -q "^summary "; then
    echo "  Removing existing environment 'summary'..."
    conda deactivate 2>/dev/null || true
    conda env remove -n summary -y
fi

conda create -n summary python=3.12 -y --no-default-packages
conda activate summary

echo -e "  ${GREEN}conda environment created${NC}"
echo -e "  Python: $(python --version)"

# ----------------------------------------------------------------------------
# Step 2: configure the pip index (auto-detect an available mirror)
# ----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[2/7] Configuring the pip index...${NC}"

# Candidate pip mirrors: "name|URL"
PIP_MIRRORS=(
    "aliyun|https://mirrors.aliyun.com/pypi/simple/"
    "ustc|https://mirrors.ustc.edu.cn/pypi/simple/"
    "tsinghua|https://pypi.tuna.tsinghua.edu.cn/simple/"
    "default|https://pypi.org/simple/"
)

PIP_INDEX=""
for entry in "${PIP_MIRRORS[@]}"; do
    name="${entry%%|*}"
    url="${entry##*|}"
    echo -n "  Testing $name ... "
    # Probe connectivity with `pip install --dry-run` (10s timeout; does not install).
    if timeout 10 pip install --dry-run --index-url "$url" pip &>/dev/null; then
        echo -e "${GREEN}available${NC}"
        PIP_INDEX="$url"
        pip config set global.index-url "$url" 2>/dev/null
        if [ "$name" != "default" ]; then
            host=$(echo "$url" | sed 's|https://||;s|/.*||')
            pip config set install.trusted-host "$host" 2>/dev/null
        fi
        break
    else
        echo -e "${RED}unavailable${NC}"
    fi
done

if [ -z "$PIP_INDEX" ]; then
    echo -e "  ${YELLOW}All mirrors unavailable; using the official PyPI index${NC}"
    pip config unset global.index-url 2>/dev/null || true
fi

echo -e "  ${GREEN}Using pip index: ${PIP_INDEX:-PyPI official}${NC}"
pip install --upgrade pip setuptools wheel --quiet

# ----------------------------------------------------------------------------
# Step 3: install the PyTorch stack (CUDA 12.8)
# ----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[3/7] Installing PyTorch 2.8.0 + CUDA 12.8...${NC}"

# Install the official PyTorch CUDA 12.8 wheels.
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

echo ""
python -c "
import torch
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version:   {torch.version.cuda}')
print(f'  GPU count:      {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  GPU name:       {torch.cuda.get_device_name(0)}')
" || {
    echo -e "${RED}  PyTorch CUDA installation verification failed! Check the CUDA driver version${NC}"
    exit 1
}

# ----------------------------------------------------------------------------
# Step 4: scientific computing packages
# ----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[4/7] Installing scientific computing packages...${NC}"

pip install numpy==2.3.4 scipy==1.17.0
pip install scikit-learn==1.8.0
pip install pandas==2.3.3

echo -e "  ${GREEN}Scientific computing packages installed${NC}"

# ----------------------------------------------------------------------------
# Step 5: SBI, BayesFlow and probabilistic programming
# ----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[5/7] Installing the SBI and probabilistic programming ecosystem...${NC}"

# --- Neural flow models ---
pip install nflows==0.14 zuko==1.5.0
pip install skorch

# --- SBI core ---
pip install sbi==0.26.1

# --- BayesFlow ---
pip install bayesflow==2.0.8
pip install keras

# --- MCMC ---
pip install pyro-ppl==1.9.1
pip install pymc==5.27.1 pytensor==2.37.0
pip install arviz==0.23.1 xarray==2026.1.0

# --- ABC ---
pip install pyabc==0.12.16

# --- Wasserstein distance ---
pip install pot

echo -e "  ${GREEN}SBI ecosystem installed${NC}"

# ----------------------------------------------------------------------------
# Step 6: visualization and other tools
# ----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[6/7] Installing visualization and other tools...${NC}"

pip install matplotlib==3.10.8 seaborn==0.13.2
pip install pillow==12.0.0
pip install altair==6.0.0
pip install streamlit==1.53.1

echo -e "  ${GREEN}Visualization tools installed${NC}"

# ----------------------------------------------------------------------------
# Step 7: full verification
# ----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[7/7] Verifying all installations...${NC}"

python << 'PYEOF'
import sys

print("\n" + "=" * 60)
print("  Environment Verification Report")
print("=" * 60)

errors = []

# -- Python --
print(f"\nPython: {sys.version.split()[0]}")

# -- PyTorch + CUDA --
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    print(f"  CUDA version   : {torch.version.cuda}")
    print(f"  GPU count      : {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"  GPU name       : {torch.cuda.get_device_name(0)}")
    else:
        errors.append("CUDA is not available; check the GPU driver")
except Exception as e:
    errors.append(f"Failed to import PyTorch: {e}")

# -- Torchvision / Torchaudio --
for pkg, name in [("torchvision", "torchvision"), ("torchaudio", "torchaudio")]:
    try:
        mod = __import__(name)
        print(f"{pkg}: {mod.__version__}")
    except Exception as e:
        errors.append(f"{pkg}: {e}")

# -- SBI --
try:
    import sbi
    print(f"sbi: {sbi.__version__}")
except Exception as e:
    errors.append(f"sbi: {e}")

# -- BayesFlow + Keras --
try:
    import bayesflow
    print(f"bayesflow: {bayesflow.__version__}")
except Exception as e:
    errors.append(f"bayesflow: {e}")

try:
    import keras
    print(f"keras: {keras.__version__}")
    backend = keras.config.backend() if hasattr(keras.config, 'backend') else 'unknown'
    print(f"  Keras backend: {backend}")
except Exception as e:
    errors.append(f"keras: {e}")

# -- Other key dependencies --
dep_list = [
    ("numpy",        "numpy"),
    ("scipy",        "scipy"),
    ("scikit-learn", "sklearn"),
    ("pandas",       "pandas"),
    ("matplotlib",   "matplotlib"),
    ("seaborn",      "seaborn"),
    ("nflows",       "nflows"),
    ("zuko",         "zuko"),
    ("skorch",       "skorch"),
    ("pyro-ppl",     "pyro"),
    ("arviz",        "arviz"),
    ("pymc",         "pymc"),
    ("pytensor",     "pytensor"),
    ("pyabc",        "pyabc"),
    ("pot",          "ot"),
    ("pillow",       "PIL"),
    ("xarray",       "xarray"),
]

print(f"\nDependency check ({len(dep_list)} packages):")
for pkg_name, import_name in dep_list:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, '__version__', '?')
        print(f"  [OK] {pkg_name}: {ver}")
    except Exception as e:
        print(f"  [FAIL] {pkg_name}: {e}")
        errors.append(pkg_name)

# -- Final summary --
print("\n" + "=" * 60)
if errors:
    print(f"  Found {len(errors)} problem(s):")
    for e in errors:
        print(f"    - {e}")
else:
    print("  All dependencies installed successfully; environment is ready!")
print("=" * 60)
print("\nTo activate the environment:")
print("  conda activate summary")
print()
PYEOF

echo ""
echo -e "${BLUE}=========================================="
echo "  Configuration complete!"
echo ""
echo "  Activate the environment and run an experiment:"
echo "    conda activate summary"
echo "    cd LotkaVoletrra"
echo "    python run_experiment.py --config config.json"
echo "==========================================${NC}"
