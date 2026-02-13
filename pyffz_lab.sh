#!/usr/bin/env bash
set -euo pipefail

############################################################
# PYFFZ-LAB GOD MODE DEPLOYMENT SYSTEM
# SELF-EXTRACTING RESEARCH APPLIANCE
############################################################

ROOT_DIR="$(pwd)/pyffz_lab"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$ROOT_DIR/.venv"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="run_$TIMESTAMP"

#############################################
# TERMINAL UX ENHANCEMENTS
#############################################

banner() {
  echo
  echo "=================================================="
  echo ">>> $1"
  echo "=================================================="
}

feature() {
  echo "[FEATURE] $1"
}

deploy_python() {
  local srcFile="$1"
  local content="$2"
  printf "%s\n" "$content" > "$ROOT_DIR/scripts/${srcFile}.py"
  echo "[DEPLOYED] scripts/${srcFile}.py"
}

hash_file() {
  sha256sum "$1" | awk '{print $1}'
}

#############################################
# ENVIRONMENT SETUP
#############################################

banner "INITIALIZING ENVIRONMENT"

if [ ! -d "$VENV_DIR" ]; then
  feature "Creating virtual environment"
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

feature "Upgrading pip"
pip install --upgrade pip > /dev/null

feature "Installing dependencies"
pip install torch numpy matplotlib > /dev/null

#############################################
# DIRECTORY ARCHITECTURE
#############################################

banner "CREATING HIGH-LEVEL ARCHITECTURE"

mkdir -p "$ROOT_DIR"/{configs,scripts,logs,plots,tables,checkpoints,runs}

feature "Directory structure established"

#############################################
# SOURCE MATERIALIZATION
#############################################

banner "MATERIALIZING PYTHON SOURCE TREE"

#############################################
# 1. FFZ CORE
#############################################

read -r -d '' FFZ_CORE_PY << 'EOF'
import torch

def ffz_operator(x, lam):
    if lam == float("inf"):
        return torch.sign(x)
    return x / (1.0 + lam * torch.abs(x))
EOF

deploy_python "ffz_core" "$FFZ_CORE_PY"

#############################################
# 2. HISTOGRAM LOGGER
#############################################

read -r -d '' HISTOGRAM_PY << 'EOF'
import numpy as np

def tensor_hist(x, bins=50):
    data = x.detach().cpu().numpy().ravel()
    return np.histogram(data, bins=bins)
EOF

deploy_python "histograms" "$HISTOGRAM_PY"

#############################################
# 3. PLATEAU DETECTOR
#############################################

read -r -d '' PLATEAU_PY << 'EOF'
import numpy as np

def plateau_metrics(hist):
    counts, bins = hist
    p = counts / (counts.sum() + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    width = counts[np.argmin(np.abs(bins))]
    return entropy, width

def detect_plateau(history, min_epochs=5):
    if len(history) < min_epochs:
        return None
    ent = [h["entropy"] for h in history[-min_epochs:]]
    if max(ent) - min(ent) < 1e-3:
        return len(history) - min_epochs
    return None
EOF

deploy_python "plateau" "$PLATEAU_PY"

#############################################
# 4. PLOTTING ENGINE
#############################################

read -r -d '' PLOTTING_PY << 'EOF'
import matplotlib.pyplot as plt

def plot_series(x, y, path, label):
    plt.figure()
    plt.plot(x, y)
    plt.title(label)
    plt.savefig(path)
    plt.close()
EOF

deploy_python "plotting" "$PLOTTING_PY"

#############################################
# 5. LATEX TABLE GENERATOR
#############################################

read -r -d '' LATEX_PY << 'EOF'
def write_table(rows, path):
    with open(path, "w") as f:
        f.write("\\begin{tabular}{ccc}\n")
        f.write("Epoch & Entropy & Width \\\\\n")
        for r in rows:
            f.write(f"{r[0]} & {r[1]:.4f} & {r[2]:.4f} \\\\\n")
        f.write("\\end{tabular}\n")
EOF

deploy_python "latex_tables" "$LATEX_PY"

#############################################
# 6. EXPERIMENT ENGINE
#############################################

read -r -d '' EXPERIMENT_PY << 'EOF'
import torch
from ffz_core import ffz_operator
from histograms import tensor_hist
from plateau import plateau_metrics

def run_experiment(lam, epochs=100):
    x = torch.randn(1024, requires_grad=True)
    history = []

    for e in range(epochs):
        y = ffz_operator(x, lam)
        loss = (y**2).mean()
        loss.backward()

        hist = tensor_hist(x)
        entropy, width = plateau_metrics(hist)

        history.append({
            "epoch": e,
            "entropy": entropy,
            "width": width
        })

        x.data -= 0.01 * x.grad
        x.grad.zero_()

    return history
EOF

deploy_python "experiment" "$EXPERIMENT_PY"

#############################################
# 7. SWEEP RUNNER
#############################################

read -r -d '' SWEEP_PY << 'EOF'
from experiment import run_experiment

def sweep():
    results = {}
    for lam in [0.0, 0.1, 1.0, 10.0, float("inf")]:
        results[lam] = run_experiment(lam)
    return results
EOF

deploy_python "sweep" "$SWEEP_PY"

#############################################
# 8. GODMODE ORCHESTRATOR
#############################################

read -r -d '' GODMODE_PY << 'EOF'
from sweep import sweep
from plotting import plot_series
from latex_tables import write_table
from plateau import detect_plateau
import os

def main():
    print("\n[FFZ] Starting sweep...")
    results = sweep()

    os.makedirs("plots", exist_ok=True)
    os.makedirs("tables", exist_ok=True)

    for lam, hist in results.items():
        epochs = [h["epoch"] for h in hist]
        entropy = [h["entropy"] for h in hist]
        width = [h["width"] for h in hist]

        plot_series(
            epochs,
            entropy,
            f"plots/entropy_lambda_{lam}.png",
            f"Entropy λ={lam}"
        )

        onset = detect_plateau(hist)
        print(f"[FFZ] λ={lam} plateau onset:", onset)

        rows = [(h["epoch"], h["entropy"], h["width"]) for h in hist]
        write_table(rows, f"tables/table_lambda_{lam}.tex")

    print("\n[FFZ] Sweep complete.")

if __name__ == "__main__":
    main()
EOF

deploy_python "godmode" "$GODMODE_PY"

#############################################
# PROVENANCE HASHING
#############################################

banner "GENERATING PROVENANCE HASH"

SCRIPT_HASH=$(hash_file "$0")
echo "Script Hash: $SCRIPT_HASH"
echo "$SCRIPT_HASH" > "$ROOT_DIR/runs/${RUN_ID}_provenance.txt"

#############################################
# EXECUTION
#############################################

banner "EXECUTING FFZ LAB"

cd "$ROOT_DIR/scripts"
$PYTHON_BIN godmode.py

banner "GOD MODE COMPLETE"
