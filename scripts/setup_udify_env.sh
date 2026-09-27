#!/usr/bin/env bash
# Creates the environment UDify needs (Python 3.8, torch 1.4, allennlp 0.9).
#
# Notes on the pins:
#  - allennlp 0.9 is what UDify pins. Its torch 1.4 pin is a CUDA 10.1 build that cannot run
#    on Ampere GPUs, so we install torch 1.7.1 (CUDA 11) instead: same allennlp API, and it
#    works on both a T4 and an A100.
#  - allennlp 0.9 asks for spacy <2.2, which has no Python 3.8 wheel, so we install allennlp
#    without its dependency resolution and pin spacy 2.3 (same API for what allennlp uses).
#  - `overrides` must stay <4: allennlp 0.9 uses the old decorator behaviour.
#
# Usage:  bash scripts/setup_udify_env.sh [env_dir]      (default: .venv-udify)
set -euo pipefail

ENV_DIR="${1:-.venv-udify}"

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv (used to get a Python 3.8 interpreter)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# On Apple Silicon there is no arm64 build of torch 1.4, but the x86_64 one runs under
# Rosetta, which is enough to test the pipeline locally (training itself belongs on a GPU).
PYTHON=3.8
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  uv python install cpython-3.8.20-macos-x86_64
  PYTHON=cpython-3.8.20-macos-x86_64
fi
uv venv --clear -p "$PYTHON" "$ENV_DIR"   # --clear: never prompt when the dir exists
# Always install into this environment explicitly; relying on VIRTUAL_ENV is not enough,
# uv otherwise falls back to the system interpreter (e.g. Python 3.13 on Colab).
PIP="uv pip install --python $ENV_DIR/bin/python"

# UDify pins torch 1.4, but that is a CUDA 10.1 build and does not run on Ampere GPUs.
# torch 1.7.1 keeps allennlp 0.9 working and its CUDA 11 build covers T4 and A100 alike.
if [ "$(uname -s)" = "Linux" ]; then
  $PIP "torch==1.7.1+cu110" --extra-index-url https://download.pytorch.org/whl/cu110
else
  $PIP "torch==1.7.1"
fi
$PIP "allennlp==0.9.0" --no-deps
# allennlp 0.9's runtime dependencies (hand-picked so that modern wheels can be used)
$PIP \
  "spacy==2.3.9" "numpy<1.24" "overrides<4" "jsonnet>=0.10" "nltk" "boto3" "requests" "tqdm" \
  "editdistance" "h5py" "scikit-learn" "scipy" "pytz" "unidecode" "tensorboardX" "ftfy" \
  "jsonpickle" "parsimonious" "sqlparse" "word2number" "flask" "flask-cors" "gevent" \
  "numpydoc" "pytest" "flaky" "responses" "matplotlib" \
  "pytorch-pretrained-bert>=0.6.2" "pytorch-transformers==1.1.0" "conllu<3"

# Fail loudly if anything above did not land in the environment.
"$ENV_DIR/bin/python" - <<'CHECK'
import importlib, sys
missing = [m for m in ["torch", "allennlp", "spacy", "_jsonnet", "overrides",
                       "pytorch_pretrained_bert", "conllu", "numpydoc"]
           if not importlib.util.find_spec(m)]
if missing:
    sys.exit("UDify environment is incomplete, missing: " + ", ".join(missing))
import torch
print("torch", torch.__version__, "| allennlp ok")
CHECK

echo
echo "UDify environment ready: $ENV_DIR"
echo "Next:"
echo "  $ENV_DIR/bin/python src/run_udify.py setup"
echo "  $ENV_DIR/bin/python src/run_udify.py run --models berturk --train-sets ota --cv 5 --folds 0 --epochs 1"
