#!/usr/bin/env bash
# Creates the environment UDify needs (Python 3.8, torch 1.4, allennlp 0.9).
#
# Notes on the pins:
#  - torch 1.4 / allennlp 0.9 are what UDify pins; they exist as Linux wheels only, and
#    torch 1.4 is a CUDA 10.1 build, so the GPU must be compute capability <= 7.5
#    (on Colab: a T4, not an A100).
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

uv venv -p 3.8 "$ENV_DIR"
export VIRTUAL_ENV="$ENV_DIR"

uv pip install "torch==1.4.0"
uv pip install "allennlp==0.9.0" --no-deps
# allennlp 0.9's runtime dependencies (hand-picked so that modern wheels can be used)
uv pip install \
  "spacy==2.3.9" "numpy<1.24" "overrides<4" "jsonnet>=0.10" "nltk" "boto3" "requests" "tqdm" \
  "editdistance" "h5py" "scikit-learn" "scipy" "pytz" "unidecode" "tensorboardX" "ftfy" \
  "jsonpickle" "parsimonious" "sqlparse" "word2number" "flask" "flask-cors" "gevent" \
  "pytorch-pretrained-bert>=0.6.2" "pytorch-transformers==1.1.0" "conllu<3"

echo
echo "UDify environment ready: $ENV_DIR"
echo "Next:"
echo "  $ENV_DIR/bin/python src/run_udify.py setup"
echo "  $ENV_DIR/bin/python src/run_udify.py run --models berturk --train-sets ota --cv 5 --folds 0 --epochs 1"
