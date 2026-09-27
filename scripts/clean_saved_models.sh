#!/usr/bin/env bash
# Deletes the trained checkpoints under $STEPS_OUTPUT_DIR, keeping everything that the
# results depend on: predictions (pred-*.conllu), scores (results.json) and logs.
#
# Usage:  bash scripts/clean_saved_models.sh [output_dir] [--dry-run]
set -euo pipefail

DIR="${1:-${STEPS_OUTPUT_DIR:-experiment_outputs}}"
DRY="${2:-}"

echo "Directory: $DIR"
echo "Before: $(du -sh "$DIR" 2>/dev/null | cut -f1)"

# What we delete: model weights of every parser (and STEPS' own save dir).
find "$DIR" \( -name "model.pt" -o -name "model_*.pt" -o -name "model.tar.gz" \
              -o -name "weights.*" -o -name "*.pth" \) -type f -print0 |
  while IFS= read -r -d '' f; do
    if [ "$DRY" = "--dry-run" ]; then
      echo "would delete $(du -h "$f" | cut -f1)  $f"
    else
      rm -f "$f"
    fi
  done

if [ "$DRY" != "--dry-run" ]; then
  # empty model dirs left behind
  find "$DIR" -type d -name model -empty -delete 2>/dev/null || true
  echo "After:  $(du -sh "$DIR" 2>/dev/null | cut -f1)"
fi
