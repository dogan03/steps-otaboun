"""Grid runner for the MaChAmp parser (benchmark experiments).

MaChAmp replaces STEPS here: same graph-based biaffine parser, but on a modern transformers
stack, so any HuggingFace encoder can be plugged in. One model per run is trained jointly on
UPOS + dependencies, then predicts every EVAL_TESTS file, and the predictions are scored with
conll18_ud_eval (UPOS / UAS / LAS, same scorer as the STEPS results).

CV folds, train sets and OUTPUT_DIR (STEPS_OUTPUT_DIR) are shared with run_sweep.py, so the
folds are identical to the STEPS experiments.

Resuming: every run (model x train set x fold x seed) lives in its own folder under
OUTPUT_DIR/machamp/ (point STEPS_OUTPUT_DIR at Drive on Colab). Re-running the same command
skips finished runs (results.json), only re-predicts runs whose model is already saved, and
retrains a run that was interrupted mid-training from its first epoch. Checkpoints are written
to local disk while training and copied to OUTPUT_DIR once the run's training finishes.

MaChAmp needs its own environment (transformers>=4,<5), separate from STEPS (transformers 3.1.0).

Usage:
    python src/run_machamp.py setup                                  # clone MaChAmp (pinned commit)
    python src/run_machamp.py run --models berturk --train-sets ota --cv 5
    python src/run_machamp.py run ... --folds 0 --epochs 1 --device -1   # quick CPU test
    python src/run_machamp.py collect --models berturk --train-sets ota --cv 5
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from run_sweep import (DEV, EVAL_TESTS as STEPS_EVAL_TESTS, OUTPUT_DIR, REPO_ROOT, TRAIN_SETS,
                       _agg, _run_to_log, make_folds)
from util.conll18_ud_eval import evaluate, load_conllu_file

MACHAMP_REPO = "https://github.com/machamp-nlp/machamp.git"
MACHAMP_COMMIT = "4048c34b37796aa496624b68b3530183dc61a690"  # master, 2026-06-03
MACHAMP_DIR = REPO_ROOT / "third_party" / "machamp"
PARAMS = REPO_ROOT / "configs" / "machamp" / "params.json"
# Local scratch space for training (per-epoch checkpoints stay off Drive).
WORK_ROOT = REPO_ROOT / "machamp_work"

# model name -> HuggingFace id (loaded with AutoModel/AutoTokenizer, so the tokenizer settings
# come from the hub; avoids the local folder's missing tokenizer_config.json).
MODELS = {
    "berturk":      "dbmdz/bert-base-turkish-cased",
    "multilingual": "bert-base-multilingual-cased",
    "xlmr":         "xlm-roberta-base",
}

EVAL_TESTS = {"ota": STEPS_EVAL_TESTS["ota"]}
METRICS = ["UPOS", "UAS", "LAS"]
SEED = 8446  # MaChAmp's default seed
# Files kept from MaChAmp's model dir (model.pt is the best epoch on dev).
MODEL_FILES = ["model.pt", "log.txt", "metrics.json", "params-config.json", "dataset-configs.json",
               "scalars.json"]


def run_dir(model, train_set, k, fold, seed):
    split = f"cv{k}_fold{fold}" if k > 1 else "dev"
    return OUTPUT_DIR / "machamp" / f"{model}_{train_set}_{split}_seed{seed}"


def _is_word(line):
    """A CoNLL-U syntactic word line (not a comment, multiword range `1-2` or empty node `1.1`)."""
    return line[:1].isdigit() and line.split("\t", 1)[0].isdigit()


def strip_multiwords(src, dst):
    """Copy a CoNLL-U file without multiword-token range lines (and empty nodes).

    MaChAmp reads every non-comment line as a word, so a `12-13 kalbimdeki` line would shift
    all later words/heads. The syntactic words themselves stay, i.e. the gold word
    segmentation is unchanged (same setup as STEPS, which also skips these lines).
    """
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip() or line.startswith("#") or _is_word(line):
                out.write(line)


def restore_multiwords(gold_path, machamp_pred, out_path):
    """Write the gold file with MaChAmp's UPOS/HEAD/DEPREL filled in (other annotation columns
    blanked), so the prediction keeps the gold multiword lines and scores with conll18_ud_eval."""
    with open(machamp_pred, encoding="utf-8") as f:
        pred = iter([l.rstrip("\n").split("\t") for l in f if _is_word(l)])
    with open(gold_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            if _is_word(line):
                cols, p = line.rstrip("\n").split("\t"), next(pred)
                assert cols[1] == p[1], f"word mismatch: gold {cols[1]!r} vs pred {p[1]!r}"
                cols[2] = cols[4] = cols[5] = "_"                    # LEMMA, XPOS, FEATS: not predicted
                cols[3], cols[6], cols[7] = p[3], p[6], p[7]         # UPOS, HEAD, DEPREL
                line = "\t".join(cols) + "\n"
            out.write(line)
    assert next(pred, None) is None, "prediction has more words than gold"


def write_configs(cfg_dir, model, train_file, dev_file, epochs):
    """Write the MaChAmp dataset + parameter configs for one run; return their paths."""
    dataset_cfg = {
        "UD": {
            "train_data_path": str(Path(train_file).resolve()),
            "dev_data_path": str(Path(dev_file).resolve()),
            "word_idx": 1,
            "tasks": {
                "upos": {"task_type": "seq", "column_idx": 3},
                "dependency": {"task_type": "dependency", "column_idx": 6},
            },
        }
    }
    params = json.loads(PARAMS.read_text())
    params["transformer_model"] = MODELS[model]
    if epochs:
        params["training"]["num_epochs"] = epochs

    dataset_path, params_path = cfg_dir / "dataset.json", cfg_dir / "params.json"
    dataset_path.write_text(json.dumps(dataset_cfg, indent=2))
    params_path.write_text(json.dumps(params, indent=2))
    return dataset_path, params_path


def _write_json_atomic(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def score(gold_path, pred_path):
    ev = evaluate(load_conllu_file(str(gold_path)), load_conllu_file(str(pred_path)))
    return {m: round(100 * ev[m].f1, 2) for m in METRICS}


def train(model, train_file, dev_file, seed, rdir, work, args, prefix):
    """Train in the local work dir, then copy the best model + logs into rdir/model."""
    if work.exists():
        shutil.rmtree(work)
    (work / "data").mkdir(parents=True)
    strip_multiwords(train_file, work / "data" / "train.conllu")
    strip_multiwords(dev_file, work / "data" / "dev.conllu")
    dataset_path, params_path = write_configs(work, model, work / "data" / "train.conllu",
                                              work / "data" / "dev.conllu", args.epochs)

    cmd = [sys.executable, str(MACHAMP_DIR / "train.py"),
           "--dataset_configs", str(dataset_path), "--parameters_config", str(params_path),
           "--model_dir", str(work / "model"), "--device", str(args.device), "--seed", str(seed)]
    rc = _run_to_log(cmd, rdir / "train.log", False, prefix=prefix)
    if rc != 0 or not (work / "model" / "model.pt").exists():
        sys.exit(f"[{prefix}] training failed (exit code {rc}); see {rdir / 'train.log'}")

    saved = rdir / "model"
    saved.mkdir(exist_ok=True)
    for name in MODEL_FILES:
        if (work / "model" / name).exists():
            shutil.copy2(work / "model" / name, saved / name)
    shutil.copy2(dataset_path, rdir / "dataset.json")
    shutil.copy2(params_path, rdir / "params.json")
    (saved / "COMPLETE").touch()  # written last: the copy above is complete


def predict_and_score(rdir, work, args, prefix):
    work.mkdir(parents=True, exist_ok=True)
    results = {}
    for tname, tpath in EVAL_TESTS.items():
        test_in, raw_pred = work / f"test-{tname}.conllu", work / f"pred-{tname}.machamp.conllu"
        strip_multiwords(tpath, test_in)
        cmd = [sys.executable, str(MACHAMP_DIR / "predict.py"), str(rdir / "model" / "model.pt"),
               str(test_in), str(raw_pred), "--device", str(args.device)]
        rc = _run_to_log(cmd, rdir / f"predict-{tname}.log", False, prefix=prefix)
        if rc != 0:
            sys.exit(f"[{prefix}] prediction failed on {tname}; see {rdir / f'predict-{tname}.log'}")
        pred = rdir / f"pred-{tname}.conllu"
        restore_multiwords(tpath, raw_pred, pred)
        results[tname] = score(tpath, pred)
        print(f"[{prefix}] test={tname}: " + "  ".join(f"{m} {v:.2f}" for m, v in results[tname].items()))
    return results


def run_one(model, train_set, k, fold, seed, train_file, dev_file, args):
    rdir = run_dir(model, train_set, k, fold, seed)
    prefix = f"machamp {model} {train_set} | fold {fold + 1}/{k} seed {seed}"
    if (rdir / "results.json").exists() and not args.force:
        print(f"[skip] {rdir.name} (done; --force to redo)")
        return
    trained = (rdir / "model" / "COMPLETE").exists() and not args.force
    if args.dry_run:
        print(f">> [{prefix}] {'predict only' if trained else 'train + predict'} -> {rdir}")
        return

    if args.force and rdir.exists():
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True, exist_ok=True)
    work = WORK_ROOT / rdir.name
    if trained:
        print(f"[{prefix}] model already trained, predicting only")
    else:
        train(model, train_file, dev_file, seed, rdir, work, args, prefix)
    _write_json_atomic(rdir / "results.json", predict_and_score(rdir, work, args, prefix))
    shutil.rmtree(work, ignore_errors=True)


def cmd_setup(args):
    if not MACHAMP_DIR.exists():
        subprocess.run(["git", "clone", "-q", MACHAMP_REPO, str(MACHAMP_DIR)], check=True)
    subprocess.run(["git", "-C", str(MACHAMP_DIR), "checkout", "-q", MACHAMP_COMMIT], check=True)
    print(f"MaChAmp ready at {MACHAMP_DIR} ({MACHAMP_COMMIT[:10]})")


def _splits(train_set, k):
    """[(fold, train_file, dev_file)] for k-fold CV, or the official dev split when k <= 1."""
    if k > 1:
        fold_dir = REPO_ROOT / "cv_folds" / train_set / f"cv{k}"
        return [(i, tr, dv) for i, (tr, dv) in enumerate(make_folds(TRAIN_SETS[train_set], k, fold_dir))]
    if train_set not in DEV:
        sys.exit(f"No dev set for '{train_set}'; use --cv.")
    return [(0, TRAIN_SETS[train_set], DEV[train_set])]


def cmd_run(args):
    if not (MACHAMP_DIR / "train.py").exists():
        sys.exit("MaChAmp not found; run `python src/run_machamp.py setup` first.")
    k = max(args.cv, 1)
    for model, train_set in itertools.product(args.models, args.train_sets):
        for fold, train_file, dev_file in _splits(train_set, k):
            if args.folds is not None and fold not in args.folds:
                continue
            for seed in args.seeds:
                run_one(model, train_set, k, fold, seed, train_file, dev_file, args)
    if not args.dry_run:
        cmd_collect(args)


def cmd_collect(args):
    k = max(args.cv, 1)
    lines = ["model,train_set,test,metric,individual,mean_std"]
    for model, train_set in itertools.product(args.models, args.train_sets):
        runs = [run_dir(model, train_set, k, fold, seed) / "results.json"
                for fold in range(k) for seed in args.seeds]
        done = [json.loads(p.read_text()) for p in runs if p.exists()]
        if not done:
            continue
        print(f"\n=== machamp | {model} | {train_set} ({len(done)}/{len(runs)} runs done) ===")
        for tname in EVAL_TESTS:
            for m in METRICS:
                vals = [r[tname][m] for r in done if tname in r]
                print(f"  test={tname:4} {m:5} " + " ".join(f"{v:6.2f}" for v in vals) + f"   {_agg(vals)}")
                lines.append(f"{model},{train_set},{tname},{m}," + "|".join(f"{v:.2f}" for v in vals)
                             + f",{_agg(vals)}")
    summary = OUTPUT_DIR / "machamp" / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {summary}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="clone MaChAmp at the pinned commit").set_defaults(func=cmd_setup)

    def add_filters(p):
        p.add_argument("--models", nargs="+", default=["berturk"], choices=list(MODELS))
        p.add_argument("--train-sets", nargs="+", default=["ota"], choices=list(TRAIN_SETS))
        p.add_argument("--cv", type=int, default=5, metavar="K", help="k-fold CV (default 5; 0/1 = dev set)")
        p.add_argument("--seeds", nargs="+", type=int, default=[SEED], help=f"one run per seed (default {SEED})")

    p_run = sub.add_parser("run", help="train + predict + score (resumes by default)")
    add_filters(p_run)
    p_run.add_argument("--folds", nargs="+", type=int, default=None, metavar="I")
    p_run.add_argument("--device", type=int, default=0, help="CUDA device; -1 for CPU")
    p_run.add_argument("--epochs", type=int, default=None, help="override num_epochs (e.g. 1 for a test)")
    p_run.add_argument("--force", action="store_true", help="redo runs from scratch")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_col = sub.add_parser("collect", help="summarize existing results")
    add_filters(p_col)
    p_col.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
