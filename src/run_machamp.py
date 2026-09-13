"""Grid runner for the MaChAmp parser (benchmark experiments).

MaChAmp replaces STEPS here: same graph-based biaffine parser, but on a modern transformers
stack, so any HuggingFace encoder can be plugged in. One model per run is trained jointly on
UPOS + dependencies, then predicts every EVAL_TESTS file, and the predictions are scored with
conll18_ud_eval (UPOS / UAS / LAS, same scorer as the STEPS results).

CV folds, train sets and OUTPUT_DIR (STEPS_OUTPUT_DIR) are shared with run_sweep.py, so the
folds are identical to the STEPS experiments.

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

# model name -> HuggingFace id (loaded with AutoModel/AutoTokenizer, so the tokenizer settings
# come from the hub; avoids the local folder's missing tokenizer_config.json).
MODELS = {
    "berturk":      "dbmdz/bert-base-turkish-cased",
    "multilingual": "bert-base-multilingual-cased",
    "xlmr":         "xlm-roberta-base",
}

EVAL_TESTS = {"ota": STEPS_EVAL_TESTS["ota"]}
METRICS = ["UPOS", "UAS", "LAS"]
SEED = 8446  # MaChAmp's default seed, fixed for every run


def run_dir(model, train_set, k, fold):
    return OUTPUT_DIR / "machamp" / f"{model}_{train_set}_cv{k}_fold{fold}"


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


def write_configs(rdir, model, train_file, dev_file, epochs):
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

    dataset_path, params_path = rdir / "dataset.json", rdir / "params.json"
    dataset_path.write_text(json.dumps(dataset_cfg, indent=2))
    params_path.write_text(json.dumps(params, indent=2))
    return dataset_path, params_path


def score(gold_path, pred_path):
    ev = evaluate(load_conllu_file(str(gold_path)), load_conllu_file(str(pred_path)))
    return {m: round(100 * ev[m].f1, 2) for m in METRICS}


def run_one(model, train_set, k, fold, train_file, dev_file, args):
    rdir = run_dir(model, train_set, k, fold)
    results_path = rdir / "results.json"
    if results_path.exists() and not args.force:
        print(f"[skip] {rdir.name} (already done; --force to redo)")
        return
    prefix = f"machamp {model} {train_set} | fold {fold + 1}/{k}"
    if args.dry_run:
        print(f">> [{prefix}] would train in {rdir}")
        return

    # MaChAmp refuses an existing model dir, so a crashed/forced run starts from scratch.
    if rdir.exists():
        shutil.rmtree(rdir)
    data_dir = rdir / "data"
    data_dir.mkdir(parents=True)
    strip_multiwords(train_file, data_dir / "train.conllu")
    strip_multiwords(dev_file, data_dir / "dev.conllu")
    dataset_path, params_path = write_configs(rdir, model, data_dir / "train.conllu",
                                              data_dir / "dev.conllu", args.epochs)
    model_dir = rdir / "model"

    train_cmd = [sys.executable, str(MACHAMP_DIR / "train.py"),
                 "--dataset_configs", str(dataset_path), "--parameters_config", str(params_path),
                 "--model_dir", str(model_dir), "--device", str(args.device), "--seed", str(SEED)]
    _run_to_log(train_cmd, rdir / "train.log", False, prefix=prefix)
    if not (model_dir / "model.pt").exists():
        sys.exit(f"Training failed, no model.pt in {model_dir} (see {rdir / 'train.log'})")

    results = {}
    for tname, tpath in EVAL_TESTS.items():
        test_in, raw_pred = data_dir / f"test-{tname}.conllu", data_dir / f"pred-{tname}.machamp.conllu"
        strip_multiwords(tpath, test_in)
        pred_cmd = [sys.executable, str(MACHAMP_DIR / "predict.py"), str(model_dir / "model.pt"),
                    str(test_in), str(raw_pred), "--device", str(args.device)]
        _run_to_log(pred_cmd, rdir / f"predict-{tname}.log", False, prefix=prefix)
        pred = rdir / f"pred-{tname}.conllu"
        restore_multiwords(tpath, raw_pred, pred)
        results[tname] = score(tpath, pred)
        print(f"[{prefix}] test={tname}: " + "  ".join(f"{m} {v:.2f}" for m, v in results[tname].items()))
    results_path.write_text(json.dumps(results, indent=2))


def cmd_setup(args):
    if not MACHAMP_DIR.exists():
        subprocess.run(["git", "clone", "-q", MACHAMP_REPO, str(MACHAMP_DIR)], check=True)
    subprocess.run(["git", "-C", str(MACHAMP_DIR), "checkout", "-q", MACHAMP_COMMIT], check=True)
    print(f"MaChAmp ready at {MACHAMP_DIR} ({MACHAMP_COMMIT[:10]})")


def cmd_run(args):
    if not (MACHAMP_DIR / "train.py").exists():
        sys.exit("MaChAmp not found; run `python src/run_machamp.py setup` first.")
    for model, train_set in itertools.product(args.models, args.train_sets):
        if args.cv > 1:
            fold_dir = REPO_ROOT / "cv_folds" / train_set / f"cv{args.cv}"
            folds = make_folds(TRAIN_SETS[train_set], args.cv, fold_dir)
            for i, (train_file, dev_file) in enumerate(folds):
                if args.folds is None or i in args.folds:
                    run_one(model, train_set, args.cv, i, train_file, dev_file, args)
        else:
            if train_set not in DEV:
                sys.exit(f"No dev set for '{train_set}'; use --cv.")
            run_one(model, train_set, 1, 0, TRAIN_SETS[train_set], DEV[train_set], args)
    if not args.dry_run:
        cmd_collect(args)


def cmd_collect(args):
    k = args.cv if args.cv > 1 else 1
    lines = ["model,train_set,test,metric,individual,mean_std"]
    for model, train_set in itertools.product(args.models, args.train_sets):
        runs = []
        for i in range(k):
            p = run_dir(model, train_set, k, i) / "results.json"
            if p.exists():
                runs.append(json.loads(p.read_text()))
        if not runs:
            continue
        print(f"\n=== machamp | {model} | {train_set} ({len(runs)}/{k} folds) ===")
        for tname in EVAL_TESTS:
            for m in METRICS:
                vals = [r[tname][m] for r in runs if tname in r]
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

    p_run = sub.add_parser("run", help="train + predict + score")
    add_filters(p_run)
    p_run.add_argument("--folds", nargs="+", type=int, default=None, metavar="I")
    p_run.add_argument("--device", type=int, default=0, help="CUDA device; -1 for CPU")
    p_run.add_argument("--epochs", type=int, default=None, help="override num_epochs (e.g. 1 for a test)")
    p_run.add_argument("--force", action="store_true", help="redo runs that already have results")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_col = sub.add_parser("collect", help="summarize existing results")
    add_filters(p_col)
    p_col.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
