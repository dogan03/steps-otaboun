"""Grid runner for UDPipe 2 (benchmark experiments).

UDPipe 2 (Straka et al.) is run with its own published recipe: its default hyperparameters
(60 epochs, 40 at lr 1e-3 + 20 at lr 1e-4, batch 32, BiLSTM tagger/parser on top of frozen
contextual embeddings). Only the encoder differs per run, exactly like their UD models, where
the encoder is a set of pre-computed embeddings (the "wembeddings" service, last 4 layers).

Pipeline per run: contextual embeddings (cached per encoder+corpus) -> train -> predict on
every EVAL_TESTS file -> score with conll18_ud_eval (same scorer as the other parsers).
CV folds, train sets and OUTPUT_DIR are shared with run_sweep.py / run_machamp.py.

Needs the TensorFlow environment (TF2 + torch + transformers), not the MaChAmp one:
    pip install tensorflow tf-keras torch "transformers>=4,<5" ufal.chu_liu_edmonds

Usage:
    python src/run_udpipe2.py setup                                    # clone + patch UDPipe 2
    python src/run_udpipe2.py run --models berturk --train-sets ota --cv 5
    python src/run_udpipe2.py run ... --folds 0 --epochs 1:1e-3        # quick test
    python src/run_udpipe2.py collect --models berturk --train-sets ota --cv 5
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from run_machamp import EVAL_TESTS, METRICS, SEED, _write_json_atomic, score
from run_sweep import DEV, OUTPUT_DIR, REPO_ROOT, TRAIN_SETS, _agg, _run_to_log, make_folds

UDPIPE_REPO = "https://github.com/ufal/udpipe.git"
UDPIPE_BRANCH = "udpipe-2"
UDPIPE_COMMIT = "223da86f482d2242a89998599cfa614e26fb483d"
WEMB_COMMIT = "364a28979591333733b561b62bc5b4c2df3e70c0"
UDPIPE_DIR = REPO_ROOT / "third_party" / "udpipe2"
WEMB_DIR = UDPIPE_DIR / "wembedding_service"
# TF2 support for training + the encoders of this benchmark in the embeddings service.
PATCHES = [(UDPIPE_DIR, REPO_ROOT / "patches" / "udpipe2-tf2.patch"),
           (WEMB_DIR, REPO_ROOT / "patches" / "udpipe2-wembeddings-models.patch")]

# model name -> wembeddings model (mean of the last 4 layers, as in the published UD models)
MODELS = {
    "berturk":      "berturk-last4",
    "multilingual": "bert-base-multilingual-cased-last4",
    "xlmr":         "xlm-roberta-base-last4",
    "xlmr_large":   "xlm-roberta-large-last4",
    "tabibert":     "tabibert-last4",
    "mmbert":       "mmbert-last4",
    "tiny":         "bert-tiny-last4",   # only for quick pipeline tests
}

WORK_ROOT = REPO_ROOT / "udpipe2_work"       # local scratch (checkpoints while training)
EMB_CACHE = REPO_ROOT / "udpipe2_embeddings"  # cached .npz per encoder + corpus file


def run_dir(model, train_set, k, fold, seed, epochs=None):
    split = f"cv{k}_fold{fold}" if k > 1 else "dev"
    name = f"{model}_{train_set}_{split}_seed{seed}"
    if epochs:
        name += "_" + epochs.replace(":", "").replace(",", "_")
    return OUTPUT_DIR / "udpipe2" / name


def ensure_udpipe2():
    """Clone UDPipe 2 (+ the wembeddings submodule) at the pinned commits and patch them."""
    if not UDPIPE_DIR.exists():
        subprocess.run(["git", "clone", "-q", "--branch", UDPIPE_BRANCH, UDPIPE_REPO, str(UDPIPE_DIR)], check=True)
    if not (WEMB_DIR / "compute_wembeddings.py").exists():
        subprocess.run(["git", "-C", str(UDPIPE_DIR), "submodule", "update", "--init", "-q",
                        "wembedding_service"], check=True)
    for repo, patch in PATCHES:
        git = ["git", "-C", str(repo)]
        if subprocess.run(git + ["apply", "--reverse", "--check", str(patch)], capture_output=True).returncode != 0:
            commit = UDPIPE_COMMIT if repo == UDPIPE_DIR else WEMB_COMMIT
            subprocess.run(git + ["checkout", "-q", "-f", commit], check=True)
            subprocess.run(git + ["apply", str(patch)], check=True)


def embeddings_for(corpus_path, model, args, prefix):
    """Contextual embeddings for one corpus file; cached per encoder + file."""
    corpus_path = Path(corpus_path)
    cache = EMB_CACHE / model / f"{corpus_path.stem}.npz"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.npz")
        cmd = [sys.executable, str(WEMB_DIR / "compute_wembeddings.py"), str(corpus_path.resolve()), str(tmp),
               "--model", MODELS[model], "--batch_size", str(args.emb_batch_size)]
        rc = _run_to_log(cmd, cache.with_suffix(".log"), False, prefix=f"{prefix} embeddings")
        if rc != 0 or not tmp.exists():
            sys.exit(f"[{prefix}] computing embeddings failed for {corpus_path}")
        os.replace(tmp, cache)
    return cache


def stage(corpus_path, model, work, name, args, prefix):
    """Put a corpus file and its embeddings side by side in the work dir (UDPipe 2 looks for
    `<corpus>*.npz`) and return the staged corpus path."""
    staged = work / f"{name}.conllu"
    shutil.copy2(corpus_path, staged)
    shutil.copy2(embeddings_for(corpus_path, model, args, prefix), work / f"{name}.conllu.npz")
    return staged


def train(model, train_file, dev_file, seed, rdir, work, args, prefix):
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    staged_train = stage(train_file, model, work, "train", args, prefix)
    staged_dev = stage(dev_file, model, work, "dev", args, prefix)

    cmd = [sys.executable, str(UDPIPE_DIR / "udpipe2.py"), str(work / "model"),
           "--train", str(staged_train), "--dev", str(staged_dev),
           "--seed", str(seed), "--threads", str(args.threads),
           "--wembedding_model", MODELS[model]]
    if args.epochs:
        cmd += ["--epochs", args.epochs]
    rc = _run_to_log(cmd, rdir / "train.log", False, prefix=prefix)
    if rc != 0 or not (work / "model" / "options.json").exists():
        sys.exit(f"[{prefix}] training failed (exit code {rc}); see {rdir / 'train.log'}")

    saved = rdir / "model"
    if saved.exists():
        shutil.rmtree(saved)
    shutil.copytree(work / "model", saved)
    (saved / "COMPLETE").touch()  # written last: the copy above is complete


def predict_and_score(model, rdir, work, results, args, prefix):
    work.mkdir(parents=True, exist_ok=True)
    for tname, tpath in EVAL_TESTS.items():
        if tname in results:
            continue
        staged = stage(tpath, model, work, f"test-{tname}", args, prefix)
        pred = rdir / f"pred-{tname}.conllu"
        cmd = [sys.executable, str(UDPIPE_DIR / "udpipe2.py"), str(rdir / "model"), "--predict",
               "--predict_input", str(staged), "--predict_output", str(pred),
               "--threads", str(args.threads)]
        rc = _run_to_log(cmd, rdir / f"predict-{tname}.log", False, prefix=prefix)
        if rc != 0:
            sys.exit(f"[{prefix}] prediction failed on {tname}; see {rdir / f'predict-{tname}.log'}")
        results[tname] = score(tpath, pred, ["upos", "dependency"])
        print(f"[{prefix}] test={tname}: " + "  ".join(f"{m} {v:.2f}" for m, v in results[tname].items()))
        _write_json_atomic(rdir / "results.json", results)


def run_one(model, train_set, k, fold, seed, train_file, dev_file, args):
    rdir = run_dir(model, train_set, k, fold, seed, args.epochs)
    prefix = f"udpipe2 {model} {train_set} | fold {fold + 1}/{k} seed {seed}"
    results_path = rdir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() and not args.force else {}
    missing = [t for t in EVAL_TESTS if t not in results]
    if not missing:
        print(f"[skip] {rdir.name} (done; --force to redo)")
        return
    trained = (rdir / "model" / "COMPLETE").exists() and not args.force
    if args.dry_run:
        what = f"predict only ({', '.join(missing)})" if trained else "train + predict"
        print(f">> [{prefix}] {what} -> {rdir}")
        return

    if args.force and rdir.exists():
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True, exist_ok=True)
    work = WORK_ROOT / rdir.name
    if trained:
        print(f"[{prefix}] model already trained, predicting only: {', '.join(missing)}")
    else:
        train(model, train_file, dev_file, seed, rdir, work, args, prefix)
    predict_and_score(model, rdir, work, results, args, prefix)
    shutil.rmtree(work, ignore_errors=True)


def cmd_setup(args):
    ensure_udpipe2()
    print(f"UDPipe 2 ready at {UDPIPE_DIR} ({UDPIPE_COMMIT[:10]} + {len(PATCHES)} patch)")


def _splits(train_set, k):
    if k > 1:
        fold_dir = REPO_ROOT / "cv_folds" / train_set / f"cv{k}"
        return [(i, tr, dv) for i, (tr, dv) in enumerate(make_folds(TRAIN_SETS[train_set], k, fold_dir))]
    if train_set not in DEV:
        sys.exit(f"No dev set for '{train_set}'; use --cv.")
    return [(0, TRAIN_SETS[train_set], DEV[train_set])]


def cmd_run(args):
    ensure_udpipe2()
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
    epochs = getattr(args, "epochs", None)
    lines = ["model,train_set,test,metric,individual,mean_std"]
    for model, train_set in itertools.product(args.models, args.train_sets):
        runs = [run_dir(model, train_set, k, fold, seed, epochs) / "results.json"
                for fold in range(k) for seed in args.seeds]
        done = [json.loads(p.read_text()) for p in runs if p.exists()]
        if not done:
            continue
        print(f"\n=== udpipe2 | {model} | {train_set} ({len(done)}/{len(runs)} runs done) ===")
        for tname in EVAL_TESTS:
            for m in METRICS:
                vals = [r[tname][m] for r in done if m in r.get(tname, {})]
                if not vals:
                    continue
                print(f"  {tname:4} {m:5} " + " ".join(f"{v:6.2f}" for v in vals) + f"   {_agg(vals)}")
                lines.append(f"{model},{train_set},{tname},{m}," + "|".join(f"{v:.2f}" for v in vals)
                             + f",{_agg(vals)}")
    summary = OUTPUT_DIR / "udpipe2" / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {summary}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="clone + patch UDPipe 2").set_defaults(func=cmd_setup)

    def add_filters(p):
        p.add_argument("--models", nargs="+", default=["berturk"], choices=list(MODELS))
        p.add_argument("--train-sets", nargs="+", default=["ota"], choices=list(TRAIN_SETS))
        p.add_argument("--cv", type=int, default=5, metavar="K", help="k-fold CV (default 5; 0/1 = dev set)")
        p.add_argument("--seeds", nargs="+", type=int, default=[SEED])
        p.add_argument("--epochs", default=None,
                       help="override UDPipe 2's epoch schedule, e.g. '1:1e-3' for a quick test")

    p_run = sub.add_parser("run", help="train + predict + score (resumes by default)")
    add_filters(p_run)
    p_run.add_argument("--folds", nargs="+", type=int, default=None, metavar="I")
    p_run.add_argument("--threads", type=int, default=4)
    p_run.add_argument("--emb-batch-size", type=int, default=64)
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_col = sub.add_parser("collect", help="summarize existing results")
    add_filters(p_col)
    p_col.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
