"""Grid runner for the OTA-BOUN / TR-BOUN experiments.

Instead of hand-editing a config + the `tee` filename for every single run (which is
how the `bert-base-turkish-cased-nonpretrained` = mBERT and the stale `HEMS` filename
mix-ups happened), this script sweeps the whole experiment matrix:

    {model} x {train_set} x {task} x {repeat}

For each combination it calls `src/train.py` with `-m key=value` overrides (the config
override mechanism that already exists in init_config.py), writes the log to a
deterministically named file, and finally collects UAS / LAS / UPOS into a summary
table (mean +/- std over repeats) -- i.e. it reproduces the paper table automatically.

Usage:
    python src/run_sweep.py prep                 # build the UPOS vocab (needed for the upos task)
    python src/run_sweep.py run --dry-run        # print the commands without running
    python src/run_sweep.py run                  # run the whole grid
    python src/run_sweep.py run --tasks parse --models berturk --train-sets ota   # a subset
    python src/run_sweep.py run --cv 5           # k-fold CV within the training set
    python src/run_sweep.py collect              # (re)build the summary table from existing logs
    python src/run_sweep.py collect --cv 5       # collect CV results instead of repeats

Edit the PATHS block below to match your environment (e.g. Colab paths).
"""

import argparse
import itertools
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# EDIT THESE FOR YOUR ENVIRONMENT
# --------------------------------------------------------------------------------------
# IMPORTANT: the folder name must match its CONTENT. A BERTurk vocab is ~32k; mBERT is
# ~119k. Do not save mBERT into a folder called "...turkish..." (that was the old bug).
MODELS = {
    "berturk":      "data/pretrained_embeddings/bert-base-turkish-cased",
    "multilingual": "data/pretrained_embeddings/bert-base-multilingual-cased",
}

# Training corpora for each "train set" axis.
TRAIN_SETS = {
    "tr":     "data/corpora/ota_boun/tr_boun-ud-train.conllu",     # modern Turkish only
    "ota":    "data/corpora/ota_boun/ota_boun-train-2026.conllu",  # historical Turkish (2026, 1600 sents)
    "tr_ota": "data/corpora/ota_boun/best_together.conllu",        # TR + OTA combined
}

# We always evaluate on the OTA-BOUN (historical) test set.
TEST_FILE = "data/corpora/ota_boun/ota_boun-test-2026.conllu"

# One base config per task. Architecture differs (dep parsing vs sequence tagging),
# so each task needs its own base config; everything else is overridden per run.
TASKS = {
    "parse": {"config": "configs/ota_boun.json", "eval": "basic", "with_test": True},
    "upos":  {"config": "configs/ota_upos.json", "eval": "basic", "with_test": True},
}

REPEATS = [1, 2, 3]            # the paper reports mean +/- std over 3 runs
# Where logs/summary/fold files go. Point STEPS_OUTPUT_DIR at a Drive path on Colab so
# results survive the session (e.g. /content/drive/MyDrive/ota_results).
OUTPUT_DIR = Path(os.environ.get("STEPS_OUTPUT_DIR", "experiment_outputs"))
UPOS_VOCAB = "data/corpora/ota_boun/vocab/upos.vocab"

# macOS uses 'spawn' for multiprocessing, which can't pickle the data loader's lambda
# -> keep this 0 on a Mac. On Colab/Linux you can bump it to 2 for speed.
NUM_WORKERS = 0
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def out_path(task, model, train_set, repeat):
    return OUTPUT_DIR / f"output_{task}_{model}_{train_set}_{repeat}.txt"


def cv_out_path(task, model, train_set, k, fold):
    return OUTPUT_DIR / f"output_{task}_{model}_{train_set}_cv{k}_fold{fold}.txt"


def build_command(task, model, name, train_file, dev_file):
    """Build the train.py command for one run, given explicit train/dev files."""
    spec = TASKS[task]

    mods = [
        f"experiment={task}",
        f"name={name}",
        f"model.args.embeddings_processor.args.model_path={MODELS[model]}",
        f"data_loaders.paths.train={train_file}",
        f"data_loaders.paths.dev={dev_file}",
        f"data_loaders.args.num_workers={NUM_WORKERS}",
    ]
    if spec["with_test"]:
        mods.append(f"data_loaders.paths.test={TEST_FILE}")

    # `-e` before `-m` because `-m` is nargs='+' and would otherwise swallow it.
    return [
        sys.executable, "src/train.py", spec["config"],
        "-e", spec["eval"],
        "-m", *mods,
    ]


def _run_to_log(cmd, log_file, dry_run, prefix=""):
    """Run a command, streaming output to both console and log file.

    `prefix` is prepended to every console line (so you can tell which fold/run a line
    belongs to during a sweep). The log file itself stays clean (no prefix).
    """
    print(">>", " ".join(cmd))
    print("   log ->", log_file)
    if dry_run:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    tag = f"[{prefix}] " if prefix else ""
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                cwd=REPO_ROOT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(tag + line)
            sys.stdout.flush()
            f.write(line)
        proc.wait()


# ---- k-fold cross-validation helpers -------------------------------------------------
def _read_sentences(conllu_path):
    """Read a CoNLL-U file into a list of sentence blocks (lists of lines incl. newline)."""
    blocks, cur = [], []
    with open(conllu_path) as f:
        for line in f:
            if line.strip() == "":
                if cur:
                    blocks.append(cur)
                    cur = []
            else:
                cur.append(line if line.endswith("\n") else line + "\n")
    if cur:
        blocks.append(cur)
    return blocks


def _write_sentences(blocks, path):
    with open(path, "w") as f:
        for block in blocks:
            f.writelines(block)
            f.write("\n")  # blank line separates sentences


def make_folds(train_file, k, out_dir):
    """Split a training corpus into k round-robin folds; write train/dev file per fold.

    Returns a list of (train_path, dev_path) tuples, one per fold.
    """
    sents = _read_sentences(train_file)
    if len(sents) < k:
        sys.exit(f"Cannot make {k} folds from {len(sents)} sentences in {train_file}")
    buckets = [[] for _ in range(k)]
    for i, s in enumerate(sents):
        buckets[i % k].append(s)  # round-robin keeps folds balanced

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_files = []
    for i in range(k):
        dev = buckets[i]
        train = [s for j in range(k) if j != i for s in buckets[j]]
        tp = out_dir / f"fold{i}_train.conllu"
        dp = out_dir / f"fold{i}_dev.conllu"
        _write_sentences(train, tp)
        _write_sentences(dev, dp)
        fold_files.append((tp, dp))
    return fold_files


def cmd_prep(args):
    """Generate the UPOS tag vocabulary from the training corpora (column 3)."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from data_handling.custom_conll_dataset import CustomCoNLLDataset

    train_files = [TRAIN_SETS[ts] for ts in args.train_sets]
    train_files = [f for f in train_files if Path(f).exists()]
    if not train_files:
        sys.exit(f"None of the train files exist: {[TRAIN_SETS[ts] for ts in args.train_sets]}")

    layer = {"upos": {"type": "TagSequence", "source_column": 3}}
    datasets = [CustomCoNLLDataset.from_corpus_file(f, layer) for f in train_files]
    vocab = CustomCoNLLDataset.extract_label_vocab(*datasets, annotation_id="upos")

    Path(UPOS_VOCAB).parent.mkdir(parents=True, exist_ok=True)
    vocab.to_file(UPOS_VOCAB)
    print(f"Wrote UPOS vocab ({len(vocab)} tags) to {UPOS_VOCAB}")
    print("Tags:", ", ".join(t for t in vocab.token_to_ix) if hasattr(vocab, "token_to_ix") else "")


def cmd_run(args):
    if getattr(args, "cv", 0) and args.cv > 1:
        return run_cv(args)

    combos = list(itertools.product(args.tasks, args.models, args.train_sets, args.repeats))
    print(f"{len(combos)} run(s) planned\n")

    total = len(args.repeats)
    for task, model, train_set, repeat in combos:
        name = f"{task}_{model}_{train_set}_{repeat}"
        cmd = build_command(task, model, name, TRAIN_SETS[train_set], TEST_FILE)
        prefix = f"{task} {model} {train_set} | run {repeat}/{total}"
        _run_to_log(cmd, out_path(task, model, train_set, repeat), args.dry_run, prefix=prefix)

    if not args.dry_run:
        print()
        cmd_collect(args)


def run_cv(args):
    """k-fold cross-validation: split each training set into k folds; for each fold,
    train on k-1 folds, early-stop on the held-out fold, and (for parse) evaluate on
    the OTA test set. Results are averaged over folds in collect.
    """
    k = args.cv
    combos = list(itertools.product(args.tasks, args.models, args.train_sets))
    print(f"{len(combos)} config(s) x {k} folds = {len(combos) * k} run(s)\n")

    for task, model, train_set in combos:
        # Fold files go to LOCAL disk, never OUTPUT_DIR -- a Drive (FUSE) mount is flaky
        # for many small writes and throws FileNotFoundError mid-loop. They're temporary
        # deterministic splits, so they don't need to persist to Drive.
        fold_dir = REPO_ROOT / "cv_folds" / train_set / f"cv{k}"
        folds = make_folds(TRAIN_SETS[train_set], k, fold_dir)
        for i, (train_file, dev_file) in enumerate(folds):
            name = f"{task}_{model}_{train_set}_cv{k}_fold{i}"
            cmd = build_command(task, model, name, train_file, dev_file)
            prefix = f"{task} {model} {train_set} | fold {i + 1}/{k}"
            _run_to_log(cmd, cv_out_path(task, model, train_set, k, i), args.dry_run, prefix=prefix)

    if not args.dry_run:
        print()
        cmd_collect(args)


# Patterns for the metrics train.py prints to the log.
RE_LAS = re.compile(r"las_final_test:\s*([\d.]+)%")
RE_UAS = re.compile(r"uas_final_test:\s*([\d.]+)%")
RE_UPOS_TEST = re.compile(r"upos_final_test:\s*([\d.]+)%")
RE_UPOS_VALID = re.compile(r"upos-fscore_valid:\s*([\d.]+)%")


def parse_log(task, path):
    """Return the headline metric(s) from one log file, or None if not found."""
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    if task == "parse":
        las = RE_LAS.findall(text)
        uas = RE_UAS.findall(text)
        if not las:
            return None
        return {"UAS": float(uas[-1]) if uas else None, "LAS": float(las[-1])}
    else:  # upos -> prefer UPOS on the OTA test set; fall back to best validation f-score
        test = RE_UPOS_TEST.findall(text)
        if test:
            return {"UPOS": float(test[-1])}
        vals = [float(v) for v in RE_UPOS_VALID.findall(text)]
        if not vals:
            return None
        return {"UPOS": max(vals)}


def _agg(values):
    values = [v for v in values if v is not None]
    if not values:
        return "-"
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f}±{std:.2f}"


def cmd_collect(args):
    cv = getattr(args, "cv", 0)
    label = "fold" if (cv and cv > 1) else "run"
    csv_lines = ["task,model,train_set,metric,individual,mean_std"]
    any_found = False

    for task, model, train_set in itertools.product(args.tasks, args.models, args.train_sets):
        if cv and cv > 1:
            runs = [cv_out_path(task, model, train_set, cv, i) for i in range(cv)]
        else:
            runs = [out_path(task, model, train_set, r) for r in args.repeats]
        per_metric = {}
        for path in runs:
            res = parse_log(task, path)
            if res:
                for k, v in res.items():
                    per_metric.setdefault(k, []).append(v)
        if not per_metric:
            continue
        any_found = True

        print(f"\n=== {task} | {model} | {train_set} ===")
        n = max(len(v) for v in per_metric.values())
        # column header: one per fold/run
        print(f"{'':6} " + " ".join(f"{label+str(i):>8}" for i in range(n)))
        for metric, vals in per_metric.items():
            indiv = " ".join(f"{v:>8.2f}" for v in vals)         # individual results (top)
            print(f"{metric:6} {indiv}")
            print(f"{'':6} {'mean±std: ' + _agg(vals):>{9 * n}}")  # mean±std (below)
            csv_lines.append(f"{task},{model},{train_set},{metric}," +
                             "|".join(f"{v:.2f}" for v in vals) + f",{_agg(vals)}")

    if not any_found:
        print("No results found yet. Run the sweep first (logs go to "
              f"{OUTPUT_DIR}/).")
        return

    summary = OUTPUT_DIR / "summary.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(csv_lines) + "\n")
    print(f"\nWrote {summary}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def add_filters(p):
        p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
        p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
        p.add_argument("--train-sets", nargs="+", default=list(TRAIN_SETS), choices=list(TRAIN_SETS))
        p.add_argument("--repeats", nargs="+", type=int, default=REPEATS)

    p_prep = sub.add_parser("prep", help="build the UPOS tag vocabulary")
    add_filters(p_prep)
    p_prep.set_defaults(func=cmd_prep)

    p_run = sub.add_parser("run", help="run the experiment grid")
    add_filters(p_run)
    p_run.add_argument("--dry-run", action="store_true", help="print commands only")
    p_run.add_argument("--cv", type=int, default=0, metavar="K",
                       help="k-fold cross-validation within the training set (e.g. --cv 5); "
                            "replaces the repeat dimension")
    p_run.set_defaults(func=cmd_run)

    p_col = sub.add_parser("collect", help="collect results from existing logs")
    add_filters(p_col)
    p_col.add_argument("--cv", type=int, default=0, metavar="K",
                       help="collect k-fold CV logs instead of repeat logs")
    p_col.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
