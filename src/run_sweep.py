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
    python src/run_sweep.py collect              # (re)build the summary table from existing logs

Edit the PATHS block below to match your environment (e.g. Colab paths).
"""

import argparse
import itertools
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
    "tr":     "data/corpora/ota_boun/tr_boun-ud-train.conllu",   # modern Turkish only
    "ota":    "data/corpora/ota_boun/ota_boun-ud-train.conllu",  # historical Turkish only
    "tr_ota": "data/corpora/ota_boun/best_together.conllu",      # TR + OTA combined
}

# We always evaluate on the OTA-BOUN (historical) test set.
TEST_FILE = "data/corpora/ota_boun/ota_boun-ud-test.conllu"

# One base config per task. Architecture differs (dep parsing vs sequence tagging),
# so each task needs its own base config; everything else is overridden per run.
TASKS = {
    "parse": {"config": "configs/ota_boun.json", "eval": "basic", "with_test": True},
    "upos":  {"config": "configs/ota_upos.json", "eval": "basic", "with_test": False},
}

REPEATS = [1, 2, 3]            # the paper reports mean +/- std over 3 runs
OUTPUT_DIR = Path("experiment_outputs")
UPOS_VOCAB = "data/corpora/ota_boun/vocab/upos.vocab"

# macOS uses 'spawn' for multiprocessing, which can't pickle the data loader's lambda
# -> keep this 0 on a Mac. On Colab/Linux you can bump it to 2 for speed.
NUM_WORKERS = 0
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


def out_path(task, model, train_set, repeat):
    return OUTPUT_DIR / f"output_{task}_{model}_{train_set}_{repeat}.txt"


def build_command(task, model, train_set, repeat):
    """Build the train.py command for one grid cell."""
    spec = TASKS[task]
    name = f"{task}_{model}_{train_set}_{repeat}"

    mods = [
        f"experiment={task}",
        f"name={name}",
        f"model.args.embeddings_processor.args.model_path={MODELS[model]}",
        f"data_loaders.paths.train={TRAIN_SETS[train_set]}",
        f"data_loaders.paths.dev={TEST_FILE}",
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
    combos = list(itertools.product(args.tasks, args.models, args.train_sets, args.repeats))
    print(f"{len(combos)} run(s) planned\n")

    for task, model, train_set, repeat in combos:
        cmd = build_command(task, model, train_set, repeat)
        log_file = out_path(task, model, train_set, repeat)
        print(">>", " ".join(cmd))
        print("   log ->", log_file)
        if args.dry_run:
            continue
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            # Stream output to BOTH the console (live progress) and the log file.
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    cwd=REPO_ROOT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
            proc.wait()

    if not args.dry_run:
        print()
        cmd_collect(args)


# Patterns for the metrics train.py prints to the log.
RE_LAS = re.compile(r"las_final_test:\s*([\d.]+)%")
RE_UAS = re.compile(r"uas_final_test:\s*([\d.]+)%")
RE_UPOS = re.compile(r"upos-fscore[^:]*:\s*([\d.]+)%")


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
    else:  # upos -> best validation UPOS f-score
        vals = [float(v) for v in RE_UPOS.findall(text)]
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
    rows = []
    for task, model, train_set in itertools.product(args.tasks, args.models, args.train_sets):
        per_metric = {}
        for repeat in args.repeats:
            res = parse_log(task, out_path(task, model, train_set, repeat))
            if res:
                for k, v in res.items():
                    per_metric.setdefault(k, []).append(v)
        if not per_metric:
            continue
        rows.append((task, model, train_set, {k: _agg(v) for k, v in per_metric.items()}))

    if not rows:
        print("No results found yet. Run the sweep first (logs go to "
              f"{OUTPUT_DIR}/).")
        return

    metric_cols = ["UAS", "LAS", "UPOS"]
    header = f"{'task':6} {'model':13} {'train':7} " + " ".join(f"{m:>12}" for m in metric_cols)
    print(header)
    print("-" * len(header))
    csv_lines = ["task,model,train_set," + ",".join(metric_cols)]
    for task, model, train_set, metrics in rows:
        cells = [metrics.get(m, "-") for m in metric_cols]
        print(f"{task:6} {model:13} {train_set:7} " + " ".join(f"{c:>12}" for c in cells))
        csv_lines.append(f"{task},{model},{train_set}," + ",".join(cells))

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
    p_run.set_defaults(func=cmd_run)

    p_col = sub.add_parser("collect", help="collect results from existing logs")
    add_filters(p_col)
    p_col.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
