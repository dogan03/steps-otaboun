"""Grid runner for UDify (Kondratyuk & Straka 2019) — benchmark experiments.

UDify is run with its own published recipe: the `udify_bert_finetune` configuration
(multi-task UPOS/FEATS/LEMMAS/DEPS on top of a fine-tuned BERT with layer attention,
80 epochs, BertAdam + ULMFiT-sqrt schedule with gradual unfreezing). Only the BERT model
differs per run. UDify is BERT-only: it uses the old `pytorch_pretrained_bert` library, so
SentencePiece encoders (XLM-R) and ModernBERT encoders cannot be plugged in.

Pipeline per run: build a UD-style treebank folder + vocabulary -> train -> predict every
EVAL_TESTS file -> score with conll18_ud_eval (same scorer as the other parsers).
CV folds, train sets and OUTPUT_DIR are shared with run_sweep.py / run_machamp.py.

UDify needs its own old environment (Python 3.8, torch 1.4, allennlp 0.9); see
scripts/setup_udify_env.sh. It only runs on GPUs supported by CUDA 10 (e.g. Colab's T4).

Usage:
    python src/run_udify.py setup                                   # clone UDify, fetch BERT
    python src/run_udify.py run --models berturk --train-sets ota --cv 5
    python src/run_udify.py run ... --folds 0 --epochs 1 --device -1   # quick test
    python src/run_udify.py collect --models berturk --train-sets ota --cv 5
"""

import argparse
import glob
import itertools
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from run_machamp import EVAL_TESTS, METRICS, SEED, _write_json_atomic, score
from run_sweep import DEV, OUTPUT_DIR, REPO_ROOT, TRAIN_SETS, _agg, _run_to_log, make_folds

UDIFY_REPO = "https://github.com/Hyperparticle/udify.git"
UDIFY_COMMIT = "master"
UDIFY_DIR = REPO_ROOT / "third_party" / "udify"
WORK_ROOT = REPO_ROOT / "udify_work"
BERT_DIR = REPO_ROOT / "udify_bert"       # BERT models in pytorch_pretrained_bert layout

# model name -> HuggingFace id. BERT architectures only (pytorch_pretrained_bert).
MODELS = {
    "berturk":      "dbmdz/bert-base-turkish-cased",
    "multilingual": "bert-base-multilingual-cased",
}


def run_dir(model, train_set, k, fold, seed, epochs=None):
    split = f"cv{k}_fold{fold}" if k > 1 else "dev"
    name = f"{model}_{train_set}_{split}_seed{seed}"
    if epochs:
        name += f"_ep{epochs}"
    return OUTPUT_DIR / "udify" / name


def treebank_name(train_set, k, fold):
    return f"{train_set}_cv{k}_fold{fold}" if k > 1 else f"{train_set}_dev"


def ensure_udify():
    if not UDIFY_DIR.exists():
        subprocess.run(["git", "clone", "-q", UDIFY_REPO, str(UDIFY_DIR)], check=True)


def prepare_bert(model):
    """Download the model in the layout pytorch_pretrained_bert expects (bert_config.json,
    pytorch_model.bin, vocab.txt)."""
    target = BERT_DIR / model
    files = {"bert_config.json": "config.json", "pytorch_model.bin": "pytorch_model.bin",
             "vocab.txt": "vocab.txt"}
    if all((target / local).exists() for local in files):
        return target
    target.mkdir(parents=True, exist_ok=True)
    for local, remote in files.items():
        if (target / local).exists():
            continue
        url = f"https://huggingface.co/{MODELS[model]}/resolve/main/{remote}"
        print(f"  downloading {url}")
        tmp = target / (local + ".tmp")
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, target / local)
    return target


def write_config(cfg_path, bert_path, epochs, batch_size):
    """UDify's published fine-tuning configuration, with our BERT model plugged in."""
    bert_dir = str(Path(bert_path).resolve())
    cfg = {
        "dataset_reader": {
            "lazy": False,
            "token_indexers": {
                "tokens": {"type": "single_id", "lowercase_tokens": True},
                "bert": {"type": "udify-bert-pretrained",
                         "pretrained_model": f"{bert_dir}/vocab.txt",
                         "do_lowercase": False,
                         "use_starting_offsets": True},
            },
        },
        "model": {
            "word_dropout": 0.2,
            "mix_embedding": 12,
            "layer_dropout": 0.1,
            "tasks": ["upos", "feats", "lemmas", "deps"],
            "text_field_embedder": {
                "type": "udify_embedder",
                "dropout": 0.5,
                "allow_unmatched_keys": True,
                "embedder_to_indexer_map": {"bert": ["bert", "bert-offsets"]},
                "token_embedders": {
                    "bert": {"type": "udify-bert-pretrained", "pretrained_model": bert_dir,
                             "requires_grad": True, "dropout": 0.15, "layer_dropout": 0.1,
                             "combine_layers": "all"},
                },
            },
            "encoder": {"type": "pass_through", "input_dim": 768},
            "decoders": {
                "upos": {"encoder": {"type": "pass_through", "input_dim": 768}},
                "feats": {"encoder": {"type": "pass_through", "input_dim": 768}, "adaptive": True},
                "lemmas": {"encoder": {"type": "pass_through", "input_dim": 768}, "adaptive": True},
                "deps": {"tag_representation_dim": 256, "arc_representation_dim": 768,
                         "encoder": {"type": "pass_through", "input_dim": 768}},
            },
        },
        "iterator": {"batch_size": batch_size, "maximum_samples_per_batch": ["num_tokens", 32 * 100]},
        "trainer": {
            "num_epochs": epochs,
            "patience": epochs,
            "num_serialized_models_to_keep": 1,
            "should_log_learning_rate": True,
            "summary_interval": 100,
            "optimizer": {
                "type": "bert_adam", "b1": 0.9, "b2": 0.99, "weight_decay": 0.01, "lr": 1e-3,
                "parameter_groups": [
                    [["^text_field_embedder.*.bert_model.embeddings",
                      "^text_field_embedder.*.bert_model.encoder"], {}],
                    [["^text_field_embedder.*._scalar_mix", "^text_field_embedder.*.pooler",
                      "^scalar_mix", "^decoders", "^shared_encoder"], {}],
                ],
            },
            "learning_rate_scheduler": {
                "type": "ulmfit_sqrt", "model_size": 1, "warmup_steps": 392, "start_step": 392,
                "factor": 5.0, "gradual_unfreezing": True, "discriminative_fine_tuning": True,
                "decay_factor": 0.04,
            },
        },
        "udify_replace": [
            "dataset_reader.token_indexers", "model.text_field_embedder", "model.encoder",
            "model.decoders.xpos", "model.decoders.deps.encoder", "model.decoders.upos.encoder",
            "model.decoders.feats.encoder", "model.decoders.lemmas.encoder",
            "trainer.learning_rate_scheduler", "trainer.optimizer",
        ],
    }
    Path(cfg_path).write_text(json.dumps(cfg, indent=2))


def prepare_treebank(train_file, dev_file, work, tb):
    """UDify expects a UD-style folder: <dir>/<tb>/<tb>-ud-{train,dev,test}.conllu"""
    tb_dir = work / "ud" / tb
    tb_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(train_file, tb_dir / f"{tb}-ud-train.conllu")
    shutil.copy2(dev_file, tb_dir / f"{tb}-ud-dev.conllu")
    shutil.copy2(EVAL_TESTS["ota"], tb_dir / f"{tb}-ud-test.conllu")  # only for UDify's own report
    return work / "ud"


def train(model, train_file, dev_file, rdir, work, args, prefix):
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    tb = args.treebank
    dataset_dir = prepare_treebank(train_file, dev_file, work, tb)
    bert_path = prepare_bert(model)
    cfg_path = work / "config.json"
    write_config(cfg_path, bert_path, args.epochs or 80, args.batch_size)

    vocab_cmd = [sys.executable, "create_vocabs.py", "--dataset_dir", str(dataset_dir),
                 "--treebanks", tb, "--output_dir", "data/vocab"]
    if _run_to_log(vocab_cmd, rdir / "vocab.log", False, prefix=prefix, cwd=UDIFY_DIR) != 0:
        sys.exit(f"[{prefix}] building the vocabulary failed; see {rdir / 'vocab.log'}")

    train_cmd = [sys.executable, "train.py", "--name", tb, "--dataset_dir", str(dataset_dir),
                 "--config", str(cfg_path.resolve()), "--batch_size", str(args.batch_size),
                 "--device", str(args.device)]
    if _run_to_log(train_cmd, rdir / "train.log", False, prefix=prefix, cwd=UDIFY_DIR) != 0:
        sys.exit(f"[{prefix}] training failed; see {rdir / 'train.log'}")

    archives = sorted(glob.glob(str(UDIFY_DIR / "logs" / tb / "*" / "model.tar.gz")))
    if not archives:
        sys.exit(f"[{prefix}] no model.tar.gz under {UDIFY_DIR / 'logs' / tb}")
    (rdir / "model").mkdir(exist_ok=True)
    shutil.copy2(archives[-1], rdir / "model" / "model.tar.gz")
    shutil.copy2(cfg_path, rdir / "config.json")
    (rdir / "model" / "COMPLETE").touch()  # written last: the copy above is complete


def predict_and_score(rdir, work, results, args, prefix):
    work.mkdir(parents=True, exist_ok=True)
    for tname, tpath in EVAL_TESTS.items():
        if tname in results:
            continue
        pred = rdir / f"pred-{tname}.conllu"
        cmd = [sys.executable, "predict.py", str((rdir / "model" / "model.tar.gz").resolve()),
               str(Path(tpath).resolve()), str(pred.resolve()),
               "--device", str(args.device), "--batch_size", str(args.batch_size)]
        if _run_to_log(cmd, rdir / f"predict-{tname}.log", False, prefix=prefix, cwd=UDIFY_DIR) != 0:
            sys.exit(f"[{prefix}] prediction failed on {tname}; see {rdir / f'predict-{tname}.log'}")
        results[tname] = score(tpath, pred, ["upos", "dependency"])
        print(f"[{prefix}] test={tname}: " + "  ".join(f"{m} {v:.2f}" for m, v in results[tname].items()))
        _write_json_atomic(rdir / "results.json", results)


def run_one(model, train_set, k, fold, seed, train_file, dev_file, args):
    rdir = run_dir(model, train_set, k, fold, seed, args.epochs)
    prefix = f"udify {model} {train_set} | fold {fold + 1}/{k} seed {seed}"
    args.treebank = treebank_name(train_set, k, fold)
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
        train(model, train_file, dev_file, rdir, work, args, prefix)
    predict_and_score(rdir, work, results, args, prefix)
    shutil.rmtree(work, ignore_errors=True)


def cmd_setup(args):
    ensure_udify()
    for model in args.models:
        prepare_bert(model)
    print(f"UDify ready at {UDIFY_DIR}; BERT models in {BERT_DIR}")


def _splits(train_set, k):
    if k > 1:
        fold_dir = REPO_ROOT / "cv_folds" / train_set / f"cv{k}"
        return [(i, tr, dv) for i, (tr, dv) in enumerate(make_folds(TRAIN_SETS[train_set], k, fold_dir))]
    if train_set not in DEV:
        sys.exit(f"No dev set for '{train_set}'; use --cv.")
    return [(0, TRAIN_SETS[train_set], DEV[train_set])]


def cmd_run(args):
    ensure_udify()
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
        print(f"\n=== udify | {model} | {train_set} ({len(done)}/{len(runs)} runs done) ===")
        for tname in EVAL_TESTS:
            for m in METRICS:
                vals = [r[tname][m] for r in done if m in r.get(tname, {})]
                if not vals:
                    continue
                print(f"  {tname:4} {m:5} " + " ".join(f"{v:6.2f}" for v in vals) + f"   {_agg(vals)}")
                lines.append(f"{model},{train_set},{tname},{m}," + "|".join(f"{v:.2f}" for v in vals)
                             + f",{_agg(vals)}")
    summary = OUTPUT_DIR / "udify" / "summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {summary}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def add_filters(p):
        p.add_argument("--models", nargs="+", default=["berturk"], choices=list(MODELS))
        p.add_argument("--train-sets", nargs="+", default=["ota"], choices=list(TRAIN_SETS))
        p.add_argument("--cv", type=int, default=5, metavar="K")
        p.add_argument("--seeds", nargs="+", type=int, default=[SEED])
        p.add_argument("--epochs", type=int, default=None, help="override UDify's 80 epochs")

    p_setup = sub.add_parser("setup", help="clone UDify and download the BERT models")
    p_setup.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p_setup.set_defaults(func=cmd_setup)

    p_run = sub.add_parser("run", help="train + predict + score (resumes by default)")
    add_filters(p_run)
    p_run.add_argument("--folds", nargs="+", type=int, default=None, metavar="I")
    p_run.add_argument("--device", type=int, default=0, help="CUDA device; -1 for CPU")
    p_run.add_argument("--batch_size", type=int, default=32)
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
