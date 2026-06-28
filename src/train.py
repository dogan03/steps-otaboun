#  Copyright (c) 2020 Robert Bosch GmbH
#  All rights reserved.
#
#  This source code is licensed under the AGPL v3 license found in the
#  LICENSE file in the root directory of this source tree.
#
#  Author: Stefan Grünewald

"""Main script for training a parser based on a configuration file."""

import argparse
import os

# Newer MLflow refuses the filesystem (./mlruns) tracking backend unless this is set.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
# Disable MLflow's phone-home telemetry (network calls every epoch, just clutter).
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("DO_NOT_TRACK", "true")

from pathlib import Path

from init_config import ConfigParser
from parse_corpus import reset_file, parse_corpus, run_evaluation


def main(config, eval_mode="basic"):
    """Main function to initialize model, load data, and run training.

    Args:
        config: Experimental configuration.
        eval_mode: Method to use in post-training evaluation: "basic" for basic UD, "enhanced" for enhanced UD.
          Default: "basic".
    """
    model = config.init_model()

    data_loaders = config.init_data_loaders(model)

    trainer = config.init_trainer(model, data_loaders["train"], data_loaders["dev"])

    trainer.train()

    # Evaluate the best model on EVERY configured test set in one go (keys "test" or
    # "test_<name>", e.g. test_ota, test_tr) -- train once, evaluate on both, no retraining.
    test_keys = [k for k in config["data_loaders"]["paths"] if k == "test" or k.startswith("test_")]
    if not test_keys:
        return

    checkpoint_path = Path(trainer.checkpoint_dir) / "model_best.pth"
    trainer._resume_checkpoint(checkpoint_path)

    # conll18/iwpt eval needs a dependency tree (HEAD column). Tag-only models (e.g. UPOS)
    # have no heads, so they are evaluated with the model's internal tagging metrics instead.
    is_dependency = bool({"heads", "labels"} & set(model.outputs.keys()))

    for test_key in test_keys:
        name = "test" if test_key == "test" else test_key[len("test_"):]
        if is_dependency:
            evaluate_dependency_on_test(trainer, config, test_key, name, eval_mode=eval_mode)
        else:
            evaluate_tags_on_test(trainer, config, data_loaders[test_key], name)


def evaluate_tags_on_test(trainer, config, test_data_loader, name):
    """Evaluate a tag-only model (e.g. UPOS) on one test set via the model's internal
    metrics. Logs `<output>_final_<name>` (e.g. upos_final_ota)."""
    logger = config.logger
    logger.info(f"Evaluation on test set '{name}' (internal tagging metrics):")

    metrics = trainer.run_epoch(trainer.start_epoch, test_data_loader, training=False)
    for outp_id in sorted(metrics.keys() - {"_AGGREGATE_", "_loss"}):
        logger.log_metric(f"{outp_id}_final_{name}", metrics[outp_id]["fscore"], percent=True)


def evaluate_dependency_on_test(trainer, config, path_key, name, eval_mode="basic"):
    """Evaluate a dependency model on one test set via conll18/iwpt. Logs uas/las_final_<name>."""
    logger = config.logger
    logger.info(f"Evaluation on test set '{name}':")

    test_path = config["data_loaders"]["paths"][path_key]
    out_path = f"test-parsed-{name}.conllu"
    with open(test_path, "r") as gold_test_file, open(out_path, "w") as output_file:
        parse_corpus(config, gold_test_file, output_file, parser=trainer.parser)
        output_file = reset_file(output_file, out_path)
        gold_test_file = reset_file(gold_test_file, test_path)
        test_evaluation = run_evaluation(gold_test_file, output_file, mode=eval_mode)

    if eval_mode == "basic":
        logger.log_final_metrics_basic(test_evaluation, suffix=f"_{name}")
    elif eval_mode == "enhanced":
        logger.log_final_metrics_enhanced(test_evaluation, suffix=f"_{name}")
    else:
        raise Exception(f"Unknown evaluation mode {eval_mode}")

    try:
        logger.log_artifact(out_path)
    except Exception as e:
        logger.info(f"Skipping artifact logging (non-fatal): {e}")


def init_config_modification(raw_modifications):
    """Turn a "raw" config modification string into a dictionary of key-value pairs to replace."""
    modification = dict()
    for mod in raw_modifications:
        key, value = mod.split("=", 1)

        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value == "true":
                    value = True
                elif value == "false":
                    value = False

        modification[key] = value

    return modification


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Graph-based enhanced UD parser (training mode)')
    argparser.add_argument('config', type=str, help='config file path (required)')
    argparser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    argparser.add_argument('-s', '--save-dir', default=None, type=str, help='model save directory (config override)')
    argparser.add_argument('-m', '--modification', default=None, type=str, nargs='+', help='modifications to make to'
                                                                                           'specified configuration file'
                                                                                           '(config override)')
    argparser.add_argument('-e', '--eval', type=str, default="basic", help='Evaluation type (basic/enhanced).'
                                                                           'Default: basic')
    args = argparser.parse_args()

    modification = init_config_modification(args.modification) if args.modification is not None else dict()
    if args.save_dir is not None:
        modification["trainer.save_dir"] = args.save_dir

    config = ConfigParser.from_args(args, modification=modification)
    main(config, eval_mode=args.eval)
