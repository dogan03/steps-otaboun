"""Significance tests between two systems, using the predictions saved by the runners.

A "system" is written as `<parser>:<encoder>:<train set>`, e.g. `udpipe2:berturk:ota`.
Since every system has several runs (5 CV folds, or 3 seeds), the run whose score is closest
to that system's mean is used as its representative — the same choice as in the previous
paper. The test itself is the approximate randomization test in significance_test.py
(sentence-level swaps, 10k iterations by default).

Usage:
    python src/run_significance.py --test ota --metric las \\
        udpipe2:berturk:ota machamp:berturk:ota
    python src/run_significance.py --test dudu --metric las --all-pairs \\
        udpipe2:berturk:dudu udify:berturk:dudu machamp:berturk:dudu
"""

import argparse
import itertools
import json
import random
import statistics
import sys
from pathlib import Path

from run_machamp import EVAL_TESTS
from run_sweep import OUTPUT_DIR
from significance_test import overall, read_conllu, sentence_counts

METRIC_KEY = {"las": "LAS", "uas": "UAS", "upos": "UPOS"}


def representative(system, test, metric):
    """The run of this system whose score is closest to the system's mean, with its scores."""
    parser, model, train_set = system.split(":")
    key = METRIC_KEY[metric]
    runs = []
    for results_path in sorted((OUTPUT_DIR / parser).glob(f"{model}_{train_set}_*/results.json")):
        scores = json.loads(results_path.read_text())
        pred = results_path.parent / f"pred-{test}.conllu"
        if test in scores and key in scores[test] and pred.exists():
            runs.append((scores[test][key], pred))
    if not runs:
        sys.exit(f"No runs with predictions for {system} on test '{test}' under {OUTPUT_DIR / parser}")
    mean = statistics.mean(v for v, _ in runs)
    value, pred = min(runs, key=lambda vp: abs(vp[0] - mean))
    return {"system": system, "runs": len(runs), "mean": round(mean, 2),
            "value": value, "pred": pred}


def randomization_test(gold_path, pred_a, pred_b, metric, iterations, seed):
    gold, a, b = (read_conllu(str(p)) for p in (gold_path, pred_a, pred_b))
    if not (len(gold) == len(a) == len(b)):
        sys.exit(f"Sentence count mismatch: gold={len(gold)} A={len(a)} B={len(b)}")
    counts_a = [sentence_counts(g, x, metric) for g, x in zip(gold, a)]
    counts_b = [sentence_counts(g, x, metric) for g, x in zip(gold, b)]
    score_a, score_b = overall(counts_a), overall(counts_b)
    observed = abs(score_a - score_b)

    rng = random.Random(seed)
    at_least = 0
    for _ in range(iterations):
        ac = at_ = bc = bt = 0
        for i in range(len(gold)):
            x, y = (counts_b[i], counts_a[i]) if rng.random() < 0.5 else (counts_a[i], counts_b[i])
            ac += x[0]; at_ += x[1]
            bc += y[0]; bt += y[1]
        if abs(100.0 * ac / at_ - 100.0 * bc / bt) >= observed:
            at_least += 1
    return score_a, score_b, observed, (at_least + 1) / (iterations + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("systems", nargs="+", help="<parser>:<encoder>:<train set>, e.g. udpipe2:berturk:ota")
    ap.add_argument("--test", default="ota", choices=list(EVAL_TESTS))
    ap.add_argument("--metric", default="las", choices=list(METRIC_KEY))
    ap.add_argument("--all-pairs", action="store_true", help="test every pair, not just the first two")
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default=None, help="also write the results to this CSV file")
    args = ap.parse_args()

    gold = EVAL_TESTS[args.test]
    reps = {s: representative(s, args.test, args.metric) for s in args.systems}
    for r in reps.values():
        print(f"{r['system']:28} {args.metric.upper()} {r['value']:6.2f} "
              f"(mean of {r['runs']} runs: {r['mean']:.2f})  {r['pred'].name}")

    pairs = list(itertools.combinations(args.systems, 2)) if args.all_pairs else [tuple(args.systems[:2])]
    rows = [("test", "metric", "system A", "system B", "score A", "score B", "|diff|", "p", "significant")]
    print()
    for a, b in pairs:
        sa, sb, diff, p = randomization_test(gold, reps[a]["pred"], reps[b]["pred"],
                                             args.metric, args.iterations, args.seed)
        verdict = "yes" if p < 0.05 else "no"
        print(f"{a} vs {b}: {sa:.2f} / {sb:.2f} | diff {diff:.2f} | p={p:.4f} | significant: {verdict}")
        rows.append((args.test, args.metric.upper(), a, b, f"{sa:.2f}", f"{sb:.2f}",
                     f"{diff:.2f}", f"{p:.4f}", verdict))

    if args.csv:
        Path(args.csv).write_text("\n".join(",".join(r) for r in rows) + "\n")
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
