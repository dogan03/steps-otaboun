"""Approximate randomization significance test for dependency parsing and POS tagging.

Compares two systems' prediction files against the same gold file. The null hypothesis is:
"The two models are not different; their predictions are interchangeable."

Procedure (sentence-level approximate randomization, cf. Noreen 1989; Yeh 2000):
  1. Compute the observed absolute difference of the metric between system A and system B.
  2. For each of R iterations, independently swap each sentence's predictions between the
     two systems with probability 0.5, and recompute the metric difference.
  3. p-value = (number of iterations with |diff| >= observed + 1) / (R + 1).

A small p-value (< 0.05) means the observed difference is unlikely under the null
hypothesis, i.e. the two systems are significantly different.

Usage:
    python src/significance_test.py --gold data/corpora/ota_boun/ota_boun-test-2026.conllu \
        --sys-a predictions/A.conllu --sys-b predictions/B.conllu \
        --metric las --iterations 10000

    --metric las   LAS  (predicted head AND deprel match gold)
    --metric uas   UAS  (predicted head matches gold)
    --metric upos  UPOS accuracy (predicted UPOS matches gold)
"""

import argparse
import random


def read_conllu(path):
    """Read a CoNLL-U file into a list of sentences, each a list of token tuples
    (form, upos, head, deprel). Multi-word token ranges (1-2) and empty nodes (1.1)
    are skipped, as are comment lines."""
    sentences, current = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            # Match conll18_ud_eval: LAS ignores language-specific deprel subtypes
            # (e.g. "obl:tmod" -> "obl"), so strip everything after the first colon.
            deprel = cols[7].split(":")[0]
            current.append((cols[1], cols[3], cols[6], deprel))
    if current:
        sentences.append(current)
    return sentences


def sentence_counts(gold_sent, sys_sent, metric):
    """(correct, total) for one sentence under the given metric."""
    if len(gold_sent) != len(sys_sent):
        raise ValueError(f"Token count mismatch: gold={len(gold_sent)} sys={len(sys_sent)} "
                         f"(first gold token: {gold_sent[0][0]!r})")
    correct = 0
    for g, s in zip(gold_sent, sys_sent):
        if metric == "upos":
            correct += (s[1] == g[1])
        elif metric == "uas":
            correct += (s[2] == g[2])
        else:  # las
            correct += (s[2] == g[2] and s[3] == g[3])
    return correct, len(gold_sent)


def overall(counts):
    correct = sum(c for c, _ in counts)
    total = sum(t for _, t in counts)
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True, help="gold CoNLL-U file")
    ap.add_argument("--sys-a", required=True, help="system A predictions (CoNLL-U)")
    ap.add_argument("--sys-b", required=True, help="system B predictions (CoNLL-U)")
    ap.add_argument("--metric", required=True, choices=["las", "uas", "upos"])
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gold = read_conllu(args.gold)
    sys_a = read_conllu(args.sys_a)
    sys_b = read_conllu(args.sys_b)
    if not (len(gold) == len(sys_a) == len(sys_b)):
        raise ValueError(f"Sentence count mismatch: gold={len(gold)} A={len(sys_a)} B={len(sys_b)}")

    # Pre-compute per-sentence (correct, total) so each iteration is just sums.
    counts_a = [sentence_counts(g, a, args.metric) for g, a in zip(gold, sys_a)]
    counts_b = [sentence_counts(g, b, args.metric) for g, b in zip(gold, sys_b)]

    score_a, score_b = overall(counts_a), overall(counts_b)
    observed = abs(score_a - score_b)

    rng = random.Random(args.seed)
    n = len(gold)
    n_at_least = 0
    for _ in range(args.iterations):
        ca_c = ca_t = cb_c = cb_t = 0
        for i in range(n):
            if rng.random() < 0.5:  # swap this sentence's predictions
                ca_c += counts_b[i][0]; ca_t += counts_b[i][1]
                cb_c += counts_a[i][0]; cb_t += counts_a[i][1]
            else:
                ca_c += counts_a[i][0]; ca_t += counts_a[i][1]
                cb_c += counts_b[i][0]; cb_t += counts_b[i][1]
        diff = abs(100.0 * ca_c / ca_t - 100.0 * cb_c / cb_t)
        if diff >= observed:
            n_at_least += 1

    p_value = (n_at_least + 1) / (args.iterations + 1)

    print(f"Metric          : {args.metric.upper()}")
    print(f"System A        : {args.sys_a}  ->  {score_a:.2f}")
    print(f"System B        : {args.sys_b}  ->  {score_b:.2f}")
    print(f"Observed |diff| : {observed:.2f}")
    print(f"Iterations      : {args.iterations} (seed {args.seed})")
    print(f"p-value         : {p_value:.4f}  "
          f"({'significant at 0.05' if p_value < 0.05 else 'NOT significant at 0.05'})")


if __name__ == "__main__":
    main()
