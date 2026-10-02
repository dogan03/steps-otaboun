#!/usr/bin/env python3
"""A readable status table for a batch of UDPipe 2 runs.

Reads what is finished from the output directory (a run is done once it has results.json)
and how far the rest have got from their logs (UDPipe 2 prints one "Dev  epoch" line per
epoch, and its counter restarts for the second learning-rate phase, so lines are counted).

    python scripts/udpipe2_status.py --train-sets ota_dudu tr_dudu tr_ota tr_ota_dudu \
        --log-dir /content/qb --name-prefix u2_berturk_
"""

import argparse
import os
import re
import time
from pathlib import Path

EPOCHS = 60


def epochs_done(log):
    try:
        return log.read_text(errors="ignore").count("Dev  epoch")
    except OSError:
        return None


def start_times(queue_log):
    """When the queue started each job: it logs "HH:MM basladi: <name> ...".
    (A log file's own timestamps are no help -- they change on every write.)"""
    starts = {}
    try:
        text = Path(queue_log).read_text(errors="ignore")
    except (OSError, TypeError):
        return starts
    for h, m, name in re.findall(r"(\d\d):(\d\d) basladi: (\S+)", text):
        starts[name] = int(h) * 60 + int(m)
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.environ.get("STEPS_OUTPUT_DIR", "experiment_outputs"))
    ap.add_argument("--model", default="berturk")
    ap.add_argument("--train-sets", nargs="+", required=True)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=8446)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--name-prefix", default="u2_berturk_")
    ap.add_argument("--queue-log", default=None)
    args = ap.parse_args()

    root = Path(args.out_dir) / "udpipe2"
    starts = start_times(args.queue_log)
    now_min = time.localtime().tm_hour * 60 + time.localtime().tm_min
    logs = Path(args.log_dir)
    done = running = waiting = 0
    epochs_total = epochs_sum = 0
    rates = []

    print(f"{'train set':14}" + "".join(f"{'fold '+str(i):>10}" for i in range(args.cv)))
    for ts in args.train_sets:
        cells = []
        for i in range(args.cv):
            rdir = root / f"{args.model}_{ts}_cv{args.cv}_fold{i}_seed{args.seed}"
            log = logs / f"{args.name_prefix}{ts}_f{i}.log"
            epochs_total += EPOCHS
            if (rdir / "results.json").exists():
                done += 1; epochs_sum += EPOCHS
                cells.append("bitti")
                continue
            e = epochs_done(log)
            if e is None:
                waiting += 1
                cells.append("-")
                continue
            running += 1; epochs_sum += e
            cells.append(f"{e}/{EPOCHS}")
            began = starts.get(f"{args.name_prefix}{ts}_f{i}")
            # a run that has just started is still paying its embedding-extraction cost,
            # so its apparent rate would drag the average down
            if e >= 5 and began is not None:
                mins = (now_min - began) % (24 * 60)
                if mins:
                    rates.append(e / mins * 60)
        print(f"{ts:14}" + "".join(f"{c:>10}" for c in cells))

    total = done + running + waiting
    pct = 100.0 * epochs_sum / epochs_total if epochs_total else 0
    print(f"\nkosu: {done} bitti · {running} kosuyor · {waiting} sirada · toplam {total}")
    print(f"epoch: {epochs_sum}/{epochs_total} (%{pct:.1f})")
    if rates:
        rate = sum(rates) / len(rates)
        left = (epochs_total - epochs_sum) / (rate * max(running, 1)) if rate else 0
        print(f"hiz: ~{rate:.1f} epoch/saat/kosu · kalan tahmini ~{left:.1f} saat")
    if args.queue_log and Path(args.queue_log).exists():
        last = Path(args.queue_log).read_text(errors="ignore").strip().splitlines()[-1:]
        if last:
            print(f"kuyruk: {last[0]}")


if __name__ == "__main__":
    main()
