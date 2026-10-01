#!/usr/bin/env bash
# Runs a list of jobs with as much parallelism as the GPU actually has room for.
#
# Reads jobs from stdin, one per line:   <name> <command ...>
# A job starts only when (a) fewer than --slots jobs are running and (b) the GPU has at
# least --need MiB free, so a run can never die with CUDA out of memory. Each job's output
# goes to <log-dir>/<name>.log.
#
#   bash scripts/gpu_queue.sh --slots 4 --need 11000 --log-dir /content/qlogs < jobs.txt
set -o pipefail

SLOTS=4; NEED=11000; LOGDIR="queue_logs"; SETTLE=120
while [ $# -gt 0 ]; do
  case "$1" in
    --slots)   SLOTS="$2";  shift 2;;
    --need)    NEED="$2";   shift 2;;   # MiB a single run needs
    --log-dir) LOGDIR="$2"; shift 2;;
    --settle)  SETTLE="$2"; shift 2;;   # seconds to let a new job claim its memory
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
mkdir -p "$LOGDIR"

free_mb() {
  local v
  v=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  # no GPU (or no nvidia-smi): do not gate on memory
  [ -n "$v" ] && echo "$v" || echo 999999
}

running=""        # pids of the jobs still alive
alive_count() {
  local still=""
  for p in $running; do kill -0 "$p" 2>/dev/null && still="$still $p"; done
  running="$still"
  echo $running | wc -w | tr -d ' '
}

queued=0
while IFS= read -r line; do
  case "$line" in ""|\#*) continue;; esac
  name="${line%% *}"; cmd="${line#* }"
  queued=$((queued + 1))
  while :; do
    n=$(alive_count); free=$(free_mb)
    [ "$n" -lt "$SLOTS" ] && [ "$free" -ge "$NEED" ] && break
    sleep 15
  done
  echo "$(date +%H:%M) basladi: $name  (calisan: $n, bos GPU: ${free} MiB)"
  bash -c "$cmd" > "$LOGDIR/$name.log" 2>&1 &
  running="$running $!"
  sleep "$SETTLE"
done

wait
echo "$(date +%H:%M) kuyruk bitti ($queued is)"
