#!/usr/bin/env bash
#SBATCH --job-name=ess-sweep
#SBATCH --nodes=1
#SBATCH --output=logs/%x-%A_%a.out
# Fixes sweep: each array task takes the lines of JOBLIST whose index is its id modulo the array size and runs them
# with xargs -P $SLURM_CPUS_PER_TASK. A line is: RUN_ID ARM ARGS...; a run is skipped when its DONE exists.
# Arms are source trees in $ARMS (default: the parent of the submit folder): $ARMS/<arm>/src. ARM `random` has none.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?run through sbatch from the project root}"
cd "$ROOT"
JOBLIST="${JOBLIST:-slurm/sweep_jobs.txt}" RUN="${RUN:-fixes1}" ARMS="${ARMS:-$(dirname "$ROOT")}" SEEDS="${SEEDS:-20}"
OUT="$ROOT/results/$RUN"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 PYTHONUNBUFFERED=1 ROOT OUT ARMS SEEDS
run_one() { # RUN_ID ARM ARGS... -> $OUT/RUN_ID/ written atomically (tmp folder, then rename)
    local id="$1" arm="$2"; shift 2
    [[ -f "$OUT/$id/DONE" ]] && return 0
    local tmp="$OUT/.tmp-$id-$$" src=()
    rm -rf "$tmp"; mkdir -p "$tmp"
    if [[ "$arm" != random ]]; then
        src=(--src "$ARMS/$arm/src")
        cat "$ARMS/$arm/.slurm/SOURCE" >"$tmp/SOURCE"
    fi
    echo "job ${SLURM_JOB_ID:-} host $(hostname) start $(date -u +%FT%TZ) args $arm $*" >"$tmp/PROVENANCE"
    "$ROOT/venv/bin/python" "$ROOT/slurm/sweep_run.py" "${src[@]}" --out "$tmp" --seeds "$SEEDS" "$@" >"$tmp/run.log" 2>&1 \
        || { echo "FAIL $id" >&2; tail -3 "$tmp/run.log" >&2; return 1; }
    date -u +%FT%TZ >"$tmp/DONE"; rm -rf "${OUT:?}/$id"; mv "$tmp" "$OUT/$id"
}
export -f run_one
mkdir -p "$OUT"
cp -n venv/freeze.txt "$OUT/freeze.txt" 2>/dev/null || :
task="${SLURM_ARRAY_TASK_ID:-0}" shards="${SLURM_ARRAY_TASK_COUNT:-1}"
awk -v t="$task" -v s="$shards" '(NR - 1) % s == t' "$JOBLIST" \
    | xargs -P "${SLURM_CPUS_PER_TASK:-1}" -L1 bash -c 'run_one "$@"' _
