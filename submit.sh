#!/bin/bash
# Submit the ntupelization as a SLURM job array, one array task per output file, and (if the
# config has jets_per_file) a rechunk job that runs once the whole array has finished.
#
# Usage: [CONFIG=<domain>/<name>] ./submit.sh [hydra overrides]
#   CONFIG: a config in fm_data/<domain>/config, default ee_sim/cld; e.g. CONFIG=lep/aleph
#
# 1. saves the config to <output_dir>/ntupelize_config.yaml (mode=setup)
# 2. asks for the number of tasks (mode=n_tasks)
# 3. submits the array; each task runs `mode=run task_id=$SLURM_ARRAY_TASK_ID`
# 4. with jets_per_file: submits one `mode=rechunk` job per sample with --dependency=afterany
#    on the array. A rechunk refuses to run if a task output of its sample is missing: then
#    resubmit (step 3 skips tasks whose outputs exist) and rerun the rechunk.
# SLURM settings come from the `slurm` section of the config (or overrides).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs/slurm

CONFIG="${CONFIG:-ee_sim/cld}"
DOMAIN="${CONFIG%%/*}"
NAME="${CONFIG##*/}"
CONFIG_FILE="fm_data/${DOMAIN}/config/${NAME}.yaml"
[[ -f "$CONFIG_FILE" ]] || { echo "No config $CONFIG_FILE"; exit 1; }
SCRIPT="scripts/ntupelize.py --config-path ../fm_data/${DOMAIN}/config --config-name ${NAME}"

./run.sh python3 $SCRIPT "$@" mode=setup
N_TASKS="$(./run.sh python3 $SCRIPT "$@" mode=n_tasks | grep -oE '^[0-9]+$' | tail -n 1)"
cfg_value() {
    ./run.sh python3 -c "from omegaconf import OmegaConf; import sys; \
cfg = OmegaConf.merge(OmegaConf.load('$CONFIG_FILE'), OmegaConf.from_dotlist(sys.argv[2:])); \
print(OmegaConf.select(cfg, sys.argv[1]))" "$1" "${@:2}" | tail -n 1
}
PARTITION="$(cfg_value slurm.partition "$@")"
MEM="$(cfg_value slurm.mem "$@")"
TIME="$(cfg_value slurm.time "$@")"
MAX_CONCURRENT="$(cfg_value slurm.max_concurrent "$@")"
JETS_PER_FILE="$(cfg_value jets_per_file "$@")"

echo "--- Submitting ${N_TASKS} tasks of ${CONFIG} (at most ${MAX_CONCURRENT} at a time) to ${PARTITION}"
ARRAY_ID="$(sbatch --parsable --array="0-$((N_TASKS - 1))%${MAX_CONCURRENT}" --partition="$PARTITION" \
    --mem="$MEM" --time="$TIME" --job-name="ntupelize-${NAME}" \
    --output=logs/slurm/ntupelize-%A_%a.out --error=logs/slurm/ntupelize-%A_%a.err \
    --wrap="./run.sh python3 $SCRIPT $* mode=run task_id=\$SLURM_ARRAY_TASK_ID")"
echo "array job ${ARRAY_ID}"

if [[ "$JETS_PER_FILE" != "None" && -n "$JETS_PER_FILE" ]]; then
    # one rechunk job per sample (they are independent), each with the other samples removed
    SAMPLES="$(./run.sh python3 -c "from omegaconf import OmegaConf; import sys; \
cfg = OmegaConf.merge(OmegaConf.load('$CONFIG_FILE'), OmegaConf.from_dotlist(sys.argv[1:])); \
print(' '.join(cfg.samples))" "$@" | tail -n 1)"
    for SAMPLE in $SAMPLES; do
        DROP=""
        for OTHER in $SAMPLES; do
            [[ "$OTHER" != "$SAMPLE" ]] && DROP="$DROP ~samples.$OTHER"
        done
        RECHUNK_ID="$(sbatch --parsable --dependency="afterany:${ARRAY_ID}" --partition="$PARTITION" \
            --mem=16G --time=02:00:00 --job-name="rechunk-${SAMPLE}" \
            --output=logs/slurm/rechunk-${SAMPLE}-%j.out --error=logs/slurm/rechunk-${SAMPLE}-%j.err \
            --wrap="./run.sh python3 $SCRIPT $* mode=rechunk$DROP")"
        echo "rechunk job ${RECHUNK_ID} for ${SAMPLE} (after ${ARRAY_ID})"
    done
fi
