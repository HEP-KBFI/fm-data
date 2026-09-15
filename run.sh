#!/bin/bash
# Run a command inside the analysis container.
#
#   ./run.sh python3 lep_data/aleph/scripts/rechunk.py -i ... -o ...
#
# The generated SLURM job scripts use this as their environment_script, so the
# path they record is the one of the checkout ntupelize_all.py was run from.
#
# keras is not used, but something in the image imports it and crashes without
# a backend set.
export KERAS_BACKEND=torch
apptainer exec \
    -B /scratch/persistent,/local,/home/laurits \
    --env PYTHONPATH="$(pwd)" \
    --nv \
    /home/software/singularity/pytorch.simg:2025-09-01 "$@"
