#!/bin/bash
# Run a command in the container used for the ntupelization. It needs uproot >= 5.6 to read
# the CLD EDM4hep files (gabbro-latest.sif has uproot 5.1.2, which fails on them).
export OPENBLAS_NUM_THREADS=1
cd "$(dirname "${BASH_SOURCE[0]}")"
apptainer exec -B /scratch/persistent,/local,/home --env PYTHONPATH=`pwd` /home/software/singularity/pytorch.simg:2025-09-01 "$@"
