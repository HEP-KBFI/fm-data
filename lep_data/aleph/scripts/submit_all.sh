#!/bin/bash

PROGNAME=$0

# Parse user options.
usage() {
  cat << EOF >&2
Usage: bash scripts/$PROGNAME -s DIR
  -s DIR : Submit jobs to slurm for the specified directory.
EOF
  exit 1
}


while getopts 's:' OPTION; do
  case $OPTION in
    s) SUBMISSION_DIR=$OPTARG ;;
    ?) usage ;;
  esac
done
shift "$((OPTIND - 1))"

for file in "$SUBMISSION_DIR"/*; do
  if [ -f "$file" ]; then
    sbatch "$file"
  fi
done

