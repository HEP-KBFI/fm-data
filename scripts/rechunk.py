"""Rechunk existing jet-level .parquet files into files of N jets each (default 100k).

The production rechunks the per-task jet tables itself (scripts/ntupelize.py mode=rechunk,
see fm_data.common.ntupelizing.rechunk_jets); this is the stand-alone version for files that
already exist. It writes a numbered series <prefix>_00000.parquet, <prefix>_00001.parquet, ...

A jet-level dataset (one row per jet) hits the target exactly. Rows are never split across
files, so an event-level dataset (one row per event, jets in a list column) lands on
whole-event boundaries. Which of the two an input is gets detected from the type of its
`jet_pt` field. Inputs are read one row group at a time and each output file is written once,
so memory is bounded by a single output file.

--max-abs-eta / --min-jet-energy apply jet-level cuts while rechunking (jet-level input only).

Usage:
    ./run.sh python3 scripts/rechunk.py -i <dir, glob or file> -o <output_dir> [options]
"""

import argparse
import sys
from pathlib import Path

# Importable without installing the package: the repo root is four levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm_data.common import chunking as ch


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Directory of .parquet files, a glob, or a single file.",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, help="Where to write the chunked files."
    )
    parser.add_argument(
        "--prefix",
        default="data",
        help="Output filename prefix: <prefix>_00000.parquet (default: data).",
    )
    parser.add_argument(
        "--jets-per-file",
        type=int,
        default=ch.DEFAULT_JETS_PER_FILE,
        help=f"Jets per output file (default: {ch.DEFAULT_JETS_PER_FILE}).",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=ch.DEFAULT_ROW_GROUP_SIZE,
        help=f"Rows per parquet row group (default: {ch.DEFAULT_ROW_GROUP_SIZE}).",
    )
    parser.add_argument(
        "--jet-field",
        default=ch.JET_FIELD,
        help=f"Field whose length counts the jets of a row (default: {ch.JET_FIELD}).",
    )
    parser.add_argument(
        "--max-abs-eta",
        type=float,
        default=None,
        help="Drop jets with |jet_eta| >= this while rechunking (jet-level input only).",
    )
    parser.add_argument(
        "--min-jet-energy",
        type=float,
        default=None,
        help="Drop jets with jet_energy <= this while rechunking (jet-level input only).",
    )
    args = parser.parse_args()

    ch.rechunk(
        input_loc=args.input,
        output_dir=args.output_dir,
        prefix=args.prefix,
        jets_per_file=args.jets_per_file,
        row_group_size=args.row_group_size,
        jet_field=args.jet_field,
        max_abs_eta=args.max_abs_eta,
        min_jet_energy=args.min_jet_energy,
    )


if __name__ == "__main__":
    main()
