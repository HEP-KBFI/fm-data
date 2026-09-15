"""Rechunk existing ALEPH .parquet files into files of ~100k jets each.

The ALEPH ntupelizer writes one .parquet per input ROOT file, so the file sizes
follow the input rather than anything useful for training. This reads such a set
of files and writes a numbered series holding `--jets-per-file` jets each:

    aleph_00000.parquet, aleph_00001.parquet, ...

A jet-level dataset (one row per jet, which is what `output_level: jet`
produces) hits the target exactly. Rows are never split across files, so an
event-level dataset (one row per event, jets in a list column) instead lands on
whole-event boundaries and its files hold about rather than exactly the target.
Which of the two an input is gets detected from the type of its `jet_pt` field.

Inputs are read one row group at a time and each output file is written once, so
memory is bounded by a single output file no matter how large the dataset is.

Usage:
    python3 lep_data/aleph/scripts/rechunk.py -i <input> -o <output_dir> [options]

Examples:
    # rechunk a directory of per-ROOT-file ntuples
    python3 lep_data/aleph/scripts/rechunk.py \\
        -i /local/laurits/ALEPH/ALEPH_jet \\
        -o /local/laurits/ALEPH/ALEPH_jet_chunked

    # consolidate the per-job files that ntupelize_list.py writes
    python3 lep_data/aleph/scripts/rechunk.py \\
        -i "/local/laurits/ALEPH/ALEPH_jet/job*.parquet" \\
        -o /local/laurits/ALEPH/ALEPH_jet_chunked
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

# Importable without installing the package: the repo root is four levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lep_data.aleph.tools import chunking as ch


def natural_key(path):
    """Sort key ordering any numbers in a filename numerically."""
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", os.path.basename(path))
    )


def resolve_inputs(input_loc):
    """Return the input .parquet files: a directory, a glob, or a single file."""
    if os.path.isdir(input_loc):
        paths = glob.glob(os.path.join(input_loc, "*.parquet"))
    elif "*" in input_loc:
        paths = glob.glob(input_loc)
    elif os.path.isfile(input_loc):
        paths = [input_loc]
    else:
        raise FileNotFoundError(f"No such file, directory or glob: {input_loc}")
    if not paths:
        raise FileNotFoundError(f"No .parquet files found in {input_loc}")
    return sorted(paths, key=natural_key)


def rechunk(
    input_loc,
    output_dir,
    prefix="aleph",
    jets_per_file=ch.DEFAULT_JETS_PER_FILE,
    row_group_size=ch.DEFAULT_ROW_GROUP_SIZE,
    jet_field=ch.JET_FIELD,
):
    """Stream the inputs into numbered files of jets_per_file jets each."""
    input_paths = resolve_inputs(input_loc)
    schema = ch.plain_schema(pq.ParquetFile(input_paths[0]).schema_arrow)
    level = "event" if ch.is_event_level(schema, jet_field) else "jet"
    print(
        f"Rechunking {len(input_paths)} {level}-level file(s) into "
        f"{jets_per_file}-jet files under {output_dir}"
    )

    writer = ch.JetChunkWriter(
        output_dir=output_dir,
        prefix=prefix,
        schema=schema,
        jets_per_file=jets_per_file,
        row_group_size=row_group_size,
        jet_field=jet_field,
    )
    # Writing into the input directory is fine, writing over the inputs is not:
    # the input list is fixed up front, so a clash would silently corrupt a file
    # that has not been read yet.
    output_pattern = re.compile(rf"^{re.escape(prefix)}_\d+\.parquet$")
    clashing = [
        path
        for path in input_paths
        if os.path.dirname(os.path.abspath(path)) == os.path.abspath(output_dir)
        and output_pattern.match(os.path.basename(path))
    ]
    if clashing:
        raise ValueError(
            f"The output would overwrite its own input, e.g. {clashing[0]}. "
            f"Use a different --output-dir or --prefix."
        )

    n_rows_in = 0
    for i, path in enumerate(input_paths, 1):
        parquet_file = pq.ParquetFile(path)
        if parquet_file.metadata.num_rows == 0:
            print(f"[{i}/{len(input_paths)}] {os.path.basename(path)}: empty, skipping")
            continue
        print(f"[{i}/{len(input_paths)}] {os.path.basename(path)}", flush=True)
        for row_group in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(row_group)
            n_rows_in += table.num_rows
            writer.add(table.replace_schema_metadata(None))
    paths = writer.close()

    if writer.total_rows != n_rows_in:
        raise RuntimeError(
            f"Wrote {writer.total_rows} rows but read {n_rows_in}; refusing to "
            f"report success on a dataset that lost or duplicated rows."
        )
    print(
        f"Done: {n_rows_in} rows / {writer.total_jets} jets from "
        f"{len(input_paths)} file(s) into {len(paths)} file(s)"
    )
    return paths


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
        default="aleph",
        help="Output filename prefix: <prefix>_00000.parquet (default: aleph).",
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
    args = parser.parse_args()

    rechunk(
        input_loc=args.input,
        output_dir=args.output_dir,
        prefix=args.prefix,
        jets_per_file=args.jets_per_file,
        row_group_size=args.row_group_size,
        jet_field=args.jet_field,
    )


if __name__ == "__main__":
    main()
