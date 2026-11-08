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

--max-abs-eta / --min-jet-energy apply the same jet-level cuts the ntupelizer
itself applies (MAX_ABS_JET_ETA / MIN_JET_ENERGY_GEV in tools/create_aleph.py)
to a dataset produced before those cuts existed, without re-running fastjet
clustering from ROOT. Jet-level input only: an event-level row holds several
jets in a list column, and dropping just one of them is not a rechunk.

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

    # apply the |jet_eta| < 3 cut retroactively while rechunking
    python3 lep_data/aleph/scripts/rechunk.py \\
        -i /local/laurits/0930_ALEPH_raw \\
        -o /local/laurits/0930_ALEPH_raw_eta3 \\
        --max-abs-eta 3.0
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pyarrow.compute as pc
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


def build_row_filter(max_abs_eta=None, min_jet_energy=None):
    """Return a callable filtering a jet-level table by the given cuts, or None."""
    if max_abs_eta is None and min_jet_energy is None:
        return None

    def row_filter(table):
        mask = None
        if max_abs_eta is not None:
            mask = pc.less(pc.abs(table.column("jet_eta")), max_abs_eta)
        if min_jet_energy is not None:
            above = pc.greater(table.column("jet_energy"), min_jet_energy)
            mask = above if mask is None else pc.and_(mask, above)
        return table.filter(mask)

    return row_filter


def rechunk(
    input_loc,
    output_dir,
    prefix="aleph",
    jets_per_file=ch.DEFAULT_JETS_PER_FILE,
    row_group_size=ch.DEFAULT_ROW_GROUP_SIZE,
    jet_field=ch.JET_FIELD,
    max_abs_eta=None,
    min_jet_energy=None,
):
    """Stream the inputs into numbered files of jets_per_file jets each.

    max_abs_eta / min_jet_energy, if given, additionally drop jets failing
    |jet_eta| < max_abs_eta and/or jet_energy > min_jet_energy while rechunking
    (jet-level input only -- see module docstring).
    """
    input_paths = resolve_inputs(input_loc)
    schema = ch.plain_schema(pq.ParquetFile(input_paths[0]).schema_arrow)
    level = "event" if ch.is_event_level(schema, jet_field) else "jet"
    row_filter = build_row_filter(max_abs_eta, min_jet_energy)
    if row_filter is not None and level == "event":
        raise ValueError(
            "max_abs_eta / min_jet_energy filtering only supports jet-level "
            f"input (one row per jet); {input_paths[0]} is event-level."
        )
    filter_desc = " and ".join(
        s
        for s in [
            f"|jet_eta| < {max_abs_eta}" if max_abs_eta is not None else None,
            f"jet_energy > {min_jet_energy}" if min_jet_energy is not None else None,
        ]
        if s is not None
    )
    print(
        f"Rechunking {len(input_paths)} {level}-level file(s) into "
        f"{jets_per_file}-jet files under {output_dir}"
        + (f" (filtering: {filter_desc})" if filter_desc else "")
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
    n_rows_kept = 0
    for i, path in enumerate(input_paths, 1):
        parquet_file = pq.ParquetFile(path)
        if parquet_file.metadata.num_rows == 0:
            print(f"[{i}/{len(input_paths)}] {os.path.basename(path)}: empty, skipping")
            continue
        print(f"[{i}/{len(input_paths)}] {os.path.basename(path)}", flush=True)
        for row_group in range(parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(row_group).replace_schema_metadata(None)
            n_rows_in += table.num_rows
            if row_filter is not None:
                table = row_filter(table)
            n_rows_kept += table.num_rows
            if table.num_rows > 0:
                writer.add(table)
    paths = writer.close()

    if writer.total_rows != n_rows_kept:
        raise RuntimeError(
            f"Wrote {writer.total_rows} rows but kept {n_rows_kept} after reading "
            f"{n_rows_in}; refusing to report success on a dataset that lost or "
            f"duplicated rows."
        )
    dropped_msg = f" ({n_rows_in - n_rows_kept} dropped by the filter)" if row_filter is not None else ""
    print(
        f"Done: {n_rows_in} rows read{dropped_msg}, {n_rows_kept} kept / "
        f"{writer.total_jets} jets from {len(input_paths)} file(s) into "
        f"{len(paths)} file(s)"
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

    rechunk(
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
