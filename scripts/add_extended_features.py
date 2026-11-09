"""Add the input features of the jet foundation models (gabbro) to existing e+e- jet files.

New columns (see fm_data.common.jets.extended_features / collider_features):
    part_etarel_R  sin(theta_jet) * part_etarel / R
    part_phirel_R  sin(theta_jet) * part_phirel / R
    part_logptrel  log(part_ptrel)
    jet_R          R
    jet_is_ee      1
    jet_is_pp      0
    jet_sqrt_s     event_sqrt_s
    label_<name>   1 for every jet, for each --label (the models load label_QCD)

Every input file is rewritten to <output-dir> with the same name, the same row groups and all
original columns; the new columns are appended. The input files are not changed.

Usage:
    ./run.sh python3 scripts/add_extended_features.py \
        --input-dir  /local/laurits/20261108_ALEPH_ntupelized/ALEPH_combined/jets_ee_antikt_R04 \
        --output-dir /local/laurits/20261108_ALEPH_ntupelized/ALEPH_combined/jets_ee_antikt_R04_ext \
        --R 0.4 --label QCD --workers 16
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm_data.common import jets as jt

INPUT_COLUMNS = ["jet_theta", "part_etarel", "part_phirel", "part_ptrel", "event_sqrt_s"]


def _like(template, values):
    """List column with the offsets (and list type) of `template` and the flat `values`."""
    template = template.combine_chunks() if isinstance(template, pa.ChunkedArray) else template
    values = pa.array(np.asarray(values, dtype=np.float64), type=template.type.value_type)
    return pa.LargeListArray.from_arrays(template.offsets, values, type=template.type)


def add_columns(table, R, labels):
    for name in [n for n in INPUT_COLUMNS if n not in table.column_names]:
        raise KeyError(f"input column {name} missing")
    new_names = ["part_etarel_R", "part_phirel_R", "part_logptrel", "jet_R", "jet_is_ee",
                 "jet_is_pp", "jet_sqrt_s"] + [f"label_{label}" for label in labels]
    present = [name for name in new_names if name in table.column_names]
    if present:
        raise ValueError(f"columns already present: {present}")

    arrays = {name: ak.from_arrow(table.column(name)) for name in INPUT_COLUMNS}
    part = jt.extended_features(
        ak.to_numpy(arrays["jet_theta"]), arrays["part_etarel"], arrays["part_phirel"],
        arrays["part_ptrel"], R,
    )
    for name, values in part.items():
        table = table.append_column(
            name, _like(table.column("part_etarel"), ak.to_numpy(ak.flatten(values)))
        )
    jet = jt.collider_features(len(table), R, ak.to_numpy(arrays["event_sqrt_s"]))
    for label in labels:
        jet[f"label_{label}"] = np.ones(len(table), dtype=np.int32)
    for name, values in jet.items():
        table = table.append_column(name, pa.array(values))
    return table


def process(path, output_dir, R, labels, overwrite):
    target = os.path.join(output_dir, os.path.basename(path))
    if os.path.exists(target) and not overwrite:
        return path, "exists"
    source = pq.ParquetFile(path)
    tmp = target + ".tmp"
    writer = None
    n_rows = 0
    try:
        for row_group in range(source.metadata.num_row_groups):
            table = add_columns(source.read_row_group(row_group), R, labels)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=len(table))
            n_rows += len(table)
    finally:
        if writer is not None:
            writer.close()
    if n_rows != source.metadata.num_rows:
        raise RuntimeError(f"{path}: wrote {n_rows} rows of {source.metadata.num_rows}")
    os.replace(tmp, target)
    return path, n_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--R", type=float, required=True, help="jet radius of the clustering")
    parser.add_argument("--label", action="append", default=[], help="label_<name> = 1 column")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.realpath(args.input_dir) == os.path.realpath(args.output_dir):
        sys.exit("--output-dir must differ from --input-dir")
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))[: args.max_files]
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{len(paths)} files: {args.input_dir} -> {args.output_dir}", flush=True)
    n_done = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(process, path, args.output_dir, args.R, args.label, args.overwrite)
                   for path in paths]
        for future in futures:
            path, result = future.result()
            n_done += 1
            print(f"[{n_done}/{len(paths)}] {os.path.basename(path)}: {result}", flush=True)


if __name__ == "__main__":
    main()
