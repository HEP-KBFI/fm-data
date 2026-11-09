"""Combine the per-sample outputs of a production into one sample (e.g. all ALEPH years).

For every jet table, the chunk files of the samples are taken in the given order. Files that
already hold exactly --jets-per-file jets are copied unchanged into one numbered series
<output>/<table>/data_<name>_<NNNNN>.parquet. The partial last file of each sample is
rechunked into full files at the end, so every combined file except the last holds exactly
--jets-per-file jets. The jets of a file therefore come from one sample, except in the files
made from the leftovers. Event files (one per input ROOT file) are copied with their names.

Every output file is a copy: the combined sample does not change if the per-sample outputs are
rewritten or deleted. <output>/manifest.csv lists, for each output file, where its rows came
from.

Usage:
    ./run.sh python3 scripts/combine_samples.py --base-dir <production> --name ALEPH_combined \
        --samples ALEPH1990 ALEPH1991 ... [--tables events jets_ee_antikt_R04 ...]
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fm_data.common import chunking as ch


def copy_file(source, target):
    """Copy via a temporary name so that an interrupted copy never looks complete."""
    tmp = target + ".tmp"
    shutil.copyfile(source, tmp)
    os.replace(tmp, target)


def sample_files(base_dir, samples, table):
    files = []
    for sample in samples:
        paths = ch.resolve_inputs(os.path.join(base_dir, sample, table))
        files += [(sample, path) for path in paths]
    return files


def combine_jets(base_dir, samples, table, output_dir, prefix, jets_per_file, manifest):
    files = sample_files(base_dir, samples, table)
    n_in = sum(pq.ParquetFile(path).metadata.num_rows for _, path in files)
    full = [(s, p) for s, p in files if pq.ParquetFile(p).metadata.num_rows == jets_per_file]
    partial = [(s, p) for s, p in files if pq.ParquetFile(p).metadata.num_rows != jets_per_file]
    print(f"{table}: {len(files)} files, {n_in} jets; {len(full)} full files copied, "
          f"{len(partial)} partial files rechunked", flush=True)

    schema = pq.ParquetFile(files[0][1]).schema_arrow
    writer = ch.JetChunkWriter(output_dir, prefix, schema, jets_per_file=jets_per_file)
    for index, (sample, path) in enumerate(full):
        target = writer.path(index)
        copy_file(path, target)
        manifest.append((table, os.path.basename(target), jets_per_file, sample, path))
    writer.first_index = len(full)

    for sample, path in partial:
        parquet_file = pq.ParquetFile(path)
        start = writer.first_index + len(writer.paths)
        for row_group in range(parquet_file.metadata.num_row_groups):
            writer.add(parquet_file.read_row_group(row_group).replace_schema_metadata(None))
        # record which output files this partial file went into
        end = writer.first_index + len(writer.paths)
        manifest.append((table, f"{os.path.basename(writer.path(start))} .. "
                         f"{os.path.basename(writer.path(end))}",
                         parquet_file.metadata.num_rows, sample, path))
    writer.close()

    n_out = sum(pq.ParquetFile(writer.path(i)).metadata.num_rows
                for i in range(writer.first_index + len(writer.paths)))
    if n_out != n_in:
        raise RuntimeError(f"{table}: wrote {n_out} jets but read {n_in}")
    print(f"{table}: {n_out} jets in {writer.first_index + len(writer.paths)} files", flush=True)


def combine_events(base_dir, samples, table, output_dir, manifest):
    os.makedirs(output_dir, exist_ok=True)
    files = sample_files(base_dir, samples, table)
    for sample, path in files:
        copy_file(path, os.path.join(output_dir, os.path.basename(path)))
        manifest.append((table, os.path.basename(path),
                         pq.ParquetFile(path).metadata.num_rows, sample, path))
    print(f"{table}: {len(files)} files copied", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", required=True, help="Production directory (one subdirectory per sample).")
    parser.add_argument("--samples", nargs="+", required=True, help="Samples to combine, in order.")
    parser.add_argument("--name", required=True, help="Name of the combined sample (its directory under --base-dir).")
    parser.add_argument("--tables", nargs="+",
                        default=["events", "jets_ee_antikt_R04", "jets_ee_cambridge_exclusive2"])
    parser.add_argument("--jets-per-file", type=int, default=ch.DEFAULT_JETS_PER_FILE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = os.path.join(args.base_dir, args.name)
    if os.path.exists(output) and os.listdir(output) and not args.overwrite:
        raise SystemExit(f"{output} is not empty; use --overwrite")
    if args.overwrite and os.path.exists(output):
        shutil.rmtree(output)
    os.makedirs(output)

    manifest = []
    for table in args.tables:
        output_dir = os.path.join(output, table)
        if table == "events":
            combine_events(args.base_dir, args.samples, table, output_dir, manifest)
        else:
            combine_jets(args.base_dir, args.samples, table, output_dir, f"data_{args.name}",
                         args.jets_per_file, manifest)

    with open(os.path.join(output, "manifest.csv"), "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["table", "output_file", "rows", "sample", "source_file"])
        out.writerows(manifest)
    with open(os.path.join(output, "README.txt"), "w") as f:
        f.write(f"Combined sample {args.name}: samples {' '.join(args.samples)}, "
                f"tables {' '.join(args.tables)}.\n"
                f"Made by scripts/combine_samples.py; manifest.csv says where each file came from.\n"
                f"Jet files hold {args.jets_per_file} jets each except the last; the files made "
                f"from the leftover partial file of each sample mix samples.\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
