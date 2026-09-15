"""Ntupelize a list of ALEPH ROOT files into chunk files of ~100k jets.

Run by one SLURM job (see ntupelize_all.py, which writes the job scripts). The
files assigned to the job are processed in turn and streamed into a numbered
series of output files holding `jets_per_file` jets each:

    <output_dir>/<prefix>_00000.parquet, <prefix>_00001.parquet, ...

`prefix` is what keeps concurrent jobs from writing over each other, so it has
to be unique per job; ntupelize_all.py derives it from the job index. Since the
jobs are independent, each one's last file holds only what was left over — run
scripts/rechunk.py over the output directory afterwards to consolidate those
tails into one uniform series.

Usage (arguments are Hydra overrides, as passed by the generated job script):
    python3 ntupelize_list.py +input_paths='[a.root,b.root]' \
        +output_dir=/path/out +prefix=job0000 +output_level=event \
        [+jets_per_file=100000] [+row_group_size=1024]
"""

import os
import time

import awkward as ak
import hydra
from omegaconf import DictConfig

from lep_data.aleph.tools import chunking as ch
from lep_data.aleph.tools import create_aleph as na

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    input_files = cfg.input_paths
    output_dir = cfg.output_dir
    prefix = cfg.prefix
    output_level = cfg.get("output_level", "event")
    jets_per_file = cfg.get("jets_per_file", ch.DEFAULT_JETS_PER_FILE)
    row_group_size = cfg.get("row_group_size", ch.DEFAULT_ROW_GROUP_SIZE)
    if output_level not in ("event", "jet"):
        raise ValueError(f"output_level must be 'event' or 'jet', got {output_level!r}")

    start_time = time.time()
    writer = None
    n_failed = 0
    for i, input_path in enumerate(input_files, 1):
        print(f"[{i}/{len(input_files)}] {input_path}", flush=True)
        try:
            dataset = na.build_dataset(
                input_path=input_path,
                jet_level=output_level == "jet",
                event_level=output_level == "event",
            )
        except Exception as error:
            # One unreadable ROOT file should not throw away the rest of the job.
            print(f"WARNING: failed to ntupelize {input_path}: {error}")
            n_failed += 1
            continue
        if len(dataset) == 0:
            print(f"  no rows survived the selection in {input_path}")
            continue
        table = ak.to_arrow_table(dataset, extensionarray=False)
        if writer is None:
            # The first processed file defines the output schema.
            writer = ch.JetChunkWriter(
                output_dir=output_dir,
                prefix=prefix,
                schema=table.schema,
                jets_per_file=jets_per_file,
                row_group_size=row_group_size,
            )
        writer.add(table.replace_schema_metadata(None))

    if writer is None:
        raise RuntimeError(
            f"None of the {len(input_files)} input files produced any rows"
        )
    paths = writer.close()
    elapsed = time.time() - start_time
    print(
        f"Finished {len(input_files) - n_failed}/{len(input_files)} files in "
        f"{elapsed:.1f} s: {writer.total_rows} rows / {writer.total_jets} jets "
        f"in {len(paths)} file(s)"
    )


if __name__ == "__main__":
    main()
