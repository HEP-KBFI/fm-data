"""Split the input files into tasks (one output file each) and process a task: read each
ROOT file once, and write the event table and one jet table per jet algorithm. Optionally
(`jets_per_file`), rechunk the jet tables of each sample into files of exactly that many jets.

The format-specific part is a module with
`process_file(filepath, cfg, sample, file_number) -> (event_table, {jet algorithm: table})`,
selected by `samples.<sample>.format` (see FORMATS).

Output layout (`<sample_dir>` = `<output_dir>/<sample>` if `sample_subdirs`, else `<output_dir>`):
    <sample_dir>/events/data_<sample>_<index>_1.parquet           one file per task
    <sample_dir>/jets_<algorithm>/data_<sample>_<index>_1.parquet  one file per task, or, with
                                                                    `jets_per_file`:
    <sample_dir>/unchunked/jets_<algorithm>/data_<sample>_<index>_1.parquet   (per task)
    <sample_dir>/jets_<algorithm>/data_<sample>_<chunk>.parquet               (mode=rechunk)
"""

import glob
import importlib
import logging
import re
from pathlib import Path

import awkward as ak
import numpy as np

from fm_data.common import chunking as ch
from fm_data.common import jets as jt

log = logging.getLogger(__name__)

EVENTS_DIR = "events"
UNCHUNKED_DIR = "unchunked"
FORMATS = {
    "cld": "fm_data.ee_sim.cld",
    "aleph": "fm_data.lep.aleph",
}


def file_number(filepath, regex=r"_(\d+)\.root$"):
    """Number of the input file from its name: the first group of `regex`, followed by the
    further groups zero-padded to 3 digits. E.g. reco_p8_ee_Z_qq_ecm91_604833.root -> 604833
    (CLD, r"_(\\d+)\\.root$"), Y15224_13_AL.root -> 15224013 (ALEPH,
    r"^[A-Z]+(\\d+)_(\\d+)_AL\\.root$")."""
    match = re.search(regex, Path(filepath).name)
    if match is None:
        raise ValueError(f"Can't read a file number from {filepath} with {regex}")
    groups = match.groups()
    if any(int(group) > 999 for group in groups[1:]):
        raise ValueError(f"{filepath}: file number parts after the first must be < 1000")
    return int(groups[0] + "".join(f"{int(group):03d}" for group in groups[1:]))


def build_tasks(cfg):
    """One task per output file: {"sample", "index", "files"}, samples in config order."""
    tasks = []
    for sample, sample_cfg in cfg.samples.items():
        files = sorted(glob.glob(str(Path(sample_cfg.input_dir) / sample_cfg.file_pattern)))
        if cfg.max_files_per_sample is not None:
            files = files[: cfg.max_files_per_sample]
        if len(files) == 0:
            raise FileNotFoundError(
                f"No files matching {sample_cfg.file_pattern} in {sample_cfg.input_dir}"
            )
        numbers = [file_number(path, sample_cfg.file_number_regex) for path in files]
        if len(set(numbers)) != len(numbers):
            raise ValueError(f"Sample {sample}: file numbers are not unique (event_id would clash)")
        for index, start in enumerate(range(0, len(files), cfg.files_per_output)):
            tasks.append(
                {"sample": sample, "index": index, "files": files[start : start + cfg.files_per_output]}
            )
    return tasks


def sample_dir(cfg, sample):
    return Path(cfg.output_dir) / sample if cfg.get("sample_subdirs", False) else Path(cfg.output_dir)


def has_jet_tables(cfg, sample):
    """Jet tables are written for samples with truth objects (simulation, labelled jets) and
    for samples with `jet_tables: all` (data, all selected jets)."""
    sample_cfg = cfg.samples[sample]
    return sample_cfg.get("truth_objects") is not None or sample_cfg.get("jet_tables") == "all"


def output_paths(cfg, sample, index):
    """{"events": path, <jet algorithm>: path} of one task."""
    filename = f"data_{sample}_{index:05d}_1.parquet"
    base = sample_dir(cfg, sample)
    paths = {EVENTS_DIR: base / EVENTS_DIR / filename}
    if has_jet_tables(cfg, sample):
        jet_base = base / UNCHUNKED_DIR if cfg.get("jets_per_file") else base
        for name in cfg.jet_algorithms:
            paths[name] = jet_base / f"jets_{name}" / filename
    return paths


def process_file(filepath, cfg, sample):
    """Event table and jet tables {name: table or None} of one ROOT file."""
    sample_cfg = cfg.samples[sample]
    if sample_cfg.format not in FORMATS:
        raise ValueError(f"Unknown format {sample_cfg.format} of sample {sample} ({list(FORMATS)})")
    module = importlib.import_module(FORMATS[sample_cfg.format])
    number = file_number(filepath, sample_cfg.file_number_regex)
    return module.process_file(filepath, cfg, sample, number)


def check_event_link(event_table, jet_table):
    """Check that each jet constituent is the PFO at `part_event_index` of its event."""
    row_of_event = {event_id: row for row, event_id in enumerate(ak.to_numpy(event_table["event_id"]))}
    rows = np.array([row_of_event[event_id] for event_id in ak.to_numpy(jet_table["event_id"])])
    event_energy = event_table["part_energy"][rows]
    linked_energy = event_energy[jet_table["part_event_index"]]
    if not ak.all(linked_energy == jet_table["part_energy"]):
        raise RuntimeError("Jet constituents don't match the event PFOs at part_event_index.")


def _write(table, path):
    # write to a temporary file first, so that killed jobs don't leave partial files
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".parquet.tmp")
    ak.to_parquet(table, tmp_path, row_group_size=1024, compression="zstd")
    tmp_path.rename(path)


def process_task(task, cfg):
    """Process the files of one task and write its output files."""
    paths = output_paths(cfg, task["sample"], task["index"])
    if not cfg.overwrite and all(path.exists() for path in paths.values()):
        log.info(f"All output files of {task['sample']} {task['index']} exist - skipping.")
        return
    event_tables = []
    jet_tables = {name: [] for name in paths if name != EVENTS_DIR}
    for filepath in task["files"]:
        try:
            event_table, file_jet_tables = process_file(filepath, cfg, task["sample"])
        except Exception as ex:  # noqa: BLE001 - skip broken files, but keep going
            log.warning(f"Skipping {filepath}: {ex!r}")
            continue
        for name, table in file_jet_tables.items():
            if table is not None:
                check_event_link(event_table, table)
                jet_tables[name].append(table)
        event_tables.append(event_table)
    if len(event_tables) == 0:
        raise RuntimeError(f"None of the {len(task['files'])} input files could be read.")

    event_table = ak.concatenate(event_tables)
    _write(event_table, paths[EVENTS_DIR])
    log.info(f"Saved {len(event_table)} events to {paths[EVENTS_DIR]}")
    for name, tables in jet_tables.items():
        if len(tables) == 0:
            # rechunk_jets needs every task output, so a task without jets is an error to look
            # at, not something to skip silently
            raise RuntimeError(f"No {name} jets in {task['sample']} {task['index']}.")
        jet_table = ak.concatenate(tables)
        _write(jet_table, paths[name])
        labels = [field for field in jet_table.fields if field[len("label_"):] in jt.LABEL_CLASSES]
        counts = ", ".join(f"{label}: {int(ak.sum(jet_table[label]))}" for label in labels)
        log.info(f"Saved {len(jet_table)} {name} jets{f' ({counts})' if counts else ''} to {paths[name]}")


def rechunk_jets(cfg, tasks):
    """Rechunk the per-task jet tables of each sample into files of `jets_per_file` jets:
    <sample_dir>/jets_<algorithm>/data_<sample>_<chunk>.parquet. Refuses if a task's output is
    missing (resubmit the failed tasks first) or if the output directory is not empty."""
    jets_per_file = cfg.get("jets_per_file")
    if not jets_per_file:
        raise ValueError("mode=rechunk needs jets_per_file")
    for sample in cfg.samples:
        if not has_jet_tables(cfg, sample):
            continue
        sample_tasks = [task for task in tasks if task["sample"] == sample]
        missing = [
            str(path)
            for task in sample_tasks
            for path in output_paths(cfg, sample, task["index"]).values()
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"{len(missing)} task outputs of {sample} missing, e.g. {missing[0]}")
        for name in cfg.jet_algorithms:
            inputs = [str(output_paths(cfg, sample, task["index"])[name]) for task in sample_tasks]
            output_dir = sample_dir(cfg, sample) / f"jets_{name}"
            if output_dir.exists() and any(output_dir.iterdir()):
                if not cfg.overwrite:
                    raise FileExistsError(f"{output_dir} is not empty (overwrite=true to redo)")
                for path in output_dir.glob("*.parquet"):
                    path.unlink()
            paths = ch.rechunk(
                inputs, str(output_dir), prefix=f"data_{sample}", jets_per_file=int(jets_per_file)
            )
            log.info(f"{sample} {name}: {len(paths)} files of {jets_per_file} jets in {output_dir}")
