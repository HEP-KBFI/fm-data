"""Ntupelize EDM4hep ROOT files to event and jet parquet files.

Configs: fm_data/ee_sim/config/cld.yaml (default), fm_data/lep/config/aleph.yaml:
    ./run.sh python3 scripts/ntupelize.py [overrides]                                   # CLD
    ./run.sh python3 scripts/ntupelize.py --config-path ../fm_data/lep/config --config-name aleph
Modes: run (default), n_tasks, setup, rechunk (jet tables into files of jets_per_file jets).
"""

import logging
import os
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from fm_data.common.ntupelizing import build_tasks, process_task, rechunk_jets

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

log = logging.getLogger(__name__)


def resolve_task_id(task_id):
    """The task to process: `task_id`, else $SLURM_ARRAY_TASK_ID, else None (all tasks)."""
    if task_id is not None:
        return int(task_id)
    if os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    return None


@hydra.main(version_base=None, config_path="../fm_data/ee_sim/config", config_name="cld")
def main(cfg: DictConfig) -> None:
    tasks = build_tasks(cfg)
    if cfg.mode == "n_tasks":
        print(len(tasks))
        return
    if cfg.mode == "setup":
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, Path(cfg.output_dir) / "ntupelize_config.yaml")
        log.info(f"{len(tasks)} tasks, config saved to {cfg.output_dir}/ntupelize_config.yaml")
        return
    if cfg.mode == "rechunk":
        rechunk_jets(cfg, tasks)
        return
    if cfg.mode != "run":
        raise ValueError(f"Unknown mode {cfg.mode} (use run, n_tasks, setup or rechunk)")

    task_id = resolve_task_id(cfg.task_id)
    if task_id is not None:
        if task_id >= len(tasks):
            log.warning(f"There is no task {task_id} ({len(tasks)} tasks) - nothing to do.")
            return
        tasks = [tasks[task_id]]
    for task in tasks:
        start = time.time()
        log.info(f"Task {task['sample']} {task['index']}: {len(task['files'])} files")
        process_task(task, cfg)
        log.info(f"Finished in {time.time() - start:.0f} s.")


if __name__ == "__main__":
    main()
