import glob
import os
from pathlib import Path

import hydra
from jinja2 import Environment, FileSystemLoader
from omegaconf import DictConfig

from lep_data.aleph.tools import create_aleph as na  # noqa: F401

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

orchestration_dir = Path(__file__).parent.parent      # lep_data/aleph
repo_root = orchestration_dir.parent.parent           # the checkout
jinja_env = Environment(loader=FileSystemLoader(orchestration_dir / "templates"))


def submit_slurm_job(
    input_paths: str,
    output_dir: str,
    output_level: str,
    idx: int,
    jets_per_file: int,
    row_group_size: int,
    jet_R: float,
) -> str:
    """Write the SLURM job script for processing a chunk of input files.

    The job streams its files into output chunks of jets_per_file jets. Its
    output filenames carry a per-job prefix, since the jobs run concurrently
    and cannot share a file index; scripts/rechunk.py consolidates the leftover
    tail files into a single uniform series once the jobs are done.
    """
    job_dir = os.path.join(output_dir, "submission_scripts")
    err_dir = os.path.join(output_dir, "error_files")
    out_dir = os.path.join(output_dir, "out_files")
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    # Create job script
    job_script_path = os.path.join(job_dir, f"chunk_{idx}.sh")

    input_paths_str = ",".join(input_paths)

    template = jinja_env.get_template("ntupelize_files.sh.j2")
    job_script_content = template.render(
        job_name="aleph_ntupelizer",
        job_dir=job_dir,
        err_dir=err_dir,
        out_dir=out_dir,
        partition="main",
        walltime="2-00:00:00",
        memory=8000,
        cpus=1,
        ntasks=1,
        working_dir=repo_root,
        # Derived from this file's location rather than hardcoded, so the
        # generated job scripts follow the checkout they were written from.
        environment_script=repo_root / "run.sh",
        processing_script=orchestration_dir / "scripts" / "ntupelize_list.py",
        input_paths=input_paths_str,
        output_dir=output_dir,
        prefix=f"job{idx:04d}",
        output_level=output_level,
        jets_per_file=jets_per_file,
        row_group_size=row_group_size,
        jet_R=jet_R,
    )

    with open(job_script_path, "wt") as f:
        f.write(job_script_content)


def split_list_into_chunks(lst, num_chunks=20):
    chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        # Add 1 extra element to first 'remainder' chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    input_dir = cfg.input_dir
    output_dir = cfg.output_dir
    num_chunks = cfg.num_chunks
    output_level = cfg.output_level
    jets_per_file = cfg.get("jets_per_file", 100_000)
    row_group_size = cfg.get("row_group_size", 1024)
    jet_R = cfg.get("jet_R", 0.4)

    input_wcp = os.path.join(input_dir, "*", "data_*.root")
    input_paths = list(glob.glob(input_wcp))
    if num_chunks > 0:
        job_chunks = split_list_into_chunks(input_paths, num_chunks=num_chunks)
    else:
        job_chunks = [[input_path] for input_path in input_paths]
    for i, input_chunk in enumerate(job_chunks):
        submit_slurm_job(
            input_paths=input_chunk,
            output_dir=output_dir,
            output_level=output_level,
            idx=i,
            jets_per_file=jets_per_file,
            row_group_size=row_group_size,
            jet_R=jet_R,
        )


if __name__ == "__main__":
    main()
