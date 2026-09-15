# lep-data

ALEPH (LEP) ntuple production for the ML-tau project. Reads the ALEPH EDM4HEP
ROOT files, clusters jets, computes jet and event variables, and writes flat
Parquet ntuples in uniform chunks ready for ML training.

Ported out of `ml-tau-data`, where this lived under `ntupelizer/aleph/`. It is a
self-contained pipeline: it shares no code with the FCC-ee/CLD workflow beyond
the kinematic helpers now in `lep_data/tools/features.py`.

## Setup

The heavy scientific stack (uproot, awkward, vector, fastjet, numba) lives in
the Apptainer image, not on the host:

```
/home/software/singularity/pytorch.simg:2025-09-01
```

`run.sh` wraps it, putting the checkout on `PYTHONPATH` so `lep_data` imports
without installing anything:

```bash
./run.sh python3 lep_data/aleph/scripts/rechunk.py --help
```

For an editable install outside the container (e.g. for a notebook kernel that
already has the stack):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Configuration

`lep_data/aleph/config/config.yaml`:

```yaml
input_dir: /local/laurits/ALEPH/1994_old/LAST
output_dir: /local/laurits/ALEPH/ALEPH_jet
num_chunks: 189            # SLURM jobs to split the input files over; 0 = one job per file
output_level: jet          # "jet" (one row per jet) or "event" (one row per event)
jets_per_file: 100000      # jets per output file
row_group_size: 1024       # rows per parquet row group
```

`output_level: jet` gives one row per jet, which is what the chunked training
dataset is built from. `output_level: event` gives one row per event instead,
with the jets of the event in list columns.

## Running

Write the per-chunk SLURM job scripts, then submit them:

```bash
./run.sh python3 lep_data/aleph/scripts/ntupelize_all.py     # writes <output_dir>/submission_scripts/chunk_*.sh
bash lep_data/aleph/scripts/submit_all.sh -s <output_dir>/submission_scripts
```

The generated scripts point back at the checkout they were written from —
`ntupelize_all.py` derives both the `run.sh` path and the `ntupelize_list.py`
path from its own location, so a second checkout does not need editing.

Each job streams the ROOT files assigned to it straight into chunk files, with
no per-ROOT-file intermediate. Because the jobs run concurrently they cannot
share a file index, so their filenames carry a per-job prefix
(`job0000_00000.parquet`) and each job's last file holds only its leftovers.

### Rechunking

To turn those ragged tails into one uniform series:

```bash
./run.sh python3 lep_data/aleph/scripts/rechunk.py \
    -i /local/laurits/ALEPH/ALEPH_jet \
    -o /local/laurits/ALEPH/ALEPH_jet_chunked
```

`rechunk.py` streams a row group at a time and writes each output file once, so
memory is bounded by one output file. It reads the level (jet or event) off the
input schema, refuses to overwrite its own inputs, and fails rather than
reporting success if the row count in does not match the row count out. Pass
`--jets-per-file`, `--row-group-size` or `--prefix` to override the defaults.

Rows are never split across files, so at jet level the output is exactly
`jets_per_file` rows per file; an event-level dataset instead lands on a
whole-event boundary just under the target.

## Repository structure

```
run.sh                                  # apptainer wrapper; also used by the job scripts
setup.py
lep_data/
  tools/
    features.py                         # angular-distance helpers (shared with ml-tau-data)
  aleph/
    config/config.yaml                  # input/output dirs, chunking, output level
    templates/ntupelize_files.sh.j2     # SLURM job script template
    scripts/
      ntupelize_all.py                  # writes the per-chunk SLURM job scripts
      ntupelize_list.py                 # one job: ROOT files -> chunk files
      rechunk.py                        # rechunk existing .parquet into uniform files
      submit_all.sh                     # sbatch every generated job script
    tools/
      create_aleph.py                   # ROOT -> jets -> flat ntuple
      jet_variable_calculations.py      # substructure: SoftDrop, N-subjettiness, reclustering
      event_variable_calculations.py    # event-level variables
      chunking.py                       # JetChunkWriter: fixed-size parquet chunks
    notebooks/ALEPH_data_overview.ipynb
```

## Known issues with the ALEPH data

Carried over from the original notes; `create_aleph.py` works around these:

- Some constituents have a p_T of -1e3 — recoParticles need cleaning.
- Some jets have 0 particles, so particles are assigned to jets by angle.
- Because of that, `nparticles` has to be taken from the angle-associated
  particles rather than from the file.
- For some constituents the eta value is infinite / NaN.
- Some jets have an infinite eta value (from the EDM4HEP files).
