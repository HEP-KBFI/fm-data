# fm-data

Ntupelization of EDM4hep ROOT files into parquet files for the event-level foundation model
and its downstream jet-level tasks (gabbro, branch `event-level`).

Supported inputs:

- **LEP:** ALEPH data from the LEP data revival (l2e production);
- **e+e- simulation:** CLD (FCC-ee).

This repository was `lep-data` (ALEPH only, a split from ml-tau-data); on 2026-11-08 fm-data
(CLD) was merged into it, and its ALEPH pipeline (`lep_data/aleph`, `create_aleph.py`) was
replaced by the common ntupelizer described here.

```
fm_data/
  common/     the ntupelizer, shared by all inputs:
                ntupelizing.py      tasks, output files, format dispatch, rechunk step
                events.py           event table (PFOs sorted by energy, event variables)
                event_variables.py  thrust, sphericity, ... (same definitions as in gabbro)
                jets.py             jet clustering (fastjet ee_genkt) and jet tables
                lifetime.py         impact parameters (PCA, errors, signing w.r.t. a jet)
                chunking.py         rechunking parquet files into N jets per file
  lep/        ALEPH: reader aleph.py, config/aleph.yaml, notebooks/
  ee_sim/     CLD: reader cld.py and reading.py, MC truth truth.py, config/cld.yaml
scripts/      ntupelize.py (hydra entry point), rechunk.py (stand-alone rechunking)
docs/lep/     ALEPH: l2e_validation/ (validation of l2e-prod-v1, converter_issues.md),
              aleph_ip_validation.md (impact parameters), CHANGES_since_June2025_production.txt
              (converter change log), and the older lep-data notes (dR cut, overview)
tests/        ./run.sh python3 -m pytest tests (unit tests, one CLD and one ALEPH file end to end)
run.sh        runs a command in the container
submit.sh     SLURM submission (array + rechunk job)
```

pp data or simulation can be added as another domain (e.g. `fm_data/pp/`): a reader module
with `process_file(filepath, cfg, sample, file_number)`, which returns the event table and the
jet tables `{algorithm: table}` of one ROOT file, an entry in `FORMATS`
(`fm_data/common/ntupelizing.py`), and a config in `fm_data/pp/config/`. The config's
`samples.<sample>.format` selects the reader; everything else (tasks, output files, SLURM,
rechunking) is shared.

Each ROOT file is read once and produces:

- an **event table**: one row per event, with all reconstructed particles (PFOs) of the
  event, sorted by energy. This is the input for the event-level tokenizer and backbone.
- one **jet table per jet algorithm**: all PFO variables of the constituents, variables
  relative to the jet, and the link back to the event table. In simulation only the jets
  matched to a generator-level object (b, c, s, ud quark or hadronic tau) are kept and
  labelled; in data (ALEPH) all selected jets are kept, without labels.

## Running

Everything runs in `/home/software/singularity/pytorch.simg:2025-09-01` through `./run.sh`, which
puts the checkout on `PYTHONPATH` (no installation needed). The CLD files need uproot >= 5.6,
which gabbro's `gabbro-latest.sif` (uproot 5.1.2) doesn't have.

```shell
# CLD (default config, fm_data/ee_sim/config/cld.yaml): a quick test, then the production
./run.sh python3 scripts/ntupelize.py output_dir=/scratch/persistent/$USER/test \
    max_files_per_sample=4 files_per_output=2
./submit.sh                                   # or: CONFIG=ee_sim/cld ./submit.sh

# ALEPH (fm_data/lep/config/aleph.yaml): one file of 1994_old (task 5 of 7 with one file
# per year), then the production
./run.sh python3 scripts/ntupelize.py --config-path ../fm_data/lep/config --config-name aleph \
    output_dir=/scratch/persistent/$USER/test max_files_per_sample=1 task_id=5
CONFIG=lep/aleph ./submit.sh
CONFIG=lep/aleph ./submit.sh output_dir=/local/$USER/other slurm.max_concurrent=50

# only the rechunking of the jet tables (e.g. after resubmitting failed tasks)
./run.sh python3 scripts/ntupelize.py --config-path ../fm_data/lep/config --config-name aleph \
    mode=rechunk

# rechunk any existing jet-level parquet files
./run.sh python3 scripts/rechunk.py -i <dir|glob|file> -o <output_dir> --jets-per-file 100000

# tests
./run.sh python3 -m pytest tests
```

`submit.sh` (`CONFIG=<domain>/<name>`, i.e. `fm_data/<domain>/config/<name>.yaml`):

1. saves the config to `<output_dir>/ntupelize_config.yaml` (`mode=setup`);
2. submits one SLURM array task per output file (`mode=run task_id=...`);
3. if the config has `jets_per_file`, submits a `mode=rechunk` job that runs once the array
   is done (`--dependency=afterany`).

Tasks whose output files exist are skipped (`overwrite=true` redoes them), so a failed array
can simply be resubmitted. The rechunk refuses to run while a task output is missing, or if its
output directory is not empty (`overwrite=true`). Files are written to `*.tmp` first and
renamed once complete. Logs go to `logs/slurm/`.

Config settings (any value can be overridden on the command line, hydra):

- `samples`: per sample
  - `format`: `cld` or `aleph`;
  - `input_dir`, `file_pattern`;
  - `file_number_regex`: the file number is the first group followed by the further groups,
    zero-padded to 3 digits;
  - `events_per_file_max`: `event_id` = file number × this + entry;
  - `sqrt_s`;
  - `truth_objects` (simulation: `quarks` or `hadronic_taus`, labelled jet tables) or
    `jet_tables: all` (data: all selected jets);
  - `label` (ALEPH): name of the process label column, the same for all years.
- `files_per_output`: ROOT files per task (= per event-table file).
- `max_files_per_sample`: only the first N (sorted) files of each sample, for tests.
- `sample_subdirs`: write each sample to `<output_dir>/<sample>/`. Used for ALEPH, with one
  sample per year.
- `jets_per_file`: rechunk the jet tables of each sample into files of exactly this many jets;
  the last file of a sample holds the rest. The per-task jet tables stay in
  `<sample>/unchunked/`.
- `jet_algorithms`: fastjet `ee_genkt` definitions (`p`, `R`, `n_exclusive_jets`, and
  `max_match_angle` for the truth matching). Both configs have inclusive ee anti-kt with
  R = 0.4 (`ee_antikt_R04`) and exclusive ee Cambridge/Aachen with 2 jets
  (`ee_cambridge_exclusive2`).
- `jet_selection`: E > 1 GeV and |η| < 3. Always applied as well: E_jet <= √s, pt > 0, and no
  constituent with E >= √s/2.
- `mode`: `run` (default), `n_tasks` (print the number of tasks), `setup` (save the config) or
  `rechunk`.

## CLD (fm_data/ee_sim)

Input: the Z→qq and Z→ττ samples at 91 GeV
(`/local/joosep/mlpf/cld/v1.3.0_key4hep_2026-04-08/gen/`), 100 events per file, 500 files per
output file. `event_sqrt_s` is the configured 91.0 GeV, checked against the beam particles
(the CLD Z-pole samples are generated at exactly 91.0 GeV).

```
<output_dir>/ntupelize_config.yaml
<output_dir>/events/data_<sample>_<index>_1.parquet
<output_dir>/jets_ee_antikt_R04/data_<sample>_<index>_1.parquet
<output_dir>/jets_ee_cambridge_exclusive2/data_<sample>_<index>_1.parquet
```

The reader takes the PFOs from `PandoraPFOs`, the tracks from `SiTracks_Refitted` (AtIP
state), and the primary vertex from `PrimaryVertices`. The impact parameters are computed
w.r.t. the primary vertex.

Speed: about 2.2 s per ROOT file (100 events) for all outputs. The 2 × 50k Z-pole files take
about 60 CPU hours.

## ALEPH (fm_data/lep)

Input: the l2e production `/local/laurits/FM/l2e-prod-v1/DATA/{1990..1995,1994_old}` (MC is not
ntupelized), validated in `docs/lep/l2e_validation/`. The bugs listed in the change log are
fixed there; the remaining issues of the files are in `converter_issues.md`. Each year is its
own sample and is written to its own directory; years are never mixed:

```
<output_dir>/ntupelize_config.yaml
<output_dir>/ALEPH<year>/events/data_ALEPH<year>_<index>_1.parquet                    one per ROOT file
<output_dir>/ALEPH<year>/jets_ee_antikt_R04/data_ALEPH<year>_<chunk>.parquet           100k jets each
<output_dir>/ALEPH<year>/jets_ee_cambridge_exclusive2/data_ALEPH<year>_<chunk>.parquet 100k jets each
<output_dir>/ALEPH<year>/unchunked/jets_<algorithm>/data_ALEPH<year>_<index>_1.parquet per ROOT file
```

Here `<year>` is one of 1990, 1991, 1992, 1993, 1994, 1994_old or 1995. The current
production is in `/local/laurits/20261108_ALEPH_ntupelized`. The rows are in input order (by
file, then entry), not shuffled.

What the reader does about the files (details in the docstring of `fm_data/lep/aleph.py`):

- **Units and sign:** lengths are converted from cm (what the files contain) to mm, and the
  d0 sign is flipped from the ALEPH convention to LCIO.
- **Track link:** the PFO→track link goes through `_RecoParticles_tracks`, since the tracks
  are stored about twice.
- **Particle type and mass:** the PDG ids and particle-type flags come from `ParticleID.type`
  (`RecoParticles.PDG` is always 0). Types 7 and 8 (forward calorimeter objects) count as
  photons. Type 3 (a charged track from a V0) counts as a charged hadron and gets the pion
  mass, which the files don't set.
- **Reference point of the impact parameters:**
  - x, y come from the ALEPH `BeamSpot` collection, or from a per-run fit of d0 vs φ0 where it
    is missing (some 1992 runs; `event_ref_from_beamspot = 0`).
  - z is the median z of the tracks of the event at their transverse PCA.
  - `part_dxy` is the transverse (2D) distance, and the beam size (from the fit) is added to
    its error.
  - The ALEPH event vertex (`event_vertex_*`) is kept but not used: it is missing in 60–90 %
    of the events.
- **√s** is the nominal 91.2 GeV for all events, because the beam energy in `RunInformation`
  is in GeV, missing or meaningless for parts of 1990–1993 and 1995.

Extra ALEPH columns:

- event columns: `event_ref_{x,y,z}`, `event_ref_from_beamspot`, `event_beam_sigma_{x,y}`
  (beam size), `event_run_number`;
- PFO column: `part_pid_type` (ALEPH energy-flow type).

The process label is `label_ALEPH = 1`.

The column names are those of the tables below, and they differ from the earlier lep-data
jet ntuples (e.g. `part_dxy_err` instead of `part_dxyerr`). Those ntuples also had
`jet_tau1..4`, `jet_sdmass` and `part_etarel_R`, which the new jet tables don't have.

57–86 % of the input events have no PFOs (non-hadronic triggers); they are not in the event
tables. Speed: 10–35 s per ROOT file (up to ~160k events); the jobs request 30 GB memory.

## Event table

| Columns | Content |
|---|---|
| `event_id` | file number × `events_per_file_max` + entry (CLD: × 10⁴; ALEPH: × 10⁶, e.g. Y15224_13_AL.root → file number 15224013); unique within a sample |
| `event_file_number`, `event_entry`, `event_number` | input file number, entry in the file, `EventHeader.eventNumber` |
| `event_sqrt_s` | √s from the config [GeV] |
| `event_vertex_{x,y,z}`, `event_has_vertex` | primary vertex [mm]; (0, 0, 0) and `has_vertex = 0` if none was reconstructed |
| `event_nparticles`, `event_energy_vis`, `event_mass_vis`, `event_p_vis`, `event_thrust`, `event_sphericity`, `event_aplanarity`, `event_leading_energy_fraction`, `event_num_*` | event variables (same definitions as in gabbro) |
| `label_<sample>` | process label, one-hot over the samples of the config (CLD: `label_Zqq`, `label_Ztautau`; ALEPH: `label_ALEPH`) |
| `part_px/py/pz/energy/mass/charge/pdg` | PFO four-momentum [GeV], charge, PDG id |
| `part_p/pt/theta/phi/eta` | derived kinematics |
| `part_isElectron/isMuon/isPhoton/isChargedHadron/isNeutralHadron` | particle type (CLD: from the PDG id; ALEPH: from the energy-flow type) |
| `part_pfo_index` | index in the PFO collection of the file (`PandoraPFOs`, ALEPH `RecoParticles`) |
| `part_has_track` | 1 if the PFO has a track |
| `part_dxy`, `part_dz`, `part_d3` + `_err`, `_sig` | impact parameters w.r.t. the reference point (CLD: primary vertex, 3D PCA; ALEPH: beam spot and median track z, 2D PCA) [mm], errors and significances (value / error). Unsigned; 0 for PFOs without a track |
| `part_d0`, `part_z0` + `_err`, `part_phi0`, `part_tanL`, `part_omega` | track parameters at the IP (LCIO convention, mm) |
| `part_pca_{x,y,z}` | point of closest approach to the reference point [mm], used to sign the impact parameters w.r.t. any jet axis |

The PFO order (energy, descending; equal energies keep the file order) is the order of the
event-level token sequences. The jet tables refer to positions in it.

## Jet tables

Each row holds the event columns of its event (repeated per jet), and:

| Columns | Content |
|---|---|
| `jet_index` | position of the jet among the event's jets (by energy, before the selection) |
| `jet_pt/eta/phi/theta/energy/mass/px/py/pz`, `jet_*_from_p4s`, `jet_nparticles`, `jet_charge` | jet kinematics |
| `part_*` (all event-table PFO columns) | the constituents, sorted by pt |
| `part_event_index` | **position of the constituent in the event's PFO list (link to the event table)** |
| `part_ptrel`, `part_erel`, `part_etarel`, `part_phirel`, `part_deta`, `part_dphi`, `part_deltaR`, `part_angle` | constituent kinematics relative to the jet (`part_angle`: 3D opening angle) |
| `part_{dxy,dz,d3}_signed` + `_sig` | impact parameters signed w.r.t. the jet axis: positive if the PCA lies in the jet direction from the reference point |
| `label_b`, `label_c`, `label_s`, `label_ud`, `label_tau` | simulation only: one-hot label from the matched truth object |
| `jet_truth_pdg`, `jet_truth_angle`, `jet_truth_energy` | simulation only: matched quark or tau (signed PDG id), its opening angle to the jet, its (visible) energy |
| `jet_n_bhadrons`, `jet_n_chadrons` | simulation only: weakly decaying b/c hadrons in the jet (cross-check of the labels) |

Labels (simulation): each truth object is matched to the closest selected jet within
`max_match_angle` (0.4 rad for anti-kt, 0.8 rad for Cambridge). Only matched jets are
kept, and a jet that is the closest jet of two objects is dropped.

- **Quarks:** the outgoing quarks of the hard process (Pythia status 23). u and d are merged
  into `ud`.
- **Hadronic taus:** the last copy of each tau whose decay products include no electron or
  muon. They are matched with their visible momentum (tau minus neutrinos).

The ParT inputs can all be computed from these columns: per-constituent kinematics, charge,
particle type and impact parameters, and the pairwise features from the four-momenta.

## Notes

- `fm_data/common/lifetime.py` is copied from ml-tau-data (commit 83f6b1af83) with **one
  fix**: the track origin used the opposite d0 sign convention to LCIO/EDM4hep, which
  mirrored the point of closest approach through the reference point.
  - |dxy| was almost unaffected, but the lifetime sign w.r.t. the jet axis was essentially
    random: of the b-jet tracks with large significance, 54% were positive before and about
    80% after.
  - The signed impact parameters made with ml-tau-data are affected by this.
- The impact-parameter errors come from the track covariance only (no vertex error) and a
  straight-line approximation of the track near the vertex. The significances have
  non-Gaussian tails: in light jets about 3% of tracks have a signed significance below −3.
- ALEPH: the ntuples made before 2026-11-05 have the wrong d0 sign (see
  `docs/lep/aleph_ip_validation.md`). 1990 has no vertex detector, so its impact-parameter
  resolution is worse (d0 core about 0.37 mm, vs 0.15 mm in 1994).
- CLD has a 30 mrad beam crossing angle, so the events are slightly boosted along x. This
  is not corrected.
