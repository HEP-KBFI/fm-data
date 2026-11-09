# Validation of the ALEPH ntuples (2026-11-08)

Production: `/local/laurits/20261108_ALEPH_ntupelized` (l2e-prod-v1 DATA). Plots made by
`validate_ntuples.py` from 10 files of 1994_old and 3 files of every other year; CLD Z→qq is
the reference for the signed impact parameters. All numbers are in `summary.json`.

| Plot | Check | Result |
|---|---|---|
| `01_events.png` | event variables per year | all years agree; low-multiplicity (leptonic, γγ) + hadronic events, no Z selection |
| `02_pfos.png` | PFO kinematics, types, masses | type fractions stable; masses per type correct (type 3 = π mass); every charged PFO has a track, no neutral one |
| `03_reference_point.png` | IP reference point | x, y from BeamSpot in ≥ 99.85 % of events, following the runs; fitted beam size ~120 × 60 µm (1990: ~300 µm, no VDET) |
| `04_impact_parameters.png` | d0 w.r.t. reference | 1994_old core 152 µm as stored vs 1411 µm with the d0 sign flipped (all years: 151–180 µm, 1990: 373 µm); pull core 1.07–1.14 (1990: 0.95) |
| `05_signed_impact_parameters.png` | lifetime sign | positive excess in all years: R=0.4, 1994_old 10.5 % > 3σ vs 4.0 % < −3σ; heavier jets have a longer positive tail |
| `06_jets.png`, `07_two_jet_events.png` | jets | 1 < E ≤ 91.2 GeV, \|η\| < 3, no constituent ≥ 45.6 GeV; 2-jet E₁+E₂ peaks at √s |
| `summary.json` → `chunks` | rechunked jet files | every file except the last per year has exactly 100k jets; totals = unchunked totals, all years and both algorithms |

Found and fixed with these plots: the signed impact parameters first used the 3D dot product,
which for the transverse ALEPH PCA lets dz decide the sign (N(>3σ)/N(<−3σ) = 1.6 instead of
2.6). The jet tables are now signed in the transverse plane (`ip_signing="transverse"`,
`fm_data/common/jets.py`), and the production was rerun.
