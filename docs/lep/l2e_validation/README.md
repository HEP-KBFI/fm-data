# l2e-prod-v1 validation (2026-11-08)

Production: `/local/laurits/FM/l2e-prod-v1/{DATA/{1990..1995,1994_old},MC/1994}`.
Checklist: `~/ml-tau/code/docs/lep/CHANGES_since_June2025_production.txt`.
Converter source: `~/FM/code/l2e` (HEAD `6cbecbd`).

- `validate_l2e.py`: detailed checks per file. Logs: `validate_l2e.log` (1994_old ×2, 1994 ×2, 1990),
  `validate_l2e_2.log` (1990, 1991, 1992, 1993, 1995, MC QQB, MC ZMUMU); 10k events per file.
- `converter_issues.md`: the list of remaining problems of the files (for the converter authors).
- `survey_l2e.py`: all 347 DATA files (`survey_l2e.csv`): events, empty events, beam energy,
  B field, BeamSpot, d0 sign.

## Fixed bugs from the change log

| # | Bug | Status | Evidence |
|---|---|---|---|
| 1 | Jet constituents | **fixed** | every RecoParticle in exactly one jet, no duplicates or out-of-range indices; jet E = sum of constituent E to 3e-5 (all files) |
| 2 | V0 vertex x | **fixed** | x and y have the same spread; flight direction (vertex) vs V0 momentum: median cos = 0.9996 |
| 3 | Wire dE/dx status | **fixed** | wire and pad types differ (wires 0/2/3, pads 0/3/5; 1990 pads all type 4 = no pads); failed measurements = -999 with error -999, no other negative values |
| 4 | Reco mass | **fixed**, with one caveat | no NaN; π/e/μ mass for types 0/1/2; 0 for the rest. **Type 3 (charged V0-daughter tracks) gets mass 0**, deliberately ("zero mass by default"), even though they are charged hadrons |
| 5 | Cluster association | **fixed** | all RecoParticle→cluster links in range |
| 6 | Silent bank errors / -1 | no sign of them | apart from `RecoParticles.charge`, no branch has > 1 % entries equal to -1; no collection empty except `LuminousRegion` in DATA/1994 (ZD files) |
| 7 | Cluster phi | **fixed** | EM-cluster iPhi/iTheta agree with the linked PFO direction (median 2.6 mrad / 0.5–2.5 mrad) |

MC (only checked, not used for production): charges correct, generatorStatus filled, no PDG 0.
Differences from the change log: the cross sections are in `GeneratorInfo` (not
`GeneratorEventParameters`), `sqrts` there is 0 (QQB) or missing (ZMUMU), ZMUMU has no cross
sections, and **`trackMCLink`/`ecalClusterMCLink` are still written** (the change log says they
were removed; `main.cpp` writes them).

## Still true in the new files: the ntupelizers must keep these workarounds

- **Lengths in cm** (`D0`, `Z0`, reference points, vertices, BeamSpot, covariances). Omega in 1/cm
  reproduces pT within 1 % for 99 % of tracks. The source still says "unit check needed".
- **d0 sign opposite to LCIO** in every year: d0 w.r.t. the BeamSpot has a core of 0.15–0.37 mm
  negated vs 0.9–2.6 mm as stored, in all 347 files. sign(omega) = -charge.
- `RecoParticles.PDG` is 0 everywhere, so the PDG id still has to come from `ParticleID.type`.
- Tracks are stored about twice (Tracks / referenced = 1.8–2.3), so the particle→track link
  still has to go through `_RecoParticles_tracks`.
- Zero or one `Vertices` per event, missing in 60–90 % of events.

## New problems

1. **Beam energy (`RunInformation[0]`, RLEP) is unreliable outside 1992 and 1994.** It is in MeV
   (45572) in 1994, but:
   - 1990: 396k events have integer GeV values (44, 45, 46, 47, 50);
   - 1990–1992 and 1995: -1 or -1000 (~0.3–2 % of events);
   - 1991/1993: 15.0, 35750, 50000, 50250; 1995: 20200 (a few runs).

   Both ntupelizers compute sqrt_s = 2·RunInformation[0]/1000. lep-data's particle cut
   (E < E_beam) then drops these events completely.
2. **B field (`RunInformation[1]`)** is 1.1–23.2 kG for some runs in 1991 and 1993 (otherwise
   15.0–15.2).
3. **Empty events**: 57 % (1994) to 86 % (1991) of the events have no RecoParticles. These are
   non-hadronic triggers, not a converter bug; the ntuples drop them.
4. Cluster `position` is not filled (0, 0, 0); only iTheta/iPhi are. HadronicClusters are never
   linked to a PFO, and neutral hadrons (type 5) have no cluster link.
5. A `BeamSpot` collection is now present (cm, per block of events). It agrees with fm-data's
   per-run fit to ~20–30 µm, so it could replace that fit. It is (0, 0) with zero covariance for
   some 1992 runs. The covariance is the position error, not the beam size.
