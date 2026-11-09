# Run quality of the ALEPH data (2026-11-08)

`run_quality.py` checks every run in l2e-prod-v1 DATA against ALEPH's own bookkeeping and
against the data itself:

- **Bookkeeping:** the SCANBOOK good-run lists in `~/FM/code/l2e/database/data`. Per run they
  give the run quality (PERF/MAYB), the physics groups that accept the run, n(Z→hadrons), the
  LEP energy and, from 1994, the quality of each detector.
- **ROOT files:** the class word (`ClassBitset`, where class 16 is the ALEPH hadronic selection)
  and the run-header B field.
- **Ntuples:** per run, for class-16 events, ⟨E_vis⟩/√s, ⟨n charged⟩, the field the
  reconstruction used (B_eff = median pT·|ω|/0.2998), and the dxy core for tracks with
  pT > 2 GeV.

Outputs: `runs.csv` (one row per run and year), `summary.json` and `run_quality.png`.

## Result: the data are already good runs only

| Year | Runs | Class 16 | In good-run list | PERF | Off peak | Bad header B |
|---|---|---|---|---|---|---|
| 1990 | 530 | 165,672 | all | 43 % | 26 % | – |
| 1991 | 986 | 297,620 | all | 83 % | 20 % | 3 runs (591 ev.) |
| 1992 | 1446 | 725,630 | all | 53 % | 0 | – |
| 1993 | 1348 | 688,947 | all | 56 % | 31 % | 2 runs (497 ev.) |
| 1994 / 1994_old | 2691 | 1,755,945 | all | 54 % | 0 | – |
| 1995 | 706 | 767,594 | not covered (list starts at run 37854) | – | ? | – |

The table columns:
- **In good-run list:** every run in 1990–1994 is in the list. The class-16 counts agree with
  SCANBOOK n(Z→hadrons) to within 0.1–1.5 %, so the runs are also complete.
- **PERF:** the remaining runs are MAYB. MAYB is still a good run, and it accounts for most of
  the HE/HD/SH-approved data. The physics groups accept 97–100 % of the class-16 events for
  multihadronic searches (SH), and from 1991 83–100 % for VDET analyses (VD).
- **Off peak:** class-16 events with a SCANBOOK LEP energy more than 0.5 GeV from 91.2 GeV
  (Z line-shape scan points at about ±1 and ±2 GeV). Their `event_sqrt_s` = 91.2 is therefore
  wrong by up to 2 %. 1995 also had a scan, but SCANBOOK does not cover those runs.
- **Bad header B:** `RunInformation[1]` is 1.1–23 kG for these 5 runs, but B_eff = 1.500 T. The
  reconstruction used the right field, so only the stored header value is wrong. The momenta
  are fine, and so is everything else in those runs.

**B_eff** is 1.500–1.502 T in every run. The one exception is 1994 run 28125 at 1.518 T, where
the header also says 15.18 kG, so that run really had a 1 % higher field and is consistently
reconstructed.

**Detector quality in 1994** (shares of class-16 events):

| Detector | Dead (D) | MAYB (M) |
|---|---|---|
| VDET | 0.6 % | 0.1 % |
| ITC | 0 | 0.6 % |
| TPC | 0 | 11 % |
| ECAL | 0 | 10 % |
| HCAL | 3.6 % | 31 % |
| SICAL | 2.3 % | 14 % |
| LCAL | 1.7 % | 6 % |

**Outliers in the data:**
- ⟨n charged⟩ and ⟨E_vis⟩ are flat in all years.
- About 55 runs (≈ 10k class-16 events in all years) have a dxy core above 0.4 mm instead of
  0.15–0.2 mm, which means VDET was off or not aligned. Examples: 1991 12502, 12710; 1994 26097,
  27776; 1995 runs near 35500 and 37465/6. Two of these runs (1991 12502 and 1994 26097) also
  have low ⟨n charged⟩, which points to a tracking problem.
- The class-16 fraction varies by a factor of 2–3 within a year. It follows trigger and
  background conditions, and the class-16 counts match SCANBOOK.

## Conclusion

No run selection is needed beyond what the production already contains. For analyses that use
impact parameters, drop the runs with a large dxy core (`runs.csv`, `dxy_core > 0.4`) and, in
1994, the runs with VDET = D. If √s matters, per-run LEP energies for 1990–1994 are in
`runs.csv` (`sb_energy`). The 1995 scan points are not in SCANBOOK.
