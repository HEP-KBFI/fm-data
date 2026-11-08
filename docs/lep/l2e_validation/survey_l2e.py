"""Per-file survey of the l2e-prod-v1 DATA files: events, empty events, beam energy
(RunInformation[0]), B field, BeamSpot availability, d0 sign w.r.t. the BeamSpot.

    python -I survey_l2e.py <prod dir> <out.csv> [--workers N] [--max-events N]
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import awkward as ak
import numpy as np
import uproot


def robust_sigma(x):
    if len(x) < 10:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return (q75 - q25) / 1.349


def survey(path, max_events):
    try:
        t = uproot.open(path)["events"]
        n_all = t.num_entries
        ri = t["RunInformation"].array(library="ak")
        ebeam = ak.to_numpy(ri[:, 0])
        bfield = ak.to_numpy(ri[:, 1])
        n_reco = ak.to_numpy(ak.num(t["RecoParticles.energy"].array()))
        runs = ak.to_numpy(ak.firsts(t["EventHeader.runNumber"].array()))
        bsx = ak.to_numpy(ak.fill_none(ak.firsts(t["BeamSpot.position.x"].array()), np.nan))
        bsy = ak.to_numpy(ak.fill_none(ak.firsts(t["BeamSpot.position.y"].array()), np.nan))
        bs_missing = ~np.isfinite(bsx) | ((bsx == 0) & (bsy == 0))
        ev = t.arrays(
            ["RecoParticles.tracks_begin", "RecoParticles.tracks_end", "_RecoParticles_tracks.index",
             "Tracks.trackStates_begin", "_Tracks_trackStates.D0", "_Tracks_trackStates.phi",
             "_Tracks_trackStates.omega", "BeamSpot.position.x", "BeamSpot.position.y"],
            entry_stop=max_events,
        )
        tb, te = ev["RecoParticles.tracks_begin"], ev["RecoParticles.tracks_end"]
        has = (te - tb) > 0
        rel = ak.fill_none(ak.pad_none(ev["_RecoParticles_tracks.index"], 1, axis=1), 0)
        sb = ak.fill_none(ak.pad_none(ev["Tracks.trackStates_begin"], 1, axis=1), 0)
        sidx = sb[rel[ak.where(has, tb, 0)]]
        def st(n):
            a = ak.fill_none(ak.pad_none(ev[f"_Tracks_trackStates.{n}"], 1, axis=1), 0)
            return ak.to_numpy(ak.flatten(a[sidx][has]))
        d0, phi, om = st("D0"), st("phi"), st("omega")
        bx = ak.broadcast_arrays(ak.fill_none(ak.firsts(ev["BeamSpot.position.x"]), np.nan), tb)[0]
        by = ak.broadcast_arrays(ak.fill_none(ak.firsts(ev["BeamSpot.position.y"]), np.nan), tb)[0]
        bx = ak.to_numpy(ak.flatten(bx[has]))
        by = ak.to_numpy(ak.flatten(by[has]))
        pt = 0.3 * 1.5 / np.maximum(np.abs(om) * 100, 1e-9)
        ok = (pt > 2) & np.isfinite(bx) & ~((bx == 0) & (by == 0))
        n_bs = -bx * np.sin(phi) + by * np.cos(phi)
        core_p = 10 * robust_sigma((d0 - n_bs)[ok])
        core_m = 10 * robust_sigma((-d0 - n_bs)[ok])
        return dict(
            file=path, events=n_all, empty_frac=float(np.mean(n_reco == 0)),
            mean_pfos=float(n_reco.mean()), ebeam_min=float(ebeam.min()), ebeam_max=float(ebeam.max()),
            ebeam_neg_frac=float(np.mean(ebeam < 0)), bfield_min=float(bfield.min()), bfield_max=float(bfield.max()),
            runs=len(np.unique(runs)), bs_missing_frac=float(np.mean(bs_missing)),
            d0core_filesign_mm=core_p, d0core_negated_mm=core_m, error="",
        )
    except Exception as e:  # report and continue
        return dict(file=path, error=repr(e)[:200])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("prod")
    p.add_argument("out")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-events", type=int, default=3000)
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.prod, "DATA", "*", "**", "*.root"), recursive=True))
    import pandas as pd
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(survey, files, [a.max_events] * len(files)))
    df = pd.DataFrame(rows)
    df["year"] = df.file.str.split("/DATA/").str[1].str.split("/").str[0]
    df.to_csv(a.out, index=False)
    print(f"{len(df)} files, errors: {(df.error.fillna('') != '').sum()}")
