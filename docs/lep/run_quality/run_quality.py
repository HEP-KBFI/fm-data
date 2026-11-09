"""Run quality of the ALEPH data in l2e-prod-v1 and in the ntuples made from it.

For every run:
  - from the ROOT files: number of events, number of class-16 events (ALEPH hadronic selection,
    bit 16 of the class word KCLASW = `ClassBitset`), B field from the run header
    (`RunInformation[1]`, MBFL, kG);
  - from the ntuples (joined event by event through event_entry): for class-16 events the mean
    visible energy and charged multiplicity, the field the reconstruction actually used,
    B_eff = median(pT * |omega|) / 0.2998 (pT of the PFO, omega of its track), and the core
    resolution of dxy for tracks with pT > 2 GeV (sensitive to the VDET being off);
  - from the ALEPH bookkeeping (SCANBOOK good-run lists in ~/FM/code/l2e/database/data): whether
    the run is a listed good run, its run quality (PERF/MAYB), the physics groups that accept it,
    the number of hadronic Z (n_z0), LEP energy and, from 1994, the per-detector quality.

    python -I run_quality.py collect <out.csv> [--workers N]
    python -I run_quality.py analyze <runs.csv> <out dir>
"""

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import uproot

PROD = "/local/laurits/FM/l2e-prod-v1/DATA"
NTUPLES = "/local/laurits/20261108_ALEPH_ntupelized"
SCANBOOK = os.path.expanduser("~/FM/code/l2e/database/data")
YEARS = ["1990", "1991", "1992", "1993", "1994", "1994_old", "1995"]
FILE_RE = re.compile(r"^[A-Z]+(\d+)_(\d+)_AL\.root$")
SQRT_S = 91.2
DETECTORS = ["vdet", "itc", "tpc", "ecal", "hcal", "sical", "lcal", "trigger", "daq", "run_coord"]


def file_number(name):
    m = FILE_RE.match(name)
    return int(m.group(1) + m.group(2).zfill(3))


def robust_sigma(x):
    if len(x) < 30:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return (q75 - q25) / 1.349


def ntuple_map(year):
    """file_number -> events parquet of the ntuples."""
    out = {}
    for path in glob.glob(f"{NTUPLES}/ALEPH{year}/events/*.parquet"):
        pf = pq.ParquetFile(path)
        out[int(pf.read_row_group(0, columns=["event_file_number"])["event_file_number"][0].as_py())] = path
    return out


def collect_file(args):
    year, path, ntuple_path = args
    t = uproot.open(path)["events"]
    run = ak.to_numpy(ak.firsts(t["EventHeader.runNumber"].array())).astype(np.int64)
    cb = t["ClassBitset"].array()
    has_class = ak.to_numpy(ak.num(cb)) > 0
    word = ak.to_numpy(ak.fill_none(ak.firsts(cb), 0)).astype(np.int64)
    cl16 = has_class & (((word >> 15) & 1) == 1)
    ri = t["RunInformation"].array()
    bhdr = ak.to_numpy(ak.fill_none(ak.firsts(ri[:, 1:2]), np.nan)).astype(float)

    nt = pq.read_table(ntuple_path, columns=[
        "event_entry", "event_energy_vis", "event_num_charged_hadrons",
        "part_pt", "part_omega", "part_has_track", "part_dxy"])
    entry = nt["event_entry"].to_numpy()
    ev = ak.from_arrow(nt)
    nt_run = run[entry]
    nt_cl16 = cl16[entry]

    rows = []
    for r in np.unique(run):
        sel = run == r
        nsel = (nt_run == r)
        had = nsel & nt_cl16
        row = dict(year=year, file=os.path.basename(path), run=int(r),
                   n_all=int(sel.sum()), n_cl16=int((sel & cl16).sum()),
                   n_noclass=int((sel & ~has_class).sum()),
                   b_hdr_med=float(np.nanmedian(bhdr[sel])) if np.isfinite(bhdr[sel]).any() else np.nan,
                   b_hdr_min=float(np.nanmin(bhdr[sel])) if np.isfinite(bhdr[sel]).any() else np.nan,
                   b_hdr_max=float(np.nanmax(bhdr[sel])) if np.isfinite(bhdr[sel]).any() else np.nan,
                   n_reco=int(nsel.sum()), n_had=int(had.sum()))
        if had.sum() > 0:
            e = ev[had]
            row["evis_sum"] = float(np.sum(ak.to_numpy(e["event_energy_vis"])))
            row["nch_sum"] = float(np.sum(ak.to_numpy(e["event_num_charged_hadrons"])))
            trk = e["part_has_track"] == 1
            pt = ak.to_numpy(ak.flatten(e["part_pt"][trk]))
            om = np.abs(ak.to_numpy(ak.flatten(e["part_omega"][trk])))
            dxy = ak.to_numpy(ak.flatten(e["part_dxy"][trk]))
            ok = (pt > 0.5) & (om > 0)
            # omega in 1/mm: pT [GeV] = 0.2998 * B [T] / (|omega| [1/mm] * 1000)
            row["b_eff_med"] = float(np.median(pt[ok] * om[ok] * 1000 / 0.299792458)) if ok.sum() > 30 else np.nan
            row["n_trk"] = int(ok.sum())
            hi = pt > 2
            row["dxy_core"] = robust_sigma(dxy[hi])
            row["n_trk_hi"] = int(hi.sum())
        rows.append(row)
    return rows


def collect(out_csv, workers):
    tasks = []
    for year in YEARS:
        nmap = ntuple_map(year)
        for path in sorted(glob.glob(f"{PROD}/{year}/*_AL.root")):
            fn = file_number(os.path.basename(path))
            if fn not in nmap:
                print(f"no ntuple for {path}", flush=True)
                continue
            tasks.append((year, path, nmap[fn]))
    print(f"{len(tasks)} files", flush=True)
    rows = []
    with ProcessPoolExecutor(workers) as pool:
        for i, res in enumerate(pool.map(collect_file, tasks), 1):
            rows += res
            if i % 20 == 0:
                print(f"{i}/{len(tasks)}", flush=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# ---------------------------------------------------------------- SCANBOOK

def parse_scanbook():
    rows = []
    for path in sorted(glob.glob(f"{SCANBOOK}/19*_run.txt")):
        year = int(os.path.basename(path)[:4])
        for line in open(path):
            if not re.match(r"^\s*\d{4,5} \|", line):
                continue
            f = [x.strip() for x in line.rstrip("\n").split("|")]
            counts = f[3].split()
            qual_field = next(x for x in f[4:] if re.search(r"\b(PERF|MAYB|DUCK)\b", x))
            tokens = qual_field.split()
            qi = next(i for i, x in enumerate(tokens) if x in ("PERF", "MAYB", "DUCK"))
            row = dict(sb_year=year, run=int(f[0]), sb_energy=float(f[2]),
                       sb_n_ev=int(counts[0]), sb_n_z0=int(counts[1]),
                       sb_quality=tokens[qi], sb_groups=" ".join(tokens[qi + 1:]))
            lumi_status = re.findall(r"\b(OK|Prov|Unkn|Undf)\b", " ".join(f[4:]))
            row["sb_lumi_status"] = lumi_status[0] if lumi_status else ""
            last = f[-1].split()
            if len(last) == 10 and all(x in "PMD" for x in last):
                for det, q in zip(DETECTORS, last):
                    row[f"sb_{det}"] = q
            rows.append(row)
    sb = pd.DataFrame(rows)
    ranges = {}
    for path in sorted(glob.glob(f"{SCANBOOK}/19*_run.txt")):
        m = re.search(r"Selection on Run Numbers\s*:\s*(\d+)\s+(\d+)", open(path).read())
        if m:
            ranges[int(os.path.basename(path)[:4])] = (int(m.group(1)), int(m.group(2)))
    return sb, ranges


# ---------------------------------------------------------------- analysis

def per_run(df):
    sums = ["n_all", "n_cl16", "n_noclass", "n_reco", "n_had", "evis_sum", "nch_sum", "n_trk", "n_trk_hi"]
    g = df.groupby(["year", "run"])
    out = g[sums].sum()
    # median-like quantities: weight per-file values by their counts
    for col, w in [("b_eff_med", "n_trk"), ("dxy_core", "n_trk_hi")]:
        x = df.assign(_wx=df[col] * df[w]).groupby(["year", "run"])
        out[col] = x["_wx"].sum() / out[w].where(out[w] > 0)
    out["b_hdr_med"] = g["b_hdr_med"].median()
    out["b_hdr_min"] = g["b_hdr_min"].min()
    out["b_hdr_max"] = g["b_hdr_max"].max()
    out = out.reset_index()
    out["evis_mean"] = out.evis_sum / out.n_had.where(out.n_had > 0) / SQRT_S
    out["nch_mean"] = out.nch_sum / out.n_had.where(out.n_had > 0)
    out["cl16_frac"] = out.n_cl16 / out.n_all
    return out


def flag_outliers(runs, col, nmin_col, nmin, nsig):
    flags = pd.Series(False, index=runs.index)
    for year, grp in runs.groupby("year"):
        ok = grp[nmin_col] >= nmin
        x = grp.loc[ok, col]
        med = x.median()
        s = robust_sigma(x.values) if ok.sum() >= 30 else np.nan
        if not np.isfinite(s) or s == 0:
            continue
        flags.loc[x.index] = (np.abs(x - med) > nsig * s)
    return flags


def analyze(csv, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    runs = per_run(pd.read_csv(csv))
    sb, ranges = parse_scanbook()
    runs = runs.merge(sb, on="run", how="left")
    runs["in_good_list"] = runs.sb_quality.notna()
    runs["year_int"] = runs.year.str[:4].astype(int)
    runs["covered"] = [ranges[y][0] <= r <= ranges[y][1] for y, r in zip(runs.year_int, runs.run)]

    runs["flag_bhdr"] = (runs.b_hdr_min < 14.5) | (runs.b_hdr_max > 15.5)
    runs["flag_beff"] = flag_outliers(runs, "b_eff_med", "n_trk", 300, 5)
    runs["flag_evis"] = flag_outliers(runs, "evis_mean", "n_had", 100, 5)
    runs["flag_nch"] = flag_outliers(runs, "nch_mean", "n_had", 100, 5)
    runs["flag_dxy"] = flag_outliers(runs, "dxy_core", "n_trk_hi", 300, 5)
    runs["flag_cl16"] = flag_outliers(runs, "cl16_frac", "n_all", 1000, 5)
    runs["flag_off_peak"] = (runs.sb_energy - SQRT_S).abs() > 0.5
    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)

    summary = {}
    for year, g in runs.groupby("year"):
        s = dict(
            runs=len(g), events=int(g.n_all.sum()), cl16=int(g.n_cl16.sum()),
            good_list_range=ranges[int(year[:4])],
            runs_covered_by_list=int(g.covered.sum()),
            runs_in_good_list=int(g.in_good_list.sum()),
            cl16_in_good_list=int(g.n_cl16[g.in_good_list].sum()),
            cl16_covered_not_in_list=int(g.n_cl16[g.covered & ~g.in_good_list].sum()),
            cl16_not_covered=int(g.n_cl16[~g.covered].sum()),
            cl16_perf=int(g.n_cl16[g.sb_quality == "PERF"].sum()),
            cl16_maybe=int(g.n_cl16[g.sb_quality == "MAYB"].sum()),
            cl16_vs_scanbook_nz0=float(g.n_cl16[g.in_good_list].sum() / max(g.sb_n_z0[g.in_good_list].clip(lower=0).sum(), 1)),
            cl16_off_peak=int(g.n_cl16[g.flag_off_peak].sum()),
            b_hdr_bad_runs=g.run[g.flag_bhdr].tolist(),
            b_hdr_bad_cl16=int(g.n_cl16[g.flag_bhdr].sum()),
            b_eff_median=float(g.b_eff_med.median()),
            b_eff_range_of_bad_hdr=[float(g.b_eff_med[g.flag_bhdr].min()), float(g.b_eff_med[g.flag_bhdr].max())] if g.flag_bhdr.any() else None,
        )
        for f in ["flag_beff", "flag_evis", "flag_nch", "flag_dxy", "flag_cl16"]:
            s[f + "_runs"] = g.run[g[f]].tolist()
            s[f + "_cl16"] = int(g.n_cl16[g[f]].sum())
        if "sb_vdet" in g:
            for det in DETECTORS:
                col = f"sb_{det}"
                if g[col].notna().any():
                    s[f"cl16_{det}_dead"] = int(g.n_cl16[g[col] == "D"].sum())
        summary[year] = s
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)

    # plots: per year, quantities vs run
    quantities = [("b_hdr_med", "header B (MBFL) [kG]"), ("b_eff_med", "B used in reco [T]"),
                  ("cl16_frac", "class-16 fraction"), ("evis_mean", "<E_vis>/√s (class 16)"),
                  ("nch_mean", "<n charged> (class 16)"), ("dxy_core", "dxy core, pT>2 [mm]")]
    years = [y for y in YEARS if y in set(runs.year)]
    fig, axes = plt.subplots(len(quantities), len(years), figsize=(3.4 * len(years), 2.4 * len(quantities)), squeeze=False)
    for j, year in enumerate(years):
        g = runs[runs.year == year]
        for i, (col, label) in enumerate(quantities):
            ax = axes[i, j]
            for mask, c, lab in [(g.in_good_list & (g.sb_quality == "PERF"), "tab:green", "PERF"),
                                 (g.in_good_list & (g.sb_quality == "MAYB"), "tab:orange", "MAYB"),
                                 (g.covered & ~g.in_good_list, "tab:red", "not in good list"),
                                 (~g.covered, "tab:gray", "not covered by list")]:
                ax.scatter(g.run[mask], g[col][mask], s=np.clip(g.n_cl16[mask] / 200, 2, 30), c=c, label=lab, alpha=0.7)
            if col == "b_hdr_med":
                ax.set_yscale("symlog")
            if col == "dxy_core":
                ax.set_ylim(0, 1.0)
            if col == "evis_mean":
                ax.set_ylim(0.5, 1.3)
            if col == "nch_mean":
                ax.set_ylim(10, 25)
            if col == "b_eff_med":
                ax.set_ylim(1.3, 1.7)
            if i == 0:
                ax.set_title(f"ALEPH {year}")
            if j == 0:
                ax.set_ylabel(label)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "run_quality.png"), dpi=110)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["collect", "analyze"])
    parser.add_argument("inp")
    parser.add_argument("out", nargs="?")
    parser.add_argument("--workers", type=int, default=16)
    a = parser.parse_args()
    if a.mode == "collect":
        collect(a.inp, a.workers)
    else:
        analyze(a.inp, a.out)
