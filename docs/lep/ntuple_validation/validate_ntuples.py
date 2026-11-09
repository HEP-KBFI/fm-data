"""Validation plots of the ALEPH ntuples (fm_data.lep, production 20261108_ALEPH_ntupelized),
with 1994_old as the main sample, the other years overlaid, and the CLD Z->qq jet tables as
the reference for the lifetime-signed impact parameters.

    ./run.sh python3 docs/lep/ntuple_validation/validate_ntuples.py \
        [--ntuples /local/laurits/20261108_ALEPH_ntupelized] [--cld /local/laurits/CLD_ntuples] \
        [--out docs/lep/ntuple_validation] [--files-main 10] [--files-other 3]

Reads a subset of the per-ROOT-file tables (events/ and unchunked/jets_*), files spread over
each year. Writes PNGs and summary.json to --out.
"""

import argparse
import glob
import json
from pathlib import Path

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

YEARS = ["1990", "1991", "1992", "1993", "1994", "1994_old", "1995"]
MAIN = "1994_old"
COLORS = dict(zip(YEARS, ["#9e9e9e", "#b39ddb", "#80cbc4", "#ffcc80", "#64b5f6", "#d32f2f", "#81c784"]))
SQRT_S = 91.2

EVENT_COLUMNS = [
    "event_id", "event_run_number", "event_sqrt_s", "event_nparticles", "event_energy_vis",
    "event_thrust", "event_sphericity", "event_num_charged_hadrons", "event_num_photons",
    "event_num_neutral_hadrons", "event_num_electrons", "event_num_muons", "event_has_vertex",
    "event_vertex_x", "event_vertex_y", "event_vertex_z", "event_ref_x", "event_ref_y",
    "event_ref_z", "event_beam_sigma_x", "event_beam_sigma_y", "event_ref_from_beamspot",
    "part_pt", "part_eta", "part_energy", "part_mass", "part_charge", "part_pid_type",
    "part_has_track", "part_dxy", "part_dz", "part_dxy_err", "part_dz_err", "part_dxy_sig",
    "part_dz_sig", "part_d0", "part_phi0", "part_z0", "part_pca_z",
]
JET_COLUMNS = [
    "event_id", "event_sqrt_s", "jet_index", "jet_energy", "jet_eta", "jet_theta", "jet_mass",
    "jet_nparticles", "jet_charge", "part_pt", "part_energy", "part_has_track", "part_event_index",
    "part_dxy_signed_sig", "part_d3_signed_sig", "part_dxy_signed", "part_pid_type",
]
CLD_JET_COLUMNS = [
    "jet_energy", "label_b", "label_c", "label_s", "label_ud", "part_pt", "part_has_track",
    "part_dxy_signed_sig", "part_d3_signed_sig",
]


def spread(files, n):
    files = sorted(files)
    if len(files) <= n:
        return files
    return [files[i] for i in np.linspace(0, len(files) - 1, n).round().astype(int)]


def robust_sigma(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return (q75 - q25) / 1.349


def flat(array):
    return ak.to_numpy(ak.flatten(array, axis=None))


def load(ntuples, files_main, files_other):
    data = {}
    for year in YEARS:
        base = Path(ntuples) / f"ALEPH{year}"
        event_files = spread(glob.glob(str(base / "events" / "*.parquet")), files_main if year == MAIN else files_other)
        names = [Path(f).name for f in event_files]
        events = ak.concatenate([ak.from_parquet(f, columns=EVENT_COLUMNS) for f in event_files])
        jets = {}
        for alg in ("ee_antikt_R04", "ee_cambridge_exclusive2"):
            jet_files = [base / "unchunked" / f"jets_{alg}" / name for name in names]
            jets[alg] = ak.concatenate([ak.from_parquet(str(f), columns=JET_COLUMNS) for f in jet_files])
        data[year] = {"events": events, "jets": jets, "files": names}
        print(f"{year}: {len(events)} events, {len(jets['ee_antikt_R04'])} R04 jets from {len(names)} files", flush=True)
    return data


def hist_years(ax, data, getter, bins, label, logy=False, density=True, years=YEARS):
    for year in years:
        values = getter(data[year])
        values = values[np.isfinite(values)]
        ax.hist(values, bins=bins, histtype="step", density=density, lw=2.2 if year == MAIN else 1.2,
                color=COLORS[year], label=year)
    ax.set_xlabel(label)
    ax.set_ylabel("normalized" if density else "entries")
    if logy:
        ax.set_yscale("log")


def track_mask(events, min_pt=0.0):
    return (events.part_has_track == 1) & (events.part_pt > min_pt)


def d0_wrt_ref(events, sign=1.0, min_pt=2.0):
    """Transverse impact parameter (LCIO sign) w.r.t. the reference point [mm] of tracks above
    min_pt: d0_ref = sign * d0 - (-x_ref sin(phi0) + y_ref cos(phi0)); sign = -1 undoes the
    d0-sign correction of the reader."""
    mask = track_mask(events, min_pt)
    d0 = flat(events.part_d0[mask])
    phi = flat(events.part_phi0[mask])
    x = flat(ak.broadcast_arrays(events.event_ref_x, events.part_d0)[0][mask])
    y = flat(ak.broadcast_arrays(events.event_ref_y, events.part_d0)[0][mask])
    err = flat(events.part_dxy_err[mask])
    return sign * d0 - (-x * np.sin(phi) + y * np.cos(phi)), err


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(Path(out) / name, dpi=110)
    plt.close(fig)
    print(f"wrote {name}", flush=True)


def plot_events(data, out, summary):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ev = lambda d: d["events"]  # noqa: E731
    hist_years(axes[0, 0], data, lambda d: ak.to_numpy(ev(d).event_nparticles), np.arange(0, 80, 2), "PFOs per event", logy=True)
    hist_years(axes[0, 1], data, lambda d: ak.to_numpy(ev(d).event_energy_vis) / SQRT_S, np.linspace(0, 2, 81), r"$E_\mathrm{vis}/\sqrt{s}$", logy=True)
    hist_years(axes[0, 2], data, lambda d: ak.to_numpy(ev(d).event_thrust), np.linspace(0.5, 1, 51), "thrust")
    hist_years(axes[1, 0], data, lambda d: ak.to_numpy(ev(d).event_sphericity), np.linspace(0, 1, 51), "sphericity", logy=True)
    hist_years(axes[1, 1], data, lambda d: ak.to_numpy(ev(d).event_num_charged_hadrons), np.arange(0, 50), "charged hadrons per event", logy=True)
    hist_years(axes[1, 2], data, lambda d: ak.to_numpy(ev(d).event_num_photons), np.arange(0, 40), "photons per event", logy=True)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Event variables (events with >= 1 PFO; no hadronic-Z selection)")
    save(fig, out, "01_events.png")
    summary["events"] = {
        year: {
            "n_events_sample": len(ev(data[year])),
            "mean_nparticles": float(ak.mean(ev(data[year]).event_nparticles)),
            "frac_evis_gt_1.2_sqrts": float(np.mean(ak.to_numpy(ev(data[year]).event_energy_vis) > 1.2 * SQRT_S)),
            "frac_hadronic_like (>=5 charged hadrons)": float(np.mean(ak.to_numpy(ev(data[year]).event_num_charged_hadrons) >= 5)),
        }
        for year in YEARS
    }


def plot_pfos(data, out, summary):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    pf = lambda d, f: flat(d["events"][f])  # noqa: E731
    hist_years(axes[0, 0], data, lambda d: pf(d, "part_pt"), np.logspace(-2, 2, 81), r"PFO $p_T$ [GeV]", logy=True)
    axes[0, 0].set_xscale("log")
    hist_years(axes[0, 1], data, lambda d: np.tanh(pf(d, "part_eta")), np.linspace(-1, 1, 81), r"PFO cos$\theta$")
    hist_years(axes[0, 2], data, lambda d: pf(d, "part_energy"), np.linspace(0, 60, 121), "PFO energy [GeV]", logy=True)
    axes[0, 2].axvline(SQRT_S / 2, color="k", ls=":", lw=1)
    # particle-type fractions per year
    ax = axes[1, 0]
    types = [0, 1, 2, 3, 4, 5, 7, 8]
    width = 0.11
    for i, year in enumerate(YEARS):
        t = pf(data[year], "part_pid_type")
        frac = [np.mean(t == k) for k in types]
        ax.bar(np.arange(len(types)) + (i - 3) * width, frac, width, color=COLORS[year], label=year)
    ax.set_xticks(range(len(types)), ["0 ch.had", "1 e", "2 mu", "3 V0 trk", "4 gamma", "5 n.had", "7 lcal", "8 fwd"], rotation=30, fontsize=8)
    ax.set_ylabel("fraction of PFOs")
    # mass per type (main year)
    ax = axes[1, 1]
    e = data[MAIN]["events"]
    t, m = pf(data[MAIN], "part_pid_type"), pf(data[MAIN], "part_mass")
    masses = {int(k): sorted(set(np.round(m[t == k], 5).tolist())) for k in types}
    ax.axis("off")
    ax.text(0, 1, f"{MAIN}: part_mass per energy-flow type\n" + "\n".join(f"type {k}: {v}" for k, v in masses.items()), va="top", family="monospace", fontsize=9)
    # charge of tracks vs neutral
    ax = axes[1, 2]
    has = pf(data[MAIN], "part_has_track") == 1
    q = pf(data[MAIN], "part_charge")
    ax.hist([q[has], q[~has]], bins=np.arange(-1.5, 2, 1), label=["with track", "without track"], histtype="bar")
    ax.set_xlabel(f"PFO charge ({MAIN})")
    ax.legend()
    fig.suptitle("PFOs")
    save(fig, out, "02_pfos.png")
    summary["pfo_mass_per_type_" + MAIN] = masses
    summary["charged_without_track_" + MAIN] = int(np.sum((q != 0) & ~has))
    summary["neutral_with_track_" + MAIN] = int(np.sum((q == 0) & has))


def plot_reference(data, out, summary):
    e = data[MAIN]["events"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    run = ak.to_numpy(e.event_run_number)
    idx = np.arange(len(run))
    for ax, field, label in ((axes[0, 0], "event_ref_x", "reference x [mm]"), (axes[0, 1], "event_ref_y", "reference y [mm]")):
        ax.plot(idx, ak.to_numpy(e[field]), ",", color=COLORS[MAIN])
        ax.set_xlabel(f"event ({MAIN} sample, runs {run.min()}-{run.max()})")
        ax.set_ylabel(label)
    hist_years(axes[0, 2], data, lambda d: ak.to_numpy(d["events"].event_ref_z), np.linspace(-40, 40, 81), "reference z (median track z) [mm]")
    # vertex vs reference
    ax = axes[1, 0]
    hv = ak.to_numpy(e.event_has_vertex) == 1
    for axis, color in (("x", "C0"), ("y", "C1"), ("z", "C2")):
        diff = ak.to_numpy(e[f"event_vertex_{axis}"] - e[f"event_ref_{axis}"])[hv]
        ax.hist(diff, bins=np.linspace(-1, 1, 101), histtype="step", color=color, label=f"{axis}: core {robust_sigma(diff) * 1000:.0f} um")
    ax.set_xlabel(f"ALEPH vertex - reference [mm] ({MAIN}, events with a vertex)")
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    for year in YEARS:
        ev = data[year]["events"]
        ax.scatter(ak.to_numpy(ev.event_beam_sigma_x)[::500] * 1000, ak.to_numpy(ev.event_beam_sigma_y)[::500] * 1000, s=6, color=COLORS[year], label=year)
    ax.set_xlabel(r"fitted beam size $\sigma_x$ [um]")
    ax.set_ylabel(r"fitted beam size $\sigma_y$ [um]")
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    fr = [float(ak.mean(data[y]["events"].event_ref_from_beamspot)) for y in YEARS]
    ax.bar(YEARS, fr, color=[COLORS[y] for y in YEARS])
    ax.set_ylabel("fraction with x, y from BeamSpot")
    ax.set_ylim(0.9, 1.0)
    ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Reference point of the impact parameters")
    save(fig, out, "03_reference_point.png")
    summary["reference"] = {
        "frac_from_beamspot": dict(zip(YEARS, fr)),
        f"vertex_minus_ref_core_um_{MAIN}": {
            axis: float(robust_sigma(ak.to_numpy(e[f"event_vertex_{axis}"] - e[f"event_ref_{axis}"])[hv]) * 1000) for axis in "xyz"
        },
    }


def plot_ip(data, out, summary):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    # d0 w.r.t. reference: stored (correct) sign vs flipped
    ax = axes[0, 0]
    good, err = d0_wrt_ref(data[MAIN]["events"], +1)
    bad, _ = d0_wrt_ref(data[MAIN]["events"], -1)
    bins = np.linspace(-3, 3, 241)
    ax.hist(good, bins=bins, histtype="step", lw=2, color=COLORS[MAIN], label=f"as stored: core {robust_sigma(good) * 1000:.0f} um")
    ax.hist(bad, bins=bins, histtype="step", color="k", label=f"d0 sign flipped: core {robust_sigma(bad) * 1000:.0f} um")
    ax.set_yscale("log")
    ax.set_xlabel(r"$d_0$ w.r.t. reference [mm] (tracks $p_T$ > 2 GeV, " + MAIN + ")")
    ax.legend(fontsize=9)
    # pull
    ax = axes[0, 1]
    for year in YEARS:
        d, e = d0_wrt_ref(data[year]["events"], +1)
        pull = d / e
        ax.hist(pull, bins=np.linspace(-8, 8, 161), histtype="step", density=True, lw=2.2 if year == MAIN else 1.2, color=COLORS[year], label=f"{year}: core {robust_sigma(pull):.2f}")
    xs = np.linspace(-8, 8, 400)
    ax.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "k:", lw=1, label="N(0,1)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$d_0$ / $\sigma(d_{xy})$ w.r.t. reference ($p_T$ > 2 GeV)")
    ax.legend(fontsize=8)
    # dxy, dz
    sel = lambda d, f, pt=1.0: flat(d["events"][f][track_mask(d["events"], pt)])  # noqa: E731
    hist_years(axes[0, 2], data, lambda d: sel(d, "part_dxy"), np.logspace(-4, 1.5, 111), r"$d_{xy}$ [mm] (tracks $p_T$ > 1 GeV)", logy=True)
    axes[0, 2].set_xscale("log")
    hist_years(axes[1, 0], data, lambda d: sel(d, "part_dz"), np.linspace(-5, 5, 201), r"$d_z$ [mm] (tracks $p_T$ > 1 GeV)", logy=True)
    hist_years(axes[1, 1], data, lambda d: sel(d, "part_dxy_sig"), np.linspace(0, 20, 101), r"$d_{xy}/\sigma$ ($p_T$ > 1 GeV)", logy=True)
    hist_years(axes[1, 2], data, lambda d: sel(d, "part_dz_sig"), np.linspace(-20, 20, 161), r"$d_z/\sigma$ ($p_T$ > 1 GeV)", logy=True)
    axes[0, 2].legend(fontsize=8)
    fig.suptitle("Track impact parameters (w.r.t. beam spot x, y and median track z)")
    save(fig, out, "04_impact_parameters.png")
    summary["ip"] = {
        year: {
            "d0_core_um (pT>2)": float(robust_sigma(d0_wrt_ref(data[year]["events"], +1)[0]) * 1000),
            "d0_core_um_sign_flipped": float(robust_sigma(d0_wrt_ref(data[year]["events"], -1)[0]) * 1000),
            "pull_core (pT>2)": float(robust_sigma(np.divide(*d0_wrt_ref(data[year]["events"], +1)))),
            "dz_core_um (pT>1)": float(robust_sigma(sel(data[year], "part_dz")) * 1000),
            "dz_pull_core (pT>1)": float(robust_sigma(sel(data[year], "part_dz_sig"))),
        }
        for year in YEARS
    }


def signed(jets, field, min_pt=1.0, min_jet_energy=5.0):
    jet_ok = jets.jet_energy > min_jet_energy
    j = jets[jet_ok]
    mask = (j.part_has_track == 1) & (j.part_pt > min_pt)
    return flat(j[field][mask])


def asymmetry(values, xs):
    return np.array([np.sum(values > x) / max(np.sum(values < -x), 1) for x in xs])


def plot_signed(data, cld, out, summary):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    bins = np.linspace(-20, 40, 241)
    xs = np.linspace(0.5, 10, 20)
    res = {}
    for row, alg in enumerate(("ee_antikt_R04", "ee_cambridge_exclusive2")):
        ax, ax_asym = axes[row, 0], axes[row, 1]
        for year in YEARS:
            s = signed(data[year]["jets"][alg], "part_dxy_signed_sig")
            ax.hist(s, bins=bins, histtype="step", density=True, lw=2.2 if year == MAIN else 1.2, color=COLORS[year], label=year)
            ax_asym.plot(xs, asymmetry(s, xs), color=COLORS[year], lw=2.2 if year == MAIN else 1.2, label=year)
            res.setdefault(alg, {})[year] = {
                "frac_gt_3": float(np.mean(s > 3)),
                "frac_lt_-3": float(np.mean(s < -3)),
                "core": float(robust_sigma(s)),
            }
        if cld is not None and alg in cld:
            c = cld[alg]
            for name, mask, style in (("CLD Zqq udsc", c.label_b == 0, "--"), ("CLD Zqq b", c.label_b == 1, ":")):
                s = signed(c[mask], "part_dxy_signed_sig")
                ax.hist(s, bins=bins, histtype="step", density=True, color="k", ls=style, label=name)
                ax_asym.plot(xs, asymmetry(s, xs), "k" + style, label=name)
                res[alg][name] = {"frac_gt_3": float(np.mean(s > 3)), "frac_lt_-3": float(np.mean(s < -3)), "core": float(robust_sigma(s))}
        ax.set_yscale("log")
        ax.set_xlabel(rf"lifetime-signed $d_{{xy}}/\sigma$, {alg} (jets E > 5 GeV, tracks $p_T$ > 1 GeV)")
        ax_asym.set_yscale("log")
        ax_asym.set_xlabel(r"x")
        ax_asym.set_ylabel(r"N(sig > x) / N(sig < -x)")
        ax_asym.axhline(1, color="grey", lw=0.8)
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].legend(fontsize=7)
    # high-mass / high-multiplicity jets (enriched in b) vs low
    ax = axes[0, 2]
    j = data[MAIN]["jets"]["ee_cambridge_exclusive2"]
    for name, mask, color in (("jet mass > 10 GeV", j.jet_mass > 10, "C3"), ("jet mass < 5 GeV", j.jet_mass < 5, "C0")):
        s = signed(j[mask], "part_dxy_signed_sig")
        ax.hist(s, bins=bins, histtype="step", density=True, color=color, label=f"{name}: >3: {np.mean(s > 3):.3f}, <-3: {np.mean(s < -3):.3f}")
    ax.set_yscale("log")
    ax.set_xlabel(rf"lifetime-signed $d_{{xy}}/\sigma$ ({MAIN}, 2-jet)")
    ax.legend(fontsize=8)
    # signed 3D
    ax = axes[1, 2]
    for year in (MAIN, "1994", "1990"):
        s = signed(data[year]["jets"]["ee_antikt_R04"], "part_d3_signed_sig")
        ax.hist(s, bins=bins, histtype="step", density=True, color=COLORS[year], lw=2.2 if year == MAIN else 1.2, label=year)
    ax.set_yscale("log")
    ax.set_xlabel(r"lifetime-signed $d_{3}/\sigma$ (R=0.4, jets E > 5 GeV)")
    ax.legend(fontsize=8)
    fig.suptitle("Impact parameters signed w.r.t. the jet axis (positive = lifetime)")
    save(fig, out, "05_signed_impact_parameters.png")
    summary["signed_dxy_sig"] = res


def plot_jets(data, out, summary):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    res = {}
    for row, alg in enumerate(("ee_antikt_R04", "ee_cambridge_exclusive2")):
        jt = lambda d, f: ak.to_numpy(d["jets"][alg][f])  # noqa: E731
        hist_years(axes[row, 0], data, lambda d: jt(d, "jet_energy"), np.linspace(0, 95, 96), f"jet energy [GeV] ({alg})", logy=True)
        hist_years(axes[row, 1], data, lambda d: np.cos(jt(d, "jet_theta")), np.linspace(-1, 1, 81), r"jet cos$\theta$")
        hist_years(axes[row, 2], data, lambda d: jt(d, "jet_nparticles"), np.arange(0, 50), "constituents per jet", logy=True)
        hist_years(axes[row, 3], data, lambda d: jt(d, "jet_mass"), np.linspace(0, 40, 81), "jet mass [GeV]", logy=True)
        for year in YEARS:
            j = data[year]["jets"][alg]
            res.setdefault(alg, {})[year] = {
                "n_jets_sample": len(j),
                "jets_per_event": len(j) / len(data[year]["events"]),
                "max_energy": float(ak.max(j.jet_energy)),
                "min_energy": float(ak.min(j.jet_energy)),
                "max_abs_eta": float(ak.max(abs(j.jet_eta))),
                "max_constituent_energy": float(ak.max(j.part_energy)),
            }
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Jets (top: ee anti-kt R=0.4, bottom: exclusive 2-jet ee Cambridge)")
    save(fig, out, "06_jets.png")
    # 2-jet events: sum of the two jet energies
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for year in YEARS:
        j = data[year]["jets"]["ee_cambridge_exclusive2"]
        ids, counts = np.unique(ak.to_numpy(j.event_id), return_counts=True)
        both = np.isin(ak.to_numpy(j.event_id), ids[counts == 2])
        e_sum = np.bincount(np.searchsorted(ids, ak.to_numpy(j.event_id)[both]), weights=ak.to_numpy(j.jet_energy)[both])
        e_sum = e_sum[e_sum > 0]
        axes[0].hist(e_sum / SQRT_S, bins=np.linspace(0, 1.1, 111), histtype="step", density=True, color=COLORS[year], lw=2.2 if year == MAIN else 1.2, label=year)
        axes[1].bar(year, np.mean(counts == 2), color=COLORS[year])
        res["ee_cambridge_exclusive2"][year]["frac_events_both_jets_kept"] = float(np.mean(counts == 2))
    axes[0].set_xlabel(r"$(E_{j1}+E_{j2})/\sqrt{s}$, events with both jets selected")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("fraction of 2-jet events with both jets kept")
    axes[1].tick_params(axis="x", rotation=30)
    save(fig, out, "07_two_jet_events.png")
    summary["jets"] = res


def load_cld(cld_dir, n_files=2):
    if cld_dir is None or not Path(cld_dir).exists():
        return None
    out = {}
    for alg in ("ee_antikt_R04", "ee_cambridge_exclusive2"):
        files = sorted(glob.glob(str(Path(cld_dir) / f"jets_{alg}" / "data_Zqq_*.parquet")))[:n_files]
        if files:
            out[alg] = ak.concatenate([ak.from_parquet(f, columns=CLD_JET_COLUMNS) for f in files])
    return out


def check_chunks(ntuples, summary):
    """Jet counts of the rechunked files vs the unchunked ones (if the rechunk is done)."""
    import pyarrow.parquet as pq

    res = {}
    for year in YEARS:
        base = Path(ntuples) / f"ALEPH{year}"
        for alg in ("ee_antikt_R04", "ee_cambridge_exclusive2"):
            chunked = sorted(glob.glob(str(base / f"jets_{alg}" / "*.parquet")))
            unchunked = glob.glob(str(base / "unchunked" / f"jets_{alg}" / "*.parquet"))
            rows = [pq.ParquetFile(f).metadata.num_rows for f in chunked]
            total_unchunked = sum(pq.ParquetFile(f).metadata.num_rows for f in unchunked)
            res[f"{year}/{alg}"] = {
                "files": len(rows),
                "all_but_last_100k": bool(all(r == 100000 for r in rows[:-1])) if rows else None,
                "last": rows[-1] if rows else None,
                "total_chunked": sum(rows),
                "total_unchunked": total_unchunked,
                "match": sum(rows) == total_unchunked,
            }
    summary["chunks"] = res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ntuples", default="/local/laurits/20261108_ALEPH_ntupelized")
    p.add_argument("--cld", default="/local/laurits/CLD_ntuples")
    p.add_argument("--out", default=str(Path(__file__).parent))
    p.add_argument("--files-main", type=int, default=10)
    p.add_argument("--files-other", type=int, default=3)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    summary = {}
    data = load(a.ntuples, a.files_main, a.files_other)
    summary["files"] = {year: data[year]["files"] for year in YEARS}
    cld = load_cld(a.cld)
    plot_events(data, a.out, summary)
    plot_pfos(data, a.out, summary)
    plot_reference(data, a.out, summary)
    plot_ip(data, a.out, summary)
    plot_signed(data, cld, a.out, summary)
    plot_jets(data, a.out, summary)
    check_chunks(a.ntuples, summary)
    with open(Path(a.out) / "summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
