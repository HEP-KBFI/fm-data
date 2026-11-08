"""Validation of the ALEPH impact parameters in the fm-data event ntuples (CLD as reference).

Checks per dataset:
 A  units: pT from the track curvature (omega, B = 1.5 T ALEPH / 2 T CLD) vs pT of the PFO
 B  primary vertex: fraction with vertex, x/y/z location and spread
 C  beam spot from the tracks: fit d0(at reference point) = a sin(phi0) + b cos(phi0) + c
    per file (only tracks with pT > 1 GeV), compared with the mean event vertex
 D  core widths (robust sigma = IQR/1.349) of dxy w.r.t. the event vertex, of d0 w.r.t. the
    fitted beam spot, of dz; for pT > 2 GeV
 E  pulls: robust sigma of dxy/dxy_err (should be ~1 for prompt tracks if the vertex is right)
 F  lifetime sign: dxy signed w.r.t. the hemisphere axis (sum of the PFO momenta in the
    hemisphere of the track, hemispheres from the leading-PFO direction); fraction of tracks
    with S > +3 vs S < -3 (b/c decays give an excess of positive values)
"""

import glob
import json
import sys

import awkward as ak
import numpy as np

COLS = [
    "event_id", "event_has_vertex", "event_vertex_x", "event_vertex_y", "event_vertex_z",
    "part_px", "part_py", "part_pz", "part_energy", "part_charge", "part_has_track",
    "part_dxy", "part_dz", "part_dxy_err", "part_dz_err", "part_d0", "part_z0", "part_d0_err",
    "part_phi0", "part_omega", "part_pca_x", "part_pca_y", "part_pca_z", "part_dxy_sig",
]


def rsig(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return (q75 - q25) / 1.349


def flat(a):
    return ak.to_numpy(ak.flatten(a))


def analyse(files, bfield, name):
    out = {"dataset": name, "files": len(files)}
    per_file = []
    acc = {k: [] for k in ["ratio_pt", "dxy", "dxy_bs", "dz", "pull", "pull_bs", "S_life", "vx", "vy", "vz", "d0err"]}
    n_ev = n_vtx = 0
    for f in files:
        t = ak.from_parquet(f, columns=COLS)
        n_ev += len(t)
        hv = ak.to_numpy(t["event_has_vertex"]).astype(bool)
        n_vtx += hv.sum()
        vx, vy, vz = (ak.to_numpy(t[f"event_vertex_{a}"]) for a in "xyz")
        acc["vx"].append(vx[hv]); acc["vy"].append(vy[hv]); acc["vz"].append(vz[hv])
        trk = t["part_has_track"] == 1
        pt = np.sqrt(t["part_px"] ** 2 + t["part_py"] ** 2)
        # A: curvature: R[m] = pT / (0.3 B); omega stored in 1/mm
        pt_curv = 0.299792458 * bfield / (np.abs(t["part_omega"]) * 1000.0)
        sel = trk & (pt > 0.5)
        acc["ratio_pt"].append(flat((pt_curv / pt)[sel]))
        # C: beam-spot fit on tracks with pT > 1 GeV
        s1 = trk & (pt > 1.0)
        d0 = flat(t["part_d0"][s1]); phi = flat(t["part_phi0"][s1])
        A = np.stack([np.sin(phi), np.cos(phi), np.ones_like(phi)], 1)
        keep = np.abs(d0) < 5.0
        for _ in range(4):
            coef, *_ = np.linalg.lstsq(A[keep], d0[keep], rcond=None)
            res = d0 - A @ coef
            keep = np.abs(res) < 3 * max(rsig(res[keep]), 1e-3)
        bs_res = res[keep]
        per_file.append({
            "file": f.split("/")[-1],
            "fit_a_sin": float(coef[0]), "fit_b_cos": float(coef[1]), "fit_c": float(coef[2]),
            "beamspot_sigma_res_mm": float(rsig(res)),
            "vtx_mean": [float(np.mean(vx[hv])) if hv.any() else None,
                         float(np.mean(vy[hv])) if hv.any() else None,
                         float(np.mean(vz[hv])) if hv.any() else None],
            "vtx_median": [float(np.median(vx[hv])) if hv.any() else None,
                           float(np.median(vy[hv])) if hv.any() else None],
            "vtx_rsig": [float(rsig(vx[hv])), float(rsig(vy[hv])), float(rsig(vz[hv]))],
            "has_vertex": float(hv.mean()),
        })
        # D/E: with pT > 2 GeV, events with a vertex
        s2 = trk & (pt > 2.0)
        hv_b = ak.broadcast_arrays(ak.Array(hv), t["part_dxy"])[0]
        sv = s2 & hv_b
        acc["dxy"].append(flat(t["part_dxy"][sv]))
        acc["dz"].append(flat(t["part_dz"][sv]))
        acc["pull"].append(flat((t["part_dxy"] / t["part_dxy_err"])[sv]))
        acc["d0err"].append(flat(t["part_d0_err"][s2]))
        # d0 relative to the fitted beam spot of this file (all events)
        d0_2 = flat(t["part_d0"][s2]); phi_2 = flat(t["part_phi0"][s2])
        d0_bs = d0_2 - (coef[0] * np.sin(phi_2) + coef[1] * np.cos(phi_2) + coef[2])
        acc["dxy_bs"].append(d0_bs)
        acc["pull_bs"].append(d0_bs / flat(t["part_d0_err"][s2]))
        # F: lifetime sign w.r.t. the hemisphere axis, using the PCA - vertex vector (3D)
        e = t["part_energy"]
        lead = ak.argmax(e, axis=1, keepdims=True)
        nx, ny, nz = (ak.firsts(t[f"part_p{a}"][lead]) for a in "xyz")
        side = np.sign(t["part_px"] * nx + t["part_py"] * ny + t["part_pz"] * nz)
        hx = ak.sum(t["part_px"] * (side > 0), axis=1); hy = ak.sum(t["part_py"] * (side > 0), axis=1)
        hz = ak.sum(t["part_pz"] * (side > 0), axis=1)
        ax = ak.where(side > 0, hx, -hx + 0 * side) * 1.0
        ay = ak.where(side > 0, hy, -hy + 0 * side) * 1.0
        # the other hemisphere's axis is -(its sum) ~ the negative of the first one only for
        # balanced events; compute it directly instead
        gx = ak.sum(t["part_px"] * (side < 0), axis=1); gy = ak.sum(t["part_py"] * (side < 0), axis=1)
        ax = ak.where(side > 0, hx, gx); ay = ak.where(side > 0, hy, gy)
        dx = t["part_pca_x"] - ak.Array(vx); dy = t["part_pca_y"] - ak.Array(vy)
        sign = np.sign(dx * ax + dy * ay)
        S = np.abs(t["part_dxy"] / t["part_dxy_err"]) * sign
        acc["S_life"].append(flat(S[sv]))
    for k in acc:
        acc[k] = np.concatenate(acc[k]) if acc[k] else np.array([])
    r = acc["ratio_pt"]
    S = acc["S_life"]
    out.update({
        "events": int(n_ev), "frac_has_vertex": n_vtx / n_ev,
        "A_pt_curv_over_pt_median": float(np.median(r)), "A_pt_curv_over_pt_rsig": float(rsig(r)),
        "B_vertex_median_xyz_mm": [float(np.median(acc[a])) for a in ["vx", "vy", "vz"]],
        "B_vertex_rsig_xyz_mm": [float(rsig(acc[a])) for a in ["vx", "vy", "vz"]],
        "B_vertex_std_xyz_mm": [float(np.std(acc[a])) for a in ["vx", "vy", "vz"]],
        "B_vertex_frac_|x|>2mm": float(np.mean(np.abs(acc["vx"]) > 2)),
        "D_dxy_wrt_vertex_rsig_um": 1000 * rsig(acc["dxy"]),
        "D_d0_wrt_fitted_beamspot_rsig_um": 1000 * rsig(acc["dxy_bs"]),
        "D_dz_wrt_vertex_rsig_um": 1000 * rsig(acc["dz"]),
        "D_d0err_median_um": 1000 * float(np.median(acc["d0err"])),
        "E_pull_dxy_wrt_vertex_rsig": rsig(acc["pull"]),
        "E_pull_d0_wrt_beamspot_rsig": rsig(acc["pull_bs"]),
        "F_frac_S>+3": float(np.mean(S > 3)), "F_frac_S<-3": float(np.mean(S < -3)),
        "F_asym_(+3 - -3)/(+3 + -3)": float((np.mean(S > 3) - np.mean(S < -3)) / max(np.mean(np.abs(S) > 3), 1e-9)),
        "per_file": per_file,
    })
    np.savez(f"{sys.argv[1]}/arrays_{name}.npz", **{k: v[:2_000_000] for k, v in acc.items()})
    return out


if __name__ == "__main__":
    outdir = sys.argv[1]
    aleph = sorted(glob.glob("/local/laurits/ALEPH_ntuples/events/*.parquet"))
    aleph = aleph[:: max(1, len(aleph) // 12)][:12]
    cld = sorted(glob.glob("/local/laurits/CLD_ntuples/events/data_Zqq_0001?_1.parquet"))[:2]
    res = [analyse(aleph, 1.5, "ALEPH"), analyse(cld, 2.0, "CLD")]
    json.dump(res, open(f"{outdir}/ip_validation.json", "w"), indent=1, default=float)
    for r in res:
        print("=" * 30, r["dataset"])
        for k, v in r.items():
            if k != "per_file":
                print(f"  {k}: {v}")
        print("  per file (fit a, b, c [mm]; vertex mean x,y,z; vertex rsig; has_vertex):")
        for p in r["per_file"]:
            print(f"   {p['file']}: a={p['fit_a_sin']:+.4f} b={p['fit_b_cos']:+.4f} c={p['fit_c']:+.4f} "
                  f"res={p['beamspot_sigma_res_mm']*1000:.0f}um | vtx mean {np.round(p['vtx_mean'], 3)} "
                  f"median {np.round(p['vtx_median'], 3)} rsig {np.round(p['vtx_rsig'], 3)} | {p['has_vertex']:.2f}")
