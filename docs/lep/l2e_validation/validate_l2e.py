"""Check the l2e-prod-v1 ALEPH EDM4hep files for the converter bugs listed in
docs/lep/CHANGES_since_June2025_production.txt, and measure the conventions the ntupelizers
work around (units, d0 sign, PDG ids, RunInformation layout).

    python -I validate_l2e.py <file.root> [<file.root> ...] [--max-events N]

Prints one block of numbers per file. Run in pytorch.simg:2025-09-01.
"""

import argparse

import awkward as ak
import numpy as np
import uproot

PI_MASS, E_MASS, MU_MASS = 0.13957, 0.000511, 0.10566


def collection_ids(f):
    info = f["podio_metadata"]["events___CollectionTypeInfo"].array(entry_stop=1)[0]
    return dict(
        zip(
            info["events___CollectionTypeInfo.collectionID"].tolist(),
            info["events___CollectionTypeInfo.name"].tolist(),
        )
    )


def robust_sigma(x):
    q75, q25 = np.percentile(x, [75, 25])
    return (q75 - q25) / 1.349


def relation_check(ev, ids, owner, rel, label):
    """Every index of relation `rel` must lie inside the target collection of its event."""
    idx = ev[f"{rel}.index"]
    cid = ev[f"{rel}.collectionID"]
    n_bad, n_tot, targets = 0, 0, {}
    for c in np.unique(ak.to_numpy(ak.flatten(cid))):
        name = ids.get(int(c), f"?{c}")
        sel = cid == c
        n_target = ak.num(ev[f"{name}.energy" if name not in ("Tracks",) else "Tracks.chi2"])
        bad = ak.sum((idx[sel] < 0) | (idx[sel] >= n_target))
        n_bad += int(bad)
        n_tot += int(ak.sum(sel))
        targets[name] = int(ak.sum(sel))
    print(f"  {label}: {n_tot} links -> {targets}, out of range: {n_bad}")
    return n_bad


def check(path, max_events):
    print(f"\n=== {path}")
    f = uproot.open(path)
    t = f["events"]
    ids = collection_ids(f)
    names = set(t.keys(recursive=False))
    is_mc = "MCParticles" in names
    n = min(t.num_entries, max_events)
    print(f"  entries {t.num_entries}, using {n}, MC={is_mc}")

    prefixes = {
        "RecoParticles", "_RecoParticles_tracks", "_RecoParticles_clusters", "ParticleID",
        "Jets", "_Jets_particles", "Tracks", "_Tracks_trackStates", "Vertices", "BeamSpot",
        "V0Vertices", "V0Particles", "_V0Particles_decayVertex",
        "dEdxWires", "dEdxPads", "ElectromagneticClusters",
        "HadronicClusters", "EventHeader", "LuminousRegion",
    }
    want = ["RunInformation"]
    for b in t.keys():
        if "/" not in b:
            continue
        leaf = b.split("/")[-1]
        if leaf.split(".")[0] not in prefixes:
            continue
        if "[" in leaf and not leaf.endswith(("covMatrix.values[21]", "BeamSpot.covMatrix.values[6]")):
            continue
        want.append(leaf)
    ev = t.arrays(want, entry_stop=n)

    # --- RunInformation
    ri = ev["RunInformation"]
    print(f"  RunInformation length {set(ak.num(ri).tolist())}, first: {ri[0].tolist()}")

    # --- PID types, PDG ids
    ptype = ak.flatten(ev["ParticleID.type"]).to_numpy()
    print(f"  ParticleID.type counts {dict(zip(*np.unique(ptype, return_counts=True)))}")
    print(f"  ParticleID per RecoParticle: {bool(ak.all(ak.num(ev['ParticleID.type']) == ak.num(ev['RecoParticles.energy'])))}")
    rpdg = ak.flatten(ev["RecoParticles.PDG"]).to_numpy()
    print(f"  RecoParticles.PDG unique (first 10) {np.unique(rpdg)[:10]}")

    # --- bug 4: mass from the type
    mass = ak.flatten(ev["RecoParticles.mass"]).to_numpy()
    print(f"  mass: NaN {np.isnan(mass).sum()}, negative {(mass < 0).sum()}")
    for ty in np.unique(ptype):
        m = mass[ptype == ty]
        vals, cnt = np.unique(np.round(m, 5), return_counts=True)
        top = sorted(zip(cnt, vals), reverse=True)[:3]
        print(f"    type {ty}: n={len(m)}, top masses {[(float(v), int(c)) for c, v in top]}")

    # --- bug 1: jet constituents
    jb, je = ev["Jets.particles_begin"], ev["Jets.particles_end"]
    rel = ev["_Jets_particles.index"]
    relcid = ak.flatten(ev["_Jets_particles.collectionID"]).to_numpy()
    print(f"  jet constituent collections {[ids.get(int(c)) for c in np.unique(relcid)]}")
    n_reco = ak.num(ev["RecoParticles.energy"])
    n_jets = ak.num(jb)
    n_const = ak.num(rel)
    assigned = ak.sum(n_const) / ak.sum(n_reco)
    oor = ak.sum((rel < 0) | (rel >= n_reco))
    # duplicates within an event
    dup = 0
    rel_np = ak.to_list(rel[:2000])
    for r in rel_np:
        dup += len(r) - len(set(r))
    print(f"  jets/event {ak.mean(n_jets):.2f}, constituents/RecoParticles {assigned:.3f}, out of range {oor}, duplicates (2k ev) {dup}")
    # jet p4 vs sum of constituents
    safe = ak.where((rel >= 0) & (rel < n_reco), rel, 0)
    cE = ev["RecoParticles.energy"][safe]
    cpx = ev["RecoParticles.momentum.x"][safe]
    counts = ak.flatten(je - jb)
    flatE = ak.unflatten(ak.flatten(cE), counts)
    flatpx = ak.unflatten(ak.flatten(cpx), counts)
    jE = ak.flatten(ev["Jets.energy"]).to_numpy()
    jpx = ak.flatten(ev["Jets.momentum.x"]).to_numpy()
    sE = ak.to_numpy(ak.sum(flatE, axis=1))
    spx = ak.to_numpy(ak.sum(flatpx, axis=1))
    good = jE > 1
    dE = np.abs(sE[good] - jE[good]) / jE[good]
    print(
        f"  jet E vs sum constituents: median rel diff {np.median(dE):.2e}, >1%: {np.mean(dE > 0.01):.3%}, "
        f"px abs diff median {np.median(np.abs(spx - jpx)[good]):.2e}; sum jet E / sum reco E "
        f"{ak.sum(ev['Jets.energy']) / ak.sum(ev['RecoParticles.energy']):.3f}"
    )

    # --- track link, units, d0 sign
    tb, te = ev["RecoParticles.tracks_begin"], ev["RecoParticles.tracks_end"]
    has_track = (te - tb) > 0
    ntr_ref = ak.sum(te - tb)
    ntr = ak.sum(ak.num(ev["Tracks.chi2"]))
    print(f"  Tracks / referenced tracks {ntr / ntr_ref:.2f}; states per track {set(ak.flatten(ev['Tracks.trackStates_end'] - ev['Tracks.trackStates_begin']).tolist())}")
    trel = ev["_RecoParticles_tracks.index"]
    pos = ak.where(has_track, tb, 0)
    trel_p = ak.fill_none(ak.pad_none(trel, 1, axis=1), 0)
    tidx = trel_p[pos]
    sb = ak.fill_none(ak.pad_none(ev["Tracks.trackStates_begin"], 1, axis=1), 0)
    sidx = sb[tidx]
    TS = "_Tracks_trackStates"
    def st(name):
        a = ak.fill_none(ak.pad_none(ev[f"{TS}.{name}"], 1, axis=1), 0)
        return ak.to_numpy(ak.flatten(a[sidx][has_track]))
    D0, Z0, phi, om, tanl = st("D0"), st("Z0"), st("phi"), st("omega"), st("tanLambda")
    rx, ry, rz = st("referencePoint.x"), st("referencePoint.y"), st("referencePoint.z")
    print(f"  track reference point |max| {max(np.abs(rx).max(), np.abs(ry).max(), np.abs(rz).max()):.3g}")
    px = ak.to_numpy(ak.flatten(ev["RecoParticles.momentum.x"][has_track]))
    py = ak.to_numpy(ak.flatten(ev["RecoParticles.momentum.y"][has_track]))
    pt = np.hypot(px, py)
    bfield_T = 1.5
    if len(ri[0]) > 1:
        bfield_T = float(ri[0][1]) / 10.0  # kG -> T
    for unit, scale in (("1/cm", 100.0), ("1/mm", 1000.0)):
        pt_om = 0.299792458 * bfield_T / (np.abs(om) * scale)
        r = pt_om / pt
        print(f"  omega in {unit} (B={bfield_T:.4f} T): pt ratio median {np.median(r):.4f}, within 1%: {np.mean(np.abs(r - 1) < 0.01):.3f}")
    # charge sign vs omega sign
    q = ak.to_numpy(ak.flatten(ev["RecoParticles.charge"][has_track]))
    print(f"  sign(omega) == sign(charge): {np.mean(np.sign(om) == np.sign(q)):.3f}")
    # phi0 vs particle phi
    dphi = np.angle(np.exp(1j * (phi - np.arctan2(py, px))))
    print(f"  |phi0 - phi(p)| median {np.median(np.abs(dphi)):.2e}")
    # d0 sign: transverse distance to the event vertex, cm
    nv = ak.num(ev["Vertices.position.x"])
    print(f"  Vertices per event {dict(zip(*np.unique(ak.to_numpy(nv), return_counts=True)))}")
    ev_of_trk = ak.to_numpy(ak.flatten(ak.broadcast_arrays(ak.local_index(tb, axis=0), tb)[0][has_track]))
    vx = ak.to_numpy(ak.fill_none(ak.firsts(ev["Vertices.position.x"]), np.nan))[ev_of_trk]
    vy = ak.to_numpy(ak.fill_none(ak.firsts(ev["Vertices.position.y"]), np.nan))[ev_of_trk]
    ok = np.isfinite(vx) & (pt > 2)
    for sign in (+1, -1):
        # LCIO: PCA = (-d0 sin phi, d0 cos phi); d0 w.r.t. V along n = (-sin phi, cos phi)
        d0v = sign * D0 - (-vx * np.sin(phi) + vy * np.cos(phi))
        print(f"  d0 w.r.t. event vertex (LCIO sign x {sign:+d}), pT>2: core {10 * robust_sigma(d0v[ok]):.3f} mm")
    bs = ev["BeamSpot.position.x"]
    print(f"  BeamSpot per event {set(ak.num(bs).tolist())}, x,y,z sample {ak.firsts(bs)[0]}, {ak.firsts(ev['BeamSpot.position.y'])[0]}, {ak.firsts(ev['BeamSpot.position.z'])[0]}")
    print(f"  LuminousRegion sample {ak.firsts(ev['LuminousRegion.position.x'])[0]}, {ak.firsts(ev['LuminousRegion.position.y'])[0]}, {ak.firsts(ev['LuminousRegion.position.z'])[0]}")
    print(f"  Vertex z spread {np.nanstd(ak.to_numpy(ak.flatten(ev['Vertices.position.z']))):.3f} (cm if ~1)")

    # --- bug 2: V0 vertex x
    vtx_x = ak.flatten(ev["V0Vertices.position.x"]).to_numpy()
    vtx_y = ak.flatten(ev["V0Vertices.position.y"]).to_numpy()
    print(f"  V0Vertices {len(vtx_x)}: x mean/std {vtx_x.mean():.3f}/{vtx_x.std():.3f}, y mean/std {vtx_y.mean():.3f}/{vtx_y.std():.3f}")
    dvidx = ev["_V0Particles_decayVertex.index"]
    dvcid = ak.flatten(ev["_V0Particles_decayVertex.collectionID"]).to_numpy()
    print(f"  V0Particles {ak.sum(ak.num(ev['V0Particles.energy']))}, decayVertex links {ak.sum(ak.num(dvidx))} -> {[ids.get(int(c)) for c in np.unique(dvcid)]}")
    if ak.sum(ak.num(dvidx)) == ak.sum(ak.num(ev["V0Particles.energy"])) and len(vtx_x):
        sx = ev["V0Vertices.position.x"][dvidx]
        sy = ev["V0Vertices.position.y"][dvidx]
        x = ak.to_numpy(ak.flatten(sx)); y = ak.to_numpy(ak.flatten(sy))
        vpx = ak.to_numpy(ak.flatten(ev["V0Particles.momentum.x"]))
        vpy = ak.to_numpy(ak.flatten(ev["V0Particles.momentum.y"]))
        r = np.hypot(x, y)
        sel = r > 1.0
        cosang = (x * vpx + y * vpy)[sel] / (r[sel] * np.hypot(vpx, vpy)[sel])
        cos_swap = (y * vpy)[sel] / (r[sel] * np.hypot(vpx, vpy)[sel])
        print(f"  V0 flight direction vs momentum (r>1 cm, n={sel.sum()}): median cos {np.median(cosang):.4f}, cos>0.99: {np.mean(cosang > 0.99):.3f}")
    else:
        # fall back to the V0 tracks: the vertex should lie near both daughter helices
        print("  V0 -> vertex link not 1:1, skipped direction check")

    # --- bug 3: dE/dx
    for coll in ("dEdxWires", "dEdxPads"):
        ty = ak.flatten(ev[f"{coll}.dQdx.type"]).to_numpy()
        val = ak.flatten(ev[f"{coll}.dQdx.value"]).to_numpy()
        err = ak.flatten(ev[f"{coll}.dQdx.error"]).to_numpy()
        tyc = dict(zip(*np.unique(ty, return_counts=True)))
        print(
            f"  {coll}: n={len(val)} (/{int(ntr_ref)} track-PFOs), types {dict(list(tyc.items())[:8])}, value==-999 {np.sum(val == -999)}, "
            f"value<0 other {np.sum((val < 0) & (val != -999))}, err<=0 {np.sum(err <= 0)}, value median {np.median(val[val > 0]) if np.any(val > 0) else np.nan:.4g}"
        )

    # --- bug 5: cluster associations
    nbad = relation_check(ev, ids, "RecoParticles", "_RecoParticles_clusters", "RecoParticle->cluster")
    # cluster iPhi/iTheta vs the direction of the particle it belongs to (positions are not filled)
    cb, ce = ev["RecoParticles.clusters_begin"], ev["RecoParticles.clusters_end"]
    has_cl = (ce - cb) > 0
    crel = ak.fill_none(ak.pad_none(ev["_RecoParticles_clusters.index"], 1, axis=1), 0)
    ccid = ak.fill_none(ak.pad_none(ev["_RecoParticles_clusters.collectionID"], 1, axis=1), 0)
    first = ak.where(has_cl, cb, 0)
    cidx, cc = crel[first][has_cl], ccid[first][has_cl]
    ppx = ev["RecoParticles.momentum.x"][has_cl]
    ppy = ev["RecoParticles.momentum.y"][has_cl]
    ppz = ev["RecoParticles.momentum.z"][has_cl]
    ptype_cl = ev["ParticleID.type"][has_cl]
    print(f"  PFOs with a cluster link, by type: {dict(zip(*np.unique(ak.to_numpy(ak.flatten(ptype_cl)), return_counts=True)))}")
    for coll in ("ElectromagneticClusters", "HadronicClusters"):
        cid_num = [k for k, v in ids.items() if v == coll][0]
        sel = cc == cid_num
        if ak.sum(sel) == 0:
            print(f"  {coll}: no PFO links")
            continue
        n_cl = ak.num(ev[f"{coll}.energy"])
        idx = ak.where(sel & (cidx < n_cl), cidx, 0)
        ip = ak.fill_none(ak.pad_none(ev[f"{coll}.iPhi"], 1, axis=1), 0)[idx][sel]
        it = ak.fill_none(ak.pad_none(ev[f"{coll}.iTheta"], 1, axis=1), 0)[idx][sel]
        php = np.arctan2(ak.to_numpy(ak.flatten(ppy[sel])), ak.to_numpy(ak.flatten(ppx[sel])))
        thp = np.arctan2(np.hypot(ak.to_numpy(ak.flatten(ppx[sel])), ak.to_numpy(ak.flatten(ppy[sel]))), ak.to_numpy(ak.flatten(ppz[sel])))
        dphi = np.angle(np.exp(1j * (ak.to_numpy(ak.flatten(ip)) - php)))
        dth = ak.to_numpy(ak.flatten(it)) - thp
        print(f"  {coll} via PFO link (n={len(dphi)}): |iPhi - phi(PFO)| median {np.median(np.abs(dphi)):.3g}, |iTheta - theta(PFO)| median {np.median(np.abs(dth)):.3g}")
    bsx = ak.to_numpy(ak.firsts(ev["BeamSpot.position.x"]))
    bsy = ak.to_numpy(ak.firsts(ev["BeamSpot.position.y"]))
    runs = ak.to_numpy(ak.firsts(ev["EventHeader.runNumber"]))
    print(f"  BeamSpot: {len(np.unique(runs))} runs, {len(np.unique(bsx))} distinct x; x range [{bsx.min():.4f}, {bsx.max():.4f}] cm, y [{bsy.min():.4f}, {bsy.max():.4f}] cm; cov[0] {ev['BeamSpot.covMatrix.values[6]'][0][0].tolist()}")
    # --- bug 7: cluster phi / theta
    for coll in ("ElectromagneticClusters", "HadronicClusters"):
        x = ak.flatten(ev[f"{coll}.position.x"]).to_numpy()
        y = ak.flatten(ev[f"{coll}.position.y"]).to_numpy()
        z = ak.flatten(ev[f"{coll}.position.z"]).to_numpy()
        iphi = ak.flatten(ev[f"{coll}.iPhi"]).to_numpy()
        ith = ak.flatten(ev[f"{coll}.iTheta"]).to_numpy()
        e = ak.flatten(ev[f"{coll}.energy"]).to_numpy()
        pos_ok = (np.hypot(x, y) > 0)
        dphi = np.angle(np.exp(1j * (iphi - np.arctan2(y, x))))[pos_ok]
        dth = (ith - np.arctan2(np.hypot(x, y), z))[pos_ok]
        print(
            f"  {coll}: n={len(e)}, position != 0: {pos_ok.mean():.3f}, iPhi range [{iphi.min():.3g}, {iphi.max():.3g}], "
            f"iTheta range [{ith.min():.3g}, {ith.max():.3g}], |iPhi - phi(pos)| median {np.median(np.abs(dphi)) if pos_ok.any() else np.nan:.3g}, "
            f"|iTheta - theta(pos)| median {np.median(np.abs(dth)) if pos_ok.any() else np.nan:.3g}, E<=0 {np.sum(e <= 0)}"
        )

    # --- bug 6: -1 / empty collections
    print("  branches with > 1% entries == -1, and empty ones:")
    for b in want:
        if b.startswith("_") or b.endswith(("_begin", "_end")) or b == "RunInformation":
            continue
        a = ev[b]
        try:
            v = ak.to_numpy(ak.flatten(a, axis=None))
        except Exception:
            continue
        if v.dtype.kind not in "fiu":
            continue
        if len(v) == 0:
            print(f"    {b}: EMPTY in all {n} events")
            continue
        frac = np.mean(v == -1)
        if frac > 0.01:
            print(f"    {b}: {frac:.3f} == -1")
    return is_mc


def check_mc(path, max_events):
    t = uproot.open(path)["events"]
    n = min(t.num_entries, max_events)
    ev = t.arrays(
        ["MCParticles.PDG", "MCParticles.charge", "MCParticles.generatorStatus", "MCParticles.simulatorStatus",
         "GeneratorInfo.sqrts", "GeneratorInfo.crossSections_begin", "GeneratorInfo.crossSections_end",
         "_GeneratorInfo_crossSections", "trackMCLink.weight", "ecalClusterMCLink.weight"],
        entry_stop=n,
    )
    pdg = ak.flatten(ev["MCParticles.PDG"]).to_numpy()
    q = ak.flatten(ev["MCParticles.charge"]).to_numpy()
    gs = ak.flatten(ev["MCParticles.generatorStatus"]).to_numpy()
    print(f"  MC: particles {len(pdg)}, PDG==0 {np.sum(pdg == 0)}, PDG==-99999 {np.sum(pdg == -99999)}")
    for code, exp in ((211, 1), (-211, -1), (11, -1), (13, -1), (321, 1), (2212, 1), (22, 0), (130, 0), (111, 0)):
        s = pdg == code
        if s.any():
            print(f"    PDG {code}: n={s.sum()}, charge {dict(zip(*np.unique(q[s], return_counts=True)))} (expected {exp})")
    print(f"    generatorStatus {dict(zip(*np.unique(gs, return_counts=True)))}")
    print(f"    GeneratorInfo sqrts {ak.flatten(ev['GeneratorInfo.sqrts'])[:3].tolist()}, crossSections {ev['_GeneratorInfo_crossSections'][0].tolist()}")
    print(f"    trackMCLink per event {ak.mean(ak.num(ev['trackMCLink.weight'])):.2f}, ecalClusterMCLink {ak.mean(ak.num(ev['ecalClusterMCLink.weight'])):.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--max-events", type=int, default=20000)
    a = p.parse_args()
    for path in a.files:
        if check(path, a.max_events):
            check_mc(path, a.max_events)
