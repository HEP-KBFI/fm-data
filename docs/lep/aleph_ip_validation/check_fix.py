import sys, numpy as np, awkward as ak
sys.path.insert(0, "/home/laurits/FM/code/fm-data")
from ntupelizer.tools import aleph
def rsig(x):
    x=x[np.isfinite(x)]; q75,q25=np.percentile(x,[75,25]); return (q75-q25)/1.349
for sign in [1.0, -1.0]:
    aleph.ALEPH_D0_SIGN = sign
    inp = aleph.aleph_inputs(sys.argv[1]); p = inp["particles"]
    hv = np.asarray(inp["has_vertex"]) & (ak.to_numpy(ak.num(p["part_px"])) > 0)
    p = {k: v[hv] for k, v in p.items()}
    inp["vertex"] = {a: np.asarray(v)[hv] for a, v in inp["vertex"].items()}
    hv = hv[hv]
    pt = np.sqrt(p["part_px"]**2 + p["part_py"]**2)
    s = (p["part_has_track"] == 1) & (pt > 2) & ak.broadcast_arrays(ak.Array(hv), p["part_px"])[0]
    dxy = ak.to_numpy(ak.flatten(p["part_dxy"][s])); sig = ak.to_numpy(ak.flatten(p["part_dxy_sig"][s]))
    # lifetime sign w.r.t. the hemisphere axis (transverse), PCA - vertex
    vx = ak.Array(inp["vertex"]["x"]); vy = ak.Array(inp["vertex"]["y"])
    lead = ak.argmax(p["part_energy"], axis=1, keepdims=True)
    n = [ak.firsts(p[f"part_p{a}"][lead]) for a in "xyz"]
    side = (p["part_px"]*n[0] + p["part_py"]*n[1] + p["part_pz"]*n[2]) > 0
    hx = ak.where(side, ak.sum(p["part_px"]*side, axis=1), ak.sum(p["part_px"]*~side, axis=1))
    hy = ak.where(side, ak.sum(p["part_py"]*side, axis=1), ak.sum(p["part_py"]*~side, axis=1))
    lsign = np.sign((p["part_pca_x"]-vx)*hx + (p["part_pca_y"]-vy)*hy)
    L = ak.to_numpy(ak.flatten((np.abs(p["part_dxy"])*lsign)[s]))
    pp, mm = np.mean(L > 0.2), np.mean(L < -0.2)
    print(f"ALEPH_D0_SIGN={sign:+.0f}: tracks {len(dxy)}, dxy core {1e3*rsig(dxy):.0f} um, |sig| median {np.median(np.abs(sig)):.2f}, lifetime asym (|dxy|>200um) {(pp-mm)/(pp+mm):+.3f}")
    # 2D: d0 (LCIO after the sign) w.r.t. the vertex, linear approximation
    phi = p["part_phi0"]
    d2 = p["part_d0"] - (-vx*np.sin(phi) + vy*np.cos(phi))
    D2 = ak.to_numpy(ak.flatten(d2[s])); E2 = ak.to_numpy(ak.flatten(p["part_d0_err"][s]))
    # sign of the 2D d0: PCA_2D - V = d2 * (-sin, cos) (LCIO)
    l2 = np.sign(d2 * (-np.sin(phi)*hx + np.cos(phi)*hy))
    L2 = ak.to_numpy(ak.flatten((np.abs(d2)*l2)[s])); pp, mm = np.mean(L2 > 0.2), np.mean(L2 < -0.2)
    print(f"   2D d0 wrt vertex: core {1e3*rsig(D2):.0f} um, pull core {rsig(D2/E2):.2f}, lifetime asym {(pp-mm)/(pp+mm):+.3f}")
