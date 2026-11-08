"""Verify the dR_cut_explanation.txt hypothesis on real ALEPH jet-level ntuples.

Hypothesis: ALEPH jets are clustered with R=0.8 as a TRUE 3D OPENING ANGLE
(ee_genkt_algorithm), but the constituent-removal curve is computed with a box
cut in (eta, phi). Because Delta_eta, Delta_phi ~ alpha / sin(theta_jet), the
sharp clustering edge at alpha=0.8 gets smeared across polar angle into the
smooth eta-phi curve. Central jets (theta~90 deg) keep the sharp edge at 0.8;
forward jets have the edge stretched out to ~0.8/sin(theta).
"""
import glob

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FILES = sorted(glob.glob("/local/laurits/ALEPH/ALEPH/ALEPH_jetLevel_raw/data_*_1.parquet"))
print(f"Found {len(FILES)} parquet files")

# Use a handful of files for good statistics without loading everything.
N_FILES = 8
arrs = [ak.from_parquet(f) for f in FILES[:N_FILES]]
arr = ak.concatenate(arrs, axis=0)
print(f"Loaded {len(arr)} jets from {N_FILES} files")

# ---------------------------------------------------------------------------
# Per-particle quantities (broadcast jet-level fields onto particle axis)
# ---------------------------------------------------------------------------
jet_eta_b, part_dummy = ak.broadcast_arrays(arr.jet_eta, arr.part_deta)
jet_phi_b, _ = ak.broadcast_arrays(arr.jet_phi, arr.part_deta)
jet_pt_b, _ = ak.broadcast_arrays(arr.jet_pt, arr.part_deta)

# jet momentum components (massless-direction reconstruction from pt/eta/phi;
# only the DIRECTION matters for the opening angle, so jet mass is irrelevant)
jet_px = jet_pt_b * np.cos(jet_phi_b)
jet_py = jet_pt_b * np.sin(jet_phi_b)
jet_pz = jet_pt_b * np.sinh(jet_eta_b)
jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)

part_p = np.sqrt(arr.part_px**2 + arr.part_py**2 + arr.part_pz**2)

cos_alpha = (arr.part_px * jet_px + arr.part_py * jet_py + arr.part_pz * jet_pz) / (
    part_p * jet_p
)
cos_alpha = ak.where(cos_alpha > 1.0, 1.0, cos_alpha)
cos_alpha = ak.where(cos_alpha < -1.0, -1.0, cos_alpha)
alpha = np.arccos(cos_alpha)

costheta_jet = np.tanh(jet_eta_b)  # cos(theta) of the jet axis direction
sintheta_jet = np.sqrt(1.0 - costheta_jet**2)

dR_etaphi = arr.part_deltaR  # sqrt(deta^2 + dphi^2), already in the ntuple
deta = arr.part_deta
dphi = arr.part_dphi

# Flatten everything to a flat per-constituent array for histogramming.
f_deta = ak.to_numpy(ak.flatten(deta))
f_dphi = ak.to_numpy(ak.flatten(dphi))
f_dR = ak.to_numpy(ak.flatten(dR_etaphi))
f_alpha = ak.to_numpy(ak.flatten(alpha))
f_costheta = ak.to_numpy(ak.flatten(costheta_jet))
f_sintheta = ak.to_numpy(ak.flatten(sintheta_jet))

n_total = len(f_deta)
print(f"Total constituents: {n_total}")

# ---------------------------------------------------------------------------
# Plot 1: reproduce the original "% removed vs box cut c" curve (sanity check
# against the numbers quoted in dR_cut_explanation.txt)
# ---------------------------------------------------------------------------
cuts = np.linspace(0.05, 3.2, 80)
frac_removed_box = [
    np.mean((f_deta >= c) | (f_dphi >= c)) * 100 for c in cuts
]

check_points = [0.8, 1.15, 3.0]
print("Reproduced ALEPH box-cut curve at reference points:")
for c in check_points:
    frac = np.mean((f_deta >= c) | (f_dphi >= c)) * 100
    print(f"  c={c}: {frac:.2f}% removed")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(cuts, frac_removed_box, color="C0")
for c, label_y in zip(check_points, [10.37, 6.01, 0.60]):
    ax.axvline(c, color="gray", ls=":", lw=0.8)
ax.set_xlabel(r"box cut $c$  ($|\Delta\eta|<c$ and $|\Delta\phi|<c$)")
ax.set_ylabel("% of constituents removed")
ax.set_title("Reproduced ALEPH curve (box cut in $\\eta,\\phi$)")
ax.set_yscale("log")
fig.tight_layout()
fig.savefig("plot1_reproduced_box_cut_curve.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2: split the same box-cut curve into bins of jet |cos(theta)|
# ---------------------------------------------------------------------------
costheta_bins = [
    (0.0, 0.3),
    (0.3, 0.6),
    (0.6, 0.8),
    (0.8, 0.9),
    (0.9, 0.95),
    (0.95, 1.0),
]

fig, ax = plt.subplots(figsize=(7, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(costheta_bins)))
for (lo, hi), color in zip(costheta_bins, colors):
    mask = (np.abs(f_costheta) >= lo) & (np.abs(f_costheta) < hi)
    n_bin = mask.sum()
    if n_bin == 0:
        continue
    frac = [
        np.mean((f_deta[mask] >= c) | (f_dphi[mask] >= c)) * 100 for c in cuts
    ]
    mid_sin = np.sqrt(1 - ((lo + hi) / 2) ** 2)
    expected_edge = 0.8 / mid_sin
    ax.plot(
        cuts,
        frac,
        color=color,
        label=f"|cos θ| in [{lo},{hi}) (n={n_bin}, pred. edge≈{expected_edge:.2f})",
    )
    ax.axvline(min(expected_edge, cuts[-1]), color=color, ls=":", lw=0.8)

ax.axvline(0.8, color="red", ls="--", lw=1.2, label="c = 0.8 (clustering R)")
ax.set_xlabel(r"box cut $c$  ($|\Delta\eta|<c$ and $|\Delta\phi|<c$)")
ax.set_ylabel("% of constituents removed (within bin)")
ax.set_title(r"Box-cut curve split by jet $|\cos\theta|$")
ax.set_yscale("log")
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig("plot2_split_by_costheta.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 3: fraction removed vs a cut on the TRUE opening angle alpha
# ---------------------------------------------------------------------------
alpha_cuts = np.linspace(0.05, 3.14, 80)
frac_removed_alpha_total = [np.mean(f_alpha >= c) * 100 for c in alpha_cuts]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(alpha_cuts, frac_removed_alpha_total, color="black", lw=1.5, label="all jets")
for (lo, hi), color in zip(costheta_bins, colors):
    mask = (np.abs(f_costheta) >= lo) & (np.abs(f_costheta) < hi)
    n_bin = mask.sum()
    if n_bin == 0:
        continue
    frac = [np.mean(f_alpha[mask] >= c) * 100 for c in alpha_cuts]
    ax.plot(alpha_cuts, frac, color=color, lw=0.8, alpha=0.7, label=f"|cosθ| in [{lo},{hi})")

ax.axvline(0.8, color="red", ls="--", lw=1.2, label="α = 0.8 = clustering R")
ax.set_xlabel(r"cut on true 3D opening angle $\alpha$ [rad]")
ax.set_ylabel("% of constituents removed")
ax.set_title(r"Fraction removed vs cut on true opening angle $\alpha$")
ax.set_yscale("log")
ax.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig("plot3_alpha_cut_curve.png", dpi=150)
plt.close(fig)

print("Alpha-cut curve at reference points (all jets):")
for c in check_points:
    frac = np.mean(f_alpha >= c) * 100
    print(f"  alpha={c}: {frac:.2f}% removed")

# ---------------------------------------------------------------------------
# Plot 4: 2D histogram of dR_etaphi vs alpha/sin(theta_jet) -- shows the
# 1/sin(theta) stretching directly, should lie close to the y=x line.
# ---------------------------------------------------------------------------
alpha_over_sin = f_alpha / f_sintheta

fig, ax = plt.subplots(figsize=(6, 5.5))
finite = np.isfinite(alpha_over_sin) & (alpha_over_sin < 10)
h = ax.hist2d(
    alpha_over_sin[finite],
    f_dR[finite],
    bins=100,
    range=[[0, 6], [0, 6]],
    cmap="viridis",
    norm=matplotlib.colors.LogNorm(),
)
ax.plot([0, 6], [0, 6], color="red", lw=1, ls="--", label="y = x")
ax.set_xlabel(r"$\alpha / \sin\theta_{\rm jet}$")
ax.set_ylabel(r"$\Delta R_{\eta\phi} = \sqrt{\Delta\eta^2+\Delta\phi^2}$")
ax.set_title("Small-angle stretching: $\\Delta R_{\\eta\\phi} \\approx \\alpha/\\sin\\theta$")
fig.colorbar(h[3], ax=ax, label="constituents")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("plot4_dR_vs_alpha_over_sin.png", dpi=150)
plt.close(fig)

print("Done. Plots saved in current directory.")
