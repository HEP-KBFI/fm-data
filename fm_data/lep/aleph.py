"""Read the ALEPH data in EDM4hep format (LEP data revival, l2e) into the event table (see
`fm_data.common.events.build_event_table`) and the (unlabelled) jet tables.

Validated on the l2e-prod-v1 production (docs/lep/l2e_validation/README.md). The bugs fixed in
that production (jet constituents, V0 vertex x, dE/dx status, PFO mass, cluster links, bank
reading) need no workaround here; the conventions below are still those of the files.

Conventions taken over from lep-data (lep_data/aleph/tools/create_aleph.py, commit cda3713):
- lengths in the ALEPH EDM4hep files are in cm although EDM4hep declares mm (CM_TO_MM),
- the particle type comes from ParticleID.type (one entry per RecoParticle, same order),
- the particle -> track link goes through the relation collection _RecoParticles_tracks and
  Tracks.trackStates_begin (one track state per track, at the IP).
The impact parameters are computed with the (fixed) functions of ntupelizer.tools.lifetime,
i.e. with the LCIO d0 sign convention. The ALEPH files use the OPPOSITE d0 sign (PCA =
(+d0 sin phi0, -d0 cos phi0) w.r.t. the reference point, which is (0, 0, 0) for all ALEPH track
states), so D0 is negated on reading (ALEPH_D0_SIGN). Found 2026-11-05 (docs/aleph_ip_validation.md):
with the file sign, d0 w.r.t. the event vertex has a 1.4 mm core and a negative lifetime-sign
asymmetry; negated, 0.16 mm and +23%. The ALEPH ntuples produced before 2026-11-05 have the
wrong sign (part_d0, part_dxy/dz/d3, PCA, significances).

Reference point of the impact parameters (ALEPH_IP_VERSION = 3): the ALEPH event vertex
(Vertices) is missing in 60-90% of the events, often far off (vertices from <= 2 tracks) and
its z is known only to ~0.6 mm, which spoils the 3D-PCA dxy. Instead:
- x, y: the ALEPH beam spot (BeamSpot collection, new in l2e-prod-v1, per block of events).
  Where it is missing or (0, 0) (some 1992 runs), the beam spot of the run from a fit
  d0 = -x_bs sin(phi0) + y_bs cos(phi0) of the tracks with pT > 1 GeV (iterative 3 sigma
  clipping; runs with too few tracks use the fit of the whole file); the two agree to
  ~20-30 um. The transverse beam size (sigma_x, sigma_y) is always measured from the fit
  residuals (the BeamSpot covariance is the position error, not the beam size) and added to
  the dxy error (sigma_x^2 sin^2 phi0 + sigma_y^2 cos^2 phi0);
- z: per event, the median z of the tracks (pT > 0.5 GeV) at their transverse PCA to the beam
  spot;
- the PCA is the transverse (2D) one: part_dxy = 2D distance to the beam spot, part_dz = z at
  the 2D PCA - z_ref, part_d3 = sqrt(dxy^2 + dz^2).
The ALEPH vertex is kept in event_vertex_*/event_has_vertex; the reference point is stored in
event_ref_{x,y,z} (event_ref_from_beamspot = 1 if x, y come from the BeamSpot collection), the
beam size in event_beam_sigma_{x,y} (all in mm). part_d0/part_z0 stay the track parameters
w.r.t. (0, 0, 0) (sign-corrected). The jet tables sign the impact parameters w.r.t. event_ref_*,
with the transverse components for dxy and d3 and the z components for dz (ip_signing =
"transverse"): with the 3D dot product the z offset of the transverse PCA (the dz of the track,
~mm) decides the sign and dilutes the lifetime asymmetry (N(sig > 3) / N(sig < -3) = 1.6
instead of 2.6 in 1994_old).

Other choices (decided 2026-11-08):
- sqrt(s) is the nominal value of the config (91.2 GeV) for all events: RunInformation[0] (LEP
  beam energy) is in GeV for part of 1990, -1/-1000 or meaningless (15, 20200, 50000) for some
  runs of 1990-1993 and 1995.
- PFO type 3 (charged track from a V0) gets the charged-pion mass, like type 0; the files have
  mass 0 for it.
"""

import awkward as ak
import numpy as np
import uproot

from fm_data.common import jets as jt
from fm_data.common import lifetime as lt
from fm_data.common.events import LIFETIME_FIELDS, add_significances, build_event_table

# lengths in the ALEPH files are in cm (measured in lep-data: interpreting omega as 1/cm
# with B = 1.5 T reproduces the particle momenta), EDM4hep uses mm
CM_TO_MM = 10.0
# d0 sign of the ALEPH track states relative to LCIO, see the module docstring
ALEPH_D0_SIGN = -1.0
# reference point of the impact parameters, see the module docstring
ALEPH_IP_VERSION = 3
B_FIELD_T = 1.5
BEAMSPOT_MIN_PT = 1.0  # GeV, tracks for the beam-spot fit
BEAMSPOT_MAX_D0 = 2.0  # mm, start of the clipping
BEAMSPOT_MIN_TRACKS = 2000  # per run, otherwise the fit of the whole file
ZREF_MIN_PT = 0.5  # GeV, tracks for the reference z
CHARGED_PION_MASS = 0.13957039  # GeV, as set by the converter for PFO type 0

TRACK_STATES = "_Tracks_trackStates"

BRANCHES = (
    [
        f"RecoParticles.{name}"
        for name in [
            "energy",
            "momentum.x",
            "momentum.y",
            "momentum.z",
            "mass",
            "charge",
            "tracks_begin",
            "tracks_end",
        ]
    ]
    + ["_RecoParticles_tracks.index", "Tracks.trackStates_begin", "ParticleID.type"]
    + [
        f"{TRACK_STATES}.{name}"
        for name in [
            "D0",
            "phi",
            "omega",
            "Z0",
            "tanLambda",
            "referencePoint.x",
            "referencePoint.y",
            "referencePoint.z",
            "covMatrix.values[21]",
        ]
    ]
    + [f"Vertices.position.{axis}" for axis in "xyz"]
    + [f"BeamSpot.position.{axis}" for axis in "xy"]
    + ["EventHeader.eventNumber", "EventHeader.runNumber"]
)


def pid_flags(pid_type):
    """Particle-type flags from ParticleID.type (mapping as in lep-data, decided 2026-09-30):
    0 charged hadron, 1 electron, 2 muon, 3 V0 daughter track (-> charged hadron),
    4 photon, 5 neutral hadron, 7/8 forward calorimeter-only objects (-> photon)."""
    return {
        "part_isElectron": ak.values_astype(pid_type == 1, np.int32),
        "part_isMuon": ak.values_astype(pid_type == 2, np.int32),
        "part_isPhoton": ak.values_astype(
            (pid_type == 4) | (pid_type == 7) | (pid_type == 8), np.int32
        ),
        "part_isChargedHadron": ak.values_astype((pid_type == 0) | (pid_type == 3), np.int32),
        "part_isNeutralHadron": ak.values_astype(pid_type == 5, np.int32),
    }


def pdg_from_flags(flags, charge):
    """A PDG id per PFO from the particle type (the ALEPH RecoParticles.PDG are all 0)."""
    sign = ak.values_astype(np.sign(charge), np.int32)
    return (
        flags["part_isElectron"] * (-11 * sign)
        + flags["part_isMuon"] * (-13 * sign)
        + flags["part_isPhoton"] * 22
        + flags["part_isChargedHadron"] * (211 * sign)
        + flags["part_isNeutralHadron"] * 130
    )


def read_events(filepath, tree="events"):
    with uproot.open(filepath) as f:
        return f[tree].arrays(BRANCHES)


def _track_state_index(events):
    """Global (flat) index of the track state of each PFO and whether it has a track,
    both [events, PFOs] (from lep-data's find_linked_indices)."""
    begins = events["RecoParticles.tracks_begin"]
    has_track = (events["RecoParticles.tracks_end"] - begins) > 0
    relation = ak.fill_none(ak.pad_none(events["_RecoParticles_tracks.index"], 1, axis=1), 0)
    state_begin = ak.fill_none(ak.pad_none(events["Tracks.trackStates_begin"], 1, axis=1), 0)
    relation_position = ak.values_astype(ak.where(has_track, begins, 0), np.int64)
    track_index = ak.values_astype(relation[relation_position], np.int64)
    state_index = ak.values_astype(state_begin[track_index], np.int64)
    n_states = ak.to_numpy(ak.num(events[f"{TRACK_STATES}.D0"]))
    offsets = np.concatenate([[0], np.cumsum(n_states)])[:-1]
    global_index = state_index + offsets[:, None]
    return global_index, has_track


def _robust_sigma(x):
    q75, q25 = np.percentile(x, [75, 25])
    return (q75 - q25) / 1.349


def fit_beamspot(d0, phi0, d0_error):
    """Beam spot (x, y) and transverse beam size (sigma_x, sigma_y) in mm from the d0 (LCIO
    sign, w.r.t. (0, 0)) and phi0 of prompt-ish tracks: d0 = -x sin(phi0) + y cos(phi0),
    iterative 3 sigma clipping of the residuals."""
    design = np.stack([-np.sin(phi0), np.cos(phi0)], axis=1)
    keep = np.abs(d0) < BEAMSPOT_MAX_D0
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(design[keep], d0[keep], rcond=None)
        residual = d0 - design @ coef
        keep = np.abs(residual) < 3 * max(_robust_sigma(residual[keep]), 1e-3)
    sigmas = []
    for direction in (np.sin(phi0) ** 2, np.cos(phi0) ** 2):
        sel = keep & (direction > 0.8)
        if sel.sum() < 50:
            sigmas.append(0.0)
            continue
        # residual^2 ~ sigma_beam^2 * direction + d0_error^2
        excess = _robust_sigma(residual[sel]) ** 2 - np.median(d0_error[sel] ** 2)
        sigmas.append(float(np.sqrt(max(excess, 0.0) / np.mean(direction[sel]))))
    return float(coef[0]), float(coef[1]), sigmas[0], sigmas[1]


def beamspot_per_run(run_of_track, d0, phi0, d0_error, pt):
    """{run: (x, y, sigma_x, sigma_y)} for all runs of the file."""
    use = pt > BEAMSPOT_MIN_PT
    file_fit = fit_beamspot(d0[use], phi0[use], d0_error[use])
    result = {}
    for run in np.unique(run_of_track):
        sel = use & (run_of_track == run)
        result[run] = (
            fit_beamspot(d0[sel], phi0[sel], d0_error[sel])
            if sel.sum() >= BEAMSPOT_MIN_TRACKS
            else file_fit
        )
    return result


def _event_median(values, event_of_value, n_events, default=0.0):
    """Median of the values of each event, `default` for events without values."""
    order = np.lexsort((values, event_of_value))
    values, events = values[order], event_of_value[order]
    counts = np.bincount(events, minlength=n_events)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    median = np.full(n_events, default, dtype=np.float64)
    has = counts > 0
    lo = starts[has] + (counts[has] - 1) // 2
    hi = starts[has] + counts[has] // 2
    median[has] = 0.5 * (values[lo] + values[hi])
    return median


def _lifetime_columns(events):
    """Impact parameters etc. of all PFOs [events, PFOs] (0 for PFOs without a track) w.r.t.
    the reference point (beam spot of the run, median track z of the event), and the
    reference point per event."""
    global_index, has_track = _track_state_index(events)
    counts = ak.num(has_track)
    n_events = len(counts)
    has_track_flat = ak.to_numpy(ak.flatten(has_track))
    index_flat = ak.to_numpy(ak.flatten(global_index))[has_track_flat]

    def state(name, scale=1.0):
        return ak.to_numpy(ak.flatten(events[f"{TRACK_STATES}.{name}"]))[index_flat] * scale

    d0 = state("D0", ALEPH_D0_SIGN * CM_TO_MM)
    z0 = state("Z0", CM_TO_MM)
    phi0 = state("phi")
    tanL = state("tanLambda")
    omega = state("omega", 1 / CM_TO_MM)
    xr, yr, zr = (state(f"referencePoint.{axis}", CM_TO_MM) for axis in "xyz")
    cov = ak.to_numpy(ak.flatten(events[f"{TRACK_STATES}.covMatrix.values[21]"]))[index_flat]
    # diagonal of the (d0, phi, omega, z0, tanLambda, time) covariance (lower triangle)
    d0_error = CM_TO_MM * np.sqrt(np.maximum(cov[:, 0], 0))
    phi0_error = np.sqrt(np.maximum(cov[:, 2], 0))
    z0_error = CM_TO_MM * np.sqrt(np.maximum(cov[:, 9], 0))
    tanL_error = np.sqrt(np.maximum(cov[:, 14], 0))
    # pT from the curvature (omega in 1/mm)
    pt = 0.299792458 * B_FIELD_T / np.maximum(np.abs(omega) * 1000.0, 1e-12)

    event_of_pfo = ak.to_numpy(
        ak.flatten(ak.broadcast_arrays(ak.local_index(has_track, axis=0), has_track)[0])
    )[has_track_flat]
    run = ak.to_numpy(ak.firsts(events["EventHeader.runNumber"])).astype(np.int64)
    x0, y0, z0_prime = lt.calc_track_origin_vectorized(phi0, d0, z0, xr, yr, zr)

    # reference point: beam spot (x, y) per run (d0 w.r.t. (0, 0) since xr = yr = 0 for ALEPH)
    d0_origin = -np.sin(phi0) * x0 + np.cos(phi0) * y0
    beamspot = beamspot_per_run(run[event_of_pfo], d0_origin, phi0, d0_error, pt)
    bs = np.array([beamspot[r] for r in run]).reshape(n_events, 4)
    ref_x, ref_y, sigma_x, sigma_y = bs.T
    # x, y from the BeamSpot collection where available (cm -> mm), the fit otherwise
    file_x, file_y = (
        CM_TO_MM * ak.to_numpy(ak.fill_none(ak.firsts(events[f"BeamSpot.position.{axis}"]), np.nan))
        for axis in "xy"
    )
    from_file = np.isfinite(file_x) & np.isfinite(file_y) & ~((file_x == 0) & (file_y == 0))
    ref_x = np.where(from_file, file_x, ref_x)
    ref_y = np.where(from_file, file_y, ref_y)
    vx, vy = ref_x[event_of_pfo], ref_y[event_of_pfo]
    # z: median z of the tracks at the transverse PCA to the beam spot
    s = lt.calc_pca_arc_length_vectorized(phi0, tanL, x0, y0, z0_prime, vx, vy, 0.0, use_2d=True)
    z_at_pca = s * tanL + z0_prime
    use_z = pt > ZREF_MIN_PT
    ref_z = _event_median(z_at_pca[use_z], event_of_pfo[use_z], n_events)
    no_z = np.bincount(event_of_pfo[use_z], minlength=n_events) == 0
    if no_z.any():  # events with soft tracks only
        ref_z[no_z] = _event_median(z_at_pca, event_of_pfo, n_events)[no_z]
    vz = ref_z[event_of_pfo]

    s_error = lt.calc_pca_arc_length_error_vectorized(
        phi0, tanL, d0, x0, y0, z0_prime, d0_error, phi0_error, z0_error, tanL_error,
        vx, vy, vz, use_2d=True,
    )
    pca_x, pca_y, pca_z = lt.calc_pca_position_vectorized(s, phi0, tanL, x0, y0, z0_prime)
    pca_errors = lt.calc_pca_position_error_vectorized(
        s, s_error, phi0, d0, d0_error, phi0_error, z0_error, tanL, tanL_error
    )
    ip = lt.calc_impact_parameters_vectorized(pca_x, pca_y, pca_z, *pca_errors, vx, vy, vz)
    # beam size (across the track direction) in the dxy error
    beam_var = (sigma_x[event_of_pfo] * np.sin(phi0)) ** 2 + (sigma_y[event_of_pfo] * np.cos(phi0)) ** 2
    dxy_error = np.sqrt(ip["dxy_error"] ** 2 + beam_var)
    d3 = ip["d3"]
    d3_error = np.sqrt(
        np.where(
            d3 > 1e-10,
            ((ip["dxy"] * dxy_error) ** 2 + (ip["dz"] * ip["dz_error"]) ** 2)
            / np.where(d3 > 1e-10, d3, 1.0) ** 2,
            0.0,
        )
    )
    values = {
        "dxy": ip["dxy"],
        "dz": ip["dz"],
        "d3": d3,
        "dxy_error": dxy_error,
        "dz_error": ip["dz_error"],
        "d3_error": d3_error,
        "d0": d0,
        "z0": z0,
        "d0_error": d0_error,
        "z0_error": z0_error,
        "pca_x": pca_x,
        "pca_y": pca_y,
        "pca_z": pca_z,
        "phi0": phi0,
        "tanL": tanL,
        "omega": omega,
    }
    n_pfos = len(has_track_flat)
    columns = {}
    for name, field in LIFETIME_FIELDS.items():
        flat = np.full(n_pfos, lt.INVALID_VALUE)
        flat[has_track_flat] = values[field]
        columns[name] = ak.unflatten(flat, counts)
    columns["part_has_track"] = ak.values_astype(has_track, np.int32)
    reference = {
        "event_ref_x": ref_x,
        "event_ref_y": ref_y,
        "event_ref_z": ref_z,
        "event_beam_sigma_x": sigma_x,
        "event_beam_sigma_y": sigma_y,
        "event_ref_from_beamspot": from_file.astype(np.int32),
        "event_run_number": run,
    }
    return add_significances(columns), reference


def aleph_inputs(filepath):
    """Everything `events.build_event_table` needs, from one ALEPH EDM4hep file.

    Returns
    -------
    dict
        particles, vertex, has_vertex, event_number, event_columns (reference point of the
        impact parameters, beam size, run number per event).
    """
    events = read_events(filepath)
    n_vertices = ak.to_numpy(ak.num(events["Vertices.position.x"]))
    if np.any(n_vertices > 1):
        raise ValueError(f"{filepath}: events with more than one vertex.")
    has_vertex = n_vertices > 0
    vertex = {
        axis: CM_TO_MM
        * ak.to_numpy(ak.fill_none(ak.firsts(events[f"Vertices.position.{axis}"]), 0.0))
        for axis in "xyz"
    }
    charge = events["RecoParticles.charge"]
    pid_type = events["ParticleID.type"]
    flags = pid_flags(pid_type)
    particles = {
        "part_px": events["RecoParticles.momentum.x"],
        "part_py": events["RecoParticles.momentum.y"],
        "part_pz": events["RecoParticles.momentum.z"],
        "part_energy": events["RecoParticles.energy"],
        # type 3 (charged V0 daughter, a charged hadron here) has mass 0 in the files
        "part_mass": ak.where(pid_type == 3, CHARGED_PION_MASS, events["RecoParticles.mass"]),
        "part_charge": charge,
        "part_pdg": pdg_from_flags(flags, charge),
        "part_pid_type": events["ParticleID.type"],
    }
    particles |= flags
    lifetime_columns, reference = _lifetime_columns(events)
    particles |= lifetime_columns
    return {
        "particles": particles,
        "vertex": vertex,
        "has_vertex": has_vertex,
        "event_number": ak.to_numpy(ak.firsts(events["EventHeader.eventNumber"])),
        "event_columns": reference,
    }


def process_file(filepath, cfg, sample, file_number):
    """Event table and unlabelled jet tables {name: table or None} of one ALEPH file."""
    sample_cfg = cfg.samples[sample]
    event_table = build_event_table(
        **aleph_inputs(filepath),
        sqrt_s=float(sample_cfg.sqrt_s),
        file_number=file_number,
        events_per_file_max=int(sample_cfg.events_per_file_max),
        sample=sample_cfg.get("label", sample),
        all_samples=[sample_cfg.get("label", sample)],
    )
    jet_tables = {
        # the PCA is the transverse one, see the module docstring
        name: jt.build_jet_table(
            event_table, dict(algorithm), dict(cfg.jet_selection), ip_signing="transverse"
        )
        for name, algorithm in cfg.jet_algorithms.items()
    }
    return event_table, jet_tables
