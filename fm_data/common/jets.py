"""The jet tables: jets clustered from the PFOs of the event table, with all PFO variables of
their constituents and the link back to the event table. In simulation (truth objects given)
only the jets matched to a generator-level object (b, c, s, ud quark or hadronic tau) are kept
and labelled; in data all selected jets are kept, without labels.

Link to the event table: each jet row has the `event_id` of its event, and each
constituent its position in the event's (energy-sorted) PFO list, `part_event_index`.
"""

import warnings

import awkward as ak
import fastjet
import numpy as np
import vector

from fm_data.common import lifetime as lt
from fm_data.common.events import significance

vector.register_awkward()

# label classes: {name: PDG ids of the matched truth object}. u and d are merged, as they
# can't be told apart experimentally (the signed PDG id is kept in `jet_truth_pdg`).
LABEL_CLASSES = {"b": (5,), "c": (4,), "s": (3,), "ud": (1, 2), "tau": (15,)}

# lifetime variables that get a version signed w.r.t. the jet axis
SIGNED_IP_VARIABLES = ("dxy", "dz", "d3")


def _p4(px, py, pz, energy):
    return vector.zip({"px": px, "py": py, "pz": pz, "energy": energy})


def cluster_jets(particles, algorithm):
    """Cluster the PFOs of each event to jets.

    Parameters
    ----------
    particles : ak.Array
        PFO records [events, PFOs] (all events need at least `min_n_particles(algorithm)`
        PFOs), with at least `part_px`, `part_py`, `part_pz`, `part_energy`.
    algorithm : dict
        Jet definition with the keys p, R, n_exclusive_jets (None for inclusive jets).

    Returns
    -------
    jets : ak.Array
        Jet four-momenta [events, jets], sorted by energy (descending).
    jet_particles : ak.Array
        The PFO records of each jet [events, jets, PFOs], sorted by pt (descending), with
        the additional field `part_event_index` (position of the PFO in the event).
    """
    particles = ak.with_field(particles, ak.local_index(particles, axis=1), "part_event_index")
    p4s = vector.zip(
        {
            "px": particles["part_px"],
            "py": particles["part_py"],
            "pz": particles["part_pz"],
            "E": particles["part_energy"],
        }
    )
    jetdef = fastjet.JetDefinition2Param(
        fastjet.ee_genkt_algorithm, float(algorithm["R"]), float(algorithm["p"])
    )
    cluster = fastjet.ClusterSequence(p4s, jetdef)
    if algorithm.get("n_exclusive_jets") is None:
        jets = cluster.inclusive_jets()
        constituent_index = ak.Array(cluster.constituent_index())
    else:
        # the fastjet python wrapper warns for exclusive jets with any ee_genkt definition
        # ("... other than kt, C/A or genkt with p>=0 ..."), since it compares the jet
        # definition object to the algorithm enums - p = 0 is fine, so the warning is muted
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="dcut and exclusive jets")
            n_jets = int(algorithm["n_exclusive_jets"])
            jets = cluster.exclusive_jets(n_jets=n_jets)
            constituent_index = ak.Array(cluster.exclusive_jets_constituent_index(njets=n_jets))
    jets = _p4(jets["px"], jets["py"], jets["pz"], jets["E"])
    jet_particles = ak.unflatten(
        particles[ak.flatten(constituent_index, axis=2)],
        ak.flatten(ak.num(constituent_index, axis=2)),
        axis=1,
    )
    jet_order = ak.argsort(jets.energy, axis=1, ascending=False)
    jets = jets[jet_order]
    jet_particles = jet_particles[jet_order]
    part_pt = np.hypot(jet_particles["part_px"], jet_particles["part_py"])
    jet_particles = jet_particles[ak.argsort(part_pt, axis=-1, ascending=False)]
    return jets, jet_particles


def min_n_particles(algorithm):
    """Minimal number of PFOs per event for the clustering."""
    n_exclusive = algorithm.get("n_exclusive_jets")
    return 1 if n_exclusive is None else int(n_exclusive)


def match_jets_to_truth(jets, truth, jet_mask, max_angle):
    """Match each truth object to the closest selected jet.

    Parameters
    ----------
    jets : ak.Array
        Jet four-momenta [events, jets].
    truth : ak.Array
        Truth objects [events, objects] with px, py, pz, energy, pdg.
    jet_mask : ak.Array
        Boolean [events, jets], only these jets are considered.
    max_angle : float
        Maximal 3D opening angle between truth object and jet.

    Returns
    -------
    truth_pdg : ak.Array
        PDG id of the matched object per jet [events, jets], 0 for unmatched jets and for
        jets that are the closest jet of more than one object.
    truth_angle, truth_energy : ak.Array
        Opening angle to and energy of the matched object [events, jets] (-1 if unmatched).
    """
    truth_p4s = _p4(truth.px, truth.py, truth.pz, truth.energy)
    pairs = ak.cartesian({"jet": jets, "truth": truth_p4s}, axis=1, nested=True)
    angles = pairs.jet.deltaangle(pairs.truth)  # [events, jets, objects]
    pairs = ak.cartesian({"truth": truth_p4s, "jet": jets}, axis=1, nested=True)
    angles_per_truth = pairs.truth.deltaangle(pairs.jet)  # [events, objects, jets]
    angles_per_truth = ak.where(
        ak.broadcast_arrays(jet_mask[:, None, :], angles_per_truth)[0],
        angles_per_truth,
        np.inf,
    )
    closest_jet = ak.argmin(angles_per_truth, axis=2)  # [events, objects]
    closest_angle = ak.min(angles_per_truth, axis=2)
    closest_jet = ak.where(closest_angle < max_angle, closest_jet, -1)

    jet_index = ak.local_index(jets, axis=1)
    is_matched = jet_index[:, :, None] == closest_jet[:, None, :]  # [events, jets, objects]
    unique = ak.sum(is_matched, axis=2) == 1
    pdg_bcast = ak.broadcast_arrays(truth.pdg[:, None, :], is_matched)[0]
    energy_bcast = ak.broadcast_arrays(truth.energy[:, None, :], is_matched)[0]
    truth_pdg = ak.where(unique, ak.sum(ak.where(is_matched, pdg_bcast, 0), axis=2), 0)
    truth_angle = ak.where(unique, ak.sum(ak.where(is_matched, angles, 0), axis=2), -1.0)
    truth_energy = ak.where(unique, ak.sum(ak.where(is_matched, energy_bcast, 0), axis=2), -1.0)
    return truth_pdg, truth_angle, truth_energy


def count_heavy_hadrons_in_jets(jets, heavy_hadrons, flavour, max_angle=None):
    """Number of weakly decaying hadrons of the given flavour in each jet [events, jets].

    A hadron is counted for a jet if it is within `max_angle` of the jet axis, or, if
    `max_angle` is None (exclusive jets), for its closest jet.
    """
    hadrons = heavy_hadrons[heavy_hadrons.flavour == flavour]
    hadron_p4s = _p4(hadrons.px, hadrons.py, hadrons.pz, hadrons.energy)
    pairs = ak.cartesian({"jet": jets, "hadron": hadron_p4s}, axis=1, nested=True)
    angles = pairs.jet.deltaangle(pairs.hadron)  # [events, jets, hadrons]
    if max_angle is not None:
        return ak.sum(angles < max_angle, axis=2)
    pairs = ak.cartesian({"hadron": hadron_p4s, "jet": jets}, axis=1, nested=True)
    closest_jet = ak.argmin(pairs.hadron.deltaangle(pairs.jet), axis=2)  # [events, hadrons]
    jet_index = ak.local_index(jets, axis=1)
    return ak.sum(jet_index[:, :, None] == closest_jet[:, None, :], axis=2)


# components of (PCA - reference point) and of the jet axis used to sign each impact parameter:
# "3d" (CLD: 3D PCA to the primary vertex) signs all of them with the 3D dot product;
# "transverse" (ALEPH: transverse PCA to the beam spot, whose z offset is the poorly measured dz)
# signs dxy and d3 with the transverse dot product and dz with the z components.
IP_SIGNING = {
    "3d": {"dxy": "xyz", "dz": "xyz", "d3": "xyz"},
    "transverse": {"dxy": "xy", "dz": "z", "d3": "xy"},
}


def _signed_impact_parameters(parts, jets, vertex, ip_signing="3d"):
    """Impact parameters signed w.r.t. the jet axis (lifetime.calc_signed_impact_parameters:
    positive if the point of closest approach lies in the jet direction), with the components
    given by IP_SIGNING[ip_signing].

    Parameters
    ----------
    parts : ak.Array
        Constituents [jets, PFOs].
    jets : ak.Array
        Jet four-momenta [jets].
    vertex : dict
        {"x", "y", "z": np.ndarray [jets]}, reference point of the impact parameters of the
        jet's event (primary vertex, or the beam spot for ALEPH).
    """
    counts = ak.num(parts["part_energy"])

    def per_constituent(values):
        return ak.to_numpy(ak.flatten(ak.broadcast_arrays(values[:, None], parts["part_energy"])[0]))

    flat = {name: ak.to_numpy(ak.flatten(parts[name])) for name in ("part_pca_x", "part_pca_y", "part_pca_z")}
    jet_px, jet_py, jet_pz = (per_constituent(getattr(jets, axis)) for axis in ("px", "py", "pz"))
    vtx = {axis: per_constituent(ak.Array(vertex[axis])) for axis in "xyz"}
    columns = {}
    jet_p = {"x": jet_px, "y": jet_py, "z": jet_pz}
    for name in SIGNED_IP_VARIABLES:
        axes = IP_SIGNING[ip_signing][name]
        # components not used for the sign: PCA = reference point and jet component 0
        pca = {axis: flat[f"part_pca_{axis}"] if axis in axes else vtx[axis] for axis in "xyz"}
        jet = {axis: jet_p[axis] if axis in axes else np.zeros_like(jet_p[axis]) for axis in "xyz"}
        signed = lt.calc_signed_impact_parameters(
            ak.to_numpy(ak.flatten(parts[f"part_{name}"])),
            pca["x"],
            pca["y"],
            pca["z"],
            vtx["x"],
            vtx["y"],
            vtx["z"],
            jet["x"],
            jet["y"],
            jet["z"],
        )
        columns[f"part_{name}_signed"] = ak.unflatten(signed, counts)
        columns[f"part_{name}_signed_sig"] = significance(
            columns[f"part_{name}_signed"], parts[f"part_{name}_err"]
        )
    return columns


def _jet_and_relative_features(jets, parts):
    """Jet features and constituent features relative to the jet [jets(, PFOs)], with the
    names of the ALEPH jet-level ntuples (part_etarel, part_phirel, ...)."""
    p4s = _p4(parts["part_px"], parts["part_py"], parts["part_pz"], parts["part_energy"])
    p4s_sum = ak.sum(p4s, axis=1)
    part_etarel = p4s.eta - jets.eta[:, None]
    part_phirel = np.arctan2(
        np.sin(p4s.phi - jets.phi[:, None]), np.cos(p4s.phi - jets.phi[:, None])
    )
    return {
        "jet_pt": jets.pt,
        "jet_eta": jets.eta,
        "jet_phi": jets.phi,
        "jet_theta": jets.theta,
        "jet_energy": jets.energy,
        "jet_mass": jets.mass,
        "jet_px": jets.px,
        "jet_py": jets.py,
        "jet_pz": jets.pz,
        "jet_mass_from_p4s": p4s_sum.mass,
        "jet_pt_from_p4s": p4s_sum.pt,
        "jet_eta_from_p4s": p4s_sum.eta,
        "jet_phi_from_p4s": p4s_sum.phi,
        "jet_nparticles": ak.num(parts["part_energy"]),
        "jet_charge": ak.sum(parts["part_charge"], axis=1),
        "part_ptrel": p4s.pt / jets.pt[:, None],
        "part_erel": parts["part_energy"] / jets.energy[:, None],
        "part_etarel": part_etarel,
        "part_phirel": part_phirel,
        "part_deta": np.abs(part_etarel),
        "part_dphi": np.abs(part_phirel),
        "part_deltaR": np.hypot(part_etarel, part_phirel),
        # e+e- jets: the 3D opening angle to the jet axis
        "part_angle": p4s.deltaangle(jets[:, None]),
    }


def extended_features(jet_theta, part_etarel, part_phirel, part_ptrel, R):
    """Constituent coordinates scaled to the jet radius, as the tokeniser / backbone of the
    jet foundation models (gabbro, feature dict *_extended_features_*_new_features) expect.

    The jets are e+e- jets: R is an opening angle. Near the jet axis a step in eta is a polar
    angle step sin(theta) * deta and a step in phi an arc sin(theta) * dphi, so with the jet's
    sin(theta) the cone of opening angle R becomes the unit disk for any jet direction:
        part_etarel_R = sin(theta_jet) * part_etarel / R
        part_phirel_R = sin(theta_jet) * part_phirel / R
        part_logptrel = log(pt / pt_jet)

    Parameters
    ----------
    jet_theta : array [jets]
    part_etarel, part_phirel, part_ptrel : ak.Array [jets, PFOs]
    R : float
        Jet radius (opening angle) of the clustering.
    """
    sin_theta = np.sin(jet_theta)[:, None]
    return {
        "part_etarel_R": sin_theta * part_etarel / R,
        "part_phirel_R": sin_theta * part_phirel / R,
        "part_logptrel": np.log(part_ptrel),
    }


def collider_features(n_jets, R, sqrt_s):
    """Per-jet constants telling the foundation models which collider / jet definition the
    jet comes from (jet_R, jet_is_ee, jet_is_pp, jet_sqrt_s)."""
    return {
        "jet_R": np.full(n_jets, R, dtype=np.float32),
        "jet_is_ee": np.ones(n_jets, dtype=np.int32),
        "jet_is_pp": np.zeros(n_jets, dtype=np.int32),
        "jet_sqrt_s": np.asarray(sqrt_s, dtype=np.float32),
    }


def build_jet_table(
    event_table, algorithm, selection, truth=None, heavy_hadrons=None, ip_signing="3d"
):
    """Cluster jets from the PFOs of the event table, one row per kept jet.

    Parameters
    ----------
    event_table : ak.Array
        Output of `events.build_event_table`.
    algorithm : dict
        Jet definition: p, R, n_exclusive_jets, max_match_angle (only used with truth).
    selection : dict
        Jet selection: min_energy, max_abs_eta (and always: E_jet <= sqrt(s), no constituent
        with E >= sqrt(s)/2, pt > 0).
    truth : ak.Array, optional
        Truth objects of the same events [events, objects] (px, py, pz, energy, pdg). If
        given, only jets matched to a truth object are kept and labelled; if None (data),
        all selected jets are kept, without truth columns.
    heavy_hadrons : ak.Array, optional
        Weakly decaying b/c hadrons of the same events (`truth.MCTruth.heavy_hadrons`),
        required with `truth`.
    ip_signing : str
        "3d" or "transverse", see IP_SIGNING.

    Returns
    -------
    ak.Array or None
        The jet table, None if no jet is kept.
    """
    part_fields = [field for field in event_table.fields if field.startswith("part_")]
    event_fields = [field for field in event_table.fields if not field.startswith("part_")]
    labelled = truth is not None
    usable = ak.to_numpy(ak.num(event_table["part_energy"]) >= min_n_particles(algorithm))
    if labelled:
        usable = usable & (ak.to_numpy(ak.num(truth)) > 0)
    if not np.any(usable):
        return None
    events = event_table[usable]
    if labelled:
        truth = truth[usable]
        heavy_hadrons = heavy_hadrons[usable]
    particles = ak.zip({field: events[field] for field in part_fields})

    jets, jet_particles = cluster_jets(particles, algorithm)
    sqrt_s = ak.to_numpy(events["event_sqrt_s"])
    has_bad_particle = ak.any(
        (jet_particles["part_pt"] <= 0)
        | ~np.isfinite(jet_particles["part_pt"])
        | (jet_particles["part_energy"] >= sqrt_s[:, None, None] / 2),
        axis=2,
    )
    jet_mask = (
        (jets.pt > 0)
        & (jets.energy > selection["min_energy"])
        & (jets.energy <= sqrt_s[:, None])  # (as the earlier ALEPH ntuples) no jet above sqrt(s)
        & (np.abs(jets.eta) < selection["max_abs_eta"])
        & ~has_bad_particle
    )
    per_event_jets = {
        "jet_index": ak.local_index(jets, axis=1),
        "jet_event_row": ak.broadcast_arrays(ak.local_index(jets, axis=0), jets.pt)[0],
    }
    if labelled:
        truth_pdg, truth_angle, truth_energy = match_jets_to_truth(
            jets, truth, jet_mask, max_angle=algorithm["max_match_angle"]
        )
        hadron_max_angle = algorithm["R"] if algorithm.get("n_exclusive_jets") is None else None
        per_event_jets |= {
            "jet_truth_pdg": truth_pdg,
            "jet_truth_angle": truth_angle,
            "jet_truth_energy": truth_energy,
            "jet_n_bhadrons": count_heavy_hadrons_in_jets(jets, heavy_hadrons, 5, hadron_max_angle),
            "jet_n_chadrons": count_heavy_hadrons_in_jets(jets, heavy_hadrons, 4, hadron_max_angle),
        }
        keep = truth_pdg != 0
    else:
        keep = jet_mask

    # one row per kept jet
    if int(ak.sum(keep)) == 0:
        return None
    jets = ak.flatten(jets[keep])
    parts = ak.flatten(jet_particles[keep], axis=1)
    jet_columns = {name: ak.to_numpy(ak.flatten(values[keep])) for name, values in per_event_jets.items()}
    event_row = jet_columns.pop("jet_event_row")

    columns = {field: ak.to_numpy(events[field])[event_row] for field in event_fields}
    columns |= jet_columns
    if labelled:
        abs_pdg = np.abs(columns["jet_truth_pdg"])
        for name, pdg_ids in LABEL_CLASSES.items():
            columns[f"label_{name}"] = np.isin(abs_pdg, pdg_ids).astype(np.int32)
    columns |= _jet_and_relative_features(jets, parts)
    if algorithm.get("n_exclusive_jets") is None:
        # (exclusive jets have no radius to scale to)
        R = float(algorithm["R"])
        columns |= extended_features(
            columns["jet_theta"], columns["part_etarel"], columns["part_phirel"],
            columns["part_ptrel"], R,
        )
        columns |= collider_features(len(jets), R, columns["event_sqrt_s"])
    columns |= {field: parts[field] for field in parts.fields}
    # the PCA is computed w.r.t. the reference point of the impact parameters: the primary
    # vertex (CLD) or, if the reader provides one, event_ref_* (ALEPH: beam spot)
    origin = "event_ref" if "event_ref_x" in columns else "event_vertex"
    vertex = {axis: columns[f"{origin}_{axis}"] for axis in "xyz"}
    columns |= _signed_impact_parameters(parts, jets, vertex, ip_signing)
    return ak.Array(columns)
