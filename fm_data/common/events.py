"""The event table: one row per event with all reconstructed particles (PFOs) of the event,
sorted by energy, and the event-level variables.

The PFO order of this table is the order of the event-level token sequences, and the jet
tables refer to it (`part_event_index`), so it must not change after this step.

The format-specific readers (`fm_data.ee_sim.cld.cld_particles`, `fm_data.lep.aleph.aleph_inputs`)
return the PFO columns in the PFO order of the file; `build_event_table` does the rest for all
formats.
"""

import awkward as ak
import numpy as np
import vector

from fm_data.common.event_variables import calc_event_observables

vector.register_awkward()

# PDG ids of the particle-flow candidates -> particle-type flags (as in the ALEPH ntuples)
PID_FLAGS = {
    "part_isElectron": (11,),
    "part_isMuon": (13,),
    "part_isPhoton": (22,),
    "part_isChargedHadron": (211, 321, 2212),
    "part_isNeutralHadron": (130, 310, 2112),
}

# event-level particle counts (as in the ALEPH ntuples)
EVENT_COUNTS = {
    "event_num_electrons": "part_isElectron",
    "event_num_muons": "part_isMuon",
    "event_num_photons": "part_isPhoton",
    "event_num_charged_hadrons": "part_isChargedHadron",
    "event_num_neutral_hadrons": "part_isNeutralHadron",
}

# lifetime variables of the PFOs, from lifetime.find_all_track_pcas (3D point of closest
# approach to the primary vertex)
LIFETIME_FIELDS = {
    "part_dxy": "dxy",
    "part_dz": "dz",
    "part_d3": "d3",
    "part_dxy_err": "dxy_error",
    "part_dz_err": "dz_error",
    "part_d3_err": "d3_error",
    "part_d0": "d0",
    "part_z0": "z0",
    "part_d0_err": "d0_error",
    "part_z0_err": "z0_error",
    "part_pca_x": "pca_x",
    "part_pca_y": "pca_y",
    "part_pca_z": "pca_z",
    "part_phi0": "phi0",
    "part_tanL": "tanL",
    "part_omega": "omega",
}


def significance(value, error):
    """value / error, 0 where the error is 0 (e.g. PFOs without a track)."""
    return ak.where(error > 0, value / ak.where(error > 0, error, 1.0), 0.0)


def pid_flags_from_pdg(pdg):
    """Particle-type flags {flag: int32 [events, PFOs]} from the PFO PDG ids."""
    abs_pdg = np.abs(pdg)
    flags = {}
    for flag, pdg_ids in PID_FLAGS.items():
        is_type = abs_pdg == pdg_ids[0]
        for pdg_id in pdg_ids[1:]:
            is_type = is_type | (abs_pdg == pdg_id)
        flags[flag] = ak.values_astype(is_type, np.int32)
    return flags


def add_significances(columns):
    for name in ("dxy", "dz", "d3"):
        columns[f"part_{name}_sig"] = significance(
            columns[f"part_{name}"], columns[f"part_{name}_err"]
        )
    return columns


def build_event_table(
    particles,
    vertex,
    has_vertex,
    event_number,
    file_number,
    events_per_file_max,
    sample,
    all_samples,
    sqrt_s,
    event_columns=None,
):
    """One row per event: event id, sqrt(s), process labels, event variables and all PFOs
    sorted by energy (descending, ties broken by the PFO index in the file).

    Parameters
    ----------
    particles : dict
        PFO columns [events, PFOs] in the order of the file (see `cld_particles`).
    vertex : dict
        {"x", "y", "z": np.ndarray [events]}, primary vertex in mm.
    has_vertex : np.ndarray
        Whether a primary vertex was reconstructed, per event.
    event_number : np.ndarray
        Event number of each event (from the event header).
    file_number : int
        Number of the input file.
    events_per_file_max : int
        event_id = file_number * events_per_file_max + entry (entry = position of the event
        in the file), so it has to be larger than the number of events per file.
    sample : str
        Name of this sample (e.g. "Zqq"), one-hot `label_<sample>`.
    all_samples : list of str
        All sample names, a `label_<sample>` column is created for each of them.
    sqrt_s : float or np.ndarray
        Centre-of-mass energy in GeV (per sample or per event).
    event_columns : dict, optional
        Further event-level columns {name: np.ndarray [events]} (e.g. the ALEPH reference
        point of the impact parameters).

    Returns
    -------
    ak.Array
        The event table, events without PFOs are dropped.
    """
    n_events = len(particles["part_energy"])
    if n_events > events_per_file_max:
        raise ValueError(f"{n_events} events in file {file_number} > {events_per_file_max}.")
    entry = np.arange(n_events, dtype=np.int64)
    particles = dict(particles)
    particles["part_pfo_index"] = ak.local_index(particles["part_energy"], axis=1)

    # sort by energy (stable: equal energies stay in the order of the file)
    order = ak.argsort(-particles["part_energy"], axis=1, stable=True)
    particles = {name: values[order] for name, values in particles.items()}
    p4s = vector.zip(
        {
            "px": particles["part_px"],
            "py": particles["part_py"],
            "pz": particles["part_pz"],
            "energy": particles["part_energy"],
        }
    )
    particles |= {
        "part_p": p4s.p,
        "part_pt": p4s.pt,
        "part_theta": p4s.theta,
        "part_phi": p4s.phi,
        "part_eta": p4s.eta,
    }

    keep = ak.to_numpy(ak.num(particles["part_energy"]) > 0)
    particles = {name: values[keep] for name, values in particles.items()}
    n_kept = int(np.sum(keep))
    sqrt_s = np.broadcast_to(np.asarray(sqrt_s, dtype=np.float32), (n_events,))

    columns = {
        "event_id": file_number * events_per_file_max + entry[keep],
        "event_file_number": np.full(n_kept, file_number, dtype=np.int64),
        "event_entry": entry[keep],
        "event_number": np.asarray(event_number, dtype=np.int64)[keep],
        "event_sqrt_s": sqrt_s[keep],
        "event_has_vertex": np.asarray(has_vertex)[keep].astype(np.int32),
        "event_vertex_x": np.asarray(vertex["x"])[keep],
        "event_vertex_y": np.asarray(vertex["y"])[keep],
        "event_vertex_z": np.asarray(vertex["z"])[keep],
    }
    columns |= {name: np.asarray(values)[keep] for name, values in (event_columns or {}).items()}
    columns |= calc_event_observables(p4s[keep])
    for event_feature, flag in EVENT_COUNTS.items():
        columns[event_feature] = ak.to_numpy(ak.sum(particles[flag], axis=1)).astype(np.int64)
    columns |= {
        f"label_{name}": np.full(n_kept, int(name == sample), dtype=np.int32)
        for name in all_samples
    }
    columns |= particles
    return ak.Array(columns)
