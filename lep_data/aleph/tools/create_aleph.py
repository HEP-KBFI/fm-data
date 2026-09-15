"""This tool is clustering the jets with anti-kt for ee instead of relying on the association map and the jets coming directly from the ALEPH dataset."""

from typing import List, Optional

import awkward as ak
import fastjet
import numpy as np
import uproot
import vector

from lep_data.tools import features as f

from . import event_variable_calculations as evc
from . import jet_variable_calculations as jvc


def load_file_contents(
    path: str, tree_name: str = "events", branches: Optional[List[str]] = None
) -> ak.Array:
    with uproot.open(path) as in_file:
        tree = in_file[tree_name]
        arrays = tree.arrays(branches)
    return arrays


def fill_values(array: ak.Array, fill_value: float = -999.9):
    return ak.fill_none(ak.pad_none(array, 1, axis=-1), fill_value)


# The ALEPH EDM4HEP files store lengths in ALEPH's native centimetres even
# though the EDM4HEP schema declares millimetres, so every length taken out of
# a track state or a vertex is scaled here to make the output spec-compliant.
#
# Measured on data_2057_1.root (1994 reprocessing): interpreting omega as 1/cm
# and taking B = 1.5 T reproduces the momentum of the linked particle to within
# 1 % for 99.5 % of tracks, while 1/mm is off by exactly a factor 10 for all of
# them; the vertex z spread of 1.1 (i.e. ~1 cm) matches the ALEPH luminous
# region. THIS IS A PROPERTY OF THE CURRENT INPUT FILES, NOT OF THE FORMAT: if a
# later production writes proper EDM4HEP millimetres, set this back to 1.0.
# (Should the commented-out omega ever be used, note it is 1/cm, i.e. it scales
# the other way.)
CM_TO_MM = 10.0


def calculate_impact_parameters(
    events: ak.Array, track_states_collection: str = "_Tracks_trackStates"
):

    ##
    ## Here, with fill values, one should have the same # of entries as there are particles.
    ## Therefore a single 0.0 is not enough. Check the shapes.
    ##
    # Lengths are scaled to mm (see CM_TO_MM); phi and tanLambda are an angle
    # and a ratio, so they are unit-free.  The scaling is applied before
    # fill_values so that the -999.9 missing-value marker keeps its value.
    d0 = fill_values(CM_TO_MM * events[f"{track_states_collection}.D0"])
    z0 = fill_values(CM_TO_MM * events[f"{track_states_collection}.Z0"])
    phi0 = fill_values(events[f"{track_states_collection}.phi"])
    tanL = fill_values(events[f"{track_states_collection}.tanLambda"])
    # omega = fill_values(events[f"{track_states_collection}.omega"] / CM_TO_MM)
    xr = fill_values(CM_TO_MM * events[f"{track_states_collection}.referencePoint.x"])
    yr = fill_values(CM_TO_MM * events[f"{track_states_collection}.referencePoint.y"])
    zr = fill_values(CM_TO_MM * events[f"{track_states_collection}.referencePoint.z"])

    # Extract covariance matrix elements.  EDM4HEP packs the lower triangle of
    # the 6x6 (d0, phi, omega, z0, tanLambda, time) covariance matrix into 21
    # values, so entries 0, 2, 9 and 14 are the *variances* of d0, phi, z0 and
    # tanLambda -- take the square root to turn them into errors.  The maximum
    # guards against fits that came out with a negative diagonal element.
    cov_matrix = events[f"{track_states_collection}.covMatrix.values[21]"]
    d0_error = fill_values(CM_TO_MM * np.sqrt(np.maximum(cov_matrix[:, :, 0], 0.0)))
    phi0_error = fill_values(np.sqrt(np.maximum(cov_matrix[:, :, 2], 0.0)))
    z0_error = fill_values(CM_TO_MM * np.sqrt(np.maximum(cov_matrix[:, :, 9], 0.0)))
    tanL_error = fill_values(np.sqrt(np.maximum(cov_matrix[:, :, 14], 0.0)))

    # Vertex position (assuming you have this defined)
    # Events without a reconstructed vertex get the -999.9 fill value as their
    # vertex position, so every impact parameter measured against it (dxy, dz
    # and their errors) is meaningless for those events -- warn rather than let
    # ~1000 mm impact parameters pass for real measurements.
    n_vertices = ak.num(events["Vertices.position.x"])
    n_missing = int(ak.sum(n_vertices == 0))
    if n_missing:
        print(
            f"WARNING: no reconstructed vertex in {n_missing}/{len(n_vertices)} events; "
            "their dxy/dz impact parameters are measured against the -999.9 fill "
            "value and should not be used"
        )
    n_multiple = int(ak.sum(n_vertices > 1))
    if n_multiple:
        # The flatten() below assumes one vertex per event; more than one would
        # silently shift every event's vertex onto the wrong event.
        raise ValueError(
            f"{n_multiple} events have more than one vertex; the per-event "
            "vertex broadcast in calculate_impact_parameters assumes at most one"
        )
    vertex_x = ak.flatten(fill_values(CM_TO_MM * events["Vertices.position.x"]), axis=-1)
    vertex_y = ak.flatten(fill_values(CM_TO_MM * events["Vertices.position.y"]), axis=-1)
    vertex_z = ak.flatten(fill_values(CM_TO_MM * events["Vertices.position.z"]), axis=-1)

    # Calculate track origins (vectorized) - from lifetime.py
    x0 = xr + np.cos(np.pi / 2 - phi0) * d0
    y0 = yr - np.sin(np.pi / 2 - phi0) * d0
    z0_prime = z0 + zr

    # Calculate PCA arc length (vectorized) - simplified 3D version
    # Following the pattern from calc_pca_arc_length_vectorized
    ax, ay, az = x0, y0, z0_prime
    bx = np.cos(phi0) + x0
    by = np.sin(phi0) + y0
    bz = tanL + z0_prime

    num = -(
        (ax - vertex_x) * (bx - ax)
        + (ay - vertex_y) * (by - ay)
        + (az - vertex_z) * (bz - az)
    )
    den = (bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2
    s = num / den

    # Calculate PCA position (vectorized)
    pca_x = s * np.cos(phi0) + x0
    pca_y = s * np.sin(phi0) + y0
    pca_z = s * tanL + z0_prime

    # Calculate impact parameters
    dxy = np.sqrt((vertex_x - pca_x) ** 2 + (vertex_y - pca_y) ** 2)
    dz = np.abs(pca_z - vertex_z)

    # Calculate errors for PCA position (simplified error propagation)
    # Following pattern from calc_pca_position_error_vectorized
    s_error = np.sqrt(d0_error**2 + phi0_error**2)  # Simplified estimate

    x0_error = np.sqrt(
        (np.cos(np.pi / 2 - phi0) * d0_error) ** 2
        + (np.sin(np.pi / 2 - phi0) * d0 * phi0_error) ** 2
    )
    y0_error = np.sqrt(
        (np.sin(np.pi / 2 - phi0) * d0_error) ** 2
        + (np.cos(np.pi / 2 - phi0) * d0 * phi0_error) ** 2
    )
    z0_prime_error = z0_error

    pca_x_error = np.sqrt(
        (np.cos(phi0) * s_error) ** 2
        + x0_error**2
        + (s * np.sin(phi0) * phi0_error) ** 2
    )
    pca_y_error = np.sqrt(
        (np.sin(phi0) * s_error) ** 2
        + y0_error**2
        + (s * np.cos(phi0) * phi0_error) ** 2
    )
    pca_z_error = np.sqrt(
        (tanL * s_error) ** 2 + (s * tanL_error) ** 2 + z0_prime_error**2
    )

    # Calculate impact parameter errors
    dxy_error = np.sqrt(
        ((vertex_x - pca_x) / dxy * pca_x_error) ** 2
        + ((vertex_y - pca_y) / dxy * pca_y_error) ** 2
    )
    dz_error = pca_z_error
    return ak.zip(
        {
            "dzval": dz,
            "dzerr": dz_error,
            "d0val": d0,
            "d0err": d0_error,
            "dxy": dxy,
            "dxyerr": dxy_error,
        }
    )


def assign_pid_info(arrays: ak.Array) -> ak.Array:
    """Extract particle ID info vectorized across all events"""
    # 0:Charged hadrons, 1:Electron, 2:Muon, 3:Track from V0, 4:photons, 5:Neutral hadrons, 6:Hcalobject, 7:Lcalobject
    pid_type = arrays["ParticleID.type"]
    return ak.zip(
        {
            "is_charged_hadron": ak.values_astype(pid_type == 0, "int32"),
            "is_electron": ak.values_astype(pid_type == 1, "int32"),
            "is_muon": ak.values_astype(pid_type == 2, "int32"),
            "is_photon": ak.values_astype(pid_type == 4, "int32"),
            "is_neutral_hadron": ak.values_astype(pid_type == 5, "int32"),
        }
    )


def get_cand_info(events: ak.Array) -> ak.Array:
    """Extract and combine info from multiple collections vectorized"""
    # Extract ReconstructedParticles info (all events at once)
    cand_p4 = vector.awk(
        ak.zip(
            {
                "energy": events["RecoParticles.energy"],
                "px": events["RecoParticles.momentum.x"],
                "py": events["RecoParticles.momentum.y"],
                "pz": events["RecoParticles.momentum.z"],
            }
        )
    )

    # Get PID info
    pid_info = assign_pid_info(events)
    impact_parameters = calculate_impact_parameters(events=events)
    # Gather the track state of each particle by position, so that the values
    # line up with the particles they belong to whatever order the collection
    # is in, and give the neutral (trackless) ones the missing-value marker.
    state_index, has_track = find_linked_indices(events=events)
    impact_parameters = impact_parameters[state_index]
    impact_parameters = ak.zip(
        {
            field: ak.where(has_track, impact_parameters[field], -999.9)
            for field in impact_parameters.fields
        }
    )

    # Combine everything into one structure (all vectorized)
    particle_data = ak.zip(
        {
            "part_p4": cand_p4,
            "part_px": cand_p4.px,
            "part_py": cand_p4.py,
            "part_pz": cand_p4.pz,
            "part_energy": cand_p4.energy,
            "part_pt": cand_p4.pt,
            "part_eta": cand_p4.eta,
            "part_phi": cand_p4.phi,
            "part_mass": cand_p4.mass,
            "part_d0val": impact_parameters["d0val"],
            "part_d0err": impact_parameters["d0err"],
            "part_dzval": impact_parameters["dzval"],
            "part_dzerr": impact_parameters["dzerr"],
            "part_charge": events["RecoParticles.charge"],
            "part_isChargedHadron": pid_info["is_charged_hadron"],
            "part_isNeutralHadron": pid_info["is_neutral_hadron"],
            "part_isPhoton": pid_info["is_photon"],
            "part_isElectron": pid_info["is_electron"],
            "part_isMuon": pid_info["is_muon"],
        }
    )
    return particle_data


def find_linked_indices(
    events: ak.Array,
    begin_branch: str = "RecoParticles.tracks_begin",
    end_branch: str = "RecoParticles.tracks_end",
    relation_branch: str = "_RecoParticles_tracks.index",
    state_begin_branch: str = "Tracks.trackStates_begin",
):
    """Track state of every reconstructed particle, [events, particles].

    podio stores the particle -> track link in two hops, and both have to be
    followed:

      1. tracks_begin/end are offsets into the *relation* collection
         _RecoParticles_tracks, not into Tracks. Its entries hold the index of
         the track. Using tracks_begin directly as a track index only works
         when the two collections happen to line up; in the ALEPH files each
         track is stored twice (Tracks is 2.25x the number of referenced
         tracks), so it lands on the right track ~16% of the time and picks an
         unrelated track in the event otherwise.
      2. a track's parameters live in _Tracks_trackStates, starting at
         Tracks.trackStates_begin. The ALEPH files store exactly one state per
         track (location 0, at the IP), so this hop is the identity there, but
         following it keeps the lookup correct for files that store several.

    Returns:
        (state_index, has_track): both [events, particles]. state_index is only
        meaningful where has_track is True; elsewhere it is a valid dummy so
        that the gather stays in range.
    """
    begins = events[begin_branch]
    has_track = (events[end_branch] - begins) > 0

    # Pad to at least one entry per event so that the dummy index 0 used for
    # trackless particles stays in range even in events with no tracks at all.
    relation = ak.fill_none(ak.pad_none(events[relation_branch], 1, axis=1), 0)
    state_begin = ak.fill_none(ak.pad_none(events[state_begin_branch], 1, axis=1), 0)

    relation_position = ak.values_astype(ak.where(has_track, begins, 0), np.int64)
    track_index = ak.values_astype(relation[relation_position], np.int64)
    state_index = ak.values_astype(state_begin[track_index], np.int64)
    return state_index, has_track


def cluster_particles_to_jets(
    particle_data: ak.Array, deltar: float = 0.8, min_pt: float = 0.0
) -> tuple:
    """Cluster particles into jets using fastjet ee_genkt_algorithm.

    Returns (jet_p4s, constituent_indices) where both are [events, jets, ...].
    """
    # Use the same format as particle_filters.py: mass + px + py + pz via vector.awk
    particles_p4 = vector.awk(
        ak.zip(
            {
                "mass": particle_data.part_mass,
                "px": particle_data.part_px,
                "py": particle_data.part_py,
                "pz": particle_data.part_pz,
            }
        )
    )

    jetdef = fastjet.JetDefinition2Param(fastjet.ee_genkt_algorithm, deltar, -1)
    cluster = fastjet.ClusterSequence(particles_p4, jetdef)
    jets = vector.awk(cluster.inclusive_jets(min_pt=min_pt))
    jets = vector.awk(
        ak.zip(
            {
                "energy": jets["t"],
                "px": jets["x"],
                "py": jets["y"],
                "pz": jets["z"],
            }
        )
    )
    constituent_indices = ak.Array(cluster.constituent_index(min_pt=min_pt))
    njets = np.sum(ak.num(jets))
    print(f"Clustered {njets} jets with R={deltar}")
    return jets, constituent_indices


def get_jet_basic_properties(
    jet_p4: ak.Array,
    jet_assigned_particles: ak.Array,
    counts_per_jet: ak.Array,
):
    """Compute basic jet-level properties from clustered jet p4s and their constituents."""
    jet_data = ak.zip(
        {
            "jet_pt": jet_p4.pt,
            "jet_eta": jet_p4.eta,
            "jet_phi": jet_p4.phi,
            "jet_energy": jet_p4.energy,
            "jet_mass": jet_p4.mass,
            "jet_sdmass": jvc.calculate_sdmass_ee(jet_assigned_particles),
            "jet_tau1": jvc.calculate_tau_n_ee(
                jet_assigned_particles.part_pt,
                jet_assigned_particles.part_eta,
                jet_assigned_particles.part_phi,
                Naxes=1,
            ),
            "jet_tau2": jvc.calculate_tau_n_ee(
                jet_assigned_particles.part_pt,
                jet_assigned_particles.part_eta,
                jet_assigned_particles.part_phi,
                Naxes=2,
            ),
            "jet_tau3": jvc.calculate_tau_n_ee(
                jet_assigned_particles.part_pt,
                jet_assigned_particles.part_eta,
                jet_assigned_particles.part_phi,
                Naxes=3,
            ),
            "jet_tau4": jvc.calculate_tau_n_ee(
                jet_assigned_particles.part_pt,
                jet_assigned_particles.part_eta,
                jet_assigned_particles.part_phi,
                Naxes=4,
            ),
        }
    )

    sum_px = ak.sum(jet_assigned_particles.part_px, axis=-1)
    sum_py = ak.sum(jet_assigned_particles.part_py, axis=-1)
    sum_pz = ak.sum(jet_assigned_particles.part_pz, axis=-1)
    sum_e = ak.sum(jet_assigned_particles.part_energy, axis=-1)
    sum_m2 = sum_e**2 - sum_px**2 - sum_py**2 - sum_pz**2
    sum_mass = np.sqrt(np.maximum(sum_m2, 0.0))
    jet_constituent_p4_sums = vector.awk(
        ak.zip(
            {
                "px": sum_px,
                "py": sum_py,
                "pz": sum_pz,
                "mass": sum_mass,
            }
        )
    )
    constituent_data = ak.zip(
        {
            "jet_mass_from_p4s": jet_constituent_p4_sums.mass,
            "jet_pt_from_p4s": jet_constituent_p4_sums.pt,
            "jet_eta_from_p4s": jet_constituent_p4_sums.eta,
            "jet_phi_from_p4s": jet_constituent_p4_sums.phi,
            "jet_nparticles": counts_per_jet,
        }
    )
    jet_data = ak.zip(
        {
            **{field: jet_data[field] for field in jet_data.fields},
            **{field: constituent_data[field] for field in constituent_data.fields},
        }
    )

    return jet_data


def get_all_properties(
    jet_data: ak.Array,
    jet_assigned_particles: ak.Array,
):
    """Combine jet and particle properties at [events, jets, ...] level.

    Pads the particle axis (axis=-1) but does NOT flatten the jet axis.
    The caller is responsible for the final axis=1 flatten.
    """
    # Build combined dict with all fields at [events, jets, ...]
    combined = {
        # Jet-level fields: [events, jets]
        **{k: jet_data[k] for k in jet_data.fields},
        # Particle-relative-to-jet variables: [events, jets, particles]
        "part_ptrel": jet_assigned_particles.part_pt / jet_data.jet_pt,
        "part_erel": jet_assigned_particles.part_energy / jet_data.jet_energy,
        "part_etarel": f.signedDeltaEta(
            jet_assigned_particles.part_eta, jet_data.jet_eta
        ),  # part_signed_deta ; to have same field names as in the JetClass dataset
        "part_phirel": f.signedDeltaPhi(
            jet_assigned_particles.part_phi, jet_data.jet_phi
        ),  # part_signed_dphi ; to have same field names as in the JetClass dataset
        "part_deltaR": f.deltaR_etaPhi(
            jet_assigned_particles.part_eta,
            jet_assigned_particles.part_phi,
            jet_data.jet_eta,
            jet_data.jet_phi,
        ),
        "part_deta": f.deltaEta(jet_assigned_particles.part_eta, jet_data.jet_eta),
        "part_dphi": f.deltaPhi(jet_assigned_particles.part_phi, jet_data.jet_phi),
        # Particle identity fields: [events, jets, particles]
        **{
            field: jet_assigned_particles[field]
            for field in jet_assigned_particles.fields
            if field != "part_p4"
        },
    }

    # Pad the particle axis only (jet axis is always >= 1)
    for k in combined:
        arr = combined[k]
        if arr.ndim == 3:
            arr = ak.fill_none(ak.pad_none(arr, 1, axis=-1), -999.9)
        combined[k] = arr

    return combined


def construct_jet_based_dataset(events: ak.Array):
    """Cluster particles to jets with fastjet (R=0.8) and build the combined dict at [events, jets, ...] level.

    Returns:
        Tuple of (jet_data_dict, particle_data):
        - jet_data_dict: [events, jets, ...] combined jet + particle properties
        - particle_data: [events, particles] full particle data for event-level computations
    """
    particle_data = get_cand_info(events=events)

    # Cluster particles into jets using fastjet
    jet_p4, constituent_indices = cluster_particles_to_jets(
        particle_data, deltar=0.8, min_pt=0.0
    )

    # Group particle data per jet using cluster constituent indices
    # constituent_indices: [events, jets, particles_per_jet]
    num_ptcls_per_jet = ak.num(constituent_indices, axis=-1)  # [events, jets]

    # Flatten innermost axis to index into particle_data: [events, total_parts]
    flat_indices = ak.flatten(constituent_indices, axis=-1)
    selected_particles_flat = particle_data[flat_indices]

    # Per-event unflatten back to [events, jets, particles_per_jet]
    # (matches the pattern in get_jet_constituent_property)
    jet_assigned_particles = ak.from_iter(
        [
            ak.unflatten(selected_particles_flat[i], num_ptcls_per_jet[i], axis=-1)
            for i in range(len(num_ptcls_per_jet))
        ]
    )

    jet_data = get_jet_basic_properties(
        jet_p4=jet_p4,
        jet_assigned_particles=jet_assigned_particles,
        counts_per_jet=num_ptcls_per_jet,
    )
    all_properties = get_all_properties(
        jet_data=jet_data,
        jet_assigned_particles=jet_assigned_particles,
    )

    return ak.Array(all_properties), particle_data


def build_dataset(
    input_path: str, jet_level: bool = False, event_level: bool = True
) -> ak.Array:
    """Ntupelize one ROOT file and return the dataset, without writing it.

    Exactly one of jet_level / event_level may be set: they describe two
    different row meanings (one row per jet vs one row per event) and so cannot
    share an output.

    Returns:
        Either the [total_jets, ...] jet-level dataset or the [events, ...]
        event-level one, cleaned of bad particles and jets in both cases.
    """
    if jet_level == event_level:
        raise ValueError(
            "Set exactly one of jet_level / event_level, got "
            f"jet_level={jet_level}, event_level={event_level}"
        )
    events = load_file_contents(path=input_path, tree_name="events")
    combined_jet, particle_data = construct_jet_based_dataset(events)

    # Flatten from [events, jets, ...] to [total_jets, ...] at the very end
    if jet_level:
        jet_dataset = ak.Array(
            {k: ak.flatten(combined_jet[k], axis=1) for k in combined_jet.fields}
        )
        bad_particle = (
            (jet_dataset.part_pt <= 0)
            | (jet_dataset.part_energy >= 45.6)
            | (jet_dataset.part_pt == -999.9)
            | (~np.isfinite(jet_dataset.part_pt))
        )
        jet_contains_bad_particle = ak.any(bad_particle, axis=-1)
        valid_jets = (
            (jet_dataset.jet_pt > 0)
            & np.isfinite(jet_dataset.jet_eta)
            & (jet_dataset.jet_energy <= 91.2)
        )
        return jet_dataset[(~jet_contains_bad_particle) * valid_jets]

    if event_level:
        # Merge jet-level [events, jets, ...] with event-level [events] fields
        bad_particle = (
            (particle_data.part_pt <= 0)
            | (particle_data.part_energy >= 45.6)
            | (particle_data.part_pt == -999.9)
            | (~np.isfinite(particle_data.part_pt))
        )
        event_contains_bad_particle = ak.any(bad_particle, axis=-1)
        # jet_pt <= 0, not < 0: a zero-pt jet is as unusable as a negative one,
        # and the jet-level branch above already rejects it via `jet_pt > 0`.
        # Keeping the two branches on the same cut means a jet is not kept at one
        # output level and dropped at the other.
        bad_jets = (
            (combined_jet.jet_pt <= 0)
            | ~np.isfinite(combined_jet.jet_eta)
            | (combined_jet.jet_energy > 91.2)
        )
        event_contains_bad_jet = ak.any(bad_jets, axis=-1)
        event_mask_ = ak.num(combined_jet.jet_pt) > 0
        event_mask = (
            (~event_contains_bad_particle) & (~event_contains_bad_jet) & event_mask_
        )
        particle_data = particle_data[event_mask]
        combined_jet = combined_jet[event_mask]
        # Compute event-level variables from particle data and jet variables
        event_dataset = evc.get_event_variables(
            particle_data=particle_data, jet_data=combined_jet
        )
        return ak.zip(
            {
                **{f: event_dataset[f] for f in event_dataset.fields},
                **{f: combined_jet[f] for f in combined_jet.fields},
                **{f: particle_data[f] for f in particle_data.fields},
            },
            depth_limit=1,
        )


def ntupelize_file(
    input_path: str,
    output_path: str,
    jet_level: bool = False,
    event_level: bool = True,
    row_group_size: int = 1024,
):
    """Ntupelize one ROOT file into one .parquet file.

    Kept for one-off use; the processing jobs fill fixed-size chunk files
    instead, see scripts/ntupelize_list.py.
    """
    dataset = build_dataset(
        input_path=input_path, jet_level=jet_level, event_level=event_level
    )
    ak.to_parquet(
        dataset,
        output_path,
        row_group_size=row_group_size,
        compression="zstd",
        compression_level=6,
    )
