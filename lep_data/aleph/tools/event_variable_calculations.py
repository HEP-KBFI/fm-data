import awkward as ak
import numpy as np


def compute_met(particle_data: ak.Array) -> ak.Array:
    """Missing transverse energy from all particles.

    MET = | -sum(p_T) |, the magnitude of the negative vector sum of all particle transverse momenta.

    Args:
        particle_data: [events, particles] with part_px, part_py fields

    Returns:
        [events] scalar MET values
    """
    px_sum = ak.sum(particle_data.part_px, axis=-1)
    py_sum = ak.sum(particle_data.part_py, axis=-1)
    met = np.sqrt(px_sum**2 + py_sum**2)
    return met


def compute_num_jets(jet_data: ak.Array) -> ak.Array:
    """Number of jets per event.

    Args:
        jet_data: [events, jets] with jet_pt (or any jet-level field)

    Returns:
        [events] integer jet multiplicity per event
    """
    return ak.num(jet_data.jet_pt, axis=-1)


def compute_ht(jet_data: ak.Array) -> ak.Array:
    """HT = scalar sum of jet pT.

    Args:
        jet_data: [events, jets] with jet_pt field

    Returns:
        [events] HT values
    """
    return ak.sum(jet_data.jet_pt, axis=-1)


def thrust_3d(px, py, pz, max_iter=100):
    """For LHC data should consider probably 2d thrust."""
    p = np.stack([px, py, pz], axis=1)

    # total momentum magnitude
    p_abs = np.linalg.norm(p, axis=1)
    denom = np.sum(p_abs)

    # initial axis: direction of total momentum
    n = np.sum(p, axis=0)
    n = n / np.linalg.norm(n)

    for _ in range(max_iter):
        projections = p @ n
        signs = np.sign(projections)

        new_n = np.sum(signs[:, None] * p, axis=0)
        new_n = new_n / np.linalg.norm(new_n)

        if np.allclose(new_n, n):
            break

        n = new_n

    T = np.sum(np.abs(p @ n)) / denom

    return T, n


def compute_thrust(particle_data: ak.Array) -> ak.Array:
    """Event thrust T = max over n of sum_i |p_i . n| / sum_i |p_i|.

    Args:
        particle_data: [events, particles] with part_px, part_py, part_pz fields

    Returns:
        [events] thrust values (between 0.5 and 1.0)
    """
    thrust_values = []

    for event_px, event_py, event_pz in zip(
        particle_data.part_px, particle_data.part_py, particle_data.part_pz
    ):
        T, _ = thrust_3d(
            np.asarray(event_px), np.asarray(event_py), np.asarray(event_pz)
        )
        thrust_values.append(T)

    thrust_values = np.array(thrust_values)
    return thrust_values


def compute_aplanarity(eigenvalues: ak.Array) -> ak.Array:
    """Event aplanarity from the momentum tensor eigenvalues.

    Aplanarity = 3/2 * lambda_3 where lambda_3 is the smallest eigenvalue
    of the normalized momentum tensor.

    Args:
        eigenvalues: [events, 3] eigenvalues (ascending order: lambda3 <= lambda2 <= lambda1)

    Returns:
        [events] aplanarity values
    """
    aplanarity = (3 / 2) * eigenvalues[:, 0]
    return aplanarity


def compute_eigenvalues(particle_data: ak.Array) -> ak.Array:
    """Compute the eigenvalues of the normalized momentum tensor.

    Args:
        particle_data: [events, particles] with part_px, part_py, part_pz fields

    Returns:
        [events, 3] eigenvalues (ascending order: lambda3 <= lambda2 <= lambda1)
    """
    px = particle_data.part_px
    py = particle_data.part_py
    pz = particle_data.part_pz

    p2 = px**2 + py**2 + pz**2

    norm = ak.sum(p2, axis=-1)

    Sxx = ak.sum(px * px, axis=-1) / norm
    Syy = ak.sum(py * py, axis=-1) / norm
    Szz = ak.sum(pz * pz, axis=-1) / norm

    Sxy = ak.sum(px * py, axis=-1) / norm
    Sxz = ak.sum(px * pz, axis=-1) / norm
    Syz = ak.sum(py * pz, axis=-1) / norm

    tensor = np.stack(
        [
            np.stack([Sxx, Sxy, Sxz], axis=-1),
            np.stack([Sxy, Syy, Syz], axis=-1),
            np.stack([Sxz, Syz, Szz], axis=-1),
        ],
        axis=-2,
    )

    eigenvalues = np.linalg.eigvalsh(tensor)
    return eigenvalues


def compute_sphericity(eigenvalues: ak.Array) -> ak.Array:
    """Event sphericity from the momentum tensor eigenvalues.

    Sphericity = 3/2 * (lambda_2 + lambda_3) where lambda_i are the
    eigenvalues of the normalized momentum tensor.

    Args:
        eigenvalues: [events, 3] eigenvalues (ascending order: lambda3 <= lambda2 <= lambda1)

    Returns:
        [events] sphericity values (between 0 and 1)
    """
    lambda2 = eigenvalues[:, 1]
    lambda3 = eigenvalues[:, 0]
    sphericity = 1.5 * (lambda2 + lambda3)
    return sphericity


def compute_particle_counts(particle_data: ak.Array) -> ak.Array:
    """Counts of different particle types per event.

    Args:
        particle_data: [events, particles] with part_isElectron, part_isMuon,
                       part_isPhoton, part_isChargedHadron, part_isNeutralHadron fields

    Returns:
        ak.Array with fields num_electrons, num_muons, num_photons,
                 num_charged_hadrons, num_neutral_hadrons, each of shape [events]
    """
    return ak.zip(
        {
            "num_electrons": ak.sum(particle_data.part_isElectron, axis=-1),
            "num_muons": ak.sum(particle_data.part_isMuon, axis=-1),
            "num_photons": ak.sum(particle_data.part_isPhoton, axis=-1),
            "num_charged_hadrons": ak.sum(particle_data.part_isChargedHadron, axis=-1),
            "num_neutral_hadrons": ak.sum(particle_data.part_isNeutralHadron, axis=-1),
        }
    )


def compute_jettiness(jet_data: ak.Array) -> ak.Array:
    """Compute jettiness for each event."""
    # TODO: Maybe worth implementing also this?
    pass


def get_event_variables(particle_data: ak.Array, jet_data: ak.Array) -> dict:
    """Compute all event-level variables and return as a flat dict of [events] arrays.

    Args:
        particle_data: [events, particles] full particle record array
        jet_data: [events, jets] jet-level data (jet_pt, etc.)

    Returns:
        dict of field_name -> [events] arrays
    """
    particle_counts = compute_particle_counts(particle_data)
    eigenvalues = compute_eigenvalues(particle_data)
    return ak.Array(
        {
            "event_met": compute_met(particle_data),
            "event_num_jets": compute_num_jets(jet_data),
            "event_ht": compute_ht(jet_data),
            "event_thrust": compute_thrust(particle_data),
            "event_aplanarity": compute_aplanarity(eigenvalues=eigenvalues),
            "event_sphericity": compute_sphericity(eigenvalues=eigenvalues),
            "event_num_electrons": particle_counts.num_electrons,
            "event_num_muons": particle_counts.num_muons,
            "event_num_photons": particle_counts.num_photons,
            "event_num_charged_hadrons": particle_counts.num_charged_hadrons,
            "event_num_neutral_hadrons": particle_counts.num_neutral_hadrons,
        }
    )
