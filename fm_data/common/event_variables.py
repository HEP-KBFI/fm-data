"""Event-level variables for e+e- events, calculated from the particles of the event.

Same definitions as in gabbro (gabbro/metrics/event_observables.py on the event-level
branch), copied so that the ntupelizer does not depend on gabbro.
"""

import itertools

import awkward as ak
import numpy as np
import vector

vector.register_awkward()


def _padded_momenta(p4s):
    """Return the 3-momenta as zero-padded numpy array of shape (n_events, n_max, 3)."""
    n_max = max(int(ak.max(ak.num(p4s))), 1) if len(p4s) > 0 else 1
    return np.stack(
        [
            ak.to_numpy(ak.fill_none(ak.pad_none(getattr(p4s, comp), n_max, clip=True), 0.0))
            for comp in ["px", "py", "pz"]
        ],
        axis=-1,
    ).astype(np.float64)


def calc_thrust(momenta, n_iterations=10, n_leading=4):
    """Calculate the thrust of each event with the iterative algorithm.

    The axis is iterated as n -> sum_i sign(p_i . n) p_i, which converges to a local
    maximum of sum_i |p_i . n|. As in Pythia, the iteration is started from all
    sign combinations of the `n_leading` hardest particles (sum_j +-p_j) and the
    maximum is taken. The result is a lower bound of the exact thrust, which is
    found for the vast majority of events (the remaining ones end up in a slightly
    lower local maximum).

    Parameters
    ----------
    momenta : np.ndarray
        Zero-padded 3-momenta of shape (n_events, n_particles, 3).
    n_iterations : int, optional
        Number of iterations per starting axis.
    n_leading : int, optional
        Number of hardest particles used to build the starting axes.

    Returns
    -------
    np.ndarray
        Thrust values of shape (n_events,). Events without particles get 0.
    """
    abs_p = np.linalg.norm(momenta, axis=-1)
    sum_abs_p = abs_p.sum(axis=1)
    thrust = np.zeros(len(momenta))
    n_leading = min(n_leading, momenta.shape[1])
    order = np.argsort(-abs_p, axis=1)[:, :n_leading]
    leading = np.take_along_axis(momenta, order[:, :, None], axis=1)
    # the overall sign of the axis doesn't matter --> fix the sign of the hardest particle
    for signs_rest in itertools.product([1, -1], repeat=n_leading - 1):
        start_signs = np.array((1,) + signs_rest, dtype=float)
        axis = np.einsum("j,ejk->ek", start_signs, leading)
        for _ in range(n_iterations):
            axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
            signs = np.sign(np.einsum("eik,ek->ei", momenta, axis))
            signs[signs == 0] = 1
            axis = np.einsum("ei,eik->ek", signs, momenta)
        projection = np.linalg.norm(axis, axis=-1)
        thrust = np.maximum(thrust, projection / np.maximum(sum_abs_p, 1e-12))
    return thrust


def calc_sphericity_aplanarity(momenta):
    """Calculate sphericity and aplanarity from the sphericity tensor
    S_ab = sum_i p_ia p_ib / sum_i |p_i|^2.

    Parameters
    ----------
    momenta : np.ndarray
        Zero-padded 3-momenta of shape (n_events, n_particles, 3).

    Returns
    -------
    tuple of np.ndarray
        Sphericity and aplanarity, each of shape (n_events,).
    """
    tensor = np.einsum("eia,eib->eab", momenta, momenta)
    norm = np.einsum("eia,eia->e", momenta, momenta)
    tensor /= np.maximum(norm, 1e-12)[:, None, None]
    # eigenvalues in ascending order: lambda_3 <= lambda_2 <= lambda_1
    eigenvalues = np.linalg.eigvalsh(tensor)
    sphericity = 1.5 * (eigenvalues[:, 0] + eigenvalues[:, 1])
    aplanarity = 1.5 * eigenvalues[:, 0]
    return sphericity, aplanarity


def calc_event_observables(p4s):
    """Calculate event-level variables from the particles of each event.

    Parameters
    ----------
    p4s : ak.Array
        Momentum4D array of the particles, shape (n_events, <var>).

    Returns
    -------
    dict
        {"event_<variable>": np.ndarray} with nparticles, energy_vis, mass_vis, p_vis,
        thrust, sphericity, aplanarity and leading_energy_fraction.
    """
    p4_event = ak.sum(p4s, axis=1)
    energy_vis = ak.to_numpy(p4_event.E)
    momenta = _padded_momenta(p4s)
    sphericity, aplanarity = calc_sphericity_aplanarity(momenta)
    leading_energy = ak.to_numpy(ak.fill_none(ak.max(p4s.E, axis=1), 0.0))
    return {
        "event_nparticles": ak.to_numpy(ak.num(p4s)),
        "event_energy_vis": energy_vis,
        "event_mass_vis": ak.to_numpy(p4_event.tau),
        "event_p_vis": ak.to_numpy(p4_event.p),
        "event_thrust": calc_thrust(momenta),
        "event_sphericity": sphericity,
        "event_aplanarity": aplanarity,
        "event_leading_energy_fraction": leading_energy / np.maximum(energy_vis, 1e-12),
    }
