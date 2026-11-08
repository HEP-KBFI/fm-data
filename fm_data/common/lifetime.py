# Copied from ml-tau-data (https://github.com/HEP-KBFI/ml-tau-data),
# ntupelizer/tools/lifetime.py at commit 83f6b1af83, with one change (2026-11-02):
# calc_track_origin_vectorized used x0 = xr + sin(phi0) d0, y0 = yr - cos(phi0) d0, which
# mirrors the point of closest approach through the reference point. The LCIO/EDM4hep
# convention (ref. [1] below) is x0 = xr - sin(phi0) d0, y0 = yr + cos(phi0) d0. The mirrored
# PCA leaves |dxy| almost unchanged but randomises the lifetime sign w.r.t. the jet axis
# (b-jet tracks with large |IP| significance: 54% positive before, ~80% after the fix).
"""
Lifetime and Impact Parameter Tools (Vectorized Implementation)

This module provides vectorized helper functions for calculating impact parameters
and Point of Closest Approach (PCA) for particle tracks, which are essential
inputs for lifetime-based tagging algorithms.

The calculations use a linear extrapolation of the helix at the reference point.
Track parameter definitions (d0, z0, phi, tanL, Omega) follow:
    [1] https://flc.desy.de/lcnotes/notes/localfsExplorer_read?currentPath=/afs/desy.de/group/flc/lcnotes/LC-DET-2006-004.pdf

Math for finding the PCA is from:
    [2] https://www-h1.desy.de/psfiles/theses/h1th-134.pdf
    [3] https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

Authors: Torben Lange (KBFI), Laurits Tani (KBFI)
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
from fm_data.common import matching as m


# =============================================================================
# Type Aliases for Vectorized Operations
# =============================================================================

# Arrays can be numpy arrays or awkward arrays
ArrayLike = np.ndarray | ak.Array


# =============================================================================
# Constants
# =============================================================================

# Value for invalid/missing measurements (e.g. neutral candidates without a
# track). 0 rather than a large sentinel: the training dataloaders zero these
# entries anyway, and 0 cannot blow up an input normalisation. A missing
# measurement is still recognisable by its error being 0, since every real
# track has a positive IP error.
INVALID_VALUE = 0.0
NUM_TRACK_STATES = 4  # Number of track states stored per track


# =============================================================================
# Vectorized Track Origin Calculation
# =============================================================================


def calc_track_origin_vectorized(
    phi0: ArrayLike,
    d0: ArrayLike,
    z0: ArrayLike,
    xr: ArrayLike,
    yr: ArrayLike,
    zr: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Calculate track origin points from track parameters (vectorized).

    The track origin in the linear approximation:
        x0 = xr - sin(phi0) * d0
        y0 = yr + cos(phi0) * d0
        z0' = z0 + zr

    Args:
        phi0: Azimuthal angle array
        d0: Transverse impact parameter array
        z0: Longitudinal impact parameter array
        xr, yr, zr: Reference point coordinate arrays

    Returns:
        Tuple of (x0, y0, z0_prime) coordinate arrays
    """
    # LCIO convention (LC-DET-2006-004): the PCA to the reference point is at
    # (xr - d0 sin(phi0), yr + d0 cos(phi0)), see the note at the top of this file
    x0 = xr - np.sin(phi0) * d0
    y0 = yr + np.cos(phi0) * d0
    z0_prime = z0 + zr
    return x0, y0, z0_prime


# =============================================================================
# Vectorized PCA Arc Length Calculation
# =============================================================================


def calc_pca_arc_length_vectorized(
    phi0: ArrayLike,
    tanL: ArrayLike,
    x0: ArrayLike,
    y0: ArrayLike,
    z0_prime: ArrayLike,
    vertex_x: float,
    vertex_y: float,
    vertex_z: float,
    use_2d: bool = False,
) -> ArrayLike:
    """
    Calculate the arc length parameter s at the PCA for all tracks (vectorized).

    Uses the formula from [3] (Point-Line Distance in 3D):
        t = -[(a - vertex) · (b - a)] / |b - a|²

    where a is the track origin and b is a point unit distance along the track.

    Args:
        phi0: Azimuthal angle array
        tanL: tan(lambda) array
        x0, y0, z0_prime: Track origin coordinate arrays
        vertex_x, vertex_y, vertex_z: Primary vertex coordinates (scalars)
        use_2d: If True, calculate PCA in transverse (xy) plane only

    Returns:
        Arc length array at the PCA for each track
    """
    # Point a: track origin
    ax, ay, az = x0, y0, z0_prime

    # Direction vector (b - a) = (cos(phi0), sin(phi0), tanL)
    dx = np.cos(phi0)  # bx - ax
    dy = np.sin(phi0)  # by - ay
    dz = tanL  # bz - az

    if use_2d:
        # 2D calculation in transverse plane
        numerator = -((ax - vertex_x) * dx + (ay - vertex_y) * dy)
        denominator = dx**2 + dy**2
    else:
        # Full 3D calculation
        numerator = -(
            (ax - vertex_x) * dx + (ay - vertex_y) * dy + (az - vertex_z) * dz
        )
        denominator = dx**2 + dy**2 + dz**2

    return numerator / denominator


def calc_pca_arc_length_error_vectorized(
    phi0: ArrayLike,
    tanL: ArrayLike,
    d0: ArrayLike,
    x0: ArrayLike,
    y0: ArrayLike,
    z0_prime: ArrayLike,
    d0_error: ArrayLike,
    phi0_error: ArrayLike,
    z0_error: ArrayLike,
    tanL_error: ArrayLike,
    vertex_x: float,
    vertex_y: float,
    vertex_z: float,
    use_2d: bool = False,
) -> ArrayLike:
    """
    Calculate uncertainty on the PCA arc length parameter (vectorized).

    Propagates errors from track parameters using standard error propagation.

    Args:
        phi0, tanL, d0: Track parameter arrays
        x0, y0, z0_prime: Track origin coordinate arrays
        d0_error, phi0_error, z0_error, tanL_error: Parameter uncertainty arrays
        vertex_x, vertex_y, vertex_z: Primary vertex coordinates
        use_2d: If True, calculate error for 2D PCA only

    Returns:
        Arc length uncertainty array
    """
    # Origin errors (propagated from d0 and phi0)
    sin_phi = np.sin(phi0)
    cos_phi = np.cos(phi0)

    x0_error = np.sqrt(sin_phi**2 * d0_error**2 + (cos_phi * d0) ** 2 * phi0_error**2)
    y0_error = np.sqrt(cos_phi**2 * d0_error**2 + (sin_phi * d0) ** 2 * phi0_error**2)
    z0_prime_error = z0_error

    # Points and direction
    ax, ay, az = x0, y0, z0_prime
    ax_err, ay_err, az_err = x0_error, y0_error, z0_prime_error

    # b = a + direction
    bx = cos_phi + x0
    by = sin_phi + y0
    bz = tanL + z0_prime

    bx_err = np.sqrt(sin_phi**2 * phi0_error**2 + x0_error**2)
    by_err = np.sqrt(cos_phi**2 * phi0_error**2 + y0_error**2)
    bz_err = np.sqrt(tanL_error**2 + z0_prime_error**2)

    # Direction vector
    dx, dy, dz = cos_phi, sin_phi, tanL

    if use_2d:
        num = -((ax - vertex_x) * dx + (ay - vertex_y) * dy)
        num_err = np.sqrt(
            ax_err**2 * (2 * ax - bx - vertex_x) ** 2
            + bx_err**2 * (vertex_x - bx) ** 2
            + ay_err**2 * (2 * ay - by - vertex_y) ** 2
            + by_err**2 * (vertex_y - by) ** 2
        )
        den = dx**2 + dy**2
        den_err = np.sqrt(
            bx_err**2 * (2 * dx) ** 2
            + ax_err**2 * (2 * dx) ** 2
            + by_err**2 * (2 * dy) ** 2
            + ay_err**2 * (2 * dy) ** 2
        )
    else:
        num = -((ax - vertex_x) * dx + (ay - vertex_y) * dy + (az - vertex_z) * dz)
        num_err = np.sqrt(
            ax_err**2 * (2 * ax - bx - vertex_x) ** 2
            + bx_err**2 * (vertex_x - bx) ** 2
            + ay_err**2 * (2 * ay - by - vertex_y) ** 2
            + by_err**2 * (vertex_y - by) ** 2
            + az_err**2 * (2 * az - bz - vertex_z) ** 2
            + bz_err**2 * (vertex_z - bz) ** 2
        )
        den = dx**2 + dy**2 + dz**2
        den_err = np.sqrt(
            bx_err**2 * (2 * dx) ** 2
            + ax_err**2 * (2 * dx) ** 2
            + by_err**2 * (2 * dy) ** 2
            + ay_err**2 * (2 * dy) ** 2
            + bz_err**2 * (2 * dz) ** 2
            + az_err**2 * (2 * dz) ** 2
        )

    # Error propagation for ratio: sigma(a/b) = sqrt((sigma_a/b)^2 + (a*sigma_b/b^2)^2)
    return np.sqrt((num_err / den) ** 2 + (num * den_err / den**2) ** 2)


# =============================================================================
# Vectorized PCA Position Calculation
# =============================================================================


def calc_pca_position_vectorized(
    s: ArrayLike,
    phi0: ArrayLike,
    tanL: ArrayLike,
    x0: ArrayLike,
    y0: ArrayLike,
    z0_prime: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Calculate the 3D position of the PCA for all tracks (vectorized).

    The PCA position along the linearized track:
        x = s * cos(phi0) + x0
        y = s * sin(phi0) + y0
        z = s * tan(lambda) + z0'

    Args:
        s: Arc length parameter array
        phi0: Azimuthal angle array
        tanL: tan(lambda) array
        x0, y0, z0_prime: Track origin coordinate arrays

    Returns:
        Tuple of (pca_x, pca_y, pca_z) coordinate arrays
    """
    pca_x = s * np.cos(phi0) + x0
    pca_y = s * np.sin(phi0) + y0
    pca_z = s * tanL + z0_prime
    return pca_x, pca_y, pca_z


def calc_pca_position_error_vectorized(
    s: ArrayLike,
    s_error: ArrayLike,
    phi0: ArrayLike,
    d0: ArrayLike,
    d0_error: ArrayLike,
    phi0_error: ArrayLike,
    z0_error: ArrayLike,
    tanL: ArrayLike,
    tanL_error: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Calculate uncertainty on the PCA position (vectorized).

    Args:
        s, s_error: Arc length and its uncertainty arrays
        phi0, d0, tanL: Track parameter arrays
        d0_error, phi0_error, z0_error, tanL_error: Parameter uncertainty arrays

    Returns:
        Tuple of (x_error, y_error, z_error) arrays
    """
    sin_phi = np.sin(phi0)
    cos_phi = np.cos(phi0)

    # Origin errors
    x0_error = np.sqrt(sin_phi**2 * d0_error**2 + (cos_phi * d0) ** 2 * phi0_error**2)
    y0_error = np.sqrt(cos_phi**2 * d0_error**2 + (sin_phi * d0) ** 2 * phi0_error**2)
    z0_prime_error = z0_error

    # PCA position errors
    x_error = np.sqrt(
        (cos_phi * s_error) ** 2 + x0_error**2 + (s * sin_phi * phi0_error) ** 2
    )
    y_error = np.sqrt(
        (sin_phi * s_error) ** 2 + y0_error**2 + (s * cos_phi * phi0_error) ** 2
    )
    z_error = np.sqrt((tanL * s_error) ** 2 + (s * tanL_error) ** 2 + z0_prime_error**2)

    return x_error, y_error, z_error


# =============================================================================
# Vectorized Impact Parameter Calculation
# =============================================================================


def calc_impact_parameters_vectorized(
    pca_x: ArrayLike,
    pca_y: ArrayLike,
    pca_z: ArrayLike,
    pca_x_error: ArrayLike,
    pca_y_error: ArrayLike,
    pca_z_error: ArrayLike,
    vertex_x: float,
    vertex_y: float,
    vertex_z: float,
) -> dict[str, ArrayLike]:
    """
    Calculate impact parameters from PCA positions and vertex (vectorized).

    Args:
        pca_x, pca_y, pca_z: PCA position arrays
        pca_x_error, pca_y_error, pca_z_error: PCA position error arrays
        vertex_x, vertex_y, vertex_z: Primary vertex coordinates

    Returns:
        Dictionary with impact parameter arrays:
            - dxy: Transverse impact parameter
            - dz: Longitudinal impact parameter
            - d3: 3D impact parameter
            - dxy_error, dz_error, d3_error: Corresponding uncertainties
    """
    # Displacements
    delta_x = vertex_x - pca_x
    delta_y = vertex_y - pca_y
    delta_z = pca_z - vertex_z

    # Impact parameters
    dxy = np.sqrt(delta_x**2 + delta_y**2)
    dz = np.abs(delta_z)
    d3 = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

    # Error propagation
    # For dxy = sqrt(dx^2 + dy^2): sigma_dxy = sqrt(4*dx^2*sigma_x^2 + 4*dy^2*sigma_y^2) / (2*dxy)
    # Simplified: sigma_dxy = sqrt((dx*sigma_x)^2 + (dy*sigma_y)^2) / dxy (when dxy > 0)

    # Avoid division by zero
    dxy_safe = np.where(dxy > 1e-10, dxy, 1.0)
    d3_safe = np.where(d3 > 1e-10, d3, 1.0)

    dxy_error = (
        np.sqrt((delta_x * pca_x_error) ** 2 + (delta_y * pca_y_error) ** 2) / dxy_safe
    )
    dxy_error = np.where(dxy > 1e-10, dxy_error, 0.0)

    dz_error = pca_z_error

    d3_error = (
        np.sqrt(
            (delta_x * pca_x_error) ** 2
            + (delta_y * pca_y_error) ** 2
            + (delta_z * pca_z_error) ** 2
        )
        / d3_safe
    )
    d3_error = np.where(d3 > 1e-10, d3_error, 0.0)

    return {
        "dxy": dxy,
        "dz": dz,
        "d3": d3,
        "dxy_error": dxy_error,
        "dz_error": dz_error,
        "d3_error": d3_error,
    }


# =============================================================================
# Main Processing Functions
# =============================================================================


def _build_output_array(
    values: ArrayLike, valid_indices: np.ndarray, n_total: int
) -> np.ndarray:
    """Build output array with INVALID_VALUE for missing entries."""
    result = np.full(n_total, INVALID_VALUE)
    result[valid_indices] = values
    return result


def find_event_track_pcas(
    event: Any,
    reco_particle_collection: str = "PandoraPFOs",
    track_collection: str = "SiTracks_Refitted",
    vertex_collection: str = "PrimaryVertices",
    debug: int = -1,
) -> ak.Array:
    """
    Find PCAs and calculate impact parameters for all tracks (fully vectorized).

    This function processes all reconstructed particles in a single vectorized
    operation, avoiding Python loops for much better performance.

    Returns an awkward array with named fields accessible via dot notation:
        - result.dxy: Transverse impact parameter (3D PCA)
        - result.dz: Longitudinal impact parameter (3D PCA)
        - result.d3: 3D impact parameter (3D PCA)
        - result.d0: Track d0 parameter
        - result.z0: Track z0 parameter
        - result.dxy_2d: Transverse IP (2D PCA in xy plane)
        - result.dz_2d: Longitudinal IP (2D PCA)
        - result.d3_2d: 3D IP (2D PCA)
        - result.pca_x, result.pca_y, result.pca_z: PCA position
        - result.vertex_x, result.vertex_y, result.vertex_z: Vertex position
        - result.phi0, result.tanL, result.omega: Track extras
        - result.dxy_error, result.dz_error, result.d3_error: 3D PCA errors
        - result.d0_error, result.z0_error: Track parameter errors
        - result.dxy_2d_error, result.dz_2d_error, result.d3_2d_error: 2D PCA errors
        - result.pca_x_error, result.pca_y_error, result.pca_z_error: PCA position errors

    Args:
        event: Single event data structure (awkward array record)
        reco_particle_collection: Name of reconstructed particle collection
        track_collection: Name of track collection
        vertex_collection: Name of primary vertex collection
        debug: Debug level (-1=off, 0+=verbose)

    Returns:
        Awkward array with named fields for all impact parameters and errors
    """
    # =========================================================================
    # Extract primary vertex (scalar values)
    # =========================================================================
    vertex_x = float(event[f"{vertex_collection}.position.x"][0])
    vertex_y = float(event[f"{vertex_collection}.position.y"][0])
    vertex_z = float(event[f"{vertex_collection}.position.z"][0])

    # =========================================================================
    # Extract particle information
    # =========================================================================
    # Filter valid particles (PDG != 0 in new format)
    pdg = event[f"{reco_particle_collection}.PDG"]
    valid_particle_mask = pdg != 0

    tracks_begin_all = event[f"{reco_particle_collection}.tracks_begin"]
    tracks_end_all = event[f"{reco_particle_collection}.tracks_end"]

    # Apply mask to get only valid particles
    tracks_begin = tracks_begin_all[valid_particle_mask]
    tracks_end = tracks_end_all[valid_particle_mask]
    n_particles = len(tracks_begin)

    # =========================================================================
    # Build track index mapping (vectorized)
    # =========================================================================
    track_index_collection = f"_{reco_particle_collection}_tracks"
    track_index_map = event[f"{track_index_collection}.index"]

    # Determine which particles have tracks
    has_track = tracks_begin != tracks_end

    # Get track indices (use -1 for particles without tracks)
    # Handle empty track_index_map case
    n_track_links = len(track_index_map)
    if n_track_links == 0:
        particle_track_links = np.full(n_particles, -1, dtype=np.int64)
    else:
        # Use awkward for conditional indexing
        safe_idx = ak.where(tracks_begin < n_track_links, tracks_begin, 0)
        track_vals = track_index_map[safe_idx]
        particle_track_links = ak.to_numpy(
            ak.where(has_track & (tracks_begin < n_track_links), track_vals, -1)
        ).astype(np.int64)

    # Convert to track state indices (multiply by NUM_TRACK_STATES)
    si_track_idx = particle_track_links * NUM_TRACK_STATES

    # =========================================================================
    # Extract track states collection
    # =========================================================================
    track_states_collection = f"_{track_collection}_trackStates"

    # Get track state arrays
    n_track_states = len(event[f"{track_states_collection}.location"])

    # Create valid mask for tracks within bounds
    valid_track_mask = (particle_track_links >= 0) & (si_track_idx < n_track_states)
    valid_indices = np.where(valid_track_mask)[0]

    if len(valid_indices) == 0:
        if debug >= 0:
            print("No valid tracks found in event")
        # Return empty awkward array with correct structure
        return ak.zip(
            {
                "dxy": np.full(n_particles, INVALID_VALUE),
                "dz": np.full(n_particles, INVALID_VALUE),
                "d3": np.full(n_particles, INVALID_VALUE),
                "d0": np.full(n_particles, INVALID_VALUE),
                "z0": np.full(n_particles, INVALID_VALUE),
                "dxy_2d": np.full(n_particles, INVALID_VALUE),
                "dz_2d": np.full(n_particles, INVALID_VALUE),
                "d3_2d": np.full(n_particles, INVALID_VALUE),
                "pca_x": np.full(n_particles, INVALID_VALUE),
                "pca_y": np.full(n_particles, INVALID_VALUE),
                "pca_z": np.full(n_particles, INVALID_VALUE),
                "vertex_x": np.full(n_particles, INVALID_VALUE),
                "vertex_y": np.full(n_particles, INVALID_VALUE),
                "vertex_z": np.full(n_particles, INVALID_VALUE),
                "phi0": np.full(n_particles, INVALID_VALUE),
                "tanL": np.full(n_particles, INVALID_VALUE),
                "omega": np.full(n_particles, INVALID_VALUE),
                "dxy_error": np.full(n_particles, INVALID_VALUE),
                "dz_error": np.full(n_particles, INVALID_VALUE),
                "d3_error": np.full(n_particles, INVALID_VALUE),
                "d0_error": np.full(n_particles, INVALID_VALUE),
                "z0_error": np.full(n_particles, INVALID_VALUE),
                "dxy_2d_error": np.full(n_particles, INVALID_VALUE),
                "dz_2d_error": np.full(n_particles, INVALID_VALUE),
                "d3_2d_error": np.full(n_particles, INVALID_VALUE),
                "pca_x_error": np.full(n_particles, INVALID_VALUE),
                "pca_y_error": np.full(n_particles, INVALID_VALUE),
                "pca_z_error": np.full(n_particles, INVALID_VALUE),
            }
        )

    # Get the track state indices for valid particles
    valid_si_idx = si_track_idx[valid_track_mask]

    # =========================================================================
    # Extract track parameters for all valid tracks (vectorized)
    # =========================================================================
    d0 = ak.to_numpy(event[f"{track_states_collection}.D0"][valid_si_idx])
    z0 = ak.to_numpy(event[f"{track_states_collection}.Z0"][valid_si_idx])
    phi0 = ak.to_numpy(event[f"{track_states_collection}.phi"][valid_si_idx])
    tanL = ak.to_numpy(event[f"{track_states_collection}.tanLambda"][valid_si_idx])
    omega = ak.to_numpy(event[f"{track_states_collection}.omega"][valid_si_idx])
    xr = ak.to_numpy(event[f"{track_states_collection}.referencePoint.x"][valid_si_idx])
    yr = ak.to_numpy(event[f"{track_states_collection}.referencePoint.y"][valid_si_idx])
    zr = ak.to_numpy(event[f"{track_states_collection}.referencePoint.z"][valid_si_idx])

    # Extract covariance matrix elements
    cov_matrix = ak.to_numpy(
        event[f"{track_states_collection}.covMatrix.values[21]"][valid_si_idx]
    )
    d0_error = np.sqrt(np.abs(cov_matrix[:, 0]))
    phi0_error = np.sqrt(np.abs(cov_matrix[:, 2]))
    z0_error = np.sqrt(np.abs(cov_matrix[:, 9]))
    tanL_error = np.sqrt(np.abs(cov_matrix[:, 14]))

    # =========================================================================
    # Calculate track origins (vectorized)
    # =========================================================================
    x0, y0, z0_prime = calc_track_origin_vectorized(phi0, d0, z0, xr, yr, zr)

    # =========================================================================
    # Calculate 3D PCA (vectorized)
    # =========================================================================
    s_3d = calc_pca_arc_length_vectorized(
        phi0, tanL, x0, y0, z0_prime, vertex_x, vertex_y, vertex_z, use_2d=False
    )
    s_3d_error = calc_pca_arc_length_error_vectorized(
        phi0,
        tanL,
        d0,
        x0,
        y0,
        z0_prime,
        d0_error,
        phi0_error,
        z0_error,
        tanL_error,
        vertex_x,
        vertex_y,
        vertex_z,
        use_2d=False,
    )

    pca_x, pca_y, pca_z = calc_pca_position_vectorized(
        s_3d, phi0, tanL, x0, y0, z0_prime
    )
    pca_x_err, pca_y_err, pca_z_err = calc_pca_position_error_vectorized(
        s_3d, s_3d_error, phi0, d0, d0_error, phi0_error, z0_error, tanL, tanL_error
    )

    ip_3d = calc_impact_parameters_vectorized(
        pca_x,
        pca_y,
        pca_z,
        pca_x_err,
        pca_y_err,
        pca_z_err,
        vertex_x,
        vertex_y,
        vertex_z,
    )

    # =========================================================================
    # Calculate 2D PCA (vectorized)
    # =========================================================================
    s_2d = calc_pca_arc_length_vectorized(
        phi0, tanL, x0, y0, z0_prime, vertex_x, vertex_y, vertex_z, use_2d=True
    )
    s_2d_error = calc_pca_arc_length_error_vectorized(
        phi0,
        tanL,
        d0,
        x0,
        y0,
        z0_prime,
        d0_error,
        phi0_error,
        z0_error,
        tanL_error,
        vertex_x,
        vertex_y,
        vertex_z,
        use_2d=True,
    )

    pca_2d_x, pca_2d_y, pca_2d_z = calc_pca_position_vectorized(
        s_2d, phi0, tanL, x0, y0, z0_prime
    )
    pca_2d_x_err, pca_2d_y_err, pca_2d_z_err = calc_pca_position_error_vectorized(
        s_2d, s_2d_error, phi0, d0, d0_error, phi0_error, z0_error, tanL, tanL_error
    )

    ip_2d = calc_impact_parameters_vectorized(
        pca_2d_x,
        pca_2d_y,
        pca_2d_z,
        pca_2d_x_err,
        pca_2d_y_err,
        pca_2d_z_err,
        vertex_x,
        vertex_y,
        vertex_z,
    )

    if debug >= 0:
        n_valid = len(valid_indices)
        n_neutral = ak.sum(~has_track)
        print(f"Processed {n_valid} tracks, {n_neutral} neutral particles")

    # =========================================================================
    # Build awkward array with named fields
    # =========================================================================
    def _out(values: ArrayLike) -> np.ndarray:
        return _build_output_array(values, valid_indices, n_particles)

    return ak.zip(
        {
            "dxy": _out(ip_3d["dxy"]),
            "dz": _out(ip_3d["dz"]),
            "d3": _out(ip_3d["d3"]),
            "d0": _out(d0),
            "z0": _out(z0),
            "dxy_2d": _out(ip_2d["dxy"]),
            "dz_2d": _out(ip_2d["dz"]),
            "d3_2d": _out(ip_2d["d3"]),
            "pca_x": _out(pca_x),
            "pca_y": _out(pca_y),
            "pca_z": _out(pca_z),
            "vertex_x": _out(np.full(len(valid_indices), vertex_x)),
            "vertex_y": _out(np.full(len(valid_indices), vertex_y)),
            "vertex_z": _out(np.full(len(valid_indices), vertex_z)),
            "phi0": _out(phi0),
            "tanL": _out(tanL),
            "omega": _out(omega),
            "dxy_error": _out(ip_3d["dxy_error"]),
            "dz_error": _out(ip_3d["dz_error"]),
            "d3_error": _out(ip_3d["d3_error"]),
            "d0_error": _out(d0_error),
            "z0_error": _out(z0_error),
            "dxy_2d_error": _out(ip_2d["dxy_error"]),
            "dz_2d_error": _out(ip_2d["dz_error"]),
            "d3_2d_error": _out(ip_2d["d3_error"]),
            "pca_x_error": _out(pca_x_err),
            "pca_y_error": _out(pca_y_err),
            "pca_z_error": _out(pca_z_err),
        }
    )


def calc_signed_impact_parameters(
    impact_params: np.ndarray,
    pca_x: np.ndarray,
    pca_y: np.ndarray,
    pca_z: np.ndarray,
    vertex_x: float,
    vertex_y: float,
    vertex_z: float,
    jet_px: float,
    jet_py: float,
    jet_pz: float,
) -> np.ndarray:
    """
    Project impact parameters onto jet axis direction to determine sign (vectorized).

    The sign convention follows BTV-11-002: positive if the PCA is in the
    direction of the jet, negative otherwise.

    Args:
        impact_params: Impact parameter values array
        pca_x, pca_y, pca_z: PCA position arrays
        vertex_x, vertex_y, vertex_z: Primary vertex coordinates
        jet_px, jet_py, jet_pz: Jet momentum components

    Returns:
        Signed impact parameter array
    """
    # Normalize jet direction
    jet_norm = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)
    jet_dir_x = jet_px / jet_norm
    jet_dir_y = jet_py / jet_norm
    jet_dir_z = jet_pz / jet_norm

    # PCA direction relative to vertex
    pca_dir_x = pca_x - vertex_x
    pca_dir_y = pca_y - vertex_y
    pca_dir_z = pca_z - vertex_z

    # Normalize PCA direction
    pca_norm = np.sqrt(pca_dir_x**2 + pca_dir_y**2 + pca_dir_z**2)
    pca_norm_safe = np.where(pca_norm > 1e-10, pca_norm, 1.0)

    pca_dir_x = np.where(pca_norm > 1e-10, pca_dir_x / pca_norm_safe, jet_dir_x)
    pca_dir_y = np.where(pca_norm > 1e-10, pca_dir_y / pca_norm_safe, jet_dir_y)
    pca_dir_z = np.where(pca_norm > 1e-10, pca_dir_z / pca_norm_safe, jet_dir_z)

    # Sign based on alignment with jet direction
    dot_product = pca_dir_x * jet_dir_x + pca_dir_y * jet_dir_y + pca_dir_z * jet_dir_z
    sign = np.sign(dot_product)

    # Apply sign, keeping invalid values unchanged
    signed_ips = np.where(
        impact_params != INVALID_VALUE,
        sign * np.abs(impact_params),
        INVALID_VALUE,
    )

    return signed_ips


def find_all_track_pcas(
    events: ak.Array,
    reco_particle_collection: str = "PandoraPFOs",
    track_collection: str = "SiTracks_Refitted",
    vertex_collection: str = "PrimaryVertices",
    valid_particle_mask: ak.Array = None,
    debug: int = -1,
) -> ak.Array:
    """
    Find PCAs and calculate impact parameters for all tracks across all events.

    This function processes multiple events using flatten/unflatten for efficiency,
    returning a jagged awkward array where each event contains the PCA results
    for all its particles.

    Returns an awkward array with shape [n_events, var_particles] where each
    particle record has named fields accessible via dot notation:
        - result.dxy: Transverse impact parameter (3D PCA)
        - result.dz: Longitudinal impact parameter (3D PCA)
        - result.d3: 3D impact parameter (3D PCA)
        - result.d0: Track d0 parameter
        - result.z0: Track z0 parameter
        - result.dxy_2d: Transverse IP (2D PCA in xy plane)
        - result.dz_2d: Longitudinal IP (2D PCA)
        - result.d3_2d: 3D IP (2D PCA)
        - result.pca_x, result.pca_y, result.pca_z: PCA position
        - result.vertex_x, result.vertex_y, result.vertex_z: Vertex position
        - result.phi0, result.tanL, result.omega: Track extras
        - result.dxy_error, result.dz_error, result.d3_error: 3D PCA errors
        - result.d0_error, result.z0_error: Track parameter errors
        - result.dxy_2d_error, result.dz_2d_error, result.d3_2d_error: 2D PCA errors
        - result.pca_x_error, result.pca_y_error, result.pca_z_error: PCA position errors

    Example usage:
        >>> results = find_all_track_pcas(events)
        >>> # Access dxy for all particles in event 0
        >>> results[0].dxy
        >>> # Access dxy for all particles across all events (jagged)
        >>> results.dxy
        >>> # Flatten to get all particles
        >>> ak.flatten(results.dxy)

    Args:
        events: Awkward array containing multiple events
        reco_particle_collection: Name of reconstructed particle collection
        track_collection: Name of track collection
        vertex_collection: Name of primary vertex collection
        valid_particle_mask: Pre-computed mask for valid particles. If None,
            will filter by PDG != 0
        debug: Debug level (-1=off, 0+=verbose)

    Returns:
        Jagged awkward array with shape [n_events, var_particles] containing
        named fields for all impact parameters and errors
    """
    n_events = len(events)

    if debug >= 0:
        print(f"Processing {n_events} events...")

    # =========================================================================
    # Extract and flatten all data across events
    # =========================================================================
    track_states_collection = f"_{track_collection}_trackStates"
    track_index_collection = f"_{reco_particle_collection}_tracks"

    # Get primary vertices (one per event)
    vertex_x_per_event = events[f"{vertex_collection}.position.x"][:, 0]
    vertex_y_per_event = events[f"{vertex_collection}.position.y"][:, 0]
    vertex_z_per_event = events[f"{vertex_collection}.position.z"][:, 0]

    # Get particle data (jagged: [n_events, var_particles])
    tracks_begin = events[f"{reco_particle_collection}.tracks_begin"]
    tracks_end = events[f"{reco_particle_collection}.tracks_end"]

    # Use provided mask or filter by PDG != 0
    if valid_particle_mask is None:
        pdg = events[f"{reco_particle_collection}.PDG"]
        valid_particle_mask = pdg != 0

    tracks_begin_valid = tracks_begin[valid_particle_mask]
    tracks_end_valid = tracks_end[valid_particle_mask]

    # Get counts per event for unflattening later
    counts = ak.num(tracks_begin_valid)

    if debug >= 0:
        print(f"  Total particles: {ak.sum(counts)}")

    # Create event indices for each particle (for vertex lookup)
    event_indices = ak.flatten(
        ak.broadcast_arrays(
            ak.local_index(tracks_begin_valid, axis=0), tracks_begin_valid
        )[0]
    )

    # Flatten particle arrays
    tracks_begin_flat = ak.flatten(tracks_begin_valid)
    tracks_end_flat = ak.flatten(tracks_end_valid)

    # Broadcast vertex coordinates to each particle
    vertex_x = vertex_x_per_event[event_indices]
    vertex_y = vertex_y_per_event[event_indices]
    vertex_z = vertex_z_per_event[event_indices]

    # =========================================================================
    # Build track links for all particles (vectorized)
    # =========================================================================
    has_track = tracks_begin_flat != tracks_end_flat

    # Flatten track indices collection
    track_indices = events[f"{track_index_collection}.index"]
    track_indices_counts = ak.num(track_indices)

    # Get cumulative offsets for track indices
    track_indices_flat = ak.flatten(track_indices)
    track_indices_counts_np = ak.to_numpy(track_indices_counts)
    track_offsets = np.concatenate([[0], np.cumsum(track_indices_counts_np)])

    # Convert to numpy once for reuse
    event_indices_np = ak.to_numpy(event_indices)
    tracks_begin_np = ak.to_numpy(tracks_begin_flat)
    has_track_np = ak.to_numpy(has_track)

    # Calculate global track index for each particle
    # global_idx = track_offsets[event_idx] + tracks_begin
    event_offsets = track_offsets[event_indices_np]
    global_track_begin = event_offsets + tracks_begin_np

    # Vectorized bounds check
    in_bounds = tracks_begin_np < track_indices_counts_np[event_indices_np]
    valid_for_lookup = has_track_np & in_bounds

    # Get track indices using global flat indexing
    track_indices_flat_np = ak.to_numpy(track_indices_flat)
    particle_track_links = np.where(
        valid_for_lookup,
        track_indices_flat_np[
            np.minimum(global_track_begin, len(track_indices_flat_np) - 1)
        ],
        -1,
    ).astype(np.int64)

    # Convert to track state indices
    si_track_idx = particle_track_links * NUM_TRACK_STATES

    # Get track state counts per event for bounds checking and later offset calculation
    n_track_states_per_event = ak.to_numpy(
        ak.num(events[f"{track_states_collection}.D0"])
    )
    ts_offsets = np.concatenate([[0], np.cumsum(n_track_states_per_event)])

    # Valid mask: has track and within bounds
    valid_track_mask = (particle_track_links >= 0) & (
        si_track_idx < n_track_states_per_event[event_indices_np]
    )
    valid_indices = np.where(valid_track_mask)[0]

    if debug >= 0:
        print(f"  Valid tracks: {len(valid_indices)}")

    # Total particle count for output arrays
    n_particles = len(tracks_begin_flat)

    if len(valid_indices) == 0:
        if debug >= 0:
            print("  No valid tracks found")
        # Return all invalid values
        output_fields = [
            "dxy",
            "dz",
            "d3",
            "d0",
            "z0",
            "dxy_2d",
            "dz_2d",
            "d3_2d",
            "pca_x",
            "pca_y",
            "pca_z",
            "vertex_x",
            "vertex_y",
            "vertex_z",
            "phi0",
            "tanL",
            "omega",
            "dxy_error",
            "dz_error",
            "d3_error",
            "d0_error",
            "z0_error",
            "dxy_2d_error",
            "dz_2d_error",
            "d3_2d_error",
            "pca_x_error",
            "pca_y_error",
            "pca_z_error",
        ]
        result_flat = ak.zip(
            {f: np.full(n_particles, INVALID_VALUE) for f in output_fields}
        )
        return ak.unflatten(result_flat, counts)

    # =========================================================================
    # Extract track parameters for valid particles (vectorized)
    # =========================================================================
    # Flatten track states and use global indexing
    track_states_D0 = ak.flatten(events[f"{track_states_collection}.D0"])
    track_states_Z0 = ak.flatten(events[f"{track_states_collection}.Z0"])
    track_states_phi = ak.flatten(events[f"{track_states_collection}.phi"])
    track_states_tanL = ak.flatten(events[f"{track_states_collection}.tanLambda"])
    track_states_omega = ak.flatten(events[f"{track_states_collection}.omega"])
    track_states_xr = ak.flatten(events[f"{track_states_collection}.referencePoint.x"])
    track_states_yr = ak.flatten(events[f"{track_states_collection}.referencePoint.y"])
    track_states_zr = ak.flatten(events[f"{track_states_collection}.referencePoint.z"])
    track_states_cov = ak.flatten(
        events[f"{track_states_collection}.covMatrix.values[21]"]
    )

    # Calculate global track state indices
    valid_event_indices = event_indices_np[valid_indices]
    valid_si_track_idx = si_track_idx[valid_indices]
    global_ts_idx = ts_offsets[valid_event_indices] + valid_si_track_idx

    # Extract track parameters using global indices (fully vectorized)
    d0 = ak.to_numpy(track_states_D0[global_ts_idx])
    z0 = ak.to_numpy(track_states_Z0[global_ts_idx])
    phi0 = ak.to_numpy(track_states_phi[global_ts_idx])
    tanL = ak.to_numpy(track_states_tanL[global_ts_idx])
    omega = ak.to_numpy(track_states_omega[global_ts_idx])
    xr = ak.to_numpy(track_states_xr[global_ts_idx])
    yr = ak.to_numpy(track_states_yr[global_ts_idx])
    zr = ak.to_numpy(track_states_zr[global_ts_idx])
    cov_matrix = ak.to_numpy(track_states_cov[global_ts_idx])

    d0_error = np.sqrt(np.abs(cov_matrix[:, 0]))
    phi0_error = np.sqrt(np.abs(cov_matrix[:, 2]))
    z0_error = np.sqrt(np.abs(cov_matrix[:, 9]))
    tanL_error = np.sqrt(np.abs(cov_matrix[:, 14]))

    # Get vertex coordinates for valid particles
    vtx_x = ak.to_numpy(vertex_x[valid_indices])
    vtx_y = ak.to_numpy(vertex_y[valid_indices])
    vtx_z = ak.to_numpy(vertex_z[valid_indices])

    # =========================================================================
    # Vectorized PCA calculations (all valid particles at once)
    # =========================================================================
    x0, y0, z0_prime = calc_track_origin_vectorized(phi0, d0, z0, xr, yr, zr)

    # 3D PCA
    s_3d = calc_pca_arc_length_vectorized(
        phi0, tanL, x0, y0, z0_prime, vtx_x, vtx_y, vtx_z, use_2d=False
    )
    s_3d_error = calc_pca_arc_length_error_vectorized(
        phi0,
        tanL,
        d0,
        x0,
        y0,
        z0_prime,
        d0_error,
        phi0_error,
        z0_error,
        tanL_error,
        vtx_x,
        vtx_y,
        vtx_z,
        use_2d=False,
    )
    pca_x, pca_y, pca_z = calc_pca_position_vectorized(
        s_3d, phi0, tanL, x0, y0, z0_prime
    )
    pca_x_err, pca_y_err, pca_z_err = calc_pca_position_error_vectorized(
        s_3d, s_3d_error, phi0, d0, d0_error, phi0_error, z0_error, tanL, tanL_error
    )
    ip_3d = calc_impact_parameters_vectorized(
        pca_x, pca_y, pca_z, pca_x_err, pca_y_err, pca_z_err, vtx_x, vtx_y, vtx_z
    )

    # 2D PCA
    s_2d = calc_pca_arc_length_vectorized(
        phi0, tanL, x0, y0, z0_prime, vtx_x, vtx_y, vtx_z, use_2d=True
    )
    s_2d_error = calc_pca_arc_length_error_vectorized(
        phi0,
        tanL,
        d0,
        x0,
        y0,
        z0_prime,
        d0_error,
        phi0_error,
        z0_error,
        tanL_error,
        vtx_x,
        vtx_y,
        vtx_z,
        use_2d=True,
    )
    pca_2d_x, pca_2d_y, pca_2d_z = calc_pca_position_vectorized(
        s_2d, phi0, tanL, x0, y0, z0_prime
    )
    pca_2d_x_err, pca_2d_y_err, pca_2d_z_err = calc_pca_position_error_vectorized(
        s_2d, s_2d_error, phi0, d0, d0_error, phi0_error, z0_error, tanL, tanL_error
    )
    ip_2d = calc_impact_parameters_vectorized(
        pca_2d_x,
        pca_2d_y,
        pca_2d_z,
        pca_2d_x_err,
        pca_2d_y_err,
        pca_2d_z_err,
        vtx_x,
        vtx_y,
        vtx_z,
    )

    # =========================================================================
    # Build output and unflatten back to [n_events, var_particles] structure
    # =========================================================================
    def _out(values: ArrayLike) -> np.ndarray:
        return _build_output_array(values, valid_indices, n_particles)

    result_flat = ak.zip(
        {
            "dxy": _out(ip_3d["dxy"]),
            "dz": _out(ip_3d["dz"]),
            "d3": _out(ip_3d["d3"]),
            "d0": _out(d0),
            "z0": _out(z0),
            "dxy_2d": _out(ip_2d["dxy"]),
            "dz_2d": _out(ip_2d["dz"]),
            "d3_2d": _out(ip_2d["d3"]),
            "pca_x": _out(pca_x),
            "pca_y": _out(pca_y),
            "pca_z": _out(pca_z),
            "vertex_x": _out(vtx_x),
            "vertex_y": _out(vtx_y),
            "vertex_z": _out(vtx_z),
            "phi0": _out(phi0),
            "tanL": _out(tanL),
            "omega": _out(omega),
            "dxy_error": _out(ip_3d["dxy_error"]),
            "dz_error": _out(ip_3d["dz_error"]),
            "d3_error": _out(ip_3d["d3_error"]),
            "d0_error": _out(d0_error),
            "z0_error": _out(z0_error),
            "dxy_2d_error": _out(ip_2d["dxy_error"]),
            "dz_2d_error": _out(ip_2d["dz_error"]),
            "d3_2d_error": _out(ip_2d["d3_error"]),
            "pca_x_error": _out(pca_x_err),
            "pca_y_error": _out(pca_y_err),
            "pca_z_error": _out(pca_z_err),
        }
    )
    result = ak.unflatten(result_flat, counts)

    if debug >= 0:
        print(f"Done. Output shape: {len(result)} events")

    return result


def get_jet_constituent_lifetime_vars(
    events: ak.Array,
    reco_jet_constituent_indices: ak.Array,
    reco_jets: ak.Array,
    reco_particle_collection: str = "PandoraPFOs",
    track_collection: str = "SiTracks_Refitted",
    vertex_collection: str = "PrimaryVertices",
    lifetime_vars: list = [],
    signed_lifetime_vars: list = [],
    debug: int = -1,
) -> dict[str, ak.Array]:
    """
    Calculate all lifetime variables for jet constituents.

    This function:
    1. Computes PCA and impact parameters for all particles using vectorized code
    2. Extracts values for jet constituents
    3. Computes signed impact parameters using jet direction

    Args:
        events: Awkward array containing all events
        reco_jet_constituent_indices: Indices mapping jet constituents to particles
        reco_jets: Reconstructed jets (for signed IP calculation)
        reco_particle_collection: Name of reco particle collection
        track_collection: Name of track collection
        vertex_collection: Name of vertex collection
        debug: Debug level (-1=off, 0+=verbose)

    Returns:
        Dictionary with all lifetime variables for jet constituents:
            - dxy, dz, d3, d0, z0: Impact parameters (3D PCA)
            - dxy_2d, dz_2d, d3_2d: Impact parameters (2D PCA)
            - pca_x, pca_y, pca_z: PCA position
            - vertex_x, vertex_y, vertex_z: Primary vertex position
            - phi0, tanL, omega: Track parameters
            - *_error: Corresponding uncertainties
            - signed_*: Signed impact parameters
    """
    # =========================================================================
    # Step 1: Calculate lifetime info for all particles (vectorized)
    # =========================================================================
    lifetime_info = find_all_track_pcas(
        events,
        reco_particle_collection=reco_particle_collection,
        track_collection=track_collection,
        vertex_collection=vertex_collection,
        debug=debug,
    )

    num_ptcls_per_jet = ak.num(reco_jet_constituent_indices, axis=-1)

    # Helper to extract property for jet constituents
    def _get(field: str) -> ak.Array:
        return m.get_jet_constituent_property(
            getattr(lifetime_info, field),
            reco_jet_constituent_indices,
            num_ptcls_per_jet,
        )

    # =========================================================================
    # Step 2: Build result dictionary with all lifetime variables
    # =========================================================================
    result = {key: _get(key) for key in lifetime_vars}

    # =========================================================================
    # Step 3: Calculate signed impact parameters
    # =========================================================================
    for ip_name in signed_lifetime_vars:
        result[f"signed_{ip_name}"] = calc_all_signed_ip_for_jets(
            result[ip_name],
            result["pca_x"],
            result["pca_y"],
            result["pca_z"],
            result["vertex_x"],
            result["vertex_y"],
            result["vertex_z"],
            reco_jets,
        )

    return result


def calc_all_signed_ip_for_jets(
    ip_values: ak.Array,
    pca_x: ak.Array,
    pca_y: ak.Array,
    pca_z: ak.Array,
    vertex_x: ak.Array,
    vertex_y: ak.Array,
    vertex_z: ak.Array,
    reco_jets: ak.Array,
) -> ak.Array:
    """
    Calculate signed impact parameters for all jet constituents (fully vectorized).

    Uses vectorized computation to determine the sign based on
    the alignment of the PCA-vertex direction with the jet direction.

    Args:
        ip_values: Impact parameter values [n_events, n_jets, n_constituents]
        pca_x, pca_y, pca_z: PCA positions [n_events, n_jets, n_constituents]
        vertex_x, vertex_y, vertex_z: Vertex positions [n_events, n_jets, n_constituents]
        reco_jets: Jet 4-vectors [n_events, n_jets]

    Returns:
        Signed impact parameters [n_events, n_jets, n_constituents]
    """
    # Broadcast jet momentum to constituent level [n_events, n_jets] -> [n_events, n_jets, n_constituents]
    jet_px, _ = ak.broadcast_arrays(reco_jets.px, ip_values)
    jet_py, _ = ak.broadcast_arrays(reco_jets.py, ip_values)
    jet_pz, _ = ak.broadcast_arrays(reco_jets.pz, ip_values)

    # Normalize jet direction
    jet_norm = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)
    jet_norm_safe = ak.where(jet_norm > 1e-10, jet_norm, 1.0)
    jet_dir_x = jet_px / jet_norm_safe
    jet_dir_y = jet_py / jet_norm_safe
    jet_dir_z = jet_pz / jet_norm_safe

    # PCA direction relative to vertex
    pca_dir_x = pca_x - vertex_x
    pca_dir_y = pca_y - vertex_y
    pca_dir_z = pca_z - vertex_z

    # Normalize PCA direction
    pca_norm = np.sqrt(pca_dir_x**2 + pca_dir_y**2 + pca_dir_z**2)
    pca_norm_safe = ak.where(pca_norm > 1e-10, pca_norm, 1.0)

    pca_dir_x = ak.where(pca_norm > 1e-10, pca_dir_x / pca_norm_safe, jet_dir_x)
    pca_dir_y = ak.where(pca_norm > 1e-10, pca_dir_y / pca_norm_safe, jet_dir_y)
    pca_dir_z = ak.where(pca_norm > 1e-10, pca_dir_z / pca_norm_safe, jet_dir_z)

    # Sign based on alignment with jet direction
    dot_product = pca_dir_x * jet_dir_x + pca_dir_y * jet_dir_y + pca_dir_z * jet_dir_z
    sign = ak.where(dot_product >= 0, 1.0, -1.0)

    # Apply sign, keeping invalid values unchanged
    signed_ips = ak.where(
        ip_values != INVALID_VALUE,
        sign * np.abs(ip_values),
        INVALID_VALUE,
    )

    return signed_ips


def assign_lifetime_vars_to_jets(
    all_particle_lifetime_info: ak.Array,
    reco_jet_constituent_indices: ak.Array,
    reco_jets: ak.Array,
    lifetime_vars: list = [],
    signed_lifetime_vars: list = [],
) -> dict[str, ak.Array]:
    """
    Assign pre-computed lifetime variables to jet constituents using indices.

    Args:
        all_particle_lifetime_info: Pre-computed lifetime info for all particles
            (output of find_all_track_pcas)
        reco_jet_constituent_indices: Indices mapping jet constituents to particles
        reco_jets: Reconstructed jets (for signed IP calculation)
        lifetime_vars: List of lifetime variable names to include
        signed_lifetime_vars: List of variables to compute signed versions for

    Returns:
        Dictionary with lifetime variables for jet constituents
    """
    num_ptcls_per_jet = ak.num(reco_jet_constituent_indices, axis=-1)

    # Helper to extract property for jet constituents
    def _get(field: str) -> ak.Array:
        return m.get_jet_constituent_property(
            getattr(all_particle_lifetime_info, field),
            reco_jet_constituent_indices,
            num_ptcls_per_jet,
        )

    # Build result dictionary with all lifetime variables
    result = {key: _get(key) for key in lifetime_vars}

    # Calculate signed impact parameters
    for ip_name in signed_lifetime_vars:
        result[f"signed_{ip_name}"] = calc_all_signed_ip_for_jets(
            result[ip_name],
            result["pca_x"],
            result["pca_y"],
            result["pca_z"],
            result["vertex_x"],
            result["vertex_y"],
            result["vertex_z"],
            reco_jets,
        )

    return result


def replicate_lifetime_vars_per_jet(
    all_particle_lifetime_info: ak.Array,
    reco_jets: ak.Array,
    lifetime_vars: list = [],
    signed_lifetime_vars: list = [],
) -> dict[str, ak.Array]:
    """
    Replicate pre-computed lifetime variables for all event reco candidates per jet.

    For each jet in an event, this attaches ALL reco candidates from that event
    (not just the jet constituents). This is useful for attention-based models
    that need full event context for each jet.

    The output structure is [n_events, n_jets, n_reco_cands_in_event].

    Args:
        all_particle_lifetime_info: Pre-computed lifetime info for all particles
            (output of find_all_track_pcas)
        reco_jets: Reconstructed jets [n_events, n_jets]
        lifetime_vars: List of lifetime variable names to include
        signed_lifetime_vars: List of variables to compute signed versions for

    Returns:
        Dictionary with lifetime variables for all reco candidates per jet:
            - Each key maps to an array of shape [n_events, n_jets, n_reco_cands]
    """

    # Replicate event-level info for each jet in the event
    # For each event j, create len(reco_jets[j]) copies of the event's reco cand info
    def _replicate_for_jets(field_name: str) -> ak.Array:
        field = getattr(all_particle_lifetime_info, field_name)
        return ak.from_iter(
            [
                [field[j] for _ in range(len(reco_jets[j]))]
                for j in range(len(reco_jets))
            ]
        )

    result = {key: _replicate_for_jets(key) for key in lifetime_vars}

    # Calculate signed impact parameters
    for ip_name in signed_lifetime_vars:
        result[f"signed_{ip_name}"] = calc_all_signed_ip_for_event_reco_cands(
            result[ip_name],
            result["pca_x"],
            result["pca_y"],
            result["pca_z"],
            result["vertex_x"],
            result["vertex_y"],
            result["vertex_z"],
            reco_jets,
        )

    return result


def get_event_reco_cand_lifetime_vars(
    events: ak.Array,
    reco_jets: ak.Array,
    reco_particle_collection: str = "PandoraPFOs",
    track_collection: str = "SiTracks_Refitted",
    vertex_collection: str = "PrimaryVertices",
    lifetime_vars: list = [],
    signed_lifetime_vars: list = [],
    debug: int = -1,
) -> dict[str, ak.Array]:
    """
    Calculate lifetime variables for ALL reco candidates in each event, repeated per jet.

    For each jet in an event, this attaches ALL reco candidates from that event
    (not just the jet constituents). This is useful for attention-based models
    that need full event context for each jet.

    The output structure is [n_events, n_jets, n_reco_cands_in_event].

    Args:
        events: Awkward array containing all events
        reco_jets: Reconstructed jets [n_events, n_jets]
        reco_particle_collection: Name of reco particle collection
        track_collection: Name of track collection
        vertex_collection: Name of vertex collection
        lifetime_vars: List of lifetime variable names to include
        signed_lifetime_vars: List of variables to compute signed versions for
        debug: Debug level (-1=off, 0+=verbose)

    Returns:
        Dictionary with lifetime variables for all reco candidates per jet:
            - Each key maps to an array of shape [n_events, n_jets, n_reco_cands]
    """
    # =========================================================================
    # Step 1: Calculate lifetime info for all particles (vectorized)
    # =========================================================================
    lifetime_info = find_all_track_pcas(
        events,
        reco_particle_collection=reco_particle_collection,
        track_collection=track_collection,
        vertex_collection=vertex_collection,
        debug=debug,
    )

    # =========================================================================
    # Step 2: Replicate event-level info for each jet in the event
    # =========================================================================
    # For each event j, create len(reco_jets[j]) copies of the event's reco cand info
    def _replicate_for_jets(field_name: str) -> ak.Array:
        field = getattr(lifetime_info, field_name)
        return ak.from_iter(
            [
                [field[j] for _ in range(len(reco_jets[j]))]
                for j in range(len(reco_jets))
            ]
        )

    result = {key: _replicate_for_jets(key) for key in lifetime_vars}

    # =========================================================================
    # Step 3: Calculate signed impact parameters
    # =========================================================================
    for ip_name in signed_lifetime_vars:
        result[f"signed_{ip_name}"] = calc_all_signed_ip_for_event_reco_cands(
            result[ip_name],
            result["pca_x"],
            result["pca_y"],
            result["pca_z"],
            result["vertex_x"],
            result["vertex_y"],
            result["vertex_z"],
            reco_jets,
        )

    return result


def calc_all_signed_ip_for_event_reco_cands(
    ip_values: ak.Array,
    pca_x: ak.Array,
    pca_y: ak.Array,
    pca_z: ak.Array,
    vertex_x: ak.Array,
    vertex_y: ak.Array,
    vertex_z: ak.Array,
    reco_jets: ak.Array,
) -> ak.Array:
    """
    Calculate signed impact parameters for event-level reco candidates (fully vectorized).

    Uses vectorized computation to determine the sign based on
    the alignment of the PCA-vertex direction with the jet direction.

    Args:
        ip_values: Impact parameter values [n_events, n_jets, n_reco_cands]
        pca_x, pca_y, pca_z: PCA positions [n_events, n_jets, n_reco_cands]
        vertex_x, vertex_y, vertex_z: Vertex positions [n_events, n_jets, n_reco_cands]
        reco_jets: Jet 4-vectors [n_events, n_jets]

    Returns:
        Signed impact parameters [n_events, n_jets, n_reco_cands]
    """
    # Broadcast jet momentum to reco candidate level [n_events, n_jets] -> [n_events, n_jets, n_reco_cands]
    jet_px, _ = ak.broadcast_arrays(reco_jets.px, ip_values)
    jet_py, _ = ak.broadcast_arrays(reco_jets.py, ip_values)
    jet_pz, _ = ak.broadcast_arrays(reco_jets.pz, ip_values)

    # Normalize jet direction
    jet_norm = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)
    jet_norm_safe = ak.where(jet_norm > 1e-10, jet_norm, 1.0)
    jet_dir_x = jet_px / jet_norm_safe
    jet_dir_y = jet_py / jet_norm_safe
    jet_dir_z = jet_pz / jet_norm_safe

    # PCA direction relative to vertex
    pca_dir_x = pca_x - vertex_x
    pca_dir_y = pca_y - vertex_y
    pca_dir_z = pca_z - vertex_z

    # Normalize PCA direction
    pca_norm = np.sqrt(pca_dir_x**2 + pca_dir_y**2 + pca_dir_z**2)
    pca_norm_safe = ak.where(pca_norm > 1e-10, pca_norm, 1.0)

    pca_dir_x = ak.where(pca_norm > 1e-10, pca_dir_x / pca_norm_safe, jet_dir_x)
    pca_dir_y = ak.where(pca_norm > 1e-10, pca_dir_y / pca_norm_safe, jet_dir_y)
    pca_dir_z = ak.where(pca_norm > 1e-10, pca_dir_z / pca_norm_safe, jet_dir_z)

    # Sign based on alignment with jet direction
    dot_product = pca_dir_x * jet_dir_x + pca_dir_y * jet_dir_y + pca_dir_z * jet_dir_z
    sign = ak.where(dot_product >= 0, 1.0, -1.0)

    # Apply sign, keeping invalid values unchanged
    signed_ips = ak.where(
        ip_values != INVALID_VALUE,
        sign * np.abs(ip_values),
        INVALID_VALUE,
    )

    return signed_ips
