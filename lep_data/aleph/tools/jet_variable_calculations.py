import awkward as ak
import fastjet
import numba as nb
import numpy as np
import vector


def init_axes(pt, eta, phi, N):
    idx_sorted = ak.argsort(pt, axis=-1, ascending=False)

    topN = idx_sorted[..., :N]  # (..., Naxes) — top N particles per jet

    axes_eta = eta[topN]
    axes_phi = phi[topN]

    return axes_eta, axes_phi


def assign_axes(eta, phi, axes_eta, axes_phi):
    # phi/eta: [..., particles]    axes_*: [..., N]
    # broadcast to [..., particles, N]
    dphi = phi[..., None] - axes_phi[..., None, :]
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))

    deta = eta[..., None] - axes_eta[..., None, :]

    dR = np.sqrt(deta**2 + dphi**2)

    return ak.argmin(dR, axis=-1)


def update_axes(pt, eta, phi, assign, Naxes):

    new_eta = []
    new_phi = []

    for k in range(Naxes):
        mask = assign == k

        denom = ak.sum(pt * mask, axis=-1)

        # ---------- ETA (linear is fine) ----------
        eta_k = ak.sum(pt * eta * mask, axis=-1)

        # ---------- PHI (circular mean) ----------
        sin_sum = ak.sum(pt * np.sin(phi) * mask, axis=-1)
        cos_sum = ak.sum(pt * np.cos(phi) * mask, axis=-1)

        phi_k = np.arctan2(sin_sum, cos_sum)

        # ---------- safe division ----------
        eta_k = ak.where(
            denom > 0, eta_k / denom, eta_k
        )  # fallback: keep old-like behavior
        phi_k = ak.where(denom > 0, phi_k, phi_k)

        new_eta.append(eta_k[..., np.newaxis])
        new_phi.append(phi_k[..., np.newaxis])

    return ak.concatenate(new_eta, axis=-1), ak.concatenate(new_phi, axis=-1)


def find_axes(pt, eta, phi, Naxes=2, iters=5):

    axes_eta, axes_phi = init_axes(pt, eta, phi, Naxes)

    for _ in range(iters):
        assign = assign_axes(eta, phi, axes_eta, axes_phi)

        axes_eta, axes_phi = update_axes(pt, eta, phi, assign, Naxes)

    return axes_eta, axes_phi


def tau_N(pt, eta, phi, axes_eta, axes_phi, R0=0.8):
    # phi/eta: [..., particles]    axes_*: [..., N]
    # broadcast to [..., particles, N]
    dphi = phi[..., None] - axes_phi[..., None, :]
    dphi = np.arctan2(np.sin(dphi), np.cos(dphi))

    deta = eta[..., None] - axes_eta[..., None, :]

    dR = np.sqrt(deta**2 + dphi**2)

    min_dR = ak.min(dR, axis=-1)
    min_dR = np.minimum(min_dR, R0)

    num = ak.sum(pt * min_dR, axis=-1)
    den = ak.sum(pt, axis=-1) * R0

    return num / den


def calculate_tau_n(pt, eta, phi, Naxes=2, R0=0.8):
    axes_eta, axes_phi = find_axes(pt, eta, phi, Naxes)
    return tau_N(pt, eta, phi, axes_eta, axes_phi, R0)


@nb.njit
def _delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return np.arctan2(np.sin(dphi), np.cos(dphi))


@nb.njit
def _delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = _delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


@nb.njit
def _ca_cluster(pts, etas, phis):
    """Cambridge/Aachen reclustering of a single jet's constituents.

    Returns (merge_history, p4s) where:
      merge_history[i] = (child1, child2, dR_at_merge) for each of n-1 merges
      p4s: all 4-vectors [E, px, py, pz] — first n are original, rest are merged
    """
    n = len(pts)
    if n <= 1:
        return np.zeros((0, 3)), np.zeros((1, 4))

    max_nodes = 2 * n - 1
    p4s = np.zeros((max_nodes, 4))
    # Massless particles: E = pT * cosh(eta)
    p4s[:n, 0] = pts * np.cosh(etas)
    p4s[:n, 1] = pts * np.cos(phis)
    p4s[:n, 2] = pts * np.sin(phis)
    p4s[:n, 3] = pts * np.sinh(etas)

    all_etas = np.zeros(max_nodes)
    all_phis = np.zeros(max_nodes)
    all_etas[:n] = etas
    all_phis[:n] = phis

    active = np.zeros(max_nodes, dtype=np.bool_)
    active[:n] = True

    history = np.zeros((n - 1, 3))
    next_node = n

    for step in range(n - 1):
        min_dr = 1e10
        min_i = -1
        min_j = -1
        for i in range(next_node):
            if not active[i]:
                continue
            for j in range(i + 1, next_node):
                if not active[j]:
                    continue
                dr = _delta_r(all_etas[i], all_phis[i], all_etas[j], all_phis[j])
                if dr < min_dr:
                    min_dr = dr
                    min_i = i
                    min_j = j

        p4s[next_node] = p4s[min_i] + p4s[min_j]
        pt_merged = np.sqrt(p4s[next_node, 1] ** 2 + p4s[next_node, 2] ** 2)
        if pt_merged > 0:
            all_etas[next_node] = np.arcsinh(p4s[next_node, 3] / pt_merged)
            all_phis[next_node] = np.arctan2(p4s[next_node, 2], p4s[next_node, 1])

        active[min_i] = False
        active[min_j] = False
        active[next_node] = True

        history[step, 0] = min_i
        history[step, 1] = min_j
        history[step, 2] = min_dr

        next_node += 1

    return history, p4s


@nb.njit
def _sd_mass(history, p4s, z_cut, beta, R0):
    """Apply SoftDrop grooming on a C/A merge tree. Returns groomed mass."""
    n_total = p4s.shape[0]
    n_orig = (n_total + 1) // 2

    if n_orig <= 1:
        if n_orig == 1:
            p = p4s[0]
            m2 = p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
            return np.sqrt(max(0.0, m2))
        return 0.0

    # Build parent array
    parent = np.full(n_total, -1, dtype=np.int32)
    merge_dr = np.zeros(n_total)
    for k in range(len(history)):
        c1 = int(history[k, 0])
        c2 = int(history[k, 1])
        pnode = n_orig + k
        parent[c1] = pnode
        parent[c2] = pnode
        merge_dr[pnode] = history[k, 2]

    # Find root
    root = -1
    for i in range(n_total):
        if parent[i] == -1:
            root = i
            break

    # Iterative SD walk
    current = root
    while True:
        left = -1
        right = -1
        for i in range(n_total):
            if parent[i] == current:
                if left < 0:
                    left = i
                else:
                    right = i
                    break

        if left < 0 or right < 0:
            # Leaf
            p = p4s[current]
            m2 = p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
            return np.sqrt(max(0.0, m2))

        pt_l = np.sqrt(p4s[left, 1] ** 2 + p4s[left, 2] ** 2)
        pt_r = np.sqrt(p4s[right, 1] ** 2 + p4s[right, 2] ** 2)
        pt_sum = pt_l + pt_r

        if pt_sum == 0.0:
            return 0.0

        if min(pt_l, pt_r) / pt_sum > z_cut * (merge_dr[current] / R0) ** beta:
            p = p4s[current]
            m2 = p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
            return np.sqrt(max(0.0, m2))
        else:
            current = left if pt_l >= pt_r else right


def calculate_sdmass(
    jet_assigned_particles: ak.Array,
    z_cut: float = 0.1,
    beta: float = 0.0,
    R0: float = 0.8,
):
    """Compute SoftDrop mass for each jet.

    Reclusters each jet's constituents with Cambridge/Aachen, then applies
    the SoftDrop grooming condition:

        min(pT1, pT2) / (pT1 + pT2) > z_cut * (ΔR / R0)^beta

    Input:  jet_assigned_particles  [events, jets, constituents]
    Output: softdrop masses         [events, jets]
    """
    n_events = len(jet_assigned_particles)
    all_masses = []
    n_jets_per_event = np.zeros(n_events, dtype=np.int32)

    for ievt in range(n_events):
        jets = jet_assigned_particles[ievt]
        n_jets_per_event[ievt] = len(jets)
        for ijet in range(len(jets)):
            parts = jets[ijet]
            pts = np.asarray(ak.to_numpy(parts.part_pt))
            if len(pts) <= 1:
                all_masses.append(0.0)
            else:
                etas = np.asarray(ak.to_numpy(parts.part_eta))
                phis = np.asarray(ak.to_numpy(parts.part_phi))
                history, p4s = _ca_cluster(pts, etas, phis)
                all_masses.append(_sd_mass(history, p4s, z_cut, beta, R0))

    masses = np.array(all_masses)
    return ak.unflatten(masses, n_jets_per_event)


# ---------------------------------------------------------------------------
# ee_genkt-based variants (p=-1) for use with self-clustered jets
# ---------------------------------------------------------------------------


@nb.njit
def _ee_genkt_cluster(pts, etas, phis):
    """ee_genkt (p=-1) reclustering of a single jet's constituents.

    Distance: d_ij = min(E_i^(-2), E_j^(-2)) * (1 - cos(theta_ij))

    Returns (merge_history, p4s) where:
      merge_history[i] = (child1, child2, d_ij_at_merge)
      p4s: all 4-vectors [E, px, py, pz] — first n are original, rest are merged
    """
    n = len(pts)
    if n <= 1:
        return np.zeros((0, 3)), np.zeros((1, 4))

    max_nodes = 2 * n - 1
    p4s = np.zeros((max_nodes, 4))
    # Massless particles: E = pT * cosh(eta)
    p4s[:n, 0] = pts * np.cosh(etas)
    p4s[:n, 1] = pts * np.cos(phis)
    p4s[:n, 2] = pts * np.sin(phis)
    p4s[:n, 3] = pts * np.sinh(etas)

    all_etas = np.zeros(max_nodes)
    all_phis = np.zeros(max_nodes)
    all_etas[:n] = etas
    all_phis[:n] = phis

    active = np.zeros(max_nodes, dtype=np.bool_)
    active[:n] = True

    history = np.zeros((n - 1, 3))
    next_node = n

    for step in range(n - 1):
        min_dist = 1e300
        min_i = -1
        min_j = -1
        for i in range(next_node):
            if not active[i]:
                continue
            Ei = p4s[i, 0]
            invE2_i = 1.0 / (Ei * Ei) if Ei > 0.0 else 1e300
            for j in range(i + 1, next_node):
                if not active[j]:
                    continue
                Ej = p4s[j, 0]
                invE2_j = 1.0 / (Ej * Ej) if Ej > 0.0 else 1e300
                # cos(theta_ij) = (p_i · p_j) / (|p_i| |p_j|) = (p_i · p_j) / (E_i E_j)
                dot3 = (
                    p4s[i, 1] * p4s[j, 1]
                    + p4s[i, 2] * p4s[j, 2]
                    + p4s[i, 3] * p4s[j, 3]
                )
                cos_theta = dot3 / (Ei * Ej) if Ei > 0.0 and Ej > 0.0 else 1.0
                if cos_theta > 1.0:
                    cos_theta = 1.0
                elif cos_theta < -1.0:
                    cos_theta = -1.0
                dist = min(invE2_i, invE2_j) * (1.0 - cos_theta)
                if dist < min_dist:
                    min_dist = dist
                    min_i = i
                    min_j = j

        p4s[next_node] = p4s[min_i] + p4s[min_j]
        pt_merged = np.sqrt(p4s[next_node, 1] ** 2 + p4s[next_node, 2] ** 2)
        if pt_merged > 0.0:
            all_etas[next_node] = np.arcsinh(p4s[next_node, 3] / pt_merged)
            all_phis[next_node] = np.arctan2(p4s[next_node, 2], p4s[next_node, 1])

        active[min_i] = False
        active[min_j] = False
        active[next_node] = True

        history[step, 0] = min_i
        history[step, 1] = min_j
        history[step, 2] = min_dist

        next_node += 1

    return history, p4s


@nb.njit
def _sd_mass_ee(history, p4s, z_cut, beta, R0):
    """Apply SoftDrop grooming on an ee_genkt merge tree.

    The merge distance in the history is the raw ee_genkt distance
    d_ij = min(E_i^(-2), E_j^(-2)) * (1 - cos(theta_ij)).

    To convert to an angular scale comparable to C/A, we normalize by
    the energy weight: d_ang = d_ij / min(E_i^(-2), E_j^(-2)) = 1 - cos(theta).
    The reference scale d_ref = 1 - cos(R0).
    """
    n_total = p4s.shape[0]
    n_orig = (n_total + 1) // 2

    d_ref = 1.0 - np.cos(R0)
    if d_ref <= 0.0:
        d_ref = 1.0

    if n_orig <= 1:
        if n_orig == 1:
            p = p4s[0]
            m2 = p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
            return np.sqrt(max(0.0, m2))
        return 0.0

    # Build parent array and store both raw distance and energy weight
    parent = np.full(n_total, -1, dtype=np.int32)
    merge_dist_raw = np.zeros(n_total)
    merge_weight = np.ones(n_total)  # min(E_i^(-2), E_j^(-2))
    for k in range(len(history)):
        c1 = int(history[k, 0])
        c2 = int(history[k, 1])
        pnode = n_orig + k
        parent[c1] = pnode
        parent[c2] = pnode
        merge_dist_raw[pnode] = history[k, 2]
        Ei = p4s[c1, 0]
        Ej = p4s[c2, 0]
        invE2_i = 1.0 / (Ei * Ei) if Ei > 0.0 else 1.0
        invE2_j = 1.0 / (Ej * Ej) if Ej > 0.0 else 1.0
        merge_weight[pnode] = min(invE2_i, invE2_j)

    # Find root
    root = -1
    for i in range(n_total):
        if parent[i] == -1:
            root = i
            break

    # Iterative SD walk
    current = root
    while True:
        left = -1
        right = -1
        for i in range(n_total):
            if parent[i] == current:
                if left < 0:
                    left = i
                else:
                    right = i
                    break

        if left < 0 or right < 0:
            # Leaf
            p = p4s[current]
            m2 = p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
            return np.sqrt(max(0.0, m2))

        pt_l = np.sqrt(p4s[left, 1] ** 2 + p4s[left, 2] ** 2)
        pt_r = np.sqrt(p4s[right, 1] ** 2 + p4s[right, 2] ** 2)
        pt_sum = pt_l + pt_r

        if pt_sum == 0.0:
            return 0.0

        # Convert raw ee_genkt distance to angular scale
        d_ang = (
            merge_dist_raw[current] / merge_weight[current]
            if merge_weight[current] > 0.0
            else merge_dist_raw[current]
        )

        if min(pt_l, pt_r) / pt_sum > z_cut * (d_ang / d_ref) ** beta:
            p = p4s[current]
            m2 = p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2
            return np.sqrt(max(0.0, m2))
        else:
            current = left if pt_l >= pt_r else right


def calculate_sdmass_ee(
    jet_assigned_particles: ak.Array,
    z_cut: float = 0.1,
    beta: float = 0.0,
    R0: float = 0.8,
):
    """Compute SoftDrop mass using ee_genkt (p=-1) reclustering.

    Reclusters each jet's constituents with ee_genkt, then applies
    the SoftDrop grooming condition using the ee_genkt distance measure.
    """
    n_events = len(jet_assigned_particles)
    all_masses = []
    n_jets_per_event = np.zeros(n_events, dtype=np.int32)

    for ievt in range(n_events):
        jets = jet_assigned_particles[ievt]
        n_jets_per_event[ievt] = len(jets)
        for ijet in range(len(jets)):
            parts = jets[ijet]
            pts = np.asarray(ak.to_numpy(parts.part_pt))
            if len(pts) <= 1:
                all_masses.append(0.0)
            else:
                etas = np.asarray(ak.to_numpy(parts.part_eta))
                phis = np.asarray(ak.to_numpy(parts.part_phi))
                history, p4s = _ee_genkt_cluster(pts, etas, phis)
                all_masses.append(_sd_mass_ee(history, p4s, z_cut, beta, R0))

    masses = np.array(all_masses)
    return ak.unflatten(masses, n_jets_per_event)


def calculate_tau_n_ee(
    pt: ak.Array,
    eta: ak.Array,
    phi: ak.Array,
    Naxes: int = 2,
    R0: float = 0.8,
):
    """N-subjettiness using ee_genkt (p=-1) reclustering for subjet axes.

    Reclusters each jet's constituents with ee_genkt and uses
    exclusive_jets(n_jets=Naxes) to determine the N subjet axes.
    """
    n_events = len(pt)
    all_taus = []
    n_jets_per_event = np.zeros(n_events, dtype=np.int32)

    for ievt in range(n_events):
        evt_pt = pt[ievt]
        evt_eta = eta[ievt]
        evt_phi = phi[ievt]
        n_jets = len(evt_pt)
        n_jets_per_event[ievt] = n_jets
        for ijet in range(n_jets):
            jpt = np.asarray(ak.to_numpy(evt_pt[ijet]))
            n_parts = len(jpt)
            if n_parts <= Naxes:
                all_taus.append(0.0)
                continue
            jeta = np.asarray(ak.to_numpy(evt_eta[ijet]))
            jphi = np.asarray(ak.to_numpy(evt_phi[ijet]))

            # Build massless 4-vectors and recluster with ee_genkt
            # Use same format as particle_filters.py: mass + px + py + pz
            p4 = vector.awk(
                ak.zip(
                    {
                        "mass": np.zeros_like(jpt),
                        "px": jpt * np.cos(jphi),
                        "py": jpt * np.sin(jphi),
                        "pz": jpt * np.sinh(jeta),
                    }
                )
            )
            jetdef = fastjet.JetDefinition2Param(fastjet.ee_genkt_algorithm, R0, -1)
            cluster = fastjet.ClusterSequence(p4, jetdef)
            subjets = cluster.exclusive_jets(n_jets=Naxes)
            subjets = vector.awk(subjets)
            axes_eta = np.asarray(subjets.eta)
            axes_phi = np.asarray(subjets.phi)

            # Compute tauN for this jet
            dphi = jphi[:, None] - axes_phi[None, :]
            dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
            deta = jeta[:, None] - axes_eta[None, :]
            dR = np.sqrt(deta**2 + dphi**2)
            min_dR = np.min(dR, axis=1)
            min_dR = np.minimum(min_dR, R0)
            tau = np.sum(jpt * min_dR) / (np.sum(jpt) * R0)
            all_taus.append(tau)

    return ak.unflatten(np.array(all_taus), n_jets_per_event)
