"""Generator-level objects for the jet labels: the quarks of the hard process, hadronic
taus (with their visible momentum) and weakly decaying b/c hadrons."""

import awkward as ak
import numpy as np

# Pythia status of the outgoing particles of the hardest process (e.g. Z -> q qbar)
HARD_PROCESS_OUTGOING_STATUS = 23

_NEUTRINOS = (12, 14, 16)
_CHARGED_LEPTONS = (11, 13)


def _isin(abs_pdg, values):
    result = abs_pdg == values[0]
    for value in values[1:]:
        result = result | (abs_pdg == value)
    return result


def heaviest_quark(pdg):
    """Heaviest quark flavour of hadrons (0 for non-hadrons, e.g. leptons or diquarks)."""
    abs_pdg = np.abs(pdg) % 10000  # drop the excitation digits (e.g. 100443 -> 443)
    n_q1 = (abs_pdg // 1000) % 10
    n_q2 = (abs_pdg // 100) % 10
    n_q3 = (abs_pdg // 10) % 10
    is_hadron = (np.abs(pdg) > 100) & (n_q3 > 0)  # n_q3 == 0 for diquarks
    return ak.where(is_hadron, ak.where(n_q1 > 0, n_q1, n_q2), 0)


class MCTruth:
    """The MC particles of a batch of events with their daughter relations.

    Parameters
    ----------
    events : ak.Array
        Output of `ntupelizer.tools.reading.read_events`.
    mc_collection : str
        Name of the MC particle collection (e.g. "MCParticles").
    """

    def __init__(self, events, mc_collection="MCParticles"):
        mc = mc_collection
        self.pdg = events[f"{mc}.PDG"]
        self.status = events[f"{mc}.generatorStatus"]
        px = events[f"{mc}.momentum.x"]
        py = events[f"{mc}.momentum.y"]
        pz = events[f"{mc}.momentum.z"]
        energy = np.sqrt(px**2 + py**2 + pz**2 + events[f"{mc}.mass"] ** 2)
        self.particles = ak.zip(
            {"px": px, "py": py, "pz": pz, "energy": energy, "pdg": self.pdg}
        )
        # daughters of each particle: [events, particles, daughters]
        n_daughters = events[f"{mc}.daughters_end"] - events[f"{mc}.daughters_begin"]
        daughter_index = events[f"_{mc}_daughters.index"]
        if not ak.all(ak.sum(n_daughters, axis=1) == ak.num(daughter_index)):
            raise ValueError("Unexpected MC particle daughter relations.")
        self._n_daughters = ak.flatten(n_daughters)
        self._daughter_index = daughter_index

    def daughter_values(self, values):
        """`values` (per particle) of the daughters of each particle,
        [events, particles, daughters]."""
        return ak.unflatten(values[self._daughter_index], self._n_daughters, axis=1)

    def beam_sqrt_s(self):
        """sqrt(s) of each event from the incoming beam particles (generator status 4)."""
        is_beam = self.status == 4
        beams = self.particles[is_beam]
        p_sum = {field: ak.sum(beams[field], axis=1) for field in ("px", "py", "pz", "energy")}
        return np.sqrt(
            np.maximum(p_sum["energy"] ** 2 - p_sum["px"] ** 2 - p_sum["py"] ** 2 - p_sum["pz"] ** 2, 0)
        )

    def hard_quarks(self):
        """The outgoing quarks (d, u, s, c, b) of the hard process, with the fields px, py,
        pz, energy, pdg."""
        is_hard_quark = (self.status == HARD_PROCESS_OUTGOING_STATUS) & (np.abs(self.pdg) <= 5)
        return self.particles[is_hard_quark]

    def hadronic_taus(self):
        """The hadronically decaying taus (last copy, no electron or muon among the
        decay products), with their visible four-momentum (tau minus its neutrinos) in px,
        py, pz, energy and the tau's pdg."""
        abs_pdg = np.abs(self.pdg)
        abs_pdg_daughters = self.daughter_values(abs_pdg)
        is_last_tau = (abs_pdg == 15) & ~ak.any(abs_pdg_daughters == 15, axis=2)
        is_hadronic = ~ak.any(_isin(abs_pdg_daughters, _CHARGED_LEPTONS), axis=2)
        is_neutrino_daughter = _isin(abs_pdg_daughters, _NEUTRINOS)
        visible = {}
        for field in ("px", "py", "pz", "energy"):
            neutrino_sum = ak.sum(
                ak.where(is_neutrino_daughter, self.daughter_values(self.particles[field]), 0),
                axis=2,
            )
            visible[field] = self.particles[field] - neutrino_sum
        visible["pdg"] = self.pdg
        taus = ak.zip(visible)
        return taus[is_last_tau & is_hadronic]

    def truth_objects(self, kind):
        """The objects the jets are labelled with: "quarks" or "hadronic_taus"."""
        if kind == "quarks":
            return self.hard_quarks()
        if kind == "hadronic_taus":
            return self.hadronic_taus()
        raise ValueError(f"Unknown truth_objects: {kind} (use 'quarks' or 'hadronic_taus')")

    def heavy_hadrons(self):
        """The weakly decaying b and c hadrons (no daughter of the same heavy flavour), with
        the fields px, py, pz, energy, pdg and flavour (5 or 4). c hadrons from b-hadron
        decays are included."""
        quark = heaviest_quark(self.pdg)
        quark_daughters = self.daughter_values(quark)
        hadrons = []
        for flavour in (5, 4):
            is_weakly_decaying = (quark == flavour) & ~ak.any(quark_daughters == flavour, axis=2)
            selected = self.particles[is_weakly_decaying]
            hadrons.append(
                ak.with_field(selected, ak.full_like(selected.pdg, flavour), "flavour")
            )
        return ak.concatenate(hadrons, axis=1)
