"""Tests for the ntupelizer. Run in the container: ./run.sh python3 -m pytest tests"""

import glob
import unittest

import awkward as ak
import numpy as np
import vector
from omegaconf import OmegaConf

from fm_data.common import jets as jt
from fm_data.common import lifetime as lt
from fm_data.ee_sim.truth import MCTruth, heaviest_quark

vector.register_awkward()

CONFIG = "fm_data/ee_sim/config/cld.yaml"
ALEPH_CONFIG = "fm_data/lep/config/aleph.yaml"
ZQQ_FILES = sorted(glob.glob(OmegaConf.load(CONFIG).samples.Zqq.input_dir + "/*.root"))


def _lcio_d0(point, phi):
    """d0 of a straight track through `point` with azimuth `phi` (LCIO sign convention:
    the PCA to the origin is at (-d0 sin(phi), d0 cos(phi)))."""
    return -point[0] * np.sin(phi) + point[1] * np.cos(phi)


class TestLifetime(unittest.TestCase):
    def test_track_origin_lcio_convention(self):
        # track from a decay at (1, 0, 0) mm, going at 0.2 rad to the x axis
        phi = 0.2
        vertex = np.array([1.0, 0.0])
        direction = np.array([np.cos(phi), np.sin(phi)])
        pca = vertex - vertex @ direction * direction
        d0 = _lcio_d0(vertex, phi)
        x0, y0, _ = lt.calc_track_origin_vectorized(
            np.array([phi]), np.array([d0]), np.array([0.0]), 0.0, 0.0, 0.0
        )
        np.testing.assert_allclose([x0[0], y0[0]], pca, atol=1e-12)

    def test_displaced_track_has_positive_sign(self):
        phi = 0.2
        d0 = _lcio_d0(np.array([1.0, 0.0]), phi)
        x0, y0, z0 = lt.calc_track_origin_vectorized(
            np.array([phi]), np.array([d0]), np.array([0.0]), 0.0, 0.0, 0.0
        )
        s = lt.calc_pca_arc_length_vectorized(np.array([phi]), np.array([0.0]), x0, y0, z0, 0, 0, 0)
        pca = lt.calc_pca_position_vectorized(s, np.array([phi]), np.array([0.0]), x0, y0, z0)
        # the decay happened along the jet axis (x): positive lifetime sign
        signed = lt.calc_signed_impact_parameters(
            np.array([abs(d0)]), *pca, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
        )
        self.assertGreater(signed[0], 0)
        # ... and negative w.r.t. a jet in the opposite direction
        signed = lt.calc_signed_impact_parameters(
            np.array([abs(d0)]), *pca, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0
        )
        self.assertLess(signed[0], 0)


def _mc_events():
    """Event 0: Z -> b bbar along +-x with B*0 -> B0 -> D- -> pi+ along +x.
    Event 1: Z -> tau+ tau-, tau- -> pi- nu (hadronic), tau+ -> e+ nu nu (leptonic)."""
    mc = {
        "PDG": [[23, 5, -5, 513, 511, -411, 211], [23, 15, -15, -211, 16, -11, 12, -16]],
        "generatorStatus": [[22, 23, 23, 2, 2, 2, 1], [22, 2, 2, 1, 1, 1, 1, 1]],
        "mass": [[91.0, 4.8, 4.8, 5.3, 5.3, 1.9, 0.14], [91.0, 1.777, 1.777, 0.14, 0, 0, 0, 0]],
        "momentum.x": [[0, 45, -45, 20, 20, 10, 5], [0, 0, 0, 0, 0, 0, 0, 0]],
        "momentum.y": [[0, 0, 0, 0.5, 0.5, 0.5, 0.5], [0, 45, -45, 30, 15, -20, -15, -10]],
        "momentum.z": [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]],
        "daughters_begin": [[0, 2, 2, 2, 3, 4, 5], [0, 2, 4, 7, 7, 7, 7, 7]],
        "daughters_end": [[2, 2, 2, 3, 4, 5, 5], [2, 4, 7, 7, 7, 7, 7, 7]],
    }
    events = {f"MCParticles.{name}": ak.Array(values) for name, values in mc.items()}
    events["_MCParticles_daughters.index"] = ak.Array([[1, 2, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]])
    return ak.Array(events)


class TestTruth(unittest.TestCase):
    def test_heaviest_quark(self):
        pdg = ak.Array([[511, -521, 4122, -411, 443, 100443, 2101, 211, 11, 21, 22]])
        self.assertEqual(heaviest_quark(pdg).to_list(), [[5, 5, 4, 4, 4, 4, 0, 2, 0, 0, 0]])

    def test_truth_objects(self):
        truth = MCTruth(_mc_events())
        self.assertEqual(truth.hard_quarks().pdg.to_list(), [[5, -5], []])
        # the B*0 decays to a B0, so only the B0 and the D- decay weakly
        self.assertEqual(truth.heavy_hadrons().pdg.to_list(), [[511, -411], []])
        taus = truth.hadronic_taus()
        self.assertEqual(taus.pdg.to_list(), [[], [15]])
        # visible momentum: tau minus its neutrino
        self.assertAlmostEqual(taus.py[1][0], 30.0)


class TestMatching(unittest.TestCase):
    def test_match_and_uniqueness(self):
        jets = jt._p4(
            ak.Array([[40.0, -40.0, 1.0]]),
            ak.Array([[1.0, 1.0, 30.0]]),
            ak.Array([[0.0, 0.0, 0.0]]),
            ak.Array([[41.0, 41.0, 31.0]]),
        )
        truth = ak.Array([[{"px": 45.0, "py": 0.0, "pz": 0.0, "energy": 45.0, "pdg": 5},
                           {"px": -45.0, "py": 0.0, "pz": 0.0, "energy": 45.0, "pdg": -5}]])
        mask = ak.Array([[True, True, True]])
        pdg, angle, _ = jt.match_jets_to_truth(jets, truth, mask, max_angle=0.4)
        self.assertEqual(pdg.to_list(), [[5, -5, 0]])
        self.assertTrue(angle[0][0] < 0.05 and angle[0][2] == -1)
        # a deselected jet is not matched
        pdg, _, _ = jt.match_jets_to_truth(jets, truth, ak.Array([[False, True, True]]), 0.4)
        self.assertEqual(pdg.to_list(), [[0, -5, 0]])


@unittest.skipUnless(len(ZQQ_FILES) > 0, "CLD Z->qq files not available")
class TestRealFile(unittest.TestCase):
    """End-to-end on one CLD Z->qq file."""

    @classmethod
    def setUpClass(cls):
        from fm_data.common.ntupelizing import check_event_link, process_file

        cls.cfg = OmegaConf.load(CONFIG)
        cls.events, cls.jets = process_file(ZQQ_FILES[0], cls.cfg, "Zqq")
        cls.check_event_link = staticmethod(check_event_link)

    def test_event_table(self):
        events = self.events
        self.assertEqual(len(np.unique(ak.to_numpy(events.event_id))), len(events))
        self.assertTrue(ak.all(events.part_energy[:, :-1] >= events.part_energy[:, 1:]))
        self.assertTrue(np.all(ak.to_numpy(events.event_sqrt_s) == np.float32(91.0)))
        self.assertTrue(np.all(ak.to_numpy(events.label_Zqq) == 1))
        # neutral PFOs have no track, charged ones have one
        has_track = ak.flatten(events.part_has_track) == 1
        charged = ak.flatten(events.part_charge) != 0
        self.assertTrue(ak.all(has_track == charged))
        self.assertTrue(ak.all(ak.flatten(events.part_dxy)[~has_track] == 0))

    def test_jet_tables(self):
        for name, table in self.jets.items():
            with self.subTest(name):
                self.check_event_link(self.events, table)
                labels = np.stack([ak.to_numpy(table[f"label_{c}"]) for c in jt.LABEL_CLASSES])
                self.assertTrue(np.all(labels.sum(axis=0) == 1))
                self.assertEqual(int(labels[-1].sum()), 0)  # no taus in Z->qq
                # event variables are repeated per jet
                self.assertIn("event_sqrt_s", table.fields)
                # b-jet tracks: lifetime sign mostly positive at large significance
                b = table[table.label_b == 1]
                sig = ak.to_numpy(ak.flatten(b.part_d3_signed_sig[b.part_has_track == 1]))
                self.assertGreater(np.mean(sig > 3), 3 * np.mean(sig < -3))


def _smallest_aleph_file():
    files = glob.glob(OmegaConf.load(ALEPH_CONFIG).samples.ALEPH1990.input_dir + "/*_AL.root")
    return min(files, key=lambda path: __import__("os").path.getsize(path)) if files else None


ALEPH_FILE = _smallest_aleph_file()


@unittest.skipUnless(ALEPH_FILE is not None, "ALEPH l2e-prod-v1 files not available")
class TestAlephFile(unittest.TestCase):
    """End-to-end on one ALEPH 1990 file (l2e-prod-v1)."""

    @classmethod
    def setUpClass(cls):
        from fm_data.common.ntupelizing import check_event_link, process_file

        cls.cfg = OmegaConf.load(ALEPH_CONFIG)
        cls.events, cls.jets = process_file(ALEPH_FILE, cls.cfg, "ALEPH1990")
        cls.check_event_link = staticmethod(check_event_link)

    def test_event_table(self):
        events = self.events
        self.assertEqual(len(np.unique(ak.to_numpy(events.event_id))), len(events))
        self.assertTrue(np.all(ak.to_numpy(events.event_sqrt_s) == np.float32(91.2)))
        self.assertTrue(np.all(ak.to_numpy(events.label_ALEPH) == 1))
        self.assertGreater(np.mean(ak.to_numpy(events.event_ref_from_beamspot)), 0.9)
        # type 3 (charged V0 daughters) get the pion mass like type 0
        pid_type = ak.flatten(events.part_pid_type)
        mass = ak.flatten(events.part_mass)
        np.testing.assert_allclose(ak.to_numpy(mass[pid_type == 3]), 0.13957039, rtol=1e-6)
        # impact parameters w.r.t. the beam spot: < 1 mm (1990 has no VDET: ~0.5 mm; a cm or
        # d0-sign mistake gives several mm)
        has_track = ak.flatten(events.part_has_track) == 1
        dxy = ak.to_numpy(ak.flatten(events.part_dxy)[has_track])
        self.assertLess(np.median(dxy), 1.0)

    def test_jet_tables(self):
        self.assertEqual(set(self.jets), set(self.cfg.jet_algorithms))
        for name, table in self.jets.items():
            with self.subTest(name):
                self.check_event_link(self.events, table)
                self.assertFalse(any(field in table.fields for field in ("label_b", "jet_truth_pdg")))
                self.assertTrue(ak.all(table.jet_energy > 1.0))
                self.assertTrue(ak.all(table.jet_energy <= 91.2))
                self.assertTrue(ak.all(abs(table.jet_eta) < 3.0))


if __name__ == "__main__":
    unittest.main()
