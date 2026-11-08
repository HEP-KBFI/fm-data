"""EDM4hep files of the CLD simulation (FCC-ee): event table and truth-labelled jet tables."""

import logging

import awkward as ak
import numpy as np

from fm_data.common import jets as jt
from fm_data.common import lifetime as lt
from fm_data.common.events import (
    LIFETIME_FIELDS,
    add_significances,
    build_event_table,
    pid_flags_from_pdg,
)
from fm_data.ee_sim import reading
from fm_data.ee_sim.truth import MCTruth

log = logging.getLogger(__name__)

# max. difference [GeV] between the configured sqrt(s) and the one of the beam particles
SQRT_S_TOLERANCE = 0.05


def cld_particles(events, collections):
    """PFO columns [events, PFOs] of EDM4hep files of the CLD simulation (file order)."""
    pfo = collections["reco_particles"]
    particles = {
        "part_px": events[f"{pfo}.momentum.x"],
        "part_py": events[f"{pfo}.momentum.y"],
        "part_pz": events[f"{pfo}.momentum.z"],
        "part_energy": events[f"{pfo}.energy"],
        "part_mass": events[f"{pfo}.mass"],
        "part_charge": events[f"{pfo}.charge"],
        "part_pdg": events[f"{pfo}.PDG"],
    }
    particles |= pid_flags_from_pdg(particles["part_pdg"])
    info = lt.find_all_track_pcas(
        events,
        reco_particle_collection=pfo,
        track_collection=collections["tracks"],
        vertex_collection=collections["vertices"],
        # keep all PFOs, such that the result is aligned with the PFO collection
        valid_particle_mask=ak.ones_like(events[f"{pfo}.PDG"], dtype=bool),
    )
    particles |= {name: info[field] for name, field in LIFETIME_FIELDS.items()}
    particles["part_has_track"] = ak.values_astype(
        events[f"{pfo}.tracks_end"] > events[f"{pfo}.tracks_begin"], np.int32
    )
    return add_significances(particles)


def process_file(filepath, cfg, sample, file_number):
    """Event table and truth-labelled jet tables {name: table or None} of one CLD file."""
    sample_cfg = cfg.samples[sample]
    collections = dict(cfg.collections)
    events = reading.read_events(filepath, collections)
    events, has_vertex = reading.pad_vertices(events, collections["vertices"])
    vertices = collections["vertices"]
    event_table = build_event_table(
        cld_particles(events, collections),
        vertex={axis: ak.to_numpy(events[f"{vertices}.position.{axis}"][:, 0]) for axis in "xyz"},
        has_vertex=ak.to_numpy(has_vertex),
        event_number=ak.to_numpy(ak.firsts(events["EventHeader.eventNumber"])),
        file_number=file_number,
        events_per_file_max=int(sample_cfg.events_per_file_max),
        sample=sample,
        all_samples=list(cfg.samples),
        sqrt_s=float(sample_cfg.sqrt_s),
    )
    mc_truth = MCTruth(events, collections["mc_particles"])
    beam_sqrt_s = ak.to_numpy(mc_truth.beam_sqrt_s())
    if np.any(np.abs(beam_sqrt_s - float(sample_cfg.sqrt_s)) > SQRT_S_TOLERANCE):
        log.warning(
            f"{filepath}: sqrt(s) of the beam particles ({np.min(beam_sqrt_s):.3f}-"
            f"{np.max(beam_sqrt_s):.3f} GeV) differs from samples.{sample}.sqrt_s = "
            f"{sample_cfg.sqrt_s} GeV."
        )
    if sample_cfg.get("truth_objects") is None:
        return event_table, {}
    rows = ak.to_numpy(event_table["event_entry"])
    truth = mc_truth.truth_objects(sample_cfg.truth_objects)[rows]
    heavy_hadrons = mc_truth.heavy_hadrons()[rows]
    jet_tables = {
        name: jt.build_jet_table(
            event_table, dict(algorithm), dict(cfg.jet_selection), truth, heavy_hadrons
        )
        for name, algorithm in cfg.jet_algorithms.items()
    }
    return event_table, jet_tables
