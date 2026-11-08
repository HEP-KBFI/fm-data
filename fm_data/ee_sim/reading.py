"""Read the branches needed for the event and jet tables from EDM4hep ROOT files.

NOTE: uproot < 5.6 (e.g. in gabbro-latest.sif) can't open the CLD files
(`OverflowError: Python integer ... out of bounds for uint8`), see run.sh.
"""

from pathlib import Path

import awkward as ak
import uproot


def branches_to_read(collections):
    """All branches read from the events tree.

    Parameters
    ----------
    collections : dict
        Collection names with the keys "reco_particles", "tracks", "vertices" and
        "mc_particles" (see `collections` in config/ntupelize.yaml).

    Returns
    -------
    list of str
    """
    pfo = collections["reco_particles"]
    tracks = collections["tracks"]
    vertices = collections["vertices"]
    mc = collections["mc_particles"]
    track_state = f"_{tracks}_trackStates"
    return (
        ["EventHeader.eventNumber"]
        + [
            f"{pfo}.{name}"
            for name in [
                "PDG",
                "energy",
                "momentum.x",
                "momentum.y",
                "momentum.z",
                "mass",
                "charge",
                "tracks_begin",
                "tracks_end",
            ]
        ]
        + [f"_{pfo}_tracks.index"]
        + [
            f"{track_state}.{name}"
            for name in [
                "D0",
                "phi",
                "omega",
                "Z0",
                "tanLambda",
                "referencePoint.x",
                "referencePoint.y",
                "referencePoint.z",
                "covMatrix.values[21]",
            ]
        ]
        + [f"{vertices}.position.{axis}" for axis in "xyz"]
        + [
            f"{mc}.{name}"
            for name in [
                "PDG",
                "generatorStatus",
                "mass",
                "momentum.x",
                "momentum.y",
                "momentum.z",
                "daughters_begin",
                "daughters_end",
            ]
        ]
        + [f"_{mc}_daughters.index"]
    )


def read_events(filepath, collections, tree="events"):
    """Read all events of one EDM4hep ROOT file.

    Parameters
    ----------
    filepath : str
        Path to the ROOT file.
    collections : dict
        Collection names, see `branches_to_read`.
    tree : str, optional
        Name of the events tree (default is "events").

    Returns
    -------
    ak.Array
        One entry per event, with the branch names as fields.
    """
    with uproot.open(filepath) as f:
        return f[tree].arrays(branches_to_read(collections))


def pad_vertices(events, vertex_collection):
    """Give events without a reconstructed primary vertex a vertex at (0, 0, 0).

    The lifetime calculation takes the first vertex of each event, so every event needs one.

    Returns
    -------
    events : ak.Array
        The events with at least one vertex each.
    has_vertex : ak.Array
        Boolean per event, False for the events that got the (0, 0, 0) vertex.
    """
    has_vertex = ak.num(events[f"{vertex_collection}.position.x"]) > 0
    for axis in "xyz":
        field = f"{vertex_collection}.position.{axis}"
        events = ak.with_field(events, ak.fill_none(ak.pad_none(events[field], 1), 0.0), field)
    return events, has_vertex
