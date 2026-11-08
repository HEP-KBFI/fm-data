"""The one helper of ml-tau-data's ntupelizer/tools/matching.py (commit 83f6b1af83) that the copied
lifetime.py needs (for its per-jet functions)."""

import awkward as ak


def get_jet_constituent_property(
    property_: str, constituent_idx: int, num_ptcls_per_jet: int
):
    reco_property_flat = property_[ak.flatten(constituent_idx, axis=-1)]
    return ak.from_iter(
        [
            ak.unflatten(reco_property_flat[i], num_ptcls_per_jet[i], axis=-1)
            for i in range(len(num_ptcls_per_jet))
        ]
    )
