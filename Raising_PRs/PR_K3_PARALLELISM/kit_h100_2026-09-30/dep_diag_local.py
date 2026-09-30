"""Local diagnostic recipe (never committed) for the DEP-off step time: the widened text model with no vision tower."""

import dep_ratio_local as base


def w_notower_off():
    original = base._tower
    base._tower = lambda dim, s: None
    try:
        return base.w_dep_off()
    finally:
        base._tower = original
