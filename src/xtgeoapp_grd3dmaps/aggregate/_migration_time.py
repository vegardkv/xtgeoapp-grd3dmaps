from typing import List
import numpy as np
import xtgeo
import datetime


def generate_migration_time_property(
    co2_props: List[xtgeo.GridProperty],
    co2_threshold: float,
):
    # Calculate time since simulation start
    times = [
        datetime.datetime.strptime(_prop.date, '%Y%m%d')
        for _prop in co2_props
    ]
    time_since_start = [(t - times[0]).days / 365 for t in times]
    # Duplicate first property to ensure equal actnum
    t_prop = co2_props[0].copy('MigrationTime')
    t_prop.values[~t_prop.values.mask] = np.inf
    for co2, t in zip(co2_props, time_since_start):
        above_threshold = co2.values > co2_threshold
        t_prop.values[above_threshold] = np.minimum(t_prop.values[above_threshold], t)
    # Use nan instead of inf
    t_prop.values[np.isinf(t_prop.values)] = np.nan
    # TODO: more appropriately:
    # t_prop.values[np.isinf(t_prop.values)] = t_prop.undef
    # t_prop.mask_undef()
    return t_prop
