import argparse
import datetime
import pathlib
from typing import List, Optional, Tuple

import numpy as np
import xtgeo

from xtgeoapp_grd3dmaps.common.config import Property, Filter


def extract_properties(property_spec: List[Property], grid: Optional[xtgeo.Grid]) -> List[xtgeo.GridProperty]:
    properties = []
    for spec in property_spec:
        try:
            names = "all" if spec.name is None else [spec.name]
            props = xtgeo.gridproperties_from_file(
                spec.source, names=names, grid=grid, dates="all",
            ).props
        except (RuntimeError, ValueError):
            props = [xtgeo.gridproperty_from_file(spec.source, name=spec.name)]
        if spec.lower_threshold is not None:
            for p in props:
                p.values.mask[p.values < spec.lower_threshold] = True
        # Temporary workaround. TODO: remove
        for p in props:
            if p.date is None and "--" in spec.source:
                d = pathlib.Path(spec.source.split("--")[-1]).stem
                try:
                    # Make sure time stamp is on a valid format
                    datetime.datetime.strptime(d, "%Y%m%d")
                except ValueError:
                    continue
                p.date = d
                p.name += f"--{d}"
        # ---
        properties += props
    return properties


def extract_filters(filter_spec: List[Filter], actnum: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    filters = []
    for filter_ in filter_spec:
        prop = xtgeo.gridproperty_from_file(filter_.source)
        assert prop.isdiscrete
        for f_code, f_name in prop.codes.items():
            if f_name == "":
                continue
            filters.append(
                (f_name, prop.values1d[actnum] == f_code)
            )
    return filters


def process_args(arguments):
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument("--config", help="Path to a YAML config file")
    parser.add_argument("--mapfolder", help="Path to output map folder (overrides yaml file)")
    parser.add_argument("--plotfolder", help="Path to output plot folder (overrides yaml file)")
    return parser.parse_args(arguments)
