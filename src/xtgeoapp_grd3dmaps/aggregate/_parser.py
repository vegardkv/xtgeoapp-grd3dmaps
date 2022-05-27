import argparse
import datetime
import pathlib
from typing import List, Optional, Tuple, Union

import numpy as np
import xtgeo

from xtgeoapp_grd3dmaps.aggregate import _config
from xtgeoapp_grd3dmaps.aggregate._config import Property, Filter, parse_yaml, RootConfig


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(__file__)
    parser.add_argument("--config", help="Path to a YAML config file")
    parser.add_argument("--mapfolder", help="Path to output map folder (overrides yaml file)")
    parser.add_argument("--plotfolder", help="Path to output plot folder (overrides yaml file)")
    return parser.parse_args(arguments)


def process_arguments(arguments) -> RootConfig:
    parsed_args = parse_arguments(arguments)
    config = parse_yaml(parsed_args.config)
    if parsed_args.mapfolder is not None:
        config.output.mapfolder = parsed_args.mapfolder
    if parsed_args.plotfolder is not None:
        config.output.plotfolder = parsed_args.plotfolder
    return config


def extract_properties(
    property_spec: List[Property], grid: Optional[xtgeo.Grid]
) -> List[xtgeo.GridProperty]:
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


def extract_filters(
    filter_spec: List[Filter], actnum: np.ndarray
) -> List[Tuple[str, np.ndarray]]:
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


def create_map_template(map_settings: _config.MapSettings) -> Union[xtgeo.RegularSurface, float]:
    if map_settings.templatefile is not None:
        surf = xtgeo.surface_from_file(map_settings.templatefile)
        if surf.rotation != 0.0:
            raise NotImplementedError("Rotated surfaces are not handled correctly yet")
        return surf
    elif map_settings.xori is not None:
        surf_kwargs = dict(
            ncol=map_settings.ncol,
            nrow=map_settings.nrow,
            xinc=map_settings.xinc,
            yinc=map_settings.yinc,
            xori=map_settings.xori,
            yori=map_settings.yori,
        )
        if not all((s is not None for s in surf_kwargs.values())):
            missing = [k for k, v in surf_kwargs.items() if v is None]
            raise ValueError(
                f"Failed to create map template due to partial map specification. "
                f"Missing: {', '.join(missing)}"
            )
        return xtgeo.RegularSurface(**surf_kwargs)
    else:
        return map_settings.pixel_to_cell_ratio
