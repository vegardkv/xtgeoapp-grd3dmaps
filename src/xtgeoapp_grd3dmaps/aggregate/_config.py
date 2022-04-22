"""
Module for parsing YAML config file and processing config file contents

TODO: May want to solve this differently and more consistently with the rest of the repo,
 but for now, this is a convenient way of documenting the setup
"""
import argparse
import datetime
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import xtgeo
import yaml
from xtgeoapp_grd3dmaps.aggregate._grid_aggregation import AggregationMethod


@dataclass
class Property:
    source: str
    name: Optional[str] = None
    lower_threshold: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.lower_threshold, str):
            # Can be caused by invalid parsing by yaml package
            self.lower_threshold = float(self.lower_threshold)


@dataclass
class Input:
    grid: str
    properties: List[Property]

    def __post_init__(self):
        if (
            len(self.properties) > 0
            and isinstance(self.properties[0], dict)
        ):
            self.properties = [Property(**p) for p in self.properties]


@dataclass
class Filter:
    source: str
    name: Optional[str] = None


@dataclass
class ComputeSettings:
    aggregation: AggregationMethod = AggregationMethod.max

    def __post_init__(self):
        if not isinstance(self.aggregation, AggregationMethod):
            self.aggregation = AggregationMethod(self.aggregation)


@dataclass
class MapSettings:
    xori: Optional[float] = None
    xinc: Optional[float] = None
    yori: Optional[float] = None
    yinc: Optional[float] = None
    ncol: Optional[int] = None
    nrow: Optional[int] = None
    templatefile: Optional[str] = None
    # Specific to grid aggregation:
    pixel_cell_ratio: float = 2.0


@dataclass
class Output:
    mapfolder: str
    plotfolder: Optional[str] = None


@dataclass
class Root:
    input: Input
    filters: List[Filter]
    computesettings: ComputeSettings
    mapsettings: MapSettings
    output: Output

    @staticmethod
    def from_yaml(yaml_file: str) -> 'Root':
        config = yaml.safe_load(open(yaml_file))
        if "eclroot" in config["input"]:
            raise ValueError(
                "eclroot is not supported by this operation (yet)"
            )
        return Root(
            input=Input(**config["input"]),
            filters=[Filter(**f) for f in config.get("filters", [])],
            computesettings=ComputeSettings(**config.get("computesettings", {})),
            mapsettings=MapSettings(**config.get("mapsettings", {})),
            output=Output(**config["output"]),
        )


# TODO: move functions below to a separate module?


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
