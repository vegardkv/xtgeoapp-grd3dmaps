"""
Configuration for the `aggregate` module. Starting from `RootConfig`, it is possible to
deduce mandatory and optional parameters, as well as default values for whatever is not
explicitly provided.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class AggregationMethod(Enum):
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    SUM = "sum"


@dataclass
class Property:
    source: str
    name: Optional[str] = None
    lower_threshold: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.lower_threshold, str):
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
    aggregation: AggregationMethod = AggregationMethod.MAX

    def __post_init__(self):
        if isinstance(self.aggregation, str):
            self.aggregation = AggregationMethod(self.aggregation.lower())


@dataclass
class MapSettings:
    xori: Optional[float] = None
    xinc: Optional[float] = None
    yori: Optional[float] = None
    yinc: Optional[float] = None
    ncol: Optional[int] = None
    nrow: Optional[int] = None
    templatefile: Optional[str] = None
    pixel_to_cell_ratio: float = 2.0


@dataclass
class Output:
    mapfolder: str
    plotfolder: Optional[str] = None
    use_plotly: bool = False

    def __post_init__(self):
        if self.mapfolder == "fmu-dataio":
            raise NotImplementedError(
                "Export via fmu-dataio is not implemented for this action"
            )


@dataclass
class RootConfig:
    input: Input
    output: Output
    filters: List[Filter] = field(default_factory=lambda: [])
    computesettings: ComputeSettings = ComputeSettings()
    mapsettings: MapSettings = MapSettings()
