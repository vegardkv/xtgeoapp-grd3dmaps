from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import yaml


class AggregationMethod(Enum):
    max = "max"
    min = "min"
    mean = "mean"
    sum = "sum"


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
