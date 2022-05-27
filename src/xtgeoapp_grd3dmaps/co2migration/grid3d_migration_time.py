import os
import sys
import glob
import tempfile
from typing import Optional
import xtgeo

from xtgeoapp_grd3dmaps.common import config
from xtgeoapp_grd3dmaps.common import parser
from xtgeoapp_grd3dmaps.aggregate import grid3d_aggregate_map
from xtgeoapp_grd3dmaps.co2migration import _migration_time


def calculate_migration_time_property(
    properties_files: str,
    property_name: Optional[str],
    lower_threshold: float,
    grid_file: Optional[str],
):
    prop_spec = [
        config.Property(source=f, name=property_name)
        for f in glob.glob(properties_files, recursive=True)
    ]
    grid = None if grid_file is None else xtgeo.grid_from_file(grid_file)
    properties = parser.extract_properties(prop_spec, grid)
    t_prop = _migration_time.generate_migration_time_property(properties, lower_threshold)
    return t_prop


def main(arguments):
    config_ = parser.process_arguments(arguments)
    if len(config_.input.properties) > 1:
        raise ValueError(
            "Migration time computation is only supported for a single property"
        )
    p_spec = config_.input.properties.pop()
    t_prop = calculate_migration_time_property(
        p_spec.source,
        p_spec.name,
        p_spec.lower_threshold,
        config_.input.grid,
    )
    # Use temporary file for t_prop while executing aggregation
    config_.computesettings.aggregation = config.AggregationMethod.MIN
    temp_file, temp_path = tempfile.mkstemp()
    os.close(temp_file)
    config_.input.properties.append(config.Property(temp_path, None, None))
    t_prop.to_file(temp_path)
    grid3d_aggregate_map.generate_from_config(config_)
    os.unlink(temp_path)


if __name__ == '__main__':
    main(sys.argv[1:])
