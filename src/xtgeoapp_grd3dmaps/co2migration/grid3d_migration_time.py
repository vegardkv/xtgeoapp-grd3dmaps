"""
TODO: Not obvious that this should be part of xtgeoapp?
"""
import os
import sys
import glob
import tempfile
from typing import Optional
import xtgeo

import xtgeoapp_grd3dmaps.common.config
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
    parsed_args = parser.process_args(arguments)
    config_ = config.Root.from_yaml(parsed_args.config)
    if len(config_.input.properties) > 1:
        raise ValueError("Only a single property is supported (?)")
    p_spec = config_.input.properties.pop()
    t_prop = calculate_migration_time_property(
        p_spec.source,
        p_spec.name,
        p_spec.lower_threshold,
        config_.input.grid,
    )
    # Dump t_prop to temporary file and execute aggregation
    config_.computesettings.aggregation = xtgeoapp_grd3dmaps.common.config.AggregationMethod.min
    temp_file, temp_path = tempfile.mkstemp()
    os.close(temp_file)
    config_.input.properties.append(config.Property(temp_path, None, None))
    t_prop.to_file(temp_path)
    grid3d_aggregate_map.generate_from_config(config_)
    os.unlink(temp_path)


if __name__ == '__main__':
    main(sys.argv[1:])
