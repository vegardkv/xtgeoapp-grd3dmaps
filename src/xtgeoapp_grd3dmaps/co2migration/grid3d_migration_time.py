"""
TODO: Not obvious that this should be part of xtgeoapp?
"""
import os
import sys
import glob
import tempfile
from typing import Optional
import xtgeo

from xtgeoapp_grd3dmaps.aggregate import _config, grid3d_aggregate_map
from xtgeoapp_grd3dmaps.co2migration import _migration_time


def calculate_migration_time_property(
    properties_files: str,
    property_name: Optional[str],
    lower_threshold: float,
    grid_file: Optional[str],
):
    prop_spec = [
        _config.Property(source=f, name=property_name)
        for f in glob.glob(properties_files, recursive=True)
    ]
    grid = None if grid_file is None else xtgeo.grid_from_file(grid_file)
    properties = _config.extract_properties(prop_spec, grid)
    t_prop = _migration_time.generate_migration_time_property(properties, lower_threshold)
    return t_prop


def main(arguments):
    parsed_args = _config.process_args(arguments)
    config = _config.Root.from_yaml(parsed_args.config)
    if len(config.input.properties) > 1:
        raise ValueError("Only a single property is supported (?)")
    p_spec = config.input.properties.pop()
    t_prop = calculate_migration_time_property(
        p_spec.source,
        p_spec.name,
        p_spec.lower_threshold,
        config.input.grid,
    )
    # Dump t_prop to temporary file and execute aggregation
    config.computesettings.aggregation = _config.AggregationMethod.min
    temp_file, temp_path = tempfile.mkstemp()
    os.close(temp_file)
    config.input.properties.append(_config.Property(temp_path, None, None))
    t_prop.to_file(temp_path)
    grid3d_aggregate_map.generate_from_config(config)
    os.unlink(temp_path)


if __name__ == '__main__':
    main(sys.argv[1:])
