"""
TODO: Not obvious that this should be part of xtgeoapp?
"""
import pathlib
import argparse
import sys
import glob
from typing import Optional
from . import _config
from . import _migration_time


def calculate_migration_time_property(
    properties_files: str,
    property_name: Optional[str],
    lower_threshold: float,
):
    prop_spec = [
        _config.Property(source=f)
        for f in glob.glob(properties_files, recursive=True)
    ]
    properties = _config.extract_properties(prop_spec)
    if property_name is not None:
        properties = [p for p in properties if p.name == property_name]
    t_prop = _migration_time.generate_migration_time_property(properties, lower_threshold)
    return t_prop


def make_parser():
    desc = (
        "Converts a set of properties to a new migration time property. The properties "
        "should represent fluid saturations to be meaningful."
    )
    parser = argparse.ArgumentParser(pathlib.Path(__file__).stem, description=desc)
    parser.add_argument(
        "properties",
        help="File(s) containing the saturation properties. Can be a glob expression.",
    )
    parser.add_argument(
        "output_file",
        help="Output file for the generated grid property"
    )
    parser.add_argument(
        "--name",
        help=(
            "Name of the saturation property. If provided, properties not matching this "
            "name will be ignored."
        ),
    )
    parser.add_argument(
        "--threshold",
        help="Lower threshold for defining when migration has occured",
        default=1e-6,
        type=float,
    )
    return parser


def main(arguments):
    parsed_args = make_parser().parse_args(arguments)
    t_prop = calculate_migration_time_property(
        parsed_args.properties,
        parsed_args.name,
        parsed_args.threshold,
    )
    t_prop.to_file(parsed_args.output_file)


if __name__ == '__main__':
    main(sys.argv[1:])
