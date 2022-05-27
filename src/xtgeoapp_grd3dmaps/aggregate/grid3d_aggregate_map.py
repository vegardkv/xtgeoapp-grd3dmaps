import pathlib
import sys
from typing import Union
import xtgeo
import numpy as np
from xtgeo.common import XTGeoDialog

import xtgeoapp_grd3dmaps.common.config
from xtgeoapp_grd3dmaps.common.parser import (
    extract_properties,
    extract_filters,
    parse_arguments,
)
from . import _grid_aggregation


_XTG = XTGeoDialog()


def write_map(xn, yn, map_, filename):
    dx = xn[1] - xn[0]
    dy = yn[1] - yn[0]
    masked_map = np.ma.array(map_)
    masked_map.mask = np.isnan(map_)
    surface = xtgeo.RegularSurface(
        ncol=xn.size, nrow=yn.size, xinc=dx, yinc=dy, xori=xn[0], yori=yn[0], values=masked_map
    )
    surface.to_file(filename)


def write_plot(xn, yn, map_, filename):
    import plotly.express as px
    px.imshow(
        map_.T, x=xn, y=yn, origin="lower"
    ).write_html(filename, include_plotlyjs="cdn")


def create_map_template(map_settings: xtgeoapp_grd3dmaps.common.config.MapSettings) -> Union[xtgeo.RegularSurface, float]:
    # TODO: possible duplicate of existing functionality
    if map_settings.templatefile is not None:
        surf = xtgeo.surface_from_file(map_settings.templatefile)
        if surf.rotation != 0.0:
            raise NotImplementedError("Rotated surfaces are not handled correctly yet")
        return surf
    elif map_settings.xori is not None:
        return xtgeo.RegularSurface(
            ncol=map_settings.ncol,
            nrow=map_settings.nrow,
            xinc=map_settings.xinc,
            yinc=map_settings.yinc,
            xori=map_settings.xori,
            yori=map_settings.yori,
        )
    else:
        return map_settings.pixel_to_cell_ratio


def generate_maps(
    grid_name,
    property_spec,
    filter_spec,
    agg_method,
    output_directory,
    map_settings,
    plot_directory
):
    _XTG.say("Reading Grid")
    grid = xtgeo.grid_from_file(grid_name)
    _XTG.say("Reading properties")
    properties = extract_properties(property_spec, grid)
    _XTG.say("Reading Zones")
    _filters = [("all", None)]
    if filter_spec is not None:
        _filters += extract_filters(filter_spec, grid.actnum_indices)
    _XTG.say("Setting up map template")
    map_template = create_map_template(map_settings)
    _XTG.say("Generating Property Maps")
    xn, yn, p_maps = _grid_aggregation.aggregate_maps(
        map_template,
        grid,
        properties,
        [f[1] for f in _filters],
        agg_method,
    )
    assert len(_filters) == len(p_maps)
    for filter_, f_maps in zip(_filters, p_maps):
        f_name = filter_[0]
        # Max saturation maps
        assert len(properties) == len(f_maps)
        for prop, map_ in zip(properties, f_maps):
            # TODO: verify namestyle
            fn = f"{f_name}--{agg_method.value}_{prop.name.replace('_', '--')}"
            fn += ".gri"
            write_map(xn, yn, map_, pathlib.Path(output_directory) / fn)
            if plot_directory:
                pn = pathlib.Path(plot_directory) / fn
                pn = pn.with_suffix(".html")
                write_plot(xn, yn, map_, pn)


def generate_from_config(config: xtgeoapp_grd3dmaps.common.config.RootConfig):
    generate_maps(
        config.input.grid,
        config.input.properties,
        config.filters,
        config.computesettings.aggregation,
        config.output.mapfolder,
        config.mapsettings,
        config.output.plotfolder,
    )


def main(arguments):
    # TODO: try to use common for this to the extent possible
    args = parse_arguments(arguments)
    config = xtgeoapp_grd3dmaps.common.config.parse_yaml(args.config)
    if args.mapfolder is not None:
        config.output.mapfolder = args.mapfolder
    if args.plotfolder is not None:
        config.output.plotfolder = args.plotfolder
    generate_from_config(config)


if __name__ == '__main__':
    main(sys.argv[1:])
