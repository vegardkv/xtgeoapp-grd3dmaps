import pathlib
import sys
import xtgeo
import numpy as np
from xtgeo.common import XTGeoDialog

from xtgeoapp_grd3dmaps.common import config
from xtgeoapp_grd3dmaps.common.parser import (
    extract_properties,
    extract_filters,
    process_arguments,
    create_map_template,
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


def generate_from_config(config_: config.RootConfig):
    generate_maps(
        config_.input.grid,
        config_.input.properties,
        config_.filters,
        config_.computesettings.aggregation,
        config_.output.mapfolder,
        config_.mapsettings,
        config_.output.plotfolder,
    )


def main(arguments):
    generate_from_config(process_arguments(arguments))


if __name__ == '__main__':
    main(sys.argv[1:])
