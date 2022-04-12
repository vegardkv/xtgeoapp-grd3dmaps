import pathlib
import sys
from typing import Union
import xtgeo
from ._config import (
    extract_properties,
    extract_filters,
    process_args,
)
from . import (
    _grid_aggregation,
    _config,
)


def write_map(xn, yn, map_, filename):
    dx = xn[1] - xn[0]
    dy = yn[1] - yn[0]
    surface = xtgeo.RegularSurface(
        ncol=xn.size, nrow=yn.size, xinc=dx, yinc=dy, xori=xn[0], yori=yn[0], values=map_
    )
    # TODO: should mask map_ instead of using nans where values are undefined?
    surface.to_file(filename)


def write_plot(xn, yn, map_, filename):
    import plotly.express as px
    px.imshow(
        map_.T, x=xn, y=yn, origin="lower"
    ).write_html(filename, include_plotlyjs="cdn")


def create_map_template(map_settings: _config.MapSettings) -> Union[xtgeo.RegularSurface, float]:
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
        return map_settings.pixel_cell_ratio


def generate_maps(
    grid_name,
    property_spec,
    filter_spec,
    agg_method,
    output_directory,
    map_settings,
    plot_directory
):
    print("*** Reading Grid ***")
    grid = xtgeo.grid_from_file(grid_name)
    print("*** Reading properties ***")
    properties = extract_properties(property_spec)
    print("*** Reading Zones ***")
    _filters = [("all", None)]
    if filter_spec is not None:
        _filters += extract_filters(filter_spec, grid.actnum_indices)
    print("*** Setting up map template ***")
    map_template = create_map_template(map_settings)
    print("*** Generating Property Maps ***")
    xn, yn, p_maps = _grid_aggregation.aggregate_maps(
        map_template,
        grid,
        properties,
        [f[1] for f in _filters],
        agg_method,
    )
    assert len(p_maps) == len(_filters)
    for filter_, f_maps in zip(_filters, p_maps):
        f_name = filter_[0]
        assert len(f_maps) == len(properties)
        # Max saturation maps
        for prop, map_ in zip(properties, f_maps):
            fn = f"{f_name}--{agg_method.value}_{prop.name}"
            if prop.date:
                fn += f"--{prop.date}"
            fn += ".gri"
            write_map(xn, yn, map_, pathlib.Path(output_directory) / fn)
            if plot_directory:
                pn = pathlib.Path(plot_directory) / fn
                pn = pn.with_suffix(".html")
                write_plot(xn, yn, map_, pn)


def main(arguments):
    # TODO: try to use configparser for this to the extent possible
    args = process_args(arguments)
    config = _config.Root.from_yaml(args.config)
    generate_maps(
        config.input.grid,
        config.input.properties,
        config.filters,
        config.computesettings.aggregation,
        config.output.mapfolder,
        config.mapsettings,
        config.output.plotfolder,
    )


if __name__ == '__main__':
    main(sys.argv[1:])
