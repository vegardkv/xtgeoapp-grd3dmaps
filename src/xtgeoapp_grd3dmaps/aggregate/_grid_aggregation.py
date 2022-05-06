from enum import Enum
from typing import List, Optional, Tuple, Union
import numpy as np
import tqdm
import xtgeo
import scipy.interpolate
import scipy.spatial
import scipy.sparse


class AggregationMethod(Enum):
    max = "max"
    min = "min"
    mean = "mean"
    sum = "sum"


def aggregate_maps(
    map_template: Union[xtgeo.RegularSurface, float],
    grid: xtgeo.Grid,
    grid_props: List[xtgeo.GridProperty],
    inclusion_filters: List[Optional[np.ndarray]],
    method: AggregationMethod,
) -> Tuple[np.ndarray, np.ndarray, List[List[np.ndarray]]]:
    """
    Aggregate multiple grid properties, using multiple grid cell filters, to 2D maps.

    Args:
        map_template: Template to use for the generated maps. If a float is provided, it
            will be used as an approximate pixel-to-cell-size ratio to automatically set
            map bounds and resolution from the grid.
        grid: The 3D grid
        grid_props: List of the grid properties to be aggregated
        inclusion_filters: List containing the grid cell filters. A filter is defined by
            either a numpy array or `None`. If a numpy array is used, it must be a boolean
            1D array representing which cells (among the active cells) that are to be
            included. A `1` indicates inclusion. If `None` is provided, all of the grid
            cells are included.
        method: The aggregation method to apply for pixels that overlap more than one grid
            cell in the xy-plane

    Returns:
        Doubly nested list of maps. The first index corresponds to `ìnclusion_filters`,
        and the second index to `grid_props`.
    """
    # TODO: May want to remove type hints? (depends on repo)
    # Determine cells where properties are always masked
    active = grid.actnum_array.flatten().astype(bool)
    props = [p.values1d[active] for p in grid_props]
    props, active, inclusion_filters = _remove_where_all_props_are_masked(
        props, active, inclusion_filters
    )
    # Find cell boxes and pixel nodes
    boxes = _cell_boxes(grid, active)
    if isinstance(map_template, xtgeo.RegularSurface):
        x_nodes = map_template.xori + map_template.xinc * np.arange(0, map_template.ncol, dtype=float)
        y_nodes = map_template.yori + map_template.yinc * np.arange(0, map_template.nrow, dtype=float)
    else:
        x_nodes, y_nodes = _derive_map_nodes(boxes, pixel_to_cell_size_ratio=map_template)
    # Find connections
    connections = _connect_grid_and_map(
        x_nodes,
        y_nodes,
        boxes,
    )
    # Iterate filters
    results = []
    for incl in tqdm.tqdm(inclusion_filters, desc="Iterating inclusion filters"):
        rows0, cols0 = connections
        if incl is not None:
            to_remove = ~np.isin(connections[1], np.argwhere(incl).flatten())
            rows0 = rows0[~to_remove]
            cols0 = cols0[~to_remove]
        results.append([])
        for prop in props:
            results[-1].append(_property_to_map(
                (rows0, cols0),
                prop,
                x_nodes.size,
                y_nodes.size,
                method,
            ))
    return x_nodes, y_nodes, results


def _remove_where_all_props_are_masked(
    props,
    active,
    inclusion_filters,
):
    all_masked = np.all([p.mask for p in props], axis=0)
    active[active] = ~all_masked
    props = [p[~all_masked] for p in props]
    inclusion_filters = [
        None if inc is None else inc[~all_masked]
        for inc in inclusion_filters
    ]
    return props, active, inclusion_filters


def _derive_map_nodes(boxes, pixel_to_cell_size_ratio):
    box = np.min(boxes[0]), np.min(boxes[1]), np.max(boxes[2]), np.max(boxes[3])
    res = np.mean([np.mean(boxes[2] - boxes[0]), np.mean(boxes[3] - boxes[1])])
    res /= pixel_to_cell_size_ratio
    x_nodes = np.arange(box[0] + res / 2, box[2] - res / 2 + 1e-12, res)
    y_nodes = np.arange(box[1] + res / 2, box[3] - res / 2 + 1e-12, res)
    return x_nodes, y_nodes


def _connect_grid_and_map(
    x_nodes,
    y_nodes,
    boxes,
):
    """
    Returns a mapping between the provided grid nodes and map pixels as
    an np.ndarray pair, the first referring to pixel index and the second
    to grid index
    """
    # TODO: This method is based on an approximation of cell footprints by boxes. Should
    #  we keep this, or implement alternatives + config options?
    x_mesh, y_mesh = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    within = (
        (x_mesh.flatten()[:, np.newaxis] >= boxes[0][np.newaxis, :]) &
        (y_mesh.flatten()[:, np.newaxis] >= boxes[1][np.newaxis, :]) &
        (x_mesh.flatten()[:, np.newaxis] < boxes[2][np.newaxis, :]) &
        (y_mesh.flatten()[:, np.newaxis] < boxes[3][np.newaxis, :])
    )
    row_col = np.where(within)
    return row_col


def _cell_boxes(grid: xtgeo.Grid, active_cells):
    corners = grid.get_xyz_corners()
    xyz = [c.values1d[active_cells] for c in corners]
    avg_xyz = [(xyz[i] + xyz[i + 12]) / 2 for i in range(12)]
    x_corners = avg_xyz[::3]
    y_corners = avg_xyz[1::3]
    # z_corners = avg_xyz[2::3]
    return (
        np.minimum.reduce(x_corners),
        np.minimum.reduce(y_corners),
        np.maximum.reduce(x_corners),
        np.maximum.reduce(y_corners),
    )


def _property_to_map(
    connections: Tuple[np.ndarray, np.ndarray],
    prop: np.ndarray,
    nx: int,
    ny: int,
    method: AggregationMethod,
):
    rows, cols = connections
    assert rows.shape == cols.shape
    data = prop[cols]
    if data.mask.any():
        invalid = data.mask
        rows = rows[~invalid]
        cols = cols[~invalid]
        data = data[~invalid]

    if data.size == 0:
        return np.full((nx, ny), fill_value=np.nan)
    shape = (nx * ny, max(cols) + 1)
    # Calculate temporary data shift to avoid unintended deletion of data by tocsc:
    if method == AggregationMethod.max:
        shift = data.min() - 1
    elif method == AggregationMethod.min:
        shift = data.max() + 1
    elif method in (AggregationMethod.mean, AggregationMethod.sum):
        shift = 0.0
    else:
        raise NotImplementedError
    sarr = scipy.sparse.coo_matrix(
        (data - shift, (rows, cols)), shape=shape
    ).tocsc()
    count = scipy.sparse.coo_matrix(
        (np.ones_like(data), (rows, cols)), shape=shape
    ).tocsc().sum(axis=1)
    # Make sure to shift data to avoid
    if method == AggregationMethod.max:
        res = sarr.max(axis=1).toarray()
    elif method == AggregationMethod.min:
        res = sarr.min(axis=1).toarray()
    elif method == AggregationMethod.mean:
        div = np.where(count > 0, count, 1)  # Avoid division by zero
        res = sarr.sum(axis=1) / div
        res = np.asarray(res)
    elif method == AggregationMethod.sum:
        res = np.asarray(sarr.sum(axis=1))
    else:
        raise NotImplementedError
    count = np.array(count).flatten()
    res = res.flatten() + shift
    res[count == 0] = np.nan
    res = res.reshape(nx, ny)
    return res
