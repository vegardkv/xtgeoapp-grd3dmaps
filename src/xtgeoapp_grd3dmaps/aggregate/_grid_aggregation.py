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


def aggregate_maps(
    map_template: Union[xtgeo.RegularSurface, float],
    grid: xtgeo.Grid,
    grid_props: List[xtgeo.GridProperty],
    excludes: List[Optional[np.ndarray]],
    method: AggregationMethod,
):
    # TODO: This function needs clean-up, but seems to work as intended now.
    # TODO: write proper docstring. May want to remove type hints? (depends on repo)
    # TODO: nans are filtered, but may want to remove according to GridProperty.undef as
    #  well, and possibly also inf?
    # Determine inactive cells
    active = grid.actnum_array.flatten().astype(bool)
    props = [p.values1d[active] for p in grid_props]
    all_nan = np.all([np.isnan(p) for p in props], axis=0)
    active[active] = ~all_nan
    props = [p[~all_nan] for p in props]
    excludes = [None if ex is None else ex[~all_nan] for ex in excludes]
    # Find cell boxes and pixel nodes
    boxes = _cell_boxes(grid, active)
    if isinstance(map_template, xtgeo.RegularSurface):
        x_nodes = map_template.xori + map_template.xinc * np.arange(0, map_template.ncol)
        y_nodes = map_template.yori + map_template.yinc * np.arange(0, map_template.nrow)
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
    for excl in tqdm.tqdm(excludes, desc="Iterating exclude filters"):
        # TODO: Layer information is completely ignored
        rows0, cols0 = connections
        if excl is not None:
            to_remove = ~np.isin(connections[1], np.argwhere(excl).flatten())
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


def _derive_map_nodes(boxes, pixel_to_cell_size_ratio):
    box = np.min(boxes[0]), np.min(boxes[1]), np.max(boxes[2]), np.max(boxes[3])
    res = np.mean([np.mean(boxes[2] - boxes[0]), np.mean(boxes[3] - boxes[1])])
    res /= pixel_to_cell_size_ratio
    x_nodes = np.arange(box[0], box[2], res)
    y_nodes = np.arange(box[1], box[3], res)
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
        (x_mesh.flatten()[:, np.newaxis] > boxes[0][np.newaxis, :]) &
        (y_mesh.flatten()[:, np.newaxis] > boxes[1][np.newaxis, :]) &
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
    if np.isnan(data).any():
        invalid = np.isnan(data)
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
    elif method == AggregationMethod.mean:
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
        res = sarr.max(axis=1)
    elif method == AggregationMethod.min:
        res = sarr.min(axis=1)
    elif method == AggregationMethod.mean:
        div = np.where(count > 0, count, 1)  # Avoid division by zero
        res = sarr.sum(axis=1) / div
    else:
        raise NotImplementedError
    count = np.array(count).flatten()
    res = res.toarray().flatten() + shift
    res[count == 0] = np.nan
    res = res.reshape(nx, ny)
    return res
