from typing import List, Optional, Tuple, Union
import numpy as np
import tqdm
import xtgeo
import scipy.interpolate
import scipy.spatial
import scipy.sparse

from xtgeoapp_grd3dmaps.common.config import AggregationMethod


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
    for incl in inclusion_filters:
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
    to grid index.

    This is equivalent to
        x_mesh, y_mesh = np.meshgrid(x_nodes, y_nodes, indexing="ij")
        within = (
            (x_mesh.flatten()[:, np.newaxis] >= boxes[0][np.newaxis, :]) &
            (y_mesh.flatten()[:, np.newaxis] >= boxes[1][np.newaxis, :]) &
            (x_mesh.flatten()[:, np.newaxis] < boxes[2][np.newaxis, :]) &
            (y_mesh.flatten()[:, np.newaxis] < boxes[3][np.newaxis, :])
        )
        return np.where(within)
    but uses significantly less memory
    """
    # TODO: This method is based on an approximation of cell footprints by boxes. Should
    #  we keep this, or implement alternatives + config options?
    # ---
    i0, i_range = _find_overlapped_nodes(x_nodes, boxes[0], boxes[2])
    j0, j_range = _find_overlapped_nodes(y_nodes, boxes[1], boxes[3])
    invalid = (i_range == 0) | (j_range == 0)
    i0, i_range = i0[~invalid], i_range[~invalid]
    j0, j_range = j0[~invalid], j_range[~invalid]
    # ---
    pixels_ij, box_indices = _extract_all_overlaps(i0, i_range, j0, j_range)
    rows = np.ravel_multi_index(pixels_ij.T, (x_nodes.size, y_nodes.size))
    cols = box_indices
    return rows, cols


def _extract_all_overlaps(i0, i_range, j0, j_range):
    ij_pairs = []
    indices = []
    for ni in range(1, i_range.max() + 1):
        for nj in range(1, j_range.max() + 1):
            ix = (i_range == ni) & (j_range == nj)
            if ix.sum() == 0:
                continue
            __i0 = i0[ix]
            __j0 = j0[ix]
            for qi in range(ni):
                for qj in range(nj):
                    i = __i0 + qi
                    j = __j0 + qj
                    ij_pairs.append(np.column_stack((i, j)))
            n_tot = ni * nj
            indices.append(np.kron(
                np.ones(n_tot, dtype=int), np.argwhere(ix).flatten()
            ))
    return np.vstack(ij_pairs), np.hstack(indices)


def _find_overlapped_nodes(nodes, cell_lower, cell_upper):
    i0 = np.searchsorted(nodes, cell_lower)
    i1 = np.searchsorted(nodes, cell_upper)
    lengths = i1 - i0
    return i0, lengths


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
    if method == AggregationMethod.MAX:
        shift = data.min() - 1
    elif method == AggregationMethod.MIN:
        shift = data.max() + 1
    elif method in (AggregationMethod.MEAN, AggregationMethod.SUM):
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
    if method == AggregationMethod.MAX:
        res = sarr.max(axis=1).toarray()
    elif method == AggregationMethod.MIN:
        res = sarr.min(axis=1).toarray()
    elif method == AggregationMethod.MEAN:
        div = np.where(count > 0, count, 1)  # Avoid division by zero
        res = sarr.sum(axis=1) / div
        res = np.asarray(res)
    elif method == AggregationMethod.SUM:
        res = np.asarray(sarr.sum(axis=1))
    else:
        raise NotImplementedError
    count = np.array(count).flatten()
    res = res.flatten() + shift
    res[count == 0] = np.nan
    res = res.reshape(nx, ny)
    return res
