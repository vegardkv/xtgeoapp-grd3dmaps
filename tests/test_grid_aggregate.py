"""
TODO: tests are currently not running, but that is expected
"""
import numpy as np
import numpy.testing as npt
import xtgeo
from xtgeoapp_grd3dmaps.aggregate import aggregate_maps, AggregationMethod


def example_grid():
    xx, yy = np.meshgrid(
        np.linspace(-100, 100, 21),
        np.linspace(-60, 60, 13),
        indexing='ij',
    )
    z = np.array([100, 107, 114, 128])
    coordsv = np.dstack([
        xx, yy, np.ones_like(xx) * z[0],
        xx, yy, np.ones_like(xx) * z[-1],
    ])
    zcornsv = np.ones_like(xx)[:, :, np.newaxis] * z
    zcornsv = zcornsv[:, :, :, np.newaxis] * np.ones(4)
    zcornsv = zcornsv.astype(np.float32)
    actnum = np.ones(np.array(zcornsv.shape[:-1]) - 1, dtype=np.int32)
    grid = xtgeo.Grid(coordsv, zcornsv, actnum)
    grid.gridprops.append_props([example_property(grid)])
    return grid


def example_property(grid: xtgeo.Grid):
    values = np.full(grid.dimensions, fill_value=np.nan)
    xyz = grid.get_xyz()
    d_xy = np.sqrt(xyz[0].values ** 2 + xyz[1].values ** 2)[:, :, 0]
    values[:, :, 1] = np.exp(-(d_xy / 100) ** 2)
    values[:, :, 0] = np.exp(-(d_xy / 33) ** 2)
    return xtgeo.GridProperty(grid, values=np.ma.masked_where(np.isnan(values), values))


# TODO: move this test module to a separate folder
# TODO: replace with fixture-like stuff?
_EXAMPLE_GRID: xtgeo.Grid = example_grid()
_EXAMPLE_PROP: xtgeo.GridProperty = _EXAMPLE_GRID.props[0]
_DEFAULT_ARGS = dict(
    map_template=1.0,
    grid=_EXAMPLE_GRID,
    grid_props=[_EXAMPLE_PROP],
    inclusion_filters=[None],
    method=AggregationMethod.max,
)


def test_default_testing_args():
    xn, yn, maps = aggregate_maps(**_DEFAULT_ARGS)
    map_ = maps[0][0]
    assert xn.size == 20
    assert yn.size == 12
    assert map_.shape == (20, 12)
    npt.assert_allclose(map_, _EXAMPLE_PROP.values[:, :, 1], atol=1e-12, rtol=0)


def test_surface_template():
    surf = xtgeo.RegularSurface(
        ncol=20,
        nrow=12,
        xinc=10,
        yinc=10,
        xori=-95,
        yori=-55,
    )
    kwargs = {**_DEFAULT_ARGS, 'map_template': surf}
    xn, yn, maps = aggregate_maps(**kwargs)
    map_ = maps[0][0]
    assert xn.size == surf.ncol
    assert yn.size == surf.nrow
    npt.assert_allclose(map_, _EXAMPLE_PROP.values[:, :, 1], atol=1e-12, rtol=0)


def test_with_exclusions():
    excludes = [np.ones(_EXAMPLE_GRID.dimensions, dtype=bool) for _ in range(5)]
    excludes[0][:, :, 0] = 0
    excludes[1][:, :, 1] = 0
    excludes[2][:, :, 2] = 0
    excludes[3][:, :6, :] = 0
    excludes[4][:10, :6, :] = 0
    excludes[4] = ~excludes[4]
    includes = [~ex.flatten() for ex in excludes]
    kwargs = {**_DEFAULT_ARGS, 'inclusion_filters': includes}
    _, _, maps = aggregate_maps(**kwargs)
    tols = dict(atol=1e-12, rtol=0)
    npt.assert_allclose(maps[0][0], _EXAMPLE_PROP.values[:, :, 0], **tols)
    npt.assert_allclose(maps[1][0], _EXAMPLE_PROP.values[:, :, 1], **tols)
    assert np.all(np.isnan(maps[2][0]))
    npt.assert_allclose(maps[3][0][:, :6], _EXAMPLE_PROP.values[:, :6, 1], **tols)
    assert np.all(np.isnan(maps[3][0][excludes[3][:, :, 0]]))
    npt.assert_allclose(
        maps[4][0][~excludes[4][:, :, 0]],
        _EXAMPLE_PROP.values[:, :, 1][~excludes[4][:, :, 0]],
        **tols
    )
    assert np.all(np.isnan(maps[4][0][excludes[4][:, :, 0]]))


def test_mean_method():
    kwargs = {**_DEFAULT_ARGS, 'method': AggregationMethod.mean}
    _, _, maps = aggregate_maps(**kwargs)
    map_ = maps[0][0]
    assert np.all(map_ <= _EXAMPLE_PROP.values[:, :, 1])
    assert np.all(map_ >= _EXAMPLE_PROP.values[:, :, 0])


def test_min_method():
    kwargs = {**_DEFAULT_ARGS, 'method': AggregationMethod.min}
    _, _, maps = aggregate_maps(**kwargs)
    npt.assert_allclose(maps[0][0], _EXAMPLE_PROP.values[:, :, 0], atol=1e-12, rtol=0)
