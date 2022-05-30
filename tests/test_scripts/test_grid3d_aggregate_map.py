import pytest
import xtgeo

from xtgeoapp_grd3dmaps.aggregate import grid3d_aggregate_map


def test_aggregated_map1(datatree):
    result = datatree / "aggregate1_folder"
    result.mkdir(parents=True)
    grid3d_aggregate_map.main(
        [
            "--config",
            "tests/yaml/aggregate1.yml",
            "--mapfolder",
            str(result),
        ]
    )
    swat = xtgeo.surface_from_file(result / "all--max_SWAT--20030101.gri")
    assert swat.values.min() == pytest.approx(0.1554, abs=0.001)


def test_aggregated_map2(datatree):
    result = datatree / "aggregate2_folder"
    result.mkdir(parents=True)
    grid3d_aggregate_map.main(
        [
            "--config",
            "tests/yaml/aggregate2.yml",
            "--mapfolder",
            str(result),
        ]
    )
    swat = xtgeo.surface_from_file(result / "all--min_SWAT--20030101.gri")
    assert swat.values.mean() == pytest.approx(0.7792, abs=0.001)


def test_aggregated_map3(datatree):
    result = datatree / "aggregate3_folder"
    result.mkdir(parents=True)
    grid3d_aggregate_map.main(
        [
            "--config",
            "tests/yaml/aggregate3.yml",
            "--mapfolder",
            str(result),
        ]
    )
    poro = xtgeo.surface_from_file(result / "all--mean_PORO.gri")
    assert poro.values.mean() == pytest.approx(0.1676, abs=0.001)


def test_aggregated_map4(datatree):
    result = datatree / "aggregate4_folder"
    result.mkdir(parents=True)
    grid3d_aggregate_map.main(
        [
            "--config",
            "tests/yaml/aggregate4.yml",
            "--mapfolder",
            str(result)
        ]
    )
    z1_poro = xtgeo.surface_from_file(result / "zone1--max_SWAT--20030101.gri")
    assert z1_poro.values.max() == pytest.approx(1.0, abs=0.001)
    assert (result / "all--max_SWAT--20030101.gri").is_file()
    assert (result / "zone2--max_SWAT--20030101.gri").is_file()
    assert (result / "zone3--max_SWAT--20030101.gri").is_file()
