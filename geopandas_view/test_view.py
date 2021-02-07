import geopandas as gpd
from geopandas_view import view

nybb = gpd.read_file(gpd.datasets.get_path("nybb"))
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
cities = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


def test_simple():
    m = view(nybb)
    m = view(world)
    m = view(cities)

def test_choropleth():
    m = view(world, column='pop_est')