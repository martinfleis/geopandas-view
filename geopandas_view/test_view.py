import geopandas as gpd
from geopandas_view import view

nybb = gpd.read_file(gpd.datasets.get_path("nybb"))
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
cities = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


def test_simple_pass():
    """Make sure default pass"""
    m = view(nybb)
    m = view(world)
    m = view(cities)


def test_choropleth_pass():
    """Make sure default choropleth pass"""
    m = view(world, column="pop_est")


def test_map_settings_default():
    """Check default map settings"""
    m = view(world)
    assert m.location == [-3.1774349999999956, 2.842170943040401e-14]
    assert m.options["zoom"] == 10
    assert m.options["zoomControl"] == True
    assert m.position == "relative"
    assert m.height == (100.0, "%")
    assert m.width == (100.0, "%")
    assert m.left == (0, "%")
    assert m.top == (0, "%")
    assert m.global_switches.no_touch is False
    assert m.global_switches.disable_3d is False
    assert "openstreetmap" in m.to_dict()["children"].keys()


def test_map_settings_custom():
    """Check custom map settins"""
    m = view(nybb, zoom_control=False, width=200, height=200, tiles="CartoDB positron")
    assert m.location == [40.70582377450201, -73.9778006856748]
    assert m.options["zoom"] == 10
    assert m.options["zoomControl"] == False
    assert m.height == (200.0, "px")
    assert m.width == (200.0, "px")
    assert "cartodbpositron" in m.to_dict()["children"].keys()


def test_simple_color():
    """Check color settings"""
    # single named color
    m = view(nybb, color="red")
    out = m._parent.render()
    out_str = "".join(out.split())
    assert '"color":"red"' in out_str

    # list of colors
    colors = ["#333333", "#367324", "#95824f", "#fcaa00", "#ffcc33"]
    m2 = view(nybb, color=colors)
    out = m2._parent.render()
    out_str = "".join(out.split())
    for c in colors:
        assert f'"color":"{c}"' in out_str

    # column of colors
    df = nybb.copy()
    df["colors"] = colors
    m3 = view(df, color="colors")
    out = m3._parent.render()
    out_str = "".join(out.split())
    for c in colors:
        assert f'"color":"{c}"' in out_str
