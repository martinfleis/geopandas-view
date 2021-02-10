import geopandas as gpd
import pandas as pd
import numpy as np
import pytest
from geopandas_view import view

from .view import _BRANCA_COLORS

nybb = gpd.read_file(gpd.datasets.get_path("nybb"))
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
cities = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))


def test_simple_pass():
    """Make sure default pass"""
    m = view(nybb)
    m = view(world)
    m = view(cities)
    m = view(world.geometry)


def test_choropleth_pass():
    """Make sure default choropleth pass"""
    m = view(world, column="pop_est")


def test_map_settings_default():
    """Check default map settings"""
    m = view(world)
    assert m.location == [
        pytest.approx(-3.1774349999999956, rel=1e-6),
        pytest.approx(2.842170943040401e-14, rel=1e-6),
    ]
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
    assert m.location == [
        pytest.approx(40.70582377450201, rel=1e-6),
        pytest.approx(-73.9778006856748, rel=1e-6),
    ]
    assert m.options["zoom"] == 10
    assert m.options["zoomControl"] == False
    assert m.height == (200.0, "px")
    assert m.width == (200.0, "px")
    assert "cartodbpositron" in m.to_dict()["children"].keys()

    # custom XYZ tiles
    m = view(
        nybb,
        zoom_control=False,
        width=200,
        height=200,
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google",
    )

    out = m._parent.render()
    out_str = "".join(out.split())
    assert (
        'tileLayer("https://mt1.google.com/vt/lyrs=m\\u0026x={x}\\u0026y={y}\\u0026z={z}",{"attribution":"Google"'
        in out_str
    )


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

    # line GeoSeries
    m4 = view(nybb.boundary, color="red")
    out = m4._parent.render()
    out_str = "".join(out.split())
    assert '"color":"red"' in out_str


def test_choropleth_linear():
    """Check choropleth colors"""
    # default cmap
    m = view(nybb, column="Shape_Leng")
    out = m._parent.render()
    out_str = "".join(out.split())
    assert 'fillColor":"#08519c"' in out_str
    assert 'fillColor":"#3182bd"' in out_str
    assert 'fillColor":"#bdd7e7"' in out_str
    assert 'fillColor":"#eff3ff"' in out_str

    # named cmap
    m = view(nybb, column="Shape_Leng", cmap="PuRd")
    out = m._parent.render()
    out_str = "".join(out.split())
    assert 'fillColor":"#980043"' in out_str
    assert 'fillColor":"#dd1c77"' in out_str
    assert 'fillColor":"#d7b5d8"' in out_str
    assert 'fillColor":"#f1eef6"' in out_str

    # custom number of bins
    m = view(nybb, column="Shape_Leng", k=3)
    out = m._parent.render()
    out_str = "".join(out.split())
    assert 'fillColor":"#3182bd"' in out_str
    assert 'fillColor":"#deebf7"' in out_str
    assert (
        "tickValues([330470.010332,519094.6894756666,707719.3686193333,896344.047763])"
        in out_str
    )


def test_choropleth_mapclassify():
    """Mapclassify bins"""
    # quantiles
    m = view(nybb, column="Shape_Leng", scheme="quantiles")
    out = m._parent.render()
    out_str = "".join(out.split())
    assert (
        "tickValues([330470.010332,353533.27924319997,422355.43368280004,575068.0043608,772133.2280854001,896344.047763])"
        in out_str
    )

    # headtail
    m = view(world, column="pop_est", scheme="headtailbreaks")
    out = m._parent.render()
    out_str = "".join(out.split())
    assert (
        "tickValues([140.0,41712369.84180791,182567501.0,550193675.0,1330619341.0,1379302771.0])"
        in out_str
    )

    # custom k
    m = view(world, column="pop_est", scheme="naturalbreaks", k=3)
    out = m._parent.render()
    out_str = "".join(out.split())
    assert "tickValues([140.0,83301151.0,326625791.0,1379302771.0])" in out_str


def test_categorical():
    """Categorical maps"""
    # auto detection
    m = view(world, column="continent")
    out = m._parent.render()
    out_str = "".join(out.split())
    assert '"color":"darkred"' in out_str
    assert '"color":"orange"' in out_str
    assert '"color":"green"' in out_str
    assert '"color":"beige"' in out_str
    assert '"color":"red"' in out_str
    assert '"color":"lightred"' in out_str
    assert '"color":"blue"' in out_str
    assert '"color":"purple"' in out_str

    # forced categorical
    m = view(nybb, column="BoroCode", categorical=True)
    out = m._parent.render()
    out_str = "".join(out.split())
    assert '"color":"orange"' in out_str
    assert '"color":"green"' in out_str
    assert '"color":"red"' in out_str
    assert '"color":"blue"' in out_str
    assert '"color":"purple"' in out_str

    # pandas.Categorical
    df = world.copy()
    df["categorical"] = pd.Categorical(df["name"])
    m = view(df, column="categorical")
    out = m._parent.render()
    out_str = "".join(out.split())
    for c in _BRANCA_COLORS:
        assert f'"color":"{c}"' in out_str


def test_column_values():
    """
    Check that the dataframe plot method returns same values with an
    input string (column in df), pd.Series, or np.array
    """
    column_array = np.array(world['pop_est'])
    m1 = view(world, column="pop_est") # column name
    m2 = view(world, column=column_array) # np.array
    m3 = view(world, column=world['pop_est']) # pd.Series
    assert m1.location == m2.location == m3.location

    # GeoDataframe and the given list have different number of rows
    with pytest.raises(ValueError, match="different number of rows"):
        view(world,column=np.array([1, 2, 3]))


def test_no_crs():
    """Naive geometry get no tiles"""
    df = world.copy()
    df.crs = None
    m = view(df)
    assert "openstreetmap" not in m.to_dict()["children"].keys()


def test_style_kwds():
    """Style keywords"""
    m = view(world, style_kwds=dict(fillOpacity=0.1, weight=0.5, fillColor="orange"))
    out = m._parent.render()
    out_str = "".join(out.split())
    assert '"fillColor":"orange","fillOpacity":0.1,"weight":0.5' in out_str


def test_tooltip():
    """Test tooltip"""
    # default with no tooltip or popup
    m = view(world)
    assert "GeoJsonTooltip" not in str(m.to_dict())
    assert "GeoJsonPopup" not in str(m.to_dict())

    # True
    m = view(world, tooltip=True, popup=True)
    assert "GeoJsonTooltip" in str(m.to_dict())
    assert "GeoJsonPopup" in str(m.to_dict())
    out = m._parent.render()
    out_str = "".join(out.split())
    assert 'fields=["pop_est","continent","name","iso_a3","gdp_md_est"]' in out_str
    assert 'aliases=["pop_est","continent","name","iso_a3","gdp_md_est"]' in out_str

    # True choropleth
    m = view(world, column="pop_est", tooltip=True, popup=True)
    assert "GeoJsonTooltip" in str(m.to_dict())
    assert "GeoJsonPopup" in str(m.to_dict())
    out = m._parent.render()
    out_str = "".join(out.split())
    assert 'fields=["pop_est","continent","name","iso_a3","gdp_md_est"]' in out_str
    assert 'aliases=["pop_est","continent","name","iso_a3","gdp_md_est"]' in out_str

    # single column
    m = view(world, tooltip="pop_est", popup="iso_a3")
    out = m._parent.render()
    out_str = "".join(out.split())
    assert 'fields=["pop_est"]' in out_str
    assert 'aliases=["pop_est"]' in out_str
    assert 'fields=["iso_a3"]' in out_str
    assert 'aliases=["iso_a3"]' in out_str

    # list
    m = view(world, tooltip=["pop_est", "continent"], popup=["iso_a3", "gdp_md_est"])
    out = m._parent.render()
    out_str = "".join(out.split())
    assert 'fields=["pop_est","continent"]' in out_str
    assert 'aliases=["pop_est","continent"]' in out_str
    assert 'fields=["iso_a3","gdp_md_est"' in out_str
    assert 'aliases=["iso_a3","gdp_md_est"]' in out_str

    # number
    m = view(world, tooltip=2, popup=2)
    out = m._parent.render()
    out_str = "".join(out.split())
    assert 'fields=["pop_est","continent"]' in out_str
    assert 'aliases=["pop_est","continent"]' in out_str

    # keywords tooltip
    m = view(
        world,
        tooltip=True,
        popup=False,
        tooltip_kwds=dict(aliases=[0, 1, 2, 3, 4], sticky=False),
    )
    out = m._parent.render()
    out_str = "".join(out.split())
    assert 'fields=["pop_est","continent","name","iso_a3","gdp_md_est"]' in out_str
    assert "aliases=[0,1,2,3,4]" in out_str
    assert '"sticky":false' in out_str

    # keywords popup
    m = view(
        world, tooltip=False, popup=True, popup_kwds=dict(aliases=[0, 1, 2, 3, 4]),
    )
    out = m._parent.render()
    out_str = "".join(out.split())
    assert 'fields=["pop_est","continent","name","iso_a3","gdp_md_est"]' in out_str
    assert "aliases=[0,1,2,3,4]" in out_str
    assert "<th>${aliases[i]" in out_str

    # no labels
    m = view(
        world,
        tooltip=True,
        popup=True,
        tooltip_kwds=dict(labels=False),
        popup_kwds=dict(labels=False),
    )
    out = m._parent.render()
    out_str = "".join(out.split())
    assert "<th>${aliases[i]" not in out_str

