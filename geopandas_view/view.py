from statistics import mean

import folium
import pandas as pd
import geopandas as gpd
import mapclassify
import numpy as np


def view(
    df,
    column=None,
    cmap=None,
    color=None,
    m=None,  # folium.Map is usually referred to as `m` instead of `ax` in the case of matplotlib
    tiles="OpenStreetMap",  # Map tileset to use
    tooltip=10,  # Specify fields for tooltip, defaults to 10 first columns
    popup=None,  # Show a different popup for each feature by passing a GeoJsonPopup object.
    categorical=False,
    scheme=None,
    k=5,
    vmin=None,
    vmax=None,
    width="100%",  # where matplotlib uses `figsize`, folium normally use width and height
    height="100%",  # where matplotlib uses `figsize`, folium normally use width and height
    legend_kwds=None,
    categories=None,
    classification_kwds=None,
    missing_kwds=None,
    map_kwds={},  # keyword arguments to pass to newly created folium.Map instance if 'm' is not specified
    style_kwds={},  # Additional style to be passed to folium style_function
    **kwargs,  # Keyword arguments to pass to relevant folium class, e.g. `folium.GeoJson`
):
    """Proof of a concept of GeoDataFrame.view()
    """

    if not df.crs.equals(4326):
        df = df.to_crs(4326)

    map_kwds["tiles"] = tiles

    map_kwds["width"] = width
    map_kwds["height"] = height

    if column is None:
        return _simple(
            df,
            color=color,
            style_kwds=style_kwds,
            m=m,
            map_kwds=map_kwds,
            tooltip=tooltip,
            popup=popup,
            **kwargs,
        )
    else:
        return _choropleth(
            df,
            column=column,
            cmap=cmap,
            bins=k,
            scheme=scheme,
            style_kwds=style_kwds,
            m=m,
            map_kwds=map_kwds,
            **kwargs,
        )


def _simple(
    gdf,
    color=None,
    style_kwds={},
    m=None,
    map_kwds={},
    tooltip=None,
    popup=None,
    **kwds,
):
    """
    Plot a simple single-color map with tooltip.

    Parameters
    ----------
    gdf : GeoDataFrame
        The GeoDataFrame to be viewed.
    color : str (default None)
        If specified, all objects will be colored uniformly.
    m : folium.Map (default None)
        Existing map instance on which to draw the plot
    style_kwds : dict (default {})
        Additional style to be passed to folium style_function
    map_kwds : dict (default {})
        Keyword arguments to pass to newly created folium.Map instance
        if 'm' is not specified
    **kwds : dict
        Keyword arguments to pass to folium.GeoJson

    Returns
    -------
    m : folium.Map
        Folium map instance
    """
    # TODO: field names must be string
    # TODO: wrap custom markers

    gdf = gdf.copy()

    # Get bounds to specify location and map extent
    bounds = gdf.total_bounds
    location = map_kwds.pop("location", None)
    if location is None:
        x = mean([bounds[0], bounds[2]])
        y = mean([bounds[1], bounds[3]])
        location = (y, x)

    # create folium.Map object
    if m is None:
        m = folium.Map(location=location, control_scale=True, **map_kwds)

    if isinstance(gdf, gpd.GeoDataFrame):
        # specify fields to show in the tooltip
        tooltip = _tooltip_popup("tooltip", tooltip, gdf)
        popup = _tooltip_popup("popup", popup, gdf)
    else:
        tooltip = None
        popup = None

    # specify color
    if color is not None:
        if (
            isinstance(color, str)
            and isinstance(gdf, gpd.GeoDataFrame)
            and color in gdf.columns
        ):  # use existing column
            style_function = lambda x: {"color": x["properties"][color], **style_kwds}
        else:  # assign new column
            if isinstance(gdf, gpd.GeoSeries):
                gdf = gpd.GeoDataFrame(geometry=gdf)

            gdf["__folium_color"] = color

            style_function = lambda x: {
                "color": x["properties"]["__folium_color"],
                **style_kwds,
            }
    else:  # use folium default
        style_function = lambda x: {**style_kwds}

    # add dataframe to map
    folium.GeoJson(
        gdf, tooltip=tooltip, popup=popup, style_function=style_function, **kwds,
    ).add_to(m)

    # fit bounds to get a proper zoom level
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    return m


def _choropleth(
    gdf,
    column=None,
    cmap=None,
    style_kwds={},
    m=None,
    map_kwds={},
    tooltip=None,
    popup=None,
    bins=5,
    scheme=None,
    **kwds,
):

    gdf = gdf.copy()
    gdf["__folium_key"] = range(len(gdf))
    geom = gdf.geometry.name

    # Get bounds to specify location and map extent
    bounds = gdf.total_bounds
    location = map_kwds.pop("location", None)
    if location is None:
        x = mean([bounds[0], bounds[2]])
        y = mean([bounds[1], bounds[3]])
        location = (y, x)

    # create folium.Map object
    if m is None:
        m = folium.Map(location=location, control_scale=True, **map_kwds)

    # get bins
    if scheme is not None:
        binning = mapclassify.classify(np.asarray(gdf[column]), scheme, k=bins)
        bins = binning.bins.tolist()
        bins.insert(0, gdf[column].min())

    choro = folium.Choropleth(
        gdf,
        data=gdf[["__folium_key", column]],
        key_on="feature.properties.__folium_key",
        columns=["__folium_key", column],
        legend_name=column,
        fill_color=cmap,
        bins=bins,
        **kwds,
    )
    choro.geojson.add_child(_tooltip_popup("tooltip", [column], gdf, labels=False))
    choro.geojson.add_child(_tooltip_popup("popup", 10, gdf, labels=True))

    choro.add_to(m)

    # fit bounds to get a proper zoom level
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    return m


def _tooltip_popup(type, fields, gdf, labels=True):
    """get tooltip or popup"""
    # specify fields to show in the tooltip
    if fields is False or fields is None or fields == 0:
        return None
    else:
        if fields == "all":
            fields = gdf.columns.drop(gdf.geometry.name).to_list()
        elif isinstance(fields, int):
            fields = gdf.columns.drop(gdf.geometry.name).to_list()[:fields]

    if "__folium_key" in fields:
        fields.remove("__folium_key")

    if type == "tooltip":
        return folium.GeoJsonTooltip(fields, labels=labels)
    elif type == "popup":
        return folium.GeoJsonPopup(fields, labels=labels)
