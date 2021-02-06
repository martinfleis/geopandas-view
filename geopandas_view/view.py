from statistics import mean

import folium
import pandas as pd


def view(
    df,
    column=None,
    cmap=None,
    color=None,
    m=None,  # folium.Map is usually referred to as `m` instead of `ax` in the case of matplotlib
    tiles="OpenStreetMap",  # Map tileset to use
    categorical=False,
    legend=False,
    scheme=None,
    k=5,
    vmin=None,
    vmax=None,
    markersize=None,
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
            df, color=color, style_kwds=style_kwds, m=m, map_kwds=map_kwds, **kwargs,
        )
    # else:
    #     return _choropleth(df, **kwargs)


def _simple(gdf, color=None, style_kwds={}, m=None, map_kwds={}, **kwds):
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

    gdf = gdf.copy()

    bounds = gdf.total_bounds

    location = map_kwds.pop("location", None)

    if location is None:
        x = mean([bounds[0], bounds[2]])
        y = mean([bounds[1], bounds[3]])
        location = (y, x)

    if m is None:
        m = folium.Map(location=location, control_scale=True, **map_kwds)

    if color is not None:
        if isinstance(color, str) and color in gdf.columns:
            style_function = lambda x: {"color": x["properties"][color], **style_kwds}

        else:
            gdf["__folium_color"] = color

            style_function = lambda x: {
                "color": x["properties"]["__folium_color"],
                **style_kwds,
            }

    else:
        style_function = lambda x: {**style_kwds}

    fields = kwds.pop("fields", gdf.columns.drop(gdf.geometry.name).to_list()[:10])
    if fields == "all":
        fields = gdf.columns.drop(gdf.geometry.name).to_list()

    folium.GeoJson(
        gdf.__geo_interface__,
        tooltip=folium.GeoJsonTooltip(fields=fields),
        style_function=style_function,
        **kwds,
    ).add_to(m)

    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    return m

