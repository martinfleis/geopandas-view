from statistics import mean

import folium


def view(df, column=None, **kwargs):
    """Proof of a concept of GeoDataFrame.view()
    """

    if not df.crs.equals(4326):
        df = df.to_crs(4326)

    if column is None:
        return _simple(df, **kwargs)


def _simple(gdf, color=None, style={}, m=None, map_kwds={}, **kwds):
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
    # TODO: take care about color

    bounds = gdf.total_bounds

    location = map_kwds.pop("location", None)

    if location is None:
        x = mean([bounds[0], bounds[2]])
        y = mean([bounds[1], bounds[3]])
        location = (y, x)

    if m is None:
        m = folium.Map(location=location, control_scale=True, **map_kwds)

    fields = kwds.pop("fields", gdf.columns.drop(gdf.geometry.name).to_list()[:10])
    if fields == "all":
        fields = gdf.columns.drop(gdf.geometry.name).to_list()

    folium.GeoJson(
        gdf.__geo_interface__, tooltip=folium.GeoJsonTooltip(fields=fields, **kwds),
    ).add_to(m)

    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    return m

