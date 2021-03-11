from statistics import mean

import folium
import pandas as pd
import geopandas as gpd
import mapclassify
import numpy as np

# available named colors
_BRANCA_COLORS = [
    "red",
    "blue",
    "green",
    "purple",
    "orange",
    "darkred",
    "lightred",
    "beige",
    "darkblue",
    "darkgreen",
    "cadetblue",
    "darkpurple",
    "white",
    "pink",
    "lightblue",
    "lightgreen",
    "gray",
    "black",
    "lightgray",
]

# available color palettes
_CB_PALETTES = [
    "BuGn",
    "BuPu",
    "GnBu",
    "OrRd",
    "PuBu",
    "PuBuGn",
    "PuRd",
    "RdPu",
    "YlGn",
    "YlGnBu",
    "YlOrBr",
    "YlOrRd",
]

_MAP_KWARGS = [
    "location",
    "prefer_canvas",
    "no_touch",
    "disable_3d",
    "png_enabled",
    "zoom_control",
]


def view(
    df,
    column=None,
    cmap=None,
    color=None,
    m=None,
    tiles="OpenStreetMap",
    attr=None,
    tooltip=False,
    popup=False,
    categorical=False,
    scheme=None,
    k=5,
    vmin=None,
    vmax=None,
    width="100%",
    height="100%",
    categories=None,
    classification_kwds=None,
    control_scale=True,
    crs="EPSG3857",
    marker_type=None,
    marker_kwds={},
    style_kwds={},
    tooltip_kwds={},
    popup_kwds={},
    **kwargs,
):
    """Interactive map based on GeoPandas and folium/leaflet.js

    Generate an interactive leaflet map based on GeoDataFrame or GeoSeries

    Parameters
    ----------
    df : GeoDataFrame
        The GeoDataFrame to be plotted.
    column : str, np.array, pd.Series (default None)
        The name of the dataframe column, np.array, or pd.Series to be plotted.
        If np.array or pd.Series are used then it must have same length as dataframe.
    cmap : str (default None)
        The name of a colormap recognized by colorbrewer. Available are:
        ``["BuGn", "BuPu", "GnBu", "OrRd", "PuBu", "PuBuGn", "PuRd", "RdPu", "YlGn",
        "YlGnBu", "YlOrBr", "YlOrRd"]``
    color : str, array-like (default None)
        Named color or array-like of colors (named or hex)
    m : folium.Map (default None)
        Existing map instance on which to draw the plot
    tiles : str (default 'OpenStreetMap')
        Map tileset to use. Can choose from this list of built-in tiles:

        ``["OpenStreetMap", "Stamen Terrain", “Stamen Toner", “Stamen Watercolor"
        "CartoDB positron", “CartoDB dark_matter"]``

        You can pass a custom tileset to Folium by passing a Leaflet-style URL
        to the tiles parameter: http://{s}.yourtiles.com/{z}/{x}/{y}.png.
        You can find a list of free tile providers here:
        http://leaflet-extras.github.io/leaflet-providers/preview/. Be sure
        to check their terms and conditions and to provide attribution with the attr keyword.
    attr : str (default None)
        Map tile attribution; only required if passing custom tile URL.
    tooltip : bool, str, int, list (default False)
        Display GeoDataFrame attributes when hovering over the object.
        Integer specifies first n columns to be included, ``True`` includes all
        columns. ``False`` removes tooltip. Pass string or list of strings to specify a
        column(s). Defaults to ``False``.
    popup : bool, str, int, list (default False)
        Input GeoDataFrame attributes for object displayed when clicking.
        Integer specifies first n columns to be included, ``True`` includes all
        columns. ``False`` removes tooltip. Pass string or list of strings to specify a
        column(s). Defaults to ``False``.
    categorical : bool (default False)
        If False, cmap will reflect numerical values of the
        column being plotted. For non-numerical columns, this
        will be set to True.
    scheme : str (default None)
        Name of a choropleth classification scheme (requires mapclassify).
        A mapclassify.MapClassifier object will be used
        under the hood. Supported are all schemes provided by mapclassify (e.g.
        'BoxPlot', 'EqualInterval', 'FisherJenks', 'FisherJenksSampled',
        'HeadTailBreaks', 'JenksCaspall', 'JenksCaspallForced',
        'JenksCaspallSampled', 'MaxP', 'MaximumBreaks',
        'NaturalBreaks', 'Quantiles', 'Percentiles', 'StdMean',
        'UserDefined'). Arguments can be passed in classification_kwds.
    k : int (default 5)
        Number of classes
    vmin : None or float (default None)
        Minimum value of cmap. If None, the minimum data value
        in the column to be plotted is used.
        TODO: not implemented yet
    vmax : None or float (default None)
        Maximum value of cmap. If None, the maximum data value
        in the column to be plotted is used.
        TODO: not implemented yet
    width : pixel int or percentage string (default: '100%')
        Width of the folium.Map. If the argument
        m is given explicitly, width is ignored.
    height : pixel int or percentage string (default: '100%')
        Height of the folium.Map. If the argument
        m is given explicitly, height is ignored.
    categories : list-like
        Ordered list-like object of categories to be used for categorical plot.
        TODO: not implemented yet
    classification_kwds : dict (default None)
        Keyword arguments to pass to mapclassify
    control_scale : bool, (default True)
        Whether to add a control scale on the map.
    crs : str (default "EPSG3857")
        Defines coordinate reference systems for projecting geographical points
        into pixel (screen) coordinates and back. You can use Leaflet’s values :

        * ``'EPSG3857'`` : The most common CRS for online maps, used by almost all
        free and commercial tile providers. Uses Spherical Mercator projection.
        Set in by default in Map’s crs option.
        * ``'EPSG4326'`` : A common CRS among
        GIS enthusiasts. Uses simple Equirectangular projection.
        * ``'EPSG3395'`` : arely used by some commercial tile providers. Uses Elliptical Mercator
        projection.
        * ``'Simple'`` : A simple CRS that maps longitude and latitude
        into x and y directly. May be used for maps of flat surfaces (e.g. game
        maps).

        Note that the CRS of tiles needs to match ``crs``.
    marker_type : str, folium.Circle, folium.CircleMarker, folium.Marker (default None)
        Allowed strings are ('marker', 'circle', 'circle_marker')
    marker_kwds: dict (default {})
        Additional keywords to be passed to the selected marker_type
    style_kwds : dict (default {})
        Additional style to be passed to folium style_function
    tooltip_kwds : dict (default {})
        Additional keywords to be passed to folium.features.GeoJsonTooltip,
        e.g. ``aliases``, ``labels``, or ``sticky``. See the folium
        documentation for details:
        https://python-visualization.github.io/folium/modules.html#folium.features.GeoJsonTooltip
    popup_kwds : dict (default {})
        Additional keywords to be passed to folium.features.GeoJsonPopup,
        e.g. ``aliases`` or ``labels``. See the folium
        documentation for details:
        https://python-visualization.github.io/folium/modules.html#folium.features.GeoJsonPopup

    **kwargs : dict
        Additional options to be passed on to the folium.Map, folium.GeoJson or
        folium.Choropleth.

    Returns
    -------
    m : folium.Map
        Folium map instance

    """
    # TODO: folium.LayerControl() - works only after all layers are in

    gdf = df.copy()

    if gdf.crs is None:
        crs = "Simple"
        tiles = None
    elif not gdf.crs.equals(4326):
        gdf = gdf.to_crs(4326)

    # Get bounds to specify location and map extent
    bounds = gdf.total_bounds
    location = kwargs.pop("location", None)
    if location is None:
        x = mean([bounds[0], bounds[2]])
        y = mean([bounds[1], bounds[3]])
        location = (y, x)

    # get a subset of kwargs to be passed to folium.Map
    map_kwds = {i: kwargs[i] for i in kwargs.keys() if i in _MAP_KWARGS}
    for map_kwd in _MAP_KWARGS:
        kwargs.pop(map_kwd, None)

    # create folium.Map object
    if m is None:
        m = folium.Map(
            location=location,
            control_scale=control_scale,
            tiles=tiles,
            attr=attr,
            width=width,
            height=height,
            crs=crs,
            **map_kwds,
        )

    if column is not None:
        if isinstance(column, (np.ndarray, pd.Series)):
            if column.shape[0] != gdf.shape[0]:
                raise ValueError(
                    "The GeoDataframe and given column have different number of rows."
                )
            else:
                column_name = "__plottable_column"
                gdf[column_name] = column
                column = column_name
        elif pd.api.types.is_categorical_dtype(gdf[column]):
            categorical = True
        elif gdf[column].dtype is np.dtype("O"):
            categorical = True

    if categorical:
        cat = pd.Categorical(gdf[column])
        if len(cat.categories) > len(_BRANCA_COLORS):
            color = np.take(
                _BRANCA_COLORS * (len(cat.categories) // len(_BRANCA_COLORS) + 1),
                cat.codes,
            )
        else:
            color = np.take(_BRANCA_COLORS, cat.codes)

    if column is None or categorical:
        _simple(
            m,
            gdf,
            color=color,
            style_kwds=style_kwds,
            tooltip=tooltip,
            tooltip_kwds=tooltip_kwds,
            popup=popup,
            popup_kwds=popup_kwds,
            marker_type=marker_type,
            marker_kwds=marker_kwds,
            **kwargs,
        )
    else:
        _choropleth(
            m,
            gdf,
            column=column,
            cmap=cmap,
            bins=k,
            scheme=scheme,
            style_kwds=style_kwds,
            classification_kwds=classification_kwds,
            tooltip=tooltip,
            tooltip_kwds=tooltip_kwds,
            popup=popup,
            popup_kwds=popup_kwds,
            **kwargs,
        )

    # fit bounds to get a proper zoom level
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    return m


def _simple(
    m,
    gdf,
    color=None,
    style_kwds={},
    tooltip=False,
    popup=False,
    tooltip_kwds={},
    popup_kwds={},
    marker_type=None,
    marker_kwds={},
    **kwds,
):
    """
    Plot a simple single-color map with tooltip.

    Parameters
    ----------
    m : folium.Map (default None)
        Existing map instance on which to draw the plot
    gdf : GeoDataFrame
        The GeoDataFrame to be viewed.
    color : str (default None)
        If specified, all objects will be colored uniformly.
    style_kwds : dict (default {})
        Additional style to be passed to folium style_function
    tooltip : bool, str, int, list (default False)
        Display GeoDataFrame attributes when hovering over the object.
        Integer specifies first n columns to be included, ``True`` includes all
        columns. ``False`` removes tooltip. Pass string or list of strings to specify a
        column(s). Defaults to ``False``.
    popup : bool, str, int, list (default False)
        Input GeoDataFrame attributes for object displayed when clicking.
        Integer specifies first n columns to be included, ``True`` includes all
        columns. ``False`` removes tooltip. Pass string or list of strings to specify a
        column(s). Defaults to ``False``.
    tooltip_kwds : dict (default {})
        Additional keywords to be passed to folium.features.GeoJsonTooltip,
        e.g. ``aliases``, ``labels``, or ``sticky``. See the folium
        documentation for details:
        https://python-visualization.github.io/folium/modules.html#folium.features.GeoJsonTooltip
    popup_kwds : dict (default {})
        Additional keywords to be passed to folium.features.GeoJsonPopup,
        e.g. ``aliases`` or ``labels``. See the folium
        documentation for details:
        https://python-visualization.github.io/folium/modules.html#folium.features.GeoJsonPopup
    marker_type : str, folium.Circle, folium.CircleMarker, folium.Marker (default None)
        Allowed strings are ('marker', 'circle', 'circle_marker')
    marker_kwds: dict (default {})
        Additional keywords to be passed to the selected marker_type

    **kwds : dict
        Keyword arguments to pass to folium.GeoJson
    """

    if isinstance(gdf, gpd.GeoDataFrame):
        # specify fields to show in the tooltip
        tooltip = _tooltip_popup("tooltip", tooltip, gdf, **tooltip_kwds)
        popup = _tooltip_popup("popup", popup, gdf, **popup_kwds)
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

    marker = marker_type
    if marker_type is not None and isinstance(marker_type, str):
        if marker_type == "marker":
            marker = folium.Marker(**marker_kwds)
        elif marker_type == "circle":
            marker = folium.Circle(**marker_kwds)
        elif marker_type == "circle_marker":
            marker = folium.CircleMarker(**marker_kwds)
        else:
            raise ValueError(
                "Only marker, circle, and circle_marker are supported as marker values"
            )
    # add dataframe to map
    folium.GeoJson(
        gdf.__geo_interface__,
        tooltip=tooltip,
        popup=popup,
        marker=marker,
        style_function=style_function,
        **kwds,
    ).add_to(m)


def _choropleth(
    m,
    gdf,
    column=None,
    cmap=None,
    style_kwds={},
    tooltip=False,
    popup=False,
    bins=5,
    scheme=None,
    classification_kwds=None,
    tooltip_kwds={},
    popup_kwds={},
    **kwds,
):
    """
    Plot a choropleth map with tooltip and popup.

    Parameters
    ----------
    m : folium.Map (default None)
        Existing map instance on which to draw the plot
    gdf : GeoDataFrame
        The GeoDataFrame to be viewed.
    column : str, np.array, pd.Series (default None)
        The name of the dataframe column, np.array, or pd.Series to be plotted.
        If np.array or pd.Series are used then it must have same length as dataframe.
    cmap : str (default None)
        The name of a colormap recognized by colorbrewer. Available are:
        ``["BuGn", "BuPu", "GnBu", "OrRd", "PuBu", "PuBuGn", "PuRd", "RdPu", "YlGn",
        "YlGnBu", "YlOrBr", "YlOrRd"]``
    style_kwds : dict (default {})
        Additional style to be passed to folium style_function
    tooltip : bool, str, int, list (default False)
        Display GeoDataFrame attributes when hovering over the object.
        Integer specifies first n columns to be included, ``True`` includes all
        columns. ``False`` removes tooltip. Pass string or list of strings to specify a
        column(s). Defaults to ``False``.
    popup : bool, str, int, list (default False)
        Input GeoDataFrame attributes for object displayed when clicking.
        Integer specifies first n columns to be included, ``True`` includes all
        columns. ``False`` removes tooltip. Pass string or list of strings to specify a
        column(s). Defaults to ``False``.
    bins : int (default 5)
        Number of classes
    scheme : str (default None)
        Name of a choropleth classification scheme (requires mapclassify).
        A mapclassify.MapClassifier object will be used
        under the hood. Supported are all schemes provided by mapclassify (e.g.
        'BoxPlot', 'EqualInterval', 'FisherJenks', 'FisherJenksSampled',
        'HeadTailBreaks', 'JenksCaspall', 'JenksCaspallForced',
        'JenksCaspallSampled', 'MaxP', 'MaximumBreaks',
        'NaturalBreaks', 'Quantiles', 'Percentiles', 'StdMean',
        'UserDefined'). Arguments can be passed in classification_kwds.
    classification_kwds : dict (default None)
        Keyword arguments to pass to mapclassify
    tooltip_kwds : dict (default {})
        Additional keywords to be passed to folium.features.GeoJsonTooltip,
        e.g. ``aliases``, ``labels``, or ``sticky``. See the folium
        documentation for details:
        https://python-visualization.github.io/folium/modules.html#folium.features.GeoJsonTooltip
    popup_kwds : dict (default {})
        Additional keywords to be passed to folium.features.GeoJsonPopup,
        e.g. ``aliases`` or ``labels``. See the folium
        documentation for details:
        https://python-visualization.github.io/folium/modules.html#folium.features.GeoJsonPopup

    **kwds : dict
        Keyword arguments to pass to folium.GeoJson
    """

    gdf["__folium_key"] = range(len(gdf))

    # get bins
    if scheme is not None:

        if classification_kwds is None:
            classification_kwds = {}
        if "k" not in classification_kwds:
            classification_kwds["k"] = bins

        binning = mapclassify.classify(
            np.asarray(gdf[column]), scheme, **classification_kwds
        )
        bins = binning.bins.tolist()
        bins.insert(0, gdf[column].min())

    choro = folium.Choropleth(
        gdf.__geo_interface__,
        data=gdf[["__folium_key", column]],
        key_on="feature.properties.__folium_key",
        columns=["__folium_key", column],
        legend_name=column,
        fill_color=cmap,
        bins=bins,
        name=column,
        **kwds,
    )

    if tooltip is not False:
        choro.geojson.add_child(_tooltip_popup("tooltip", tooltip, gdf, **tooltip_kwds))
    if popup is not False:
        choro.geojson.add_child(_tooltip_popup("popup", popup, gdf, **popup_kwds))

    choro.add_to(m)


def _tooltip_popup(type, fields, gdf, **kwds):
    """get tooltip or popup"""
    # specify fields to show in the tooltip
    if fields is False or fields is None or fields == 0:
        return None
    else:
        if fields is True:
            fields = gdf.columns.drop(gdf.geometry.name).to_list()
        elif isinstance(fields, int):
            fields = gdf.columns.drop(gdf.geometry.name).to_list()[:fields]
        elif isinstance(fields, str):
            fields = [fields]

    if "__folium_key" in fields:
        fields.remove("__folium_key")
    if "__plottable_column" in fields:
        fields.remove("__plottable_column")

    # Cast fields to str
    fields = list(map(str, fields))
    if type == "tooltip":
        return folium.GeoJsonTooltip(fields, **kwds)
    elif type == "popup":
        return folium.GeoJsonPopup(fields, **kwds)
