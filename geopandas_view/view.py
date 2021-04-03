from statistics import mean
from warnings import warn

import folium
import branca as bc
import pandas as pd
import geopandas as gpd
import mapclassify
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

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
    legend=None,
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
    missing_kwds={},
    tooltip_kwds={},
    popup_kwds={},
    legend_kwds={},
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
        The name of a colormap recognized by matplotlib or a list-like of colors.
    color : str, array-like (default None)
        Named color or a list-like of colors (named or hex).
    m : folium.Map (default None)
        Existing map instance on which to draw the plot.
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
    legend : bool (default None)
        Plot a categorical legend in categorical plots.
        Ignored if no `column` is given, or if `color` is given.
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
        in the column to be plotted is used. Cannot be higher than minimum data value.
    vmax : None or float (default None)
        Maximum value of cmap. If None, the maximum data value
        in the column to be plotted is used. Cannot be lower than maximum data value.
    width : pixel int or percentage string (default: '100%')
        Width of the folium.Map. If the argument
        m is given explicitly, width is ignored.
    height : pixel int or percentage string (default: '100%')
        Height of the folium.Map. If the argument
        m is given explicitly, height is ignored.
    categories : list-like
        Ordered list-like object of categories to be used for categorical plot.
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
        * ``'EPSG3395'`` : rarely used by some commercial tile providers. Uses Elliptical Mercator
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
    legend_kwds : dict (default {})
        Additional keywords to be passed to the legend.

        Currently supported customisation:

        caption : string
            Custom caption of the legend. Defaults to the column name.

        Additional accepted keywords when `scheme` is specified:

        colorbar : bool (default True)
            An option to control the style of the legend. If True, continuous
            colorbar will be used. If False, categorical legend will be used for bins.
        scale : bool (default True)
            Scale bins along the colorbar axis according to the bin edges (True)
            or use the equal length for each bin (False)
        fmt : string (default "{:.2f}")
            A formatting specification for the bin edges of the classes in the
            legend. For example, to have no decimals: ``{"fmt": "{:.0f}"}``. Applies
            if ``colorbar=False``.
        labels : list-like
            A list of legend labels to override the auto-generated labels.
            Needs to have the same number of elements as the number of
            classes (`k`). Applies if ``colorbar=False``.
        interval : boolean (default False)
            An option to control brackets from mapclassify legend.
            If True, open/closed interval brackets are shown in the legend.
            Applies if ``colorbar=False``.

    **kwargs : dict
        Additional options to be passed on to the folium.Map, folium.GeoJson or
        folium.Choropleth.

    Returns
    -------
    m : folium.Map
        Folium map instance

    """
    gdf = df.copy()

    if gdf.crs is None:
        crs = "Simple"
        tiles = None
    elif not gdf.crs.equals(4326):
        gdf = gdf.to_crs(4326)

    # create folium.Map object
    if m is None:
        # Get bounds to specify location and map extent
        bounds = gdf.total_bounds
        location = kwargs.pop("location", None)
        if location is None:
            x = mean([bounds[0], bounds[2]])
            y = mean([bounds[1], bounds[3]])
            location = (y, x)

        # get a subset of kwargs to be passed to folium.Map
        map_kwds = {i: kwargs[i] for i in kwargs.keys() if i in _MAP_KWARGS}

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

    for map_kwd in _MAP_KWARGS:
        kwargs.pop(map_kwd, None)

    nan_idx = None

    if column is not None:
        if pd.api.types.is_list_like(column):
            if len(column) != gdf.shape[0]:
                raise ValueError(
                    "The GeoDataframe and given column have different number of rows."
                )
            else:
                column_name = "__plottable_column"
                gdf[column_name] = column
                column = column_name
        elif pd.api.types.is_categorical_dtype(gdf[column]):
            if categories is not None:
                raise ValueError(
                    "Cannot specify 'categories' when column has categorical dtype"
                )
            categorical = True
        elif gdf[column].dtype is np.dtype("O") or categories:
            categorical = True

        nan_idx = pd.isna(gdf[column])

        if categorical:
            cat = pd.Categorical(gdf[column][~nan_idx], categories=categories)
            N = len(cat.categories)
            cmap = cmap if cmap else "tab20"

            # colormap exists in matplotlib
            if cmap in plt.colormaps():

                color = np.apply_along_axis(
                    colors.to_hex, 1, cm.get_cmap(cmap, N)(cat.codes)
                )
                legend_colors = np.apply_along_axis(
                    colors.to_hex, 1, cm.get_cmap(cmap, N)(range(N))
                )

            # custom list of colors
            elif pd.api.types.is_list_like(cmap):
                if N > len(cmap):
                    cmap = cmap * (N // len(cmap) + 1)
                color = np.take(cmap, cat.codes)
                legend_colors = np.take(cmap, range(N))

            else:
                raise ValueError(
                    "'cmap' is invalid. For categorical plots, pass either valid "
                    "named matplotlib colormap or a list-like of colors."
                )
        else:
            vmin = gdf[column].min() if not vmin else vmin
            vmax = gdf[column].max() if not vmax else vmax

            if vmin > gdf[column].min():
                warn(
                    "'vmin' cannot be higher than minimum value. Setting vmin to minimum.",
                    UserWarning,
                    stacklevel=3,
                )
                vmin = gdf[column].min()
            if vmax < gdf[column].max():
                warn(
                    "'vmax' cannot be lower than maximum value. Setting vmax to maximum.",
                    UserWarning,
                    stacklevel=3,
                )
                vmax = gdf[column].max()

            # get bins
            if scheme is not None:

                if classification_kwds is None:
                    classification_kwds = {}
                if "k" not in classification_kwds:
                    classification_kwds["k"] = k

                binning = mapclassify.classify(
                    np.asarray(gdf[column][~nan_idx]), scheme, **classification_kwds
                )
                color = np.apply_along_axis(
                    colors.to_hex, 1, cm.get_cmap(cmap, k)(binning.yb)
                )

            else:

                bins = np.linspace(vmin, vmax, 257)[1:]
                binning = mapclassify.classify(
                    np.asarray(gdf[column][~nan_idx]), "UserDefined", bins=bins
                )

                color = np.apply_along_axis(
                    colors.to_hex, 1, cm.get_cmap(cmap, 256)(binning.yb)
                )

        # we cannot color default 'marker'
        if marker_type is None:
            marker_type = "circle"

    # set default style
    if "fillOpacity" not in style_kwds:
        style_kwds["fillOpacity"] = 0.5
    if "weight" not in style_kwds:
        style_kwds["weight"] = 1

    # specify color
    if color is not None:
        if (
            isinstance(color, str)
            and isinstance(gdf, gpd.GeoDataFrame)
            and color in gdf.columns
        ):  # use existing column
            style_function = lambda x: {
                "fillColor": x["properties"][color],
                **style_kwds,
            }
        else:  # assign new column
            if isinstance(gdf, gpd.GeoSeries):
                gdf = gpd.GeoDataFrame(geometry=gdf)

            if nan_idx is not None and nan_idx.any():
                nan_color = missing_kwds.pop("color", None)

                gdf["__folium_color"] = nan_color
                gdf.loc[~nan_idx, "__folium_color"] = color
            else:
                gdf["__folium_color"] = color

            stroke_color = style_kwds.pop("color", None)
            if not stroke_color:
                style_function = lambda x: {
                    "fillColor": x["properties"]["__folium_color"],
                    "color": x["properties"]["__folium_color"],
                    **style_kwds,
                }
            else:
                style_function = lambda x: {
                    "fillColor": x["properties"]["__folium_color"],
                    "color": stroke_color,
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
                "Only 'marker', 'circle', and 'circle_marker' are supported as marker values"
            )

    # preprare tooltip and popup
    if isinstance(gdf, gpd.GeoDataFrame):
        # specify fields to show in the tooltip
        tooltip = _tooltip_popup("tooltip", tooltip, gdf, **tooltip_kwds)
        popup = _tooltip_popup("popup", popup, gdf, **popup_kwds)
    else:
        tooltip = None
        popup = None

    # add dataframe to map
    folium.GeoJson(
        gdf.__geo_interface__,
        tooltip=tooltip,
        popup=popup,
        marker=marker,
        style_function=style_function,
        **kwargs,
    ).add_to(m)

    # fit bounds to get a proper zoom level
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    if legend:
        # NOTE: overlaps should be resolved in branca https://github.com/python-visualization/branca/issues/88
        caption = column if not column == "__plottable_column" else ""
        caption = legend_kwds.pop("caption", caption)
        if categorical:
            categories = cat.categories.to_list()
            legend_colors = legend_colors.tolist()

            if nan_idx.any() and nan_color:
                categories.append(missing_kwds.pop("label", "NaN"))
                legend_colors.append(nan_color)

            _categorical_legend(m, caption, categories, legend_colors)
        elif column is not None:

            cbar = legend_kwds.pop("colorbar", True)
            if scheme:
                cb_colors = np.apply_along_axis(
                    colors.to_hex, 1, cm.get_cmap(cmap, binning.k)(range(binning.k))
                )
                if cbar:
                    if legend_kwds.pop("scale", True):
                        index = [vmin] + binning.bins.tolist()
                    else:
                        index = None
                    colorbar = bc.colormap.StepColormap(
                        cb_colors, vmin=vmin, vmax=vmax, caption=caption, index=index
                    )
                else:
                    fmt = legend_kwds.pop("fmt", "{:.2f}")
                    if "labels" in legend_kwds:
                        categories = legend_kwds["labels"]
                    else:
                        categories = binning.get_legend_classes(fmt)
                        show_interval = legend_kwds.pop("interval", False)
                        if not show_interval:
                            categories = [c[1:-1] for c in categories]

                    if nan_idx.any() and nan_color:
                        categories.append(missing_kwds.pop("label", "NaN"))
                        cb_colors = np.append(cb_colors, nan_color)
                    _categorical_legend(m, caption, categories, cb_colors)

            else:
                mp_cmap = cm.get_cmap(cmap)
                cb_colors = np.apply_along_axis(
                    colors.to_hex, 1, mp_cmap(range(mp_cmap.N))
                )
                # linear legend
                if mp_cmap.N > 20:
                    colorbar = bc.colormap.LinearColormap(
                        cb_colors, vmin=vmin, vmax=vmax, caption=caption
                    )

                # steps
                else:
                    colorbar = bc.colormap.StepColormap(
                        cb_colors, vmin=vmin, vmax=vmax, caption=caption
                    )

            if cbar:
                if nan_idx.any() and nan_color:
                    _categorical_legend(
                        m, "", [missing_kwds.pop("label", "NaN")], [nan_color]
                    )
                m.add_child(colorbar)

    return m


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

    for field in ["__plottable_column", "__folium_color"]:
        if field in fields:
            fields.remove(field)

    # Cast fields to str
    fields = list(map(str, fields))
    if type == "tooltip":
        return folium.GeoJsonTooltip(fields, **kwds)
    elif type == "popup":
        return folium.GeoJsonPopup(fields, **kwds)


def _categorical_legend(m, title, categories, colors):
    """
    Add categorical legend to a map

    The implementation is using the code originally written by Michel Metran
    (@michelmetran) and released on GitHub
    (https://github.com/michelmetran/package_folium) under MIT license.

    Copyright (c) 2020 Michel Metran

    Parameters
    ----------
    m : folium.Map
        Existing map instance on which to draw the plot
    title : str
        title of the legend (e.g. column name)
    categories : list-like
        list of categories
    colors : list-like
        list of colors (in the same order as categories)
    """

    # Header to Add
    head = """
    {% macro header(this, kwargs) %}
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
    <script>$( function() {
        $( ".maplegend" ).draggable({
            start: function (event, ui) {
                $(this).css({
                    right: "auto",
                    top: "auto",
                    bottom: "auto"
                });
            }
        });
    });
    </script>
    <style type='text/css'>
      .maplegend {
        position: absolute;
        z-index:9999;
        background-color: rgba(255, 255, 255, .8);
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0,0,0,0.2);
        padding: 10px;
        font: 12px/14px Arial, Helvetica, sans-serif;
        right: 10px;
        bottom: 20px;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 0px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        list-style: none;
        margin-left: 0;
        line-height: 16px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 14px;
        width: 14px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}
    """

    # Add CSS (on Header)
    macro = bc.element.MacroElement()
    macro._template = bc.element.Template(head)
    m.get_root().add_child(macro)

    body = f"""
    <div id='maplegend {title}' class='maplegend'>
        <div class='legend-title'>{title}</div>
        <div class='legend-scale'>
            <ul class='legend-labels'>"""

    # Loop Categories
    for label, color in zip(categories, colors):
        body += f"""
                <li><span style='background:{color}'></span>{label}</li>"""

    body += """
            </ul>
        </div>
    </div>
    """

    # Add Body
    body = bc.element.Element(body, "legend")
    m.get_root().html.add_child(body)
