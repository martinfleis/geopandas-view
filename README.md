# geopandas-view

**DISCLAIMER: THIS HAS BECOME AN INTERNAL MODULE OF GEOPANDAS:**

```py
gdf.explore()
```

Interactive exploration of GeoPandas GeoDataFrames

Proof-of-a-concept based on folium, closely mimcking API of `GeoDataFrame.plot()`.

For details see RFC document: https://github.com/martinfleis/geopandas-view/issues/1

## Installation

```
pip install git+https://github.com/martinfleis/geopandas-view.git
```

Requires `geopandas`, `folium` and `mapclassify`.


## Usage

```python

import geopandas
from geopandas_view import view

df = geopandas.read_file(geopandas.datasets.get_path('nybb'))
view(df)
```
