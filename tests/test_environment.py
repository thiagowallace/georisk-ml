def test_core_imports():
    import geopandas
    import numpy
    import pandas
    import rasterio
    import sklearn

    assert geopandas is not None
    assert numpy is not None
    assert pandas is not None
    assert rasterio is not None
    assert sklearn is not None