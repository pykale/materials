from kalematerials.loaddata.element_tables import DEFAULT_MIEDEMA_PATH, load_element_properties, load_miedema


def test_packaged_tables_load():
    periodic_table, miedema = load_element_properties()
    assert periodic_table.loc["Fe", "period"] == 4
    assert miedema.loc["Fe", "Co"] == miedema.loc["Co", "Fe"] != 0
    assert (miedema.dtypes == float).all() and (miedema.values.diagonal() == 0).all()


def test_explicit_path():
    assert load_miedema(DEFAULT_MIEDEMA_PATH).shape[0] == 73
