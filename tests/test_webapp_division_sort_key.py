import importlib.util


def _load_app_module():
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def should_sort_divisions_by_age_level_then_external():
    mod = _load_app_module()
    divs = [
        "External 10B",
        "10B",
        "10AA",
        "External 10AA",
        "10AAA",
        "10BB",
        "10A",
        "10C",
        "12AA",
        "External 12AA",
    ]
    out = sorted(divs, key=mod.division_sort_key)
    assert out == [
        "10AAA",
        "10AA",
        "External 10AA",
        "10A",
        "10BB",
        "10B",
        "External 10B",
        "10C",
        "12AA",
        "External 12AA",
    ]
