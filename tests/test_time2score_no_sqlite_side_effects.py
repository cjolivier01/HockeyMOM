import importlib
import sys


def should_not_create_hockey_league_db_on_import(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Ensure a fresh import so we exercise module top-level code.
    for modname in [
        "hmlib.time2score.caha_lib",
        "hmlib.time2score.sharks_ice_lib",
        "hmlib.time2score.direct",
        "hmlib.time2score.normalize",
    ]:
        sys.modules.pop(modname, None)

    importlib.import_module("hmlib.time2score.caha_lib")
    importlib.import_module("hmlib.time2score.sharks_ice_lib")
    importlib.import_module("hmlib.time2score.direct")
    importlib.import_module("hmlib.time2score.normalize")

    assert not (tmp_path / "hockey_league.db").exists()


def should_use_caha_league_id_3():
    from hmlib.time2score import caha_lib

    assert int(caha_lib.CAHA_LEAGUE) == 3

