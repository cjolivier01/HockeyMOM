import os


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hmwebapp.settings")

import django

django.setup()

from hmwebapp.webapp import urls, utils  # type: ignore


def should_import_webapp_without_db_init():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    paths = {p.pattern._route for p in urls.urlpatterns}
    expected = {
        "",
        "teams",
        "teams/new",
        "teams/<int:team_id>",
        "teams/<int:team_id>/edit",
        "teams/<int:team_id>/players/new",
        "teams/<int:team_id>/players/<int:player_id>/edit",
        "teams/<int:team_id>/players/<int:player_id>/delete",
        "schedule",
        "schedule/new",
        "hky/games/<int:game_id>",
        "game_types",
        "media/team_logo/<int:team_id>",
    }
    assert expected.issubset(paths)


def should_parse_date_formats():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    f = utils.parse_dt_or_none
    assert f("2025-01-02") == "2025-01-02 00:00:00"
    assert f("2025-01-02T12:34") == "2025-01-02 12:34:00"
    assert f("") is None
    assert f(None) is None


def should_compute_team_stats_from_rows():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    # team_id=1: wins one, loses one, ties one
    rows = [
        {"team1_id": 1, "team2_id": 2, "team1_score": 3, "team2_score": 1},  # win
        {"team1_id": 2, "team2_id": 1, "team1_score": 2, "team2_score": 2},  # tie
        {"team1_id": 3, "team2_id": 1, "team1_score": 4, "team2_score": 2},  # loss
    ]
    stats = utils.compute_team_stats(rows, team_id=1)
    assert stats["wins"] == 1
    assert stats["losses"] == 1
    assert stats["ties"] == 1
    assert stats["gf"] == 3 + 2 + 2
    assert stats["ga"] == 1 + 2 + 4
    assert stats["points"] == 1 * 2 + 1 * 1


def should_aggregate_player_totals_from_rows():
    os.environ["HM_WEBAPP_SKIP_DB_INIT"] = "1"
    rows = [
        {"player_id": 10, "goals": 2, "assists": 1, "pim": 0, "shots": 5},
        {"player_id": 11, "goals": 0, "assists": 2, "pim": 2, "shots": 1},
    ]
    agg = utils.aggregate_players_totals(rows)
    assert agg[10]["goals"] == 2 and agg[10]["assists"] == 1 and agg[10]["points"] == 3
    assert agg[11]["goals"] == 0 and agg[11]["assists"] == 2 and agg[11]["points"] == 2
