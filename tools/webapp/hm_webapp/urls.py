from __future__ import annotations

from django.urls import path


def _import_views():
    try:
        from tools.webapp.django_app import views  # type: ignore

        return views
    except Exception:  # pragma: no cover
        from django_app import views  # type: ignore

        return views


v = _import_views()

urlpatterns = [
    path("", v.index, name="index"),
    path("register", v.register, name="register"),
    path("login", v.login_view, name="login"),
    path("forgot", v.forgot_password, name="forgot"),
    path("reset/<str:token>", v.reset_password, name="reset"),
    path("logout", v.logout_view, name="logout"),
    path("league/select", v.league_select, name="league_select"),
    # API
    path("api/user/video_clip_len", v.api_user_video_clip_len, name="api_user_video_clip_len"),
    path(
        "api/leagues/<int:league_id>/page_views",
        v.api_league_page_views,
        name="api_league_page_views",
    ),
    path("api/hky/games/<int:game_id>/events", v.api_hky_game_events, name="api_hky_game_events"),
    path(
        "api/hky/teams/<int:team_id>/players/<int:player_id>/events",
        v.api_hky_team_player_events,
        name="api_hky_team_player_events",
    ),
    path(
        "api/hky/teams/<int:team_id>/goalies/<int:player_id>/stats",
        v.api_hky_team_goalie_stats,
        name="api_hky_team_goalie_stats",
    ),
    path(
        "api/import/hockey/ensure_league",
        v.api_import_ensure_league,
        name="api_import_ensure_league",
    ),
    path("api/import/hockey/teams", v.api_import_teams, name="api_import_teams"),
    path("api/import/hockey/game", v.api_import_game, name="api_import_game"),
    path("api/import/hockey/games_batch", v.api_import_games_batch, name="api_import_games_batch"),
    path(
        "api/import/hockey/shift_package",
        v.api_import_shift_package,
        name="api_import_shift_package",
    ),
    path(
        "api/internal/reset_league_data",
        v.api_internal_reset_league_data,
        name="api_internal_reset_league_data",
    ),
    path(
        "api/internal/ensure_league_owner",
        v.api_internal_ensure_league_owner,
        name="api_internal_ensure_league_owner",
    ),
    path("api/internal/ensure_user", v.api_internal_ensure_user, name="api_internal_ensure_user"),
    path(
        "api/internal/recalc_div_ratings",
        v.api_internal_recalc_div_ratings,
        name="api_internal_recalc_div_ratings",
    ),
    path(
        "api/internal/apply_event_corrections",
        v.api_internal_apply_event_corrections,
        name="api_internal_apply_event_corrections",
    ),
    # Legacy uploads/jobs UI
    path("games", v.games, name="games"),
    path("games/new", v.new_game, name="new_game"),
    path("games/<int:gid>", v.game_detail, name="game_detail"),
    path("games/<int:gid>/delete", v.delete_game, name="delete_game"),
    path("games/<int:gid>/upload", v.upload_to_game, name="upload_to_game"),
    path("games/<int:gid>/run", v.run_game, name="run_game"),
    path("uploads/<int:gid>/<path:name>", v.serve_upload, name="serve_upload"),
    path("jobs", v.jobs, name="jobs"),
    # Hockey UI
    path("media/team_logo/<int:team_id>", v.media_team_logo, name="media_team_logo"),
    path("teams", v.teams, name="teams"),
    path("teams/new", v.new_team, name="new_team"),
    path("teams/<int:team_id>", v.team_detail, name="team_detail"),
    path("teams/<int:team_id>/edit", v.team_edit, name="team_edit"),
    path("teams/<int:team_id>/players/new", v.player_new, name="player_new"),
    path("teams/<int:team_id>/players/<int:player_id>/edit", v.player_edit, name="player_edit"),
    path(
        "teams/<int:team_id>/players/<int:player_id>/delete", v.player_delete, name="player_delete"
    ),
    path("schedule", v.schedule, name="schedule"),
    path("schedule/new", v.schedule_new, name="schedule_new"),
    path("hky/games/<int:game_id>", v.hky_game_detail, name="hky_game_detail"),
    path(
        "hky/games/<int:game_id>/import_shift_stats",
        v.hky_game_import_shift_stats,
        name="hky_game_import_shift_stats",
    ),
    path("game_types", v.game_types, name="game_types"),
    # Leagues UI
    path("leagues", v.leagues_index, name="leagues_index"),
    path("leagues/new", v.leagues_new, name="leagues_new"),
    path("leagues/<int:league_id>/update", v.leagues_update, name="leagues_update"),
    path("leagues/<int:league_id>/delete", v.leagues_delete, name="leagues_delete"),
    path("leagues/<int:league_id>/members", v.league_members, name="league_members"),
    path(
        "leagues/<int:league_id>/members/remove",
        v.league_members_remove,
        name="league_members_remove",
    ),
    path(
        "leagues/recalc_div_ratings",
        v.leagues_recalc_div_ratings,
        name="leagues_recalc_div_ratings",
    ),
    # Public leagues
    path("public/leagues", v.public_leagues_index, name="public_leagues_index"),
    path("public/leagues/<int:league_id>", v.public_league_home, name="public_league_home"),
    path(
        "public/leagues/<int:league_id>/media/team_logo/<int:team_id>",
        v.public_media_team_logo,
        name="public_media_team_logo",
    ),
    path("public/leagues/<int:league_id>/teams", v.public_league_teams, name="public_league_teams"),
    path(
        "public/leagues/<int:league_id>/teams/<int:team_id>",
        v.public_league_team_detail,
        name="public_league_team_detail",
    ),
    path(
        "public/leagues/<int:league_id>/schedule",
        v.public_league_schedule,
        name="public_league_schedule",
    ),
    path(
        "public/leagues/<int:league_id>/hky/games/<int:game_id>",
        v.public_hky_game_detail,
        name="public_hky_game_detail",
    ),
]
