from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("register", views.register, name="register"),
    path("login", views.login_view, name="login"),
    path("forgot", views.forgot_password, name="forgot_password"),
    path("reset/<str:token>", views.reset_password, name="reset_password"),
    path("logout", views.logout_view, name="logout"),
    path("league/select", views.league_select, name="league_select"),
    path("games", views.games, name="games"),
    path("games/new", views.new_game, name="new_game"),
    path("games/<int:gid>", views.game_detail, name="game_detail"),
    path("games/<int:gid>/delete", views.delete_game, name="delete_game"),
    path("games/<int:gid>/upload", views.upload_files, name="upload_files"),
    path("games/<int:gid>/run", views.run_game, name="run_game"),
    path("uploads/<int:gid>/<path:name>", views.serve_upload, name="serve_upload"),
    path("jobs", views.jobs, name="jobs"),
    path("media/team_logo/<int:team_id>", views.media_team_logo, name="media_team_logo"),
    path("teams", views.teams, name="teams"),
    path("teams/new", views.new_team, name="new_team"),
    path("teams/<int:team_id>", views.team_detail, name="team_detail"),
    path("teams/<int:team_id>/edit", views.team_edit, name="team_edit"),
    path(
        "teams/<int:team_id>/players/new",
        views.player_new,
        name="player_new",
    ),
    path(
        "teams/<int:team_id>/players/<int:player_id>/edit",
        views.player_edit,
        name="player_edit",
    ),
    path(
        "teams/<int:team_id>/players/<int:player_id>/delete",
        views.player_delete,
        name="player_delete",
    ),
    path("leagues", views.leagues_index, name="leagues_index"),
    path("leagues/new", views.leagues_new, name="leagues_new"),
    path("leagues/<int:league_id>/delete", views.leagues_delete, name="leagues_delete"),
    path(
        "leagues/<int:league_id>/members",
        views.league_members,
        name="league_members",
    ),
    path(
        "leagues/<int:league_id>/members/remove",
        views.league_members_remove,
        name="league_members_remove",
    ),
    path(
        "leagues/<int:league_id>/members/add",
        views.league_members_add,
        name="league_members_add",
    ),
    path("schedule", views.schedule, name="schedule"),
    path("schedule/new", views.schedule_new, name="schedule_new"),
    path("hky/games/<int:game_id>", views.hky_game_detail, name="hky_game_detail"),
    path("game_types", views.game_types, name="game_types"),
    path("healthz", views.healthcheck, name="healthcheck"),
]
