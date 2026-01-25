from __future__ import annotations

import datetime as dt
import json

import pytest


def _set_session(client, *, user_id: int, email: str) -> None:
    sess = client.session
    sess["user_id"] = int(user_id)
    sess["user_email"] = str(email)
    sess.save()


@pytest.fixture()
def client_and_models(monkeypatch, webapp_db):
    _django_orm, m = webapp_db
    monkeypatch.setenv("HM_WEBAPP_SKIP_DB_INIT", "1")
    from django.test import Client

    return Client(), m


def should_increment_player_events_page_views_from_player_events_api(client_and_models):
    client, m = client_and_models
    from django.test import Client

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10,
        email="owner@example.com",
        password_hash="x",
        name="Owner",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    m.League.objects.create(
        id=1,
        name="Public League",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=True,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    team = m.Team.objects.create(
        id=1,
        user_id=int(owner.id),
        name="Team A",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(
        league_id=1,
        team_id=int(team.id),
        division_name="12AA",
    )
    player = m.Player.objects.create(
        id=100,
        user_id=int(owner.id),
        team_id=int(team.id),
        name="Skater One",
        jersey_number="9",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )

    client_anon = Client()
    r = client_anon.get(
        f"/api/hky/teams/{int(team.id)}/players/{int(player.id)}/events?league_id=1"
    )
    assert r.status_code == 200
    r = client_anon.get(
        f"/api/hky/teams/{int(team.id)}/players/{int(player.id)}/events?league_id=1"
    )
    assert r.status_code == 200

    pv = (
        m.LeaguePageView.objects.filter(
            league_id=1, page_kind="player_events", entity_id=int(player.id)
        )
        .values_list("view_count", flat=True)
        .first()
    )
    assert int(pv or 0) == 2

    client_owner = client
    _set_session(client_owner, user_id=int(owner.id), email=str(owner.email))
    r = client_owner.get(
        f"/api/hky/teams/{int(team.id)}/players/{int(player.id)}/events?league_id=1"
    )
    assert r.status_code == 200
    pv2 = (
        m.LeaguePageView.objects.filter(
            league_id=1, page_kind="player_events", entity_id=int(player.id)
        )
        .values_list("view_count", flat=True)
        .first()
    )
    assert int(pv2 or 0) == 2


def should_record_player_events_and_event_clip_views_via_record_endpoint(client_and_models):
    client, m = client_and_models
    from django.test import Client

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10,
        email="owner@example.com",
        password_hash="x",
        name="Owner",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    m.League.objects.create(
        id=1,
        name="Public League",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=True,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    team = m.Team.objects.create(
        id=1,
        user_id=int(owner.id),
        name="Team A",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(
        league_id=1,
        team_id=int(team.id),
        division_name="12AA",
    )
    player = m.Player.objects.create(
        id=100,
        user_id=int(owner.id),
        team_id=int(team.id),
        name="Skater One",
        jersey_number="9",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    game = m.HkyGame.objects.create(
        id=200,
        user_id=int(owner.id),
        team1_id=int(team.id),
        team2_id=int(team.id),
        game_type_id=None,
        starts_at=None,
        location=None,
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        timetoscore_game_id=None,
        external_game_key=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=1,
        game_id=int(game.id),
        division_name="12AA",
        sort_order=1,
    )
    et = m.HkyEventType.objects.create(key="goal", name="Goal", created_at=now)
    ev = m.HkyGameEventRow.objects.create(
        id=500,
        game_id=int(game.id),
        event_type_id=int(et.id),
        import_key="k",
        created_at=now,
        updated_at=None,
    )

    client_anon = Client()
    r = client_anon.post(
        "/api/leagues/1/page_views/record",
        data=json.dumps({"kind": "player_events", "entity_id": int(player.id)}),
        content_type="application/json",
    )
    assert r.status_code == 200
    r = client_anon.post(
        "/api/leagues/1/page_views/record",
        data=json.dumps({"kind": "event_clip", "entity_id": int(ev.id)}),
        content_type="application/json",
    )
    assert r.status_code == 200

    pv_player = (
        m.LeaguePageView.objects.filter(
            league_id=1, page_kind="player_events", entity_id=int(player.id)
        )
        .values_list("view_count", flat=True)
        .first()
    )
    pv_clip = (
        m.LeaguePageView.objects.filter(league_id=1, page_kind="event_clip", entity_id=int(ev.id))
        .values_list("view_count", flat=True)
        .first()
    )
    assert int(pv_player or 0) == 1
    assert int(pv_clip or 0) == 1

    client_owner = client
    _set_session(client_owner, user_id=int(owner.id), email=str(owner.email))
    r = client_owner.post(
        "/api/leagues/1/page_views/record",
        data=json.dumps({"kind": "player_events", "entity_id": int(player.id)}),
        content_type="application/json",
    )
    assert r.status_code == 200
    r = client_owner.post(
        "/api/leagues/1/page_views/record",
        data=json.dumps({"kind": "event_clip", "entity_id": int(ev.id)}),
        content_type="application/json",
    )
    assert r.status_code == 200

    pv_player2 = (
        m.LeaguePageView.objects.filter(
            league_id=1, page_kind="player_events", entity_id=int(player.id)
        )
        .values_list("view_count", flat=True)
        .first()
    )
    pv_clip2 = (
        m.LeaguePageView.objects.filter(league_id=1, page_kind="event_clip", entity_id=int(ev.id))
        .values_list("view_count", flat=True)
        .first()
    )
    assert int(pv_player2 or 0) == 1
    assert int(pv_clip2 or 0) == 1


def should_mark_event_clip_baseline_with_mark_batch(client_and_models):
    client_owner, m = client_and_models

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10,
        email="owner@example.com",
        password_hash="x",
        name="Owner",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    m.League.objects.create(
        id=1,
        name="Public League",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=True,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    m.LeaguePageView.objects.create(
        league_id=1,
        page_kind="event_clip",
        entity_id=55,
        view_count=7,
        created_at=now,
        updated_at=now,
    )

    _set_session(client_owner, user_id=int(owner.id), email=str(owner.email))
    r = client_owner.get("/api/leagues/1/page_views/batch?kind=event_clip&entity_ids=55")
    assert r.status_code == 200
    j = json.loads(r.content)
    assert j["ok"] is True
    assert j["results"]["55"]["count"] == 7
    assert j["results"]["55"]["baseline_count"] is None
    assert j["results"]["55"]["delta_count"] is None

    r = client_owner.post(
        "/api/leagues/1/page_views/mark_batch", data={"kind": "event_clip", "entity_ids": "55"}
    )
    assert r.status_code == 200
    j = json.loads(r.content)
    assert j["ok"] is True
    assert j["results"]["55"]["count"] == 7
    assert j["results"]["55"]["baseline_count"] == 7
    assert j["results"]["55"]["delta_count"] == 0

    m.LeaguePageView.objects.filter(league_id=1, page_kind="event_clip", entity_id=55).update(
        view_count=10, updated_at=now
    )
    r = client_owner.get("/api/leagues/1/page_views/batch?kind=event_clip&entity_ids=55")
    assert r.status_code == 200
    j = json.loads(r.content)
    assert j["ok"] is True
    assert j["results"]["55"]["count"] == 10
    assert j["results"]["55"]["baseline_count"] == 7
    assert j["results"]["55"]["delta_count"] == 3


def should_allow_league_admin_to_read_and_mark_page_views(client_and_models):
    client_admin, m = client_and_models

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10,
        email="owner@example.com",
        password_hash="x",
        name="Owner",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    admin = m.User.objects.create(
        id=20,
        email="admin@example.com",
        password_hash="x",
        name="Admin",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    m.League.objects.create(
        id=1,
        name="Shared League",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=False,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueMember.objects.create(
        league_id=1,
        user_id=int(admin.id),
        role="admin",
        created_at=now,
    )
    m.LeaguePageView.objects.create(
        league_id=1,
        page_kind="event_clip",
        entity_id=55,
        view_count=7,
        created_at=now,
        updated_at=now,
    )

    _set_session(client_admin, user_id=int(admin.id), email=str(admin.email))

    r = client_admin.get("/api/leagues/1/page_views/batch?kind=event_clip&entity_ids=55")
    assert r.status_code == 200
    j = json.loads(r.content)
    assert j["ok"] is True
    assert j["results"]["55"]["count"] == 7

    r = client_admin.post(
        "/api/leagues/1/page_views/mark_batch", data={"kind": "event_clip", "entity_ids": "55"}
    )
    assert r.status_code == 200
    j = json.loads(r.content)
    assert j["ok"] is True
    assert j["results"]["55"]["baseline_count"] == 7

    r = client_admin.get("/api/leagues/1/page_views?kind=event_clip&entity_id=55")
    assert r.status_code == 200
    j = json.loads(r.content)
    assert j["ok"] is True
    assert j["count"] == 7


def should_record_page_views_for_shared_league_member(client_and_models):
    _, m = client_and_models
    from django.test import Client

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10,
        email="owner@example.com",
        password_hash="x",
        name="Owner",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    member = m.User.objects.create(
        id=20,
        email="member@example.com",
        password_hash="x",
        name="Member",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    m.League.objects.create(
        id=1,
        name="Shared League",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=False,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    team = m.Team.objects.create(
        id=1,
        user_id=int(owner.id),
        name="Team A",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(league_id=1, team_id=int(team.id), division_name="12AA")
    player = m.Player.objects.create(
        id=100,
        user_id=int(owner.id),
        team_id=int(team.id),
        name="Skater One",
        jersey_number="9",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueMember.objects.create(
        league_id=1,
        user_id=int(member.id),
        role="viewer",
        created_at=now,
    )

    client_member = Client()
    _set_session(client_member, user_id=int(member.id), email=str(member.email))
    r = client_member.post(
        "/api/leagues/1/page_views/record",
        data=json.dumps({"kind": "player_events", "entity_id": int(player.id)}),
        content_type="application/json",
    )
    assert r.status_code == 200
    pv = (
        m.LeaguePageView.objects.filter(
            league_id=1, page_kind="player_events", entity_id=int(player.id)
        )
        .values_list("view_count", flat=True)
        .first()
    )
    assert int(pv or 0) == 1


def should_render_event_clip_views_in_game_events_table_for_league_admin(client_and_models):
    client, m = client_and_models

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10,
        email="owner@example.com",
        password_hash="x",
        name="Owner",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    admin = m.User.objects.create(
        id=20,
        email="admin@example.com",
        password_hash="x",
        name="Admin",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    m.League.objects.create(
        id=1,
        name="Shared League",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=False,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueMember.objects.create(
        league_id=1,
        user_id=int(admin.id),
        role="admin",
        created_at=now,
    )
    team1 = m.Team.objects.create(
        id=1,
        user_id=int(owner.id),
        name="Team A",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    team2 = m.Team.objects.create(
        id=2,
        user_id=int(owner.id),
        name="Team B",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(league_id=1, team_id=int(team1.id), division_name="12AA")
    m.LeagueTeam.objects.create(league_id=1, team_id=int(team2.id), division_name="12AA")

    player1 = m.Player.objects.create(
        id=100,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="Skater One",
        jersey_number="9",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    m.Player.objects.create(
        id=101,
        user_id=int(owner.id),
        team_id=int(team2.id),
        name="Skater Two",
        jersey_number="10",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )

    game = m.HkyGame.objects.create(
        id=200,
        user_id=int(owner.id),
        team1_id=int(team1.id),
        team2_id=int(team2.id),
        game_type_id=None,
        starts_at=None,
        location=None,
        notes=None,
        team1_score=None,
        team2_score=None,
        is_final=False,
        stats_imported_at=None,
        timetoscore_game_id=None,
        external_game_key=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=1, game_id=int(game.id), division_name="12AA", sort_order=1
    )

    et = m.HkyEventType.objects.create(key="goal", name="Goal", created_at=now)
    m.HkyGameEventRow.objects.create(
        id=500,
        game_id=int(game.id),
        event_type_id=int(et.id),
        import_key="k",
        team_id=int(team1.id),
        player_id=int(player1.id),
        team_side="home",
        period=1,
        game_time="10:00",
        video_time="00:30",
        game_seconds=600,
        video_seconds=30,
        details="Goal",
        created_at=now,
        updated_at=None,
    )

    _set_session(client, user_id=int(admin.id), email=str(admin.email))
    sess = client.session
    sess["league_id"] = 1
    sess.save()

    r = client.get(f"/hky/games/{int(game.id)}")
    assert r.status_code == 200
    html = r.content.decode("utf-8", errors="replace")
    assert "<th>Views</th>" in html
    assert 'data-hm-clip-views="1"' in html


def should_infer_team_page_views_league_id_when_not_selected(client_and_models):
    client, m = client_and_models

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10,
        email="owner@example.com",
        password_hash="x",
        name="Owner",
        created_at=now,
        default_league_id=None,
        video_clip_len_s=None,
    )
    m.League.objects.create(
        id=1,
        name="League One",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=False,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    team = m.Team.objects.create(
        id=44,
        user_id=int(owner.id),
        name="Team 44",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(league_id=1, team_id=int(team.id), division_name="12AA")

    _set_session(client, user_id=int(owner.id), email=str(owner.email))
    r = client.get(f"/teams/{int(team.id)}")
    assert r.status_code == 200
    html = r.content.decode("utf-8", errors="replace")
    assert "var pageViewsLeagueId = 1" in html
    assert "hmCanViewLeaguePageViews = true" in html


def should_include_event_row_id_in_team_player_events_api(client_and_models):
    client, m = client_and_models
    from django.test import Client

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10,
        email="owner@example.com",
        password_hash="x",
        name="Owner",
        created_at=now,
        default_league_id=1,
        video_clip_len_s=None,
    )
    m.League.objects.create(
        id=1,
        name="League One",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=False,
        show_goalie_stats=False,
        show_shift_data=False,
        source=None,
        external_key=None,
        created_at=now,
        updated_at=None,
    )
    team = m.Team.objects.create(
        id=44,
        user_id=int(owner.id),
        name="Team 44",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    opp = m.Team.objects.create(
        id=45,
        user_id=int(owner.id),
        name="Opp",
        logo_path=None,
        is_external=True,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(league_id=1, team_id=int(team.id), division_name="12AA")
    m.LeagueTeam.objects.create(league_id=1, team_id=int(opp.id), division_name="12AA")
    player = m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(team.id),
        name="Skater",
        jersey_number="13",
        position="F",
        shoots=None,
        created_at=now,
        updated_at=None,
    )
    gt, _created = m.GameType.objects.get_or_create(
        name="Regular Season", defaults={"is_default": True}
    )
    notes = json.dumps({"game_video": "https://youtu.be/abc123"}, sort_keys=True)
    game = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(team.id),
        team2_id=int(opp.id),
        game_type_id=int(gt.id),
        starts_at=now,
        location="Rink",
        notes=notes,
        team1_score=2,
        team2_score=1,
        is_final=True,
        stats_imported_at=None,
        timetoscore_game_id=None,
        external_game_key="g-1001",
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(
        league_id=1, game_id=int(game.id), division_name="12AA", sort_order=1
    )
    ev_goal, _created = m.HkyEventType.objects.get_or_create(
        key="goal", defaults={"name": "Goal", "created_at": now}
    )
    event_row_id = 9001
    m.HkyGameEventRow.objects.create(
        id=int(event_row_id),
        game_id=int(game.id),
        event_type_id=int(ev_goal.id),
        import_key="g-1",
        team_id=int(team.id),
        player_id=int(player.id),
        source="timetoscore",
        event_id=1,
        team_raw="Home",
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=1,
        game_time="11:50",
        video_time="00:10",
        game_seconds=10,
        game_seconds_end=None,
        video_seconds=10,
        details="Goal",
        attributed_players=str(player.name),
        attributed_jerseys=str(player.jersey_number),
        created_at=now,
        updated_at=None,
    )

    client_owner = Client()
    _set_session(client_owner, user_id=int(owner.id), email=str(owner.email))
    sess = client_owner.session
    sess["league_id"] = 1
    sess.save()

    r = client_owner.get(f"/api/hky/teams/{int(team.id)}/players/{int(player.id)}/events")
    assert r.status_code == 200
    j = json.loads(r.content)
    assert j["ok"] is True
    assert isinstance(j["events"], list)
    assert j["events"], j
    assert int(j["events"][0].get("id") or 0) == int(event_row_id)
