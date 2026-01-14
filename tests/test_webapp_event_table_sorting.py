from __future__ import annotations

import datetime as dt
import re


def _set_session(client, *, user_id: int, email: str) -> None:
    sess = client.session
    sess["user_id"] = int(user_id)
    sess["user_email"] = str(email)
    sess.save()


def _extract_game_events_table_event_ids(html: str) -> list[int]:
    m = re.search(
        r'<table[^>]*id="game-events-table"[^>]*>.*?<tbody>(?P<tbody>.*?)</tbody>',
        html,
        re.DOTALL,
    )
    assert m is not None, "Events table not found"
    tbody = m.group("tbody")
    pairs = re.findall(
        r"<tr[^>]*data-event-row-index=\"\d+\"[^>]*>\s*"
        r"<td[^>]*>(?P<etype>.*?)</td>\s*"
        r"<td[^>]*>(?P<eid>.*?)</td>",
        tbody,
        re.DOTALL,
    )
    out: list[int] = []
    for _etype_raw, eid_raw in pairs:
        eid_txt = re.sub(r"<[^>]*>", "", str(eid_raw or "")).strip()
        if not eid_txt:
            continue
        out.append(int(eid_txt))
    return out


def should_sort_hky_game_events_table_by_type_time_period(webapp_db):
    _django_orm, m = webapp_db

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now
    )
    team1 = m.Team.objects.create(
        id=101,
        user_id=int(owner.id),
        name="Home",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    team2 = m.Team.objects.create(
        id=102,
        user_id=int(owner.id),
        name="Away",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    game = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(team1.id),
        team2_id=int(team2.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 2, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=3,
        team2_score=2,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )

    et_assist, _ = m.HkyEventType.objects.get_or_create(
        key="assist", defaults={"name": "Assist", "created_at": now}
    )
    et_goal, _ = m.HkyEventType.objects.get_or_create(
        key="goal", defaults={"name": "Goal", "created_at": now}
    )
    et_penalty, _ = m.HkyEventType.objects.get_or_create(
        key="penalty", defaults={"name": "Penalty", "created_at": now}
    )

    # Chronological within a game: period asc, then time desc.
    m.HkyGameEventRow.objects.create(
        game_id=int(game.id),
        event_type_id=int(et_assist.id),
        import_key="a1",
        source="timetoscore",
        event_id=1,
        team_id=int(team1.id),
        player_id=None,
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=1,
        game_time="1:15",
        game_seconds=75,
        details="A1",
        created_at=now,
        updated_at=None,
    )
    m.HkyGameEventRow.objects.create(
        game_id=int(game.id),
        event_type_id=int(et_assist.id),
        import_key="a2",
        source="timetoscore",
        event_id=2,
        team_id=int(team1.id),
        player_id=None,
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=1,
        game_time="1:40",
        game_seconds=100,
        details="A2",
        created_at=now,
        updated_at=None,
    )

    # Goal: tie on time=100, period 1 before 2; then time=50.
    m.HkyGameEventRow.objects.create(
        game_id=int(game.id),
        event_type_id=int(et_goal.id),
        import_key="g1",
        source="timetoscore",
        event_id=3,
        team_id=int(team1.id),
        player_id=None,
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=1,
        game_time="1:40",
        game_seconds=100,
        details="G1",
        created_at=now,
        updated_at=None,
    )
    m.HkyGameEventRow.objects.create(
        game_id=int(game.id),
        event_type_id=int(et_goal.id),
        import_key="g2",
        source="timetoscore",
        event_id=4,
        team_id=int(team1.id),
        player_id=None,
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=2,
        game_time="1:40",
        game_seconds=100,
        details="G2",
        created_at=now,
        updated_at=None,
    )
    m.HkyGameEventRow.objects.create(
        game_id=int(game.id),
        event_type_id=int(et_goal.id),
        import_key="g3",
        source="timetoscore",
        event_id=5,
        team_id=int(team1.id),
        player_id=None,
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=2,
        game_time="0:50",
        game_seconds=50,
        details="G3",
        created_at=now,
        updated_at=None,
    )

    m.HkyGameEventRow.objects.create(
        game_id=int(game.id),
        event_type_id=int(et_penalty.id),
        import_key="p1",
        source="timetoscore",
        event_id=6,
        team_id=int(team1.id),
        player_id=None,
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=1,
        game_time="1:30",
        game_seconds=90,
        details="P1",
        created_at=now,
        updated_at=None,
    )

    from django.test import Client

    client = Client()
    _set_session(client, user_id=int(owner.id), email=str(owner.email))
    r = client.get(f"/hky/games/{int(game.id)}")
    assert r.status_code == 200
    event_ids = _extract_game_events_table_event_ids(r.content.decode("utf-8"))
    assert event_ids == [2, 3, 6, 1, 4, 5]


def should_sort_api_hky_team_player_events_by_type_time_period_date(webapp_db):
    _django_orm, m = webapp_db

    now = dt.datetime.now()
    owner = m.User.objects.create(
        id=10, email="owner@example.com", password_hash="x", name="Owner", created_at=now
    )
    league = m.League.objects.create(
        id=1,
        name="Test League",
        owner_user_id=int(owner.id),
        is_shared=True,
        is_public=False,
        created_at=now,
        updated_at=None,
    )
    team1 = m.Team.objects.create(
        id=101,
        user_id=int(owner.id),
        name="Team 1",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    team2 = m.Team.objects.create(
        id=102,
        user_id=int(owner.id),
        name="Team 2",
        logo_path=None,
        is_external=False,
        created_at=now,
        updated_at=None,
    )
    m.LeagueTeam.objects.create(league_id=int(league.id), team_id=int(team1.id))
    m.LeagueTeam.objects.create(league_id=int(league.id), team_id=int(team2.id))

    player = m.Player.objects.create(
        id=501,
        user_id=int(owner.id),
        team_id=int(team1.id),
        name="Skater",
        jersey_number="8",
        position="F",
        shoots="R",
        created_at=now,
        updated_at=None,
    )

    game1 = m.HkyGame.objects.create(
        id=1001,
        user_id=int(owner.id),
        team1_id=int(team1.id),
        team2_id=int(team2.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 2, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=5,
        team2_score=0,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    game2 = m.HkyGame.objects.create(
        id=1002,
        user_id=int(owner.id),
        team1_id=int(team2.id),
        team2_id=int(team1.id),
        game_type_id=None,
        starts_at=dt.datetime(2026, 1, 3, 10, 0, 0),
        location="Rink",
        notes=None,
        team1_score=0,
        team2_score=4,
        is_final=True,
        stats_imported_at=None,
        created_at=now,
        updated_at=None,
    )
    m.LeagueGame.objects.create(league_id=int(league.id), game_id=int(game1.id), sort_order=1)
    m.LeagueGame.objects.create(league_id=int(league.id), game_id=int(game2.id), sort_order=2)

    # Cap goal/assist events to match PlayerStat totals (what the team page shows).
    m.PlayerStat.objects.create(
        user_id=int(owner.id),
        team_id=int(team1.id),
        game_id=int(game1.id),
        player_id=int(player.id),
        goals=1,
        assists=3,
    )
    m.PlayerStat.objects.create(
        user_id=int(owner.id),
        team_id=int(team1.id),
        game_id=int(game2.id),
        player_id=int(player.id),
        goals=1,
        assists=1,
    )

    et_assist, _ = m.HkyEventType.objects.get_or_create(
        key="assist", defaults={"name": "Assist", "created_at": now}
    )
    et_goal, _ = m.HkyEventType.objects.get_or_create(
        key="goal", defaults={"name": "Goal", "created_at": now}
    )

    # Newest games first, then chronological within a game.
    m.HkyGameEventRow.objects.create(
        game_id=int(game1.id),
        event_type_id=int(et_assist.id),
        import_key="a1",
        source="timetoscore",
        event_id=101,
        team_id=int(team1.id),
        player_id=int(player.id),
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=2,
        game_time="0:50",
        game_seconds=50,
        details="A1",
        created_at=now,
        updated_at=None,
    )
    m.HkyGameEventRow.objects.create(
        game_id=int(game1.id),
        event_type_id=int(et_assist.id),
        import_key="a2",
        source="timetoscore",
        event_id=102,
        team_id=int(team1.id),
        player_id=int(player.id),
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=1,
        game_time="1:40",
        game_seconds=100,
        details="A2",
        created_at=now,
        updated_at=None,
    )
    m.HkyGameEventRow.objects.create(
        game_id=int(game2.id),
        event_type_id=int(et_assist.id),
        import_key="a3",
        source="timetoscore",
        event_id=103,
        team_id=int(team1.id),
        player_id=int(player.id),
        team_side="Away",
        for_against="For",
        team_rel="Away",
        period=1,
        game_time="1:40",
        game_seconds=100,
        details="A3",
        created_at=now,
        updated_at=None,
    )
    m.HkyGameEventRow.objects.create(
        game_id=int(game1.id),
        event_type_id=int(et_assist.id),
        import_key="a4",
        source="timetoscore",
        event_id=104,
        team_id=int(team1.id),
        player_id=int(player.id),
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=2,
        game_time="1:40",
        game_seconds=100,
        details="A4",
        created_at=now,
        updated_at=None,
    )

    # Goal events (Goal sorts after Assist).
    m.HkyGameEventRow.objects.create(
        game_id=int(game1.id),
        event_type_id=int(et_goal.id),
        import_key="g1",
        source="timetoscore",
        event_id=201,
        team_id=int(team1.id),
        player_id=int(player.id),
        team_side="Home",
        for_against="For",
        team_rel="Home",
        period=1,
        game_time="3:20",
        game_seconds=200,
        details="G1",
        created_at=now,
        updated_at=None,
    )
    m.HkyGameEventRow.objects.create(
        game_id=int(game2.id),
        event_type_id=int(et_goal.id),
        import_key="g2",
        source="timetoscore",
        event_id=202,
        team_id=int(team1.id),
        player_id=int(player.id),
        team_side="Away",
        for_against="For",
        team_rel="Away",
        period=1,
        game_time="2:30",
        game_seconds=150,
        details="G2",
        created_at=now,
        updated_at=None,
    )

    from django.test import Client

    client = Client()
    _set_session(client, user_id=int(owner.id), email=str(owner.email))
    r = client.get(
        f"/api/hky/teams/{int(team1.id)}/players/{int(player.id)}/events?league_id={int(league.id)}&limit=1000"
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    event_ids = [int(e["event_id"]) for e in (out.get("events") or []) if e.get("event_id")]
    assert event_ids == [202, 103, 201, 102, 104, 101]
