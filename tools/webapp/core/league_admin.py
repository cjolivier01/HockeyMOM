from typing import Optional

from .orm import _orm_modules


def reset_league_data(
    db_conn, league_id: int, *, owner_user_id: Optional[int] = None
) -> dict[str, int]:
    """
    Wipe imported hockey data for a league (games/teams/players/stats) while keeping:
      - users
      - league record and memberships

    This is used by `tools/webapp/scripts/reset_league_data.py` and the hidden REST endpoint.
    """
    stats: dict[str, int] = {
        "hky_game_players": 0,
        "hky_game_event_rows": 0,
        "hky_game_shift_rows": 0,
        "league_games": 0,
        "hky_games": 0,
        "league_teams": 0,
        "players": 0,
        "teams": 0,
    }
    del db_conn
    _django_orm, m = _orm_modules()
    from django.db import transaction
    from django.db.models import Q

    stats["league_games"] = m.LeagueGame.objects.filter(league_id=int(league_id)).count()
    stats["league_teams"] = m.LeagueTeam.objects.filter(league_id=int(league_id)).count()

    league_game_ids = list(
        m.LeagueGame.objects.filter(league_id=int(league_id)).values_list("game_id", flat=True)
    )
    other_game_ids = set()
    if league_game_ids:
        other_game_ids = set(
            m.LeagueGame.objects.exclude(league_id=int(league_id))
            .filter(game_id__in=league_game_ids)
            .values_list("game_id", flat=True)
        )
    exclusive_game_ids = sorted(
        {int(gid) for gid in league_game_ids if gid is not None and gid not in other_game_ids}
    )
    if exclusive_game_ids:
        stats["hky_games"] = m.HkyGame.objects.filter(id__in=exclusive_game_ids).count()
        stats["hky_game_players"] = m.HkyGamePlayer.objects.filter(
            game_id__in=exclusive_game_ids
        ).count()
        stats["hky_game_event_rows"] = m.HkyGameEventRow.objects.filter(
            game_id__in=exclusive_game_ids
        ).count()
        stats["hky_game_shift_rows"] = m.HkyGameShiftRow.objects.filter(
            game_id__in=exclusive_game_ids
        ).count()

    league_team_ids = list(
        m.LeagueTeam.objects.filter(league_id=int(league_id)).values_list("team_id", flat=True)
    )
    other_team_ids = set()
    if league_team_ids:
        other_team_ids = set(
            m.LeagueTeam.objects.exclude(league_id=int(league_id))
            .filter(team_id__in=league_team_ids)
            .values_list("team_id", flat=True)
        )
    exclusive_team_ids = sorted(
        {int(tid) for tid in league_team_ids if tid is not None and tid not in other_team_ids}
    )

    with transaction.atomic():
        # Remove league mappings (this is the "reset" behavior).
        m.LeagueGame.objects.filter(league_id=int(league_id)).delete()
        m.LeagueTeam.objects.filter(league_id=int(league_id)).delete()

        # Delete exclusive games (cascades to hky_game_* tables).
        if exclusive_game_ids:
            m.HkyGame.objects.filter(id__in=exclusive_game_ids).delete()

        if exclusive_team_ids:
            eligible_qs = m.Team.objects.filter(id__in=exclusive_team_ids, is_external=True)
            if owner_user_id is not None:
                eligible_qs = eligible_qs.filter(user_id=int(owner_user_id))
            eligible_ids = list(eligible_qs.values_list("id", flat=True))

            if eligible_ids:
                eligible_set = {int(tid) for tid in eligible_ids}
                still_used: set[int] = set()
                for team1_id, team2_id in m.HkyGame.objects.filter(
                    Q(team1_id__in=eligible_ids) | Q(team2_id__in=eligible_ids)
                ).values_list("team1_id", "team2_id"):
                    if team1_id in eligible_set:
                        still_used.add(int(team1_id))
                    if team2_id in eligible_set:
                        still_used.add(int(team2_id))

                safe_team_ids = sorted(
                    [int(tid) for tid in eligible_set if int(tid) not in still_used]
                )
                if safe_team_ids:
                    stats["players"] = m.Player.objects.filter(team_id__in=safe_team_ids).count()
                    stats["teams"] = m.Team.objects.filter(id__in=safe_team_ids).count()
                    m.Team.objects.filter(id__in=safe_team_ids).delete()

    return stats
