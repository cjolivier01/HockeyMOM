from __future__ import annotations

from django.db import models


class League(models.Model):
    name = models.CharField(max_length=255, unique=True)
    owner_user = models.ForeignKey(
        "User",
        on_delete=models.CASCADE,
        db_column="owner_user_id",
        db_constraint=False,
        related_name="owned_leagues",
    )
    is_shared = models.BooleanField(default=False)
    is_public = models.BooleanField(default=False)
    source = models.CharField(max_length=64, null=True, blank=True)
    external_key = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "leagues"


class User(models.Model):
    email = models.CharField(max_length=255, unique=True)
    password_hash = models.TextField()
    name = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField()

    default_league = models.ForeignKey(
        League,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        db_column="default_league_id",
        db_constraint=False,
        related_name="default_for_users",
    )
    video_clip_len_s = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "users"


class LeagueMember(models.Model):
    league = models.ForeignKey(League, on_delete=models.CASCADE, related_name="members")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="league_memberships")
    role = models.CharField(max_length=32, default="viewer")
    created_at = models.DateTimeField()

    class Meta:
        db_table = "league_members"
        constraints = [
            models.UniqueConstraint(fields=["league", "user"], name="uniq_member"),
        ]


class LeaguePageView(models.Model):
    league = models.ForeignKey(League, on_delete=models.CASCADE, related_name="page_views")
    page_kind = models.CharField(max_length=32)
    entity_id = models.IntegerField(default=0)
    view_count = models.BigIntegerField(default=0)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()

    class Meta:
        db_table = "league_page_views"
        constraints = [
            models.UniqueConstraint(fields=["league", "page_kind", "entity_id"], name="uniq_page"),
        ]


class Game(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="games")
    name = models.CharField(max_length=255)
    dir_path = models.TextField()
    status = models.CharField(max_length=32, default="new")
    created_at = models.DateTimeField()

    class Meta:
        db_table = "games"


class Job(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="jobs", db_constraint=False
    )
    game_id = models.IntegerField(null=True, blank=True)
    dir_path = models.TextField()
    slurm_job_id = models.CharField(max_length=64, null=True, blank=True)
    status = models.CharField(max_length=32)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    user_email = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        db_table = "jobs"


class Reset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="password_resets")
    token = models.CharField(max_length=128, unique=True)
    expires_at = models.DateTimeField()
    used_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField()

    class Meta:
        db_table = "resets"


class Team(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="teams", db_constraint=False
    )
    name = models.CharField(max_length=255)
    logo_path = models.TextField(null=True, blank=True)
    is_external = models.BooleanField(default=False)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "teams"
        constraints = [
            models.UniqueConstraint(fields=["user", "name"], name="uniq_team_user_name"),
        ]


class Player(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="players", db_constraint=False
    )
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="players")
    name = models.CharField(max_length=255)
    jersey_number = models.CharField(max_length=16, null=True, blank=True)
    position = models.CharField(max_length=32, null=True, blank=True)
    shoots = models.CharField(max_length=8, null=True, blank=True)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "players"


class GameType(models.Model):
    name = models.CharField(max_length=64, unique=True)
    is_default = models.BooleanField(default=False)

    class Meta:
        db_table = "game_types"


class HkyGame(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="hky_games", db_constraint=False
    )
    team1 = models.ForeignKey(
        Team,
        on_delete=models.RESTRICT,
        related_name="hky_games_as_team1",
        db_column="team1_id",
    )
    team2 = models.ForeignKey(
        Team,
        on_delete=models.RESTRICT,
        related_name="hky_games_as_team2",
        db_column="team2_id",
    )
    game_type = models.ForeignKey(GameType, on_delete=models.SET_NULL, null=True, blank=True)
    starts_at = models.DateTimeField(null=True, blank=True)
    location = models.CharField(max_length=255, null=True, blank=True)
    notes = models.TextField(null=True, blank=True)
    team1_score = models.IntegerField(null=True, blank=True)
    team2_score = models.IntegerField(null=True, blank=True)
    is_final = models.BooleanField(default=False)
    stats_imported_at = models.DateTimeField(null=True, blank=True)
    timetoscore_game_id = models.BigIntegerField(null=True, blank=True)
    external_game_key = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "hky_games"
        constraints = [
            models.UniqueConstraint(fields=["timetoscore_game_id"], name="uniq_hky_tts_id"),
            models.UniqueConstraint(
                fields=["user", "external_game_key"], name="uniq_hky_user_ext_key"
            ),
        ]


class PlayerStat(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="player_stats", db_constraint=False
    )
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="player_stats")
    game = models.ForeignKey(HkyGame, on_delete=models.CASCADE, related_name="player_stats")
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="stats")

    goals = models.IntegerField(null=True, blank=True)
    assists = models.IntegerField(null=True, blank=True)
    shots = models.IntegerField(null=True, blank=True)
    pim = models.IntegerField(null=True, blank=True)
    plus_minus = models.IntegerField(null=True, blank=True)
    hits = models.IntegerField(null=True, blank=True)
    blocks = models.IntegerField(null=True, blank=True)
    toi_seconds = models.IntegerField(null=True, blank=True)
    faceoff_wins = models.IntegerField(null=True, blank=True)
    faceoff_attempts = models.IntegerField(null=True, blank=True)
    goalie_saves = models.IntegerField(null=True, blank=True)
    goalie_ga = models.IntegerField(null=True, blank=True)
    goalie_sa = models.IntegerField(null=True, blank=True)

    sog = models.IntegerField(null=True, blank=True)
    expected_goals = models.IntegerField(null=True, blank=True)
    giveaways = models.IntegerField(null=True, blank=True)
    turnovers_forced = models.IntegerField(null=True, blank=True)
    created_turnovers = models.IntegerField(null=True, blank=True)
    takeaways = models.IntegerField(null=True, blank=True)
    controlled_entry_for = models.IntegerField(null=True, blank=True)
    controlled_entry_against = models.IntegerField(null=True, blank=True)
    controlled_exit_for = models.IntegerField(null=True, blank=True)
    controlled_exit_against = models.IntegerField(null=True, blank=True)
    gt_goals = models.IntegerField(null=True, blank=True)
    gw_goals = models.IntegerField(null=True, blank=True)
    ot_goals = models.IntegerField(null=True, blank=True)
    ot_assists = models.IntegerField(null=True, blank=True)
    shifts = models.IntegerField(null=True, blank=True)
    gf_counted = models.IntegerField(null=True, blank=True)
    ga_counted = models.IntegerField(null=True, blank=True)
    video_toi_seconds = models.IntegerField(null=True, blank=True)
    sb_avg_shift_seconds = models.IntegerField(null=True, blank=True)
    sb_median_shift_seconds = models.IntegerField(null=True, blank=True)
    sb_longest_shift_seconds = models.IntegerField(null=True, blank=True)
    sb_shortest_shift_seconds = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "player_stats"
        constraints = [
            models.UniqueConstraint(fields=["game", "player"], name="uniq_game_player"),
        ]


class PlayerPeriodStat(models.Model):
    game = models.ForeignKey(HkyGame, on_delete=models.CASCADE, related_name="player_period_stats")
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="period_stats")
    period = models.IntegerField()
    toi_seconds = models.IntegerField(null=True, blank=True)
    shifts = models.IntegerField(null=True, blank=True)
    gf = models.IntegerField(null=True, blank=True)
    ga = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "player_period_stats"
        constraints = [
            models.UniqueConstraint(fields=["game", "player", "period"], name="uniq_period"),
        ]


class HkyGameStat(models.Model):
    game = models.OneToOneField(
        HkyGame,
        primary_key=True,
        on_delete=models.CASCADE,
        db_column="game_id",
        related_name="stats_row",
    )
    stats_json = models.TextField(null=True, blank=True)
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "hky_game_stats"


class HkyGameEvent(models.Model):
    game = models.OneToOneField(
        HkyGame,
        primary_key=True,
        on_delete=models.CASCADE,
        db_column="game_id",
        related_name="events_row",
    )
    events_csv = models.TextField(null=True, blank=True)
    source_label = models.CharField(max_length=255, null=True, blank=True)
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "hky_game_events"


class HkyGamePlayerStatsCsv(models.Model):
    game = models.OneToOneField(
        HkyGame,
        primary_key=True,
        on_delete=models.CASCADE,
        db_column="game_id",
        related_name="player_stats_csv_row",
    )
    player_stats_csv = models.TextField(null=True, blank=True)
    source_label = models.CharField(max_length=255, null=True, blank=True)
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "hky_game_player_stats_csv"


class HkyEventType(models.Model):
    # A stable normalized key for matching across sources (e.g. "Expected Goal" vs "ExpectedGoal").
    key = models.CharField(max_length=64, unique=True)
    name = models.CharField(max_length=64)
    created_at = models.DateTimeField()

    class Meta:
        db_table = "hky_event_types"


class HkyGameEventRow(models.Model):
    game = models.ForeignKey(HkyGame, on_delete=models.CASCADE, related_name="event_rows")
    event_type = models.ForeignKey(HkyEventType, on_delete=models.RESTRICT, related_name="events")

    # Idempotency key for upserting events from repeated imports.
    import_key = models.CharField(max_length=64)

    # Optional resolved references for querying/aggregation.
    team = models.ForeignKey(Team, on_delete=models.SET_NULL, null=True, blank=True)
    player = models.ForeignKey(Player, on_delete=models.SET_NULL, null=True, blank=True)

    # Columns mirroring `all_events_summary.csv` and TimeToScore event CSVs.
    source = models.CharField(max_length=255, null=True, blank=True)
    event_id = models.IntegerField(null=True, blank=True)
    team_raw = models.CharField(max_length=64, null=True, blank=True)
    team_side = models.CharField(max_length=16, null=True, blank=True)
    for_against = models.CharField(max_length=16, null=True, blank=True)
    team_rel = models.CharField(max_length=16, null=True, blank=True)
    period = models.IntegerField(null=True, blank=True)
    game_time = models.CharField(max_length=32, null=True, blank=True)
    video_time = models.CharField(max_length=32, null=True, blank=True)
    game_seconds = models.IntegerField(null=True, blank=True)
    game_seconds_end = models.IntegerField(null=True, blank=True)
    video_seconds = models.IntegerField(null=True, blank=True)
    details = models.TextField(null=True, blank=True)
    attributed_players = models.TextField(null=True, blank=True)
    attributed_jerseys = models.TextField(null=True, blank=True)
    on_ice_players = models.TextField(null=True, blank=True)
    on_ice_players_home = models.TextField(null=True, blank=True)
    on_ice_players_away = models.TextField(null=True, blank=True)

    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "hky_game_event_rows"
        constraints = [
            models.UniqueConstraint(fields=["game", "import_key"], name="uniq_game_import_key"),
        ]
        indexes = [
            models.Index(fields=["game", "event_type"], name="idx_event_game_type"),
            models.Index(fields=["game", "player"], name="idx_event_game_player"),
            models.Index(fields=["game", "period", "game_seconds"], name="idx_event_game_time"),
        ]


class HkyGamePlayer(models.Model):
    """
    Cross-reference table for knowing which players were present in a game.
    This is intentionally separate from PlayerStat so that "present but no stats" is representable.
    """

    game = models.ForeignKey(HkyGame, on_delete=models.CASCADE, related_name="player_links")
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="game_links")
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="game_player_links")
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "hky_game_players"
        constraints = [
            models.UniqueConstraint(fields=["game", "player"], name="uniq_game_player_link"),
        ]


class HkyGameEventSuppression(models.Model):
    """
    Records event keys that should be ignored for a game (idempotent correction mechanism).
    Ingesters should skip suppressed keys, and views should exclude suppressed events.
    """

    game = models.ForeignKey(HkyGame, on_delete=models.CASCADE, related_name="event_suppressions")
    import_key = models.CharField(max_length=64)
    reason = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "hky_game_event_suppressions"
        constraints = [
            models.UniqueConstraint(
                fields=["game", "import_key"], name="uniq_game_event_suppression"
            ),
        ]


class LeagueTeam(models.Model):
    league = models.ForeignKey(League, on_delete=models.CASCADE, related_name="league_teams")
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="league_teams")
    division_name = models.CharField(max_length=255, null=True, blank=True)
    division_id = models.IntegerField(null=True, blank=True)
    conference_id = models.IntegerField(null=True, blank=True)
    mhr_div_rating = models.FloatField(null=True, blank=True)
    mhr_rating = models.FloatField(null=True, blank=True)
    mhr_agd = models.FloatField(null=True, blank=True)
    mhr_sched = models.FloatField(null=True, blank=True)
    mhr_games = models.IntegerField(null=True, blank=True)
    mhr_updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "league_teams"
        constraints = [
            models.UniqueConstraint(fields=["league", "team"], name="uniq_league_team"),
        ]


class LeagueGame(models.Model):
    league = models.ForeignKey(League, on_delete=models.CASCADE, related_name="league_games")
    game = models.ForeignKey(HkyGame, on_delete=models.CASCADE, related_name="league_games")
    division_name = models.CharField(max_length=255, null=True, blank=True)
    division_id = models.IntegerField(null=True, blank=True)
    conference_id = models.IntegerField(null=True, blank=True)
    sort_order = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "league_games"
        constraints = [
            models.UniqueConstraint(fields=["league", "game"], name="uniq_league_game"),
        ]
