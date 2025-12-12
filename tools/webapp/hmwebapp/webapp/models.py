from __future__ import annotations

from django.db import models


class User(models.Model):
    email = models.CharField(max_length=255, unique=True)
    password_hash = models.TextField()
    name = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField()
    default_league = models.ForeignKey(
        "League",
        null=True,
        blank=True,
        db_column="default_league_id",
        on_delete=models.SET_NULL,
    )

    class Meta:
        db_table = "users"
        managed = False
        app_label = "webapp"

    def __str__(self) -> str:
        return self.email


class League(models.Model):
    name = models.CharField(max_length=255, unique=True)
    owner_user = models.ForeignKey(
        User,
        db_column="owner_user_id",
        on_delete=models.CASCADE,
        related_name="owned_leagues",
    )
    is_shared = models.BooleanField(default=False)
    source = models.CharField(max_length=64, null=True, blank=True)
    external_key = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "leagues"
        managed = False
        app_label = "webapp"

    def __str__(self) -> str:
        return self.name


class LeagueMember(models.Model):
    league = models.ForeignKey(
        League,
        on_delete=models.CASCADE,
        related_name="memberships",
    )
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="league_memberships",
    )
    role = models.CharField(max_length=32, default="viewer")
    created_at = models.DateTimeField()

    class Meta:
        db_table = "league_members"
        managed = False
        app_label = "webapp"
        unique_together = ("league", "user")


class Game(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="games")
    name = models.CharField(max_length=255)
    dir_path = models.TextField()
    status = models.CharField(max_length=32, default="new")
    created_at = models.DateTimeField()

    class Meta:
        db_table = "games"
        managed = False
        app_label = "webapp"


class Job(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="jobs")
    game = models.ForeignKey(Game, null=True, blank=True, on_delete=models.SET_NULL)
    dir_path = models.TextField()
    slurm_job_id = models.CharField(max_length=64, null=True, blank=True)
    status = models.CharField(max_length=32)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    user_email = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        db_table = "jobs"
        managed = False
        app_label = "webapp"


class Reset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="resets")
    token = models.CharField(max_length=128, unique=True)
    expires_at = models.DateTimeField()
    used_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField()

    class Meta:
        db_table = "resets"
        managed = False
        app_label = "webapp"


class Team(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="teams")
    name = models.CharField(max_length=255)
    logo_path = models.TextField(null=True, blank=True)
    is_external = models.BooleanField(default=False)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "teams"
        managed = False
        app_label = "webapp"
        unique_together = ("user", "name")

    def __str__(self) -> str:
        return self.name


class Player(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="players")
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="players")
    name = models.CharField(max_length=255)
    jersey_number = models.CharField(max_length=16, null=True, blank=True)
    position = models.CharField(max_length=32, null=True, blank=True)
    shoots = models.CharField(max_length=8, null=True, blank=True)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "players"
        managed = False
        app_label = "webapp"


class GameType(models.Model):
    name = models.CharField(max_length=64, unique=True)
    is_default = models.BooleanField(default=False)

    class Meta:
        db_table = "game_types"
        managed = False
        app_label = "webapp"

    def __str__(self) -> str:
        return self.name


class HkyGame(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="hky_games")
    team1 = models.ForeignKey(
        Team,
        on_delete=models.RESTRICT,
        related_name="home_games",
        db_column="team1_id",
    )
    team2 = models.ForeignKey(
        Team,
        on_delete=models.RESTRICT,
        related_name="away_games",
        db_column="team2_id",
    )
    game_type = models.ForeignKey(
        GameType,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
    )
    starts_at = models.DateTimeField(null=True, blank=True)
    location = models.CharField(max_length=255, null=True, blank=True)
    notes = models.TextField(null=True, blank=True)
    team1_score = models.IntegerField(null=True, blank=True)
    team2_score = models.IntegerField(null=True, blank=True)
    is_final = models.BooleanField(default=False)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "hky_games"
        managed = False
        app_label = "webapp"


class PlayerStat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="player_stats")
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

    class Meta:
        db_table = "player_stats"
        managed = False
        app_label = "webapp"
        unique_together = ("game", "player")


class LeagueTeam(models.Model):
    league = models.ForeignKey(
        League,
        on_delete=models.CASCADE,
        related_name="league_teams",
    )
    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="league_teams",
    )

    class Meta:
        db_table = "league_teams"
        managed = False
        app_label = "webapp"
        unique_together = ("league", "team")


class LeagueGame(models.Model):
    league = models.ForeignKey(
        League,
        on_delete=models.CASCADE,
        related_name="league_games",
    )
    game = models.ForeignKey(
        HkyGame,
        on_delete=models.CASCADE,
        related_name="league_games",
    )

    class Meta:
        db_table = "league_games"
        managed = False
        app_label = "webapp"
        unique_together = ("league", "game")
