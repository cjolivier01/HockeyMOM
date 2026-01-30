from __future__ import annotations

from tools.webapp.scripts.import_time2score import build_timetoscore_goal_and_assist_events


def should_parse_timetoscore_scoring_dot_seconds_as_final_minute_seconds() -> None:
    # TimeToScore scoring tables can report final-minute clocks as seconds with tenths (e.g. "13.9").
    # These should be interpreted as ~0:14 remaining, not "13:09".
    stats = {
        "homeScoring": [
            {
                "period": "2",
                "time": "13.9",
                "goal": "9",
                "assist1": "",
                "assist2": "",
            }
        ],
        "awayScoring": [],
    }

    goals, assists = build_timetoscore_goal_and_assist_events(
        stats=stats,
        period_len_s=15 * 60,
        num_to_name_home={},
        num_to_name_away={},
    )
    assert assists == []
    assert goals == [
        {
            "Event Type": "Goal",
            "Source": "timetoscore",
            "Team Side": "Home",
            "For/Against": "For",
            "Team Rel": "Home",
            "Team Raw": "Home",
            "Period": 2,
            "Game Time": "0:14",
            "Game Seconds": 14,
            "Game Seconds End": "",
            "Details": "9",
            "Attributed Players": "9",
            "Attributed Jerseys": "9",
        }
    ]
