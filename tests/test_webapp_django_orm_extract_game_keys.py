from tools.webapp import django_orm


def should_extract_timetoscore_game_id_from_notes_json() -> None:
    assert django_orm._extract_timetoscore_game_id_from_notes('{"timetoscore_game_id": 123}') == 123


def should_extract_timetoscore_game_id_from_notes_assignment() -> None:
    assert django_orm._extract_timetoscore_game_id_from_notes("game_id=456") == 456
    assert django_orm._extract_timetoscore_game_id_from_notes("x game_id = 789") == 789
    assert django_orm._extract_timetoscore_game_id_from_notes("x;game_id = 42") == 42
    assert django_orm._extract_timetoscore_game_id_from_notes("x|game_id=7") == 7


def should_extract_timetoscore_game_id_from_notes_json_fragment() -> None:
    assert django_orm._extract_timetoscore_game_id_from_notes('"timetoscore_game_id": 99') == 99


def should_extract_external_game_key_from_notes_json() -> None:
    assert (
        django_orm._extract_external_game_key_from_notes('{"external_game_key": "stockton-r1"}')
        == "stockton-r1"
    )


def should_extract_external_game_key_from_notes_json_fragment() -> None:
    assert (
        django_orm._extract_external_game_key_from_notes('{"x": 1, "external_game_key": "abc-123"}')
        == "abc-123"
    )
    assert django_orm._extract_external_game_key_from_notes('"external_game_key": "z"') == "z"


def should_return_none_for_missing_game_keys() -> None:
    assert django_orm._extract_timetoscore_game_id_from_notes("") is None
    assert django_orm._extract_timetoscore_game_id_from_notes(None) is None
    assert django_orm._extract_external_game_key_from_notes("") is None
    assert django_orm._extract_external_game_key_from_notes(None) is None
