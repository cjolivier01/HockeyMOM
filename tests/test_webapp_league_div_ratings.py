from __future__ import annotations

import importlib.util
import os
from typing import Any

import pytest


def _load_app_module():
    os.environ.setdefault("HM_WEBAPP_SKIP_DB_INIT", "1")
    os.environ.setdefault("HM_WATCH_ROOT", "/tmp/hm-incoming-test")
    spec = importlib.util.spec_from_file_location("webapp_app", "tools/webapp/app.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore
    return mod


class _DummyPyMySQL:
    class cursors:
        DictCursor = object


class FakeConn:
    def __init__(self, *, admin_ok: bool) -> None:
        self.admin_ok = bool(admin_ok)

    def cursor(self, cursorclass: Any = None):
        return FakeCursor(self, dict_mode=cursorclass is not None)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeCursor:
    def __init__(self, conn: FakeConn, *, dict_mode: bool) -> None:
        self._conn = conn
        self._dict_mode = dict_mode
        self._rows: list[Any] = []
        self._idx = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query: str, params: Any = None) -> int:
        q = " ".join(str(query).split())
        p = tuple(params or ())
        self._rows = []
        self._idx = 0

        def t(*vals: Any) -> tuple[Any, ...]:
            return tuple(vals)

        def d(row: dict[str, Any]) -> dict[str, Any]:
            return dict(row)

        # open_db() league access validation
        if q.startswith("SELECT 1 FROM leagues l LEFT JOIN league_members m"):
            self._rows = [t(1)]
            return 1

        # inject_user_leagues() / leagues_index() query
        if q.startswith("SELECT l.id, l.name, l.is_shared, l.is_public"):
            self._rows = [d({"id": 1, "name": "L1", "is_shared": 0, "is_public": 0, "is_owner": 0, "is_admin": 1})]
            return 1

        # _is_league_admin() checks
        if q == "SELECT 1 FROM leagues WHERE id=%s AND owner_user_id=%s":
            self._rows = []
            return 1
        if q == "SELECT 1 FROM league_members WHERE league_id=%s AND user_id=%s AND role IN ('admin','owner')":
            self._rows = [t(1)] if self._conn.admin_ok else []
            return 1

        raise AssertionError(f"Unhandled SQL: {q!r} params={p!r}")

    def fetchone(self) -> Any:
        if self._idx >= len(self._rows):
            return None
        row = self._rows[self._idx]
        self._idx += 1
        return row

    def fetchall(self) -> list[Any]:
        if self._idx >= len(self._rows):
            return []
        out = self._rows[self._idx :]
        self._idx = len(self._rows)
        return out


@pytest.fixture()
def mod_and_client(monkeypatch):
    mod = _load_app_module()
    monkeypatch.setattr(mod, "pymysql", _DummyPyMySQL(), raising=False)
    fake_db = FakeConn(admin_ok=True)
    monkeypatch.setattr(mod, "get_db", lambda: fake_db)
    app = mod.create_app()
    app.testing = True
    client = app.test_client()
    return mod, client


def should_allow_league_admin_to_recalc_div_ratings_for_active_league(monkeypatch, mod_and_client):
    mod, client = mod_and_client
    called: list[int] = []

    def _fake_recompute(db_conn, league_id: int, *args: Any, **kwargs: Any):
        called.append(int(league_id))
        return {}

    monkeypatch.setattr(mod, "recompute_league_mhr_ratings", _fake_recompute, raising=True)

    with client.session_transaction() as sess:
        sess["user_id"] = 20
        sess["user_email"] = "admin@example.com"
        sess["league_id"] = 1

    r = client.post("/leagues/recalc_div_ratings")
    assert r.status_code == 302
    assert called == [1]


def should_reject_recalc_when_not_admin(monkeypatch, mod_and_client):
    mod, _client = mod_and_client
    # New app instance with a non-admin DB view.
    fake_db = FakeConn(admin_ok=False)
    monkeypatch.setattr(mod, "get_db", lambda: fake_db)
    app = mod.create_app()
    app.testing = True
    client = app.test_client()

    monkeypatch.setattr(
        mod,
        "recompute_league_mhr_ratings",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")),
        raising=True,
    )

    with client.session_transaction() as sess:
        sess["user_id"] = 99
        sess["user_email"] = "u@example.com"
        sess["league_id"] = 1

    r = client.post("/leagues/recalc_div_ratings")
    assert r.status_code == 302
