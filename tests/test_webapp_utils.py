import os
import pathlib

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "hmwebapp.settings")

from hmwebapp.webapp import utils  # type: ignore


def should_read_dirwatch_state_return_empty_on_error(monkeypatch):
    def fake_read_text(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError

    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)
    state = utils.read_dirwatch_state()
    assert state == {"processed": {}, "active": {}}


def should_read_dirwatch_state_parse_json(monkeypatch):
    payload = '{"processed": {"d1": {"status": "DONE"}}, "active": {"d2": {}}}'

    def fake_read_text(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return payload

    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)
    state = utils.read_dirwatch_state()
    assert state["processed"]["d1"]["status"] == "DONE"
    assert "active" in state


def should_send_email_noop_without_sendmail(monkeypatch):
    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda name: None)  # type: ignore[arg-type]
    called = {"run": False}

    def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        called["run"] = True

    monkeypatch.setattr(subprocess, "run", fake_run)
    utils.send_email("user@example.com", "Subject", "Body")
    assert called["run"] is False

