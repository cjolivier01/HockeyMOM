from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib import request

if ("TEST_SRCDIR" in os.environ or "RUNFILES_DIR" in os.environ) and "hmlib" not in sys.modules:
    hmlib_root = Path(__file__).resolve().parents[1] / "hmlib"
    hmlib_package = ModuleType("hmlib")
    hmlib_package.__file__ = str(hmlib_root / "__init__.py")
    hmlib_package.__path__ = [str(hmlib_root)]
    sys.modules["hmlib"] = hmlib_package


def _selector_module() -> Any:
    from hmlib.scoreboard import selector as selector_module

    return selector_module


def _make_selector() -> Any:
    selector_module = _selector_module()
    return selector_module.ScoreboardSelector(
        image=selector_module.Image.new("RGB", (64, 48), color=(17, 43, 71)),
        game_id="test-game",
        bind_host="127.0.0.1",
        open_browser=False,
    )


def _read_text(url: str) -> str:
    with request.urlopen(url, timeout=3) as response:
        return response.read().decode("utf-8")


def _post_json(url: str, payload: dict) -> str:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=3) as response:
        return response.read().decode("utf-8")


def should_serve_scoreboard_selector_page_and_image():
    selector = _make_selector()
    selector._start_server()
    try:
        html = _read_text(selector.primary_url)
        assert "Pin the four scoreboard corners" in html
        assert "Game: test-game" in html

        with request.urlopen(f"{selector.primary_url}image", timeout=3) as response:
            assert response.headers.get_content_type() == "image/png"
            image_bytes = response.read()

        assert image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        selector.close()


def should_save_clockwise_points_from_web_submission():
    selector = _make_selector()
    selector._start_server()
    try:
        response_html = _post_json(
            f"{selector.primary_url}api/complete",
            {
                "action": "save",
                "points": [[55, 41], [6, 8], [54, 7], [5, 40]],
            },
        )

        assert "Thank you for playing!" in response_html
        assert selector.points == [(6, 8), (54, 7), (55, 41), (5, 40)]
    finally:
        selector.close()


def should_mark_missing_scoreboard_from_web_submission():
    selector = _make_selector()
    selector._start_server()
    try:
        selector_module = _selector_module()
        response_html = _post_json(
            f"{selector.primary_url}api/complete",
            {
                "action": "none",
                "points": [],
            },
        )

        assert "Thank you for playing!" in response_html
        assert selector.points == selector_module.ScoreboardSelector.NULL_POINTS
    finally:
        selector.close()
