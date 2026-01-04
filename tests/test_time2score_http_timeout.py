import os


def should_use_timeout_for_time2score_http(monkeypatch):
    monkeypatch.setenv("HM_T2S_HTTP_TIMEOUT", "12")
    from hmlib.time2score import util

    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None, **_kwargs):  # noqa: ANN001
        seen["url"] = url
        seen["params"] = params
        seen["headers"] = headers
        seen["timeout"] = timeout

        class _Resp:
            text = "<html><body>ok</body></html>"

        return _Resp()

    monkeypatch.setattr(util, "requests", type("R", (), {"get": staticmethod(fake_get)})())
    util.get_html("http://example.com", params={"a": "b"})
    assert seen["timeout"] == 12.0

