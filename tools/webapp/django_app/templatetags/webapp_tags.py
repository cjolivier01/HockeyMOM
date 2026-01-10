from __future__ import annotations

import json
from typing import Any

from django import template
from django.core.serializers.json import DjangoJSONEncoder
from django.utils.safestring import mark_safe


register = template.Library()


def _import_logic():
    try:
        from tools.webapp import app as logic  # type: ignore

        return logic
    except Exception:  # pragma: no cover
        import app as logic  # type: ignore

        return logic


@register.filter("fmt_toi")
def fmt_toi(seconds: Any) -> str:
    logic = _import_logic()
    return logic.format_seconds_to_mmss_or_hhmmss(seconds)


@register.filter("fmt_date")
def fmt_date(value: Any) -> str:
    logic = _import_logic()
    d = logic.to_dt(value)
    return d.strftime("%Y-%m-%d") if d else ""


@register.filter("fmt_time")
def fmt_time(value: Any) -> str:
    logic = _import_logic()
    d = logic.to_dt(value)
    return d.strftime("%H:%M") if d else ""


@register.filter("fmt_stat")
def fmt_stat(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
        return f"{f:.2f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)


@register.filter("youtube_best_quality_url")
def youtube_best_quality_url(url: Any) -> str:
    s = str(url or "").strip()
    if not s:
        return ""
    try:
        from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

        u = urlparse(s)
        host = (u.hostname or "").lower()
        is_youtube = ("youtube.com" in host) or ("youtu.be" in host) or ("youtube-nocookie.com" in host)
        if not is_youtube:
            return s
        q = dict(parse_qsl(u.query or "", keep_blank_values=True))
        q.setdefault("vq", "hd1080")
        new_u = u._replace(query=urlencode(q, doseq=True))
        return urlunparse(new_u)
    except Exception:
        return s


@register.filter("tojson")
def tojson(value: Any) -> str:
    return mark_safe(json.dumps(value, cls=DjangoJSONEncoder))


@register.filter("commas_to_newlines")
def commas_to_newlines(value: Any) -> str:
    s = str(value or "")
    return s.replace(",", "\n")


@register.filter("get_item")
def get_item(value: Any, key: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    try:
        return value[key]
    except Exception:
        return None


@register.simple_tag(takes_context=True)
def url_with_args(context, **kwargs: Any) -> str:
    request = context.get("request")
    if request is None:
        return ""
    params = dict(getattr(request, "GET", {}).items())
    for k, v in (kwargs or {}).items():
        key = str(k)
        if v is None or str(v).strip() == "":
            params.pop(key, None)
        else:
            params[key] = str(v)
    from urllib.parse import urlencode

    qs = urlencode(params)
    return request.path + (f"?{qs}" if qs else "")


@register.simple_tag
def sort_label(sort_key: Any, sort_dir: Any, col_key: Any, label: Any) -> str:
    sk = str(sort_key or "")
    ck = str(col_key or "")
    if sk != ck:
        return str(label)
    sd = str(sort_dir or "").strip().lower()
    arrow = "▲" if sd == "asc" else "▼"
    return f"{label} {arrow}"


@register.simple_tag(takes_context=True)
def sort_toggle_url(
    context,
    col_key: Any,
    *,
    sort_key: Any = None,
    sort_dir: Any = None,
    sort_param: str = "recent_sort",
    dir_param: str = "recent_dir",
    default_dir: str = "desc",
) -> str:
    request = context.get("request")
    if request is None:
        return ""
    params = dict(getattr(request, "GET", {}).items())
    cur_key = str(sort_key if sort_key is not None else (params.get(sort_param) or ""))
    cur_dir = str(sort_dir if sort_dir is not None else (params.get(dir_param) or default_dir))

    next_dir = "desc"
    if str(cur_key) == str(col_key) and cur_dir.strip().lower() == "desc":
        next_dir = "asc"

    params[sort_param] = str(col_key or "")
    params[dir_param] = next_dir
    from urllib.parse import urlencode

    qs = urlencode(params)
    return request.path + (f"?{qs}" if qs else "")
