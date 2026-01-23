import datetime as dt
import os
import secrets
from pathlib import Path
from typing import Optional

from werkzeug.security import generate_password_hash

from .orm import _orm_modules

WATCH_ROOT = os.environ.get("HM_WATCH_ROOT", "/data/incoming")


def get_user_by_email(email: str) -> Optional[dict]:
    _django_orm, m = _orm_modules()
    return m.User.objects.filter(email=str(email or "")).values().first()


def create_user(email: str, password: str, name: str) -> int:
    pw = generate_password_hash(password)
    _django_orm, m = _orm_modules()
    u = m.User.objects.create(
        email=str(email or ""),
        password_hash=pw,
        name=str(name or ""),
        created_at=dt.datetime.now(),
        default_league_id=None,
        video_clip_len_s=None,
    )
    return int(u.id)


def create_game(user_id: int, name: str, email: str):
    # Create dedicated dir: <watch_root>/<user_id>_<timestamp>_<rand>
    ts = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    token = secrets.token_hex(4)
    watch_root = os.environ.get("HM_WATCH_ROOT", WATCH_ROOT)
    d = Path(watch_root) / f"game_{user_id}_{ts}_{token}"
    d.mkdir(parents=True, exist_ok=True)
    # Create meta with user email
    try:
        (d / ".dirwatch_meta.json").write_text(
            f'{{"user_email":"{email}","created":"{dt.datetime.now().isoformat()}"}}\n'
        )
    except Exception:
        pass
    _django_orm, m = _orm_modules()
    g = m.Game.objects.create(
        user_id=int(user_id),
        name=str(name or ""),
        dir_path=str(d),
        status="new",
        created_at=dt.datetime.now(),
    )
    return int(g.id), str(d)


def read_dirwatch_state():
    state_path = Path("/var/lib/dirwatcher/state.json")
    try:
        import json

        return json.loads(state_path.read_text())
    except Exception:
        return {"processed": {}, "active": {}}


def send_email(to_addr: str, subject: str, body: str, from_addr: Optional[str] = None) -> None:
    # Use system sendmail preferred
    from_addr = from_addr or ("no-reply@" + os.uname().nodename)
    msg = (
        f"From: {from_addr}\nTo: {to_addr}\nSubject: {subject}\n"
        f"Content-Type: text/plain; charset=utf-8\n\n{body}\n"
    )
    import shutil as _sh
    import subprocess as _sp

    sendmail = _sh.which("sendmail")
    if sendmail:
        try:
            _sp.run([sendmail, "-t"], input=msg.encode("utf-8"), check=True)
            return
        except Exception:
            pass
    # no-op if email fails
    return


def get_user_video_clip_len_s(db_conn, user_id: Optional[int]) -> int:
    """
    Per-user clip length preference for timeline video clips.
    Defaults to 30 seconds when unset/unknown.
    """
    if not user_id:
        return 30
    try:
        del db_conn
        _django_orm, m = _orm_modules()
        v = (
            m.User.objects.filter(id=int(user_id))
            .values_list("video_clip_len_s", flat=True)
            .first()
        )
        try:
            iv = int(v) if v is not None else None
        except Exception:
            iv = None
        if iv in {15, 20, 30, 45, 60, 90}:
            return int(iv)
    except Exception:
        return 30
    return 30
