from __future__ import annotations

# Back-compat shim for older installs/tests that reference `wsgi:application`.
try:
    from tools.webapp.hm_webapp.wsgi import application  # type: ignore
except Exception:  # pragma: no cover
    from hm_webapp.wsgi import application  # type: ignore
