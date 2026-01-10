from __future__ import annotations

# Back-compat shim for older installs/tests that reference `django_settings`.
#
# Canonical Django project settings now live in `hm_webapp.settings`.
try:
    from tools.webapp.hm_webapp.settings import *  # type: ignore
except Exception:  # pragma: no cover
    from hm_webapp.settings import *  # type: ignore
