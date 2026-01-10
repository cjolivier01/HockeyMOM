from __future__ import annotations

import os

from django.core.wsgi import get_wsgi_application


def _default_settings_module() -> str:
    return "tools.webapp.hm_webapp.settings" if __name__.startswith("tools.webapp.") else "hm_webapp.settings"


os.environ.setdefault("DJANGO_SETTINGS_MODULE", _default_settings_module())

application = get_wsgi_application()

# Best-effort DB/schema bootstrap (mirrors legacy startup behavior).
if os.environ.get("HM_WEBAPP_SKIP_DB_INIT") != "1":  # pragma: no cover
    try:
        try:
            from tools.webapp import app as logic  # type: ignore
        except Exception:
            import app as logic  # type: ignore

        logic.init_db()
    except Exception:
        raise

