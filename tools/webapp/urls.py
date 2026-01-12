from __future__ import annotations

# Back-compat shim for older installs/tests that reference `urls.urlpatterns`.
try:
    from tools.webapp.hm_webapp.urls import urlpatterns  # type: ignore
except Exception:  # pragma: no cover
    from hm_webapp.urls import urlpatterns  # type: ignore
