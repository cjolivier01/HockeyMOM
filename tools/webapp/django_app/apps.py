from __future__ import annotations

from django.apps import AppConfig


class HMWebappConfig(AppConfig):
    if __name__.startswith("tools.webapp."):
        name = "tools.webapp.django_app"
    else:
        name = "django_app"
    label = "hm_webapp"
    default_auto_field = "django.db.models.AutoField"
