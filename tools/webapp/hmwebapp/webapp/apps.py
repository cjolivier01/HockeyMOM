from django.apps import AppConfig


class WebAppConfig(AppConfig):
    default_auto_field = "django.db.models.AutoField"
    name = "hmwebapp.webapp"

    def ready(self) -> None:  # pragma: no cover - side-effect wiring
        from . import db_init
        import os

        if os.environ.get("HM_WEBAPP_SKIP_DB_INIT") == "1":
            return
        try:
            db_init.init_db()
        except Exception:
            # DB init is best-effort; failures should not crash the app
            pass

