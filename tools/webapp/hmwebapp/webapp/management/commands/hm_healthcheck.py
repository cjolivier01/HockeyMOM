from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import connections


class Command(BaseCommand):
    help = "Run a simple health check against the default database connection."

    def handle(self, *args, **options):
        conn = connections["default"]
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        except Exception as exc:  # pragma: no cover - best-effort util
            self.stderr.write(self.style.ERROR(f"Healthcheck failed: {exc!r}"))
            raise SystemExit(1)
        self.stdout.write(self.style.SUCCESS("Healthcheck OK: DB reachable and responding."))

