from __future__ import annotations

from django.core.management.base import BaseCommand

from ... import db_init


class Command(BaseCommand):
    help = "Initialize the HM WebApp database schema (users/games/teams/etc.)."

    def handle(self, *args, **options):
        self.stdout.write("Initializing HM WebApp schema via db_init.init_db()...")
        db_init.init_db()
        self.stdout.write(self.style.SUCCESS("HM WebApp schema initialized (or already up to date)."))

