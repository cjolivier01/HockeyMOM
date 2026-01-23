import json
import os
import secrets
from pathlib import Path


def load_or_create_app_secret(*, instance_dir: Path, config_path: Path) -> str:
    env = os.environ.get("HM_WEBAPP_SECRET")
    if env:
        return str(env)

    # Best-effort: allow config.json to pin the secret for multi-worker deployments.
    try:
        cfg_path = os.environ.get("HM_DB_CONFIG", str(config_path))
        with open(cfg_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        for k in ("app_secret", "secret_key", "webapp_secret"):
            v = cfg.get(k)
            if v:
                return str(v)
    except Exception:
        pass

    # Fall back to a persistent secret under instance/ so all gunicorn workers share it.
    try:
        instance_dir.mkdir(parents=True, exist_ok=True)
        secret_path = instance_dir / "app_secret.txt"
        if secret_path.exists():
            s = secret_path.read_text(encoding="utf-8").strip()
            if s:
                return s
        s = secrets.token_hex(32)
        secret_path.write_text(s + "\n", encoding="utf-8")
        try:
            os.chmod(secret_path, 0o600)
        except Exception:
            pass
        return s
    except Exception:
        # Last resort: non-persistent secret (may break sessions across workers).
        return secrets.token_hex(16)
