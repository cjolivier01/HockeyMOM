"""Bridge between PlayTracker camera controls and the Rust hm-ui sidecar."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from hmlib.log import logger


@dataclass
class _Control:
    name: str
    max_value: int
    value: int
    default_value: int


class HmUiProcess:
    """Owns the Rust hm-ui process and its JSON spec/state files."""

    def __init__(self, *, title: str = "HM UI", tmpdir: Optional[Path] = None) -> None:
        self.title = title
        self._tmpdir = (
            Path(tmpdir) if tmpdir is not None else Path(tempfile.mkdtemp(prefix="hm-ui-"))
        )
        self.spec_path = self._tmpdir / "spec.json"
        self.state_path = self._tmpdir / "state.json"
        self._windows: Dict[str, List[_Control]] = {}
        self._process: Optional[subprocess.Popen] = None
        self.stderr_path = self._tmpdir / "hm-ui.stderr.log"
        self._last_state_mtime_ns: Optional[int] = None
        self._last_action_seq = 0
        self._pending_actions: List[str] = []
        self._closed = False

    def add_window(self, name: str) -> None:
        self._windows.setdefault(name, [])
        self._write_spec()
        self.ensure_started()

    def add_slider(self, window_name: str, name: str, max_value: int, initial_value: int) -> None:
        controls = self._windows.setdefault(window_name, [])
        for control in controls:
            if control.name == name:
                control.max_value = max(1, int(max_value))
                control.value = self._clamp(initial_value, control.max_value)
                control.default_value = control.value
                break
        else:
            max_i = max(1, int(max_value))
            value_i = self._clamp(initial_value, max_i)
            controls.append(
                _Control(
                    name=name,
                    max_value=max_i,
                    value=value_i,
                    default_value=value_i,
                )
            )
        self._write_spec()

    def get_value(self, window_name: str, control_name: str) -> int:
        self.poll()
        control = self._find_control(window_name, control_name)
        return control.value

    def set_value(
        self, window_name: str, control_name: str, value: int, *, notify: bool = True
    ) -> bool:
        control = self._find_control(window_name, control_name)
        new_value = self._clamp(value, control.max_value)
        if not notify:
            control.default_value = new_value
        if new_value == control.value:
            if not notify:
                self._write_state()
                self._write_spec()
            return False
        control.value = new_value
        self._write_state()
        self._write_spec()
        if notify:
            return True
        return False

    def poll(self) -> bool:
        if self._process is not None and self._process.poll() is not None:
            logger.warning(
                "hm-ui exited with status %s; disabling Rust camera UI. stderr log: %s",
                self._process.returncode,
                self.stderr_path,
            )
            self._process = None
            self._closed = True
            return True
        if not self.state_path.exists():
            return False
        try:
            mtime_ns = self.state_path.stat().st_mtime_ns
        except OSError:
            return False
        if mtime_ns == self._last_state_mtime_ns:
            return False
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (OSError, json.JSONDecodeError) as ex:
            logger.warning("Failed to read hm-ui state: %s", ex)
            return False
        self._last_state_mtime_ns = mtime_ns
        changed = self._apply_state_values(state)
        action = state.get("last_action")
        if isinstance(action, dict):
            seq = int(action.get("seq") or 0)
            kind = str(action.get("kind") or "")
            if seq > self._last_action_seq and kind:
                self._last_action_seq = seq
                self._pending_actions.append(kind)
                changed = True
        return changed

    def consume_actions(self) -> List[str]:
        self.poll()
        actions = self._pending_actions
        self._pending_actions = []
        return actions

    def ensure_started(self) -> None:
        if self._closed:
            raise RuntimeError("hm-ui was closed")
        if self._process is not None and self._process.poll() is None:
            return
        cmd = self._resolve_command()
        if cmd is None:
            raise RuntimeError(
                "hm-ui binary not found. Build it with `bazelisk build //hm-ui:hm-ui` "
                "or `cargo build --manifest-path hm-ui/Cargo.toml`, or set HM_UI_BIN=/path/to/hm-ui."
            )
        self._write_spec()
        self._write_state()
        full_cmd = [
            *cmd,
            "--spec",
            str(self.spec_path),
            "--state",
            str(self.state_path),
            "--title",
            self.title,
        ]
        self._process = subprocess.Popen(
            full_cmd,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self.stderr_path.open("ab"),
        )
        time.sleep(0.05)
        if self._process.poll() is not None:
            stderr_tail = self._read_stderr_tail()
            raise RuntimeError(
                f"hm-ui exited during startup with status {self._process.returncode}. "
                f"stderr log: {self.stderr_path}. {stderr_tail}"
            )
        self._last_action_seq = 0

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2.0)
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def _apply_state_values(self, state: Dict) -> bool:
        windows = state.get("windows")
        if not isinstance(windows, dict):
            return False
        changed = False
        for window_name, values in windows.items():
            if not isinstance(values, dict):
                continue
            controls = self._windows.get(str(window_name), [])
            by_name = {control.name: control for control in controls}
            for control_name, raw_value in values.items():
                control = by_name.get(str(control_name))
                if control is None:
                    continue
                try:
                    new_value = self._clamp(int(raw_value), control.max_value)
                except (TypeError, ValueError):
                    continue
                if new_value != control.value:
                    control.value = new_value
                    changed = True
        return changed

    def _write_spec(self) -> None:
        payload = {
            "version": 1,
            "title": self.title,
            "subtitle": "Runtime tracking, stitch, and camera controls",
            "windows": [
                {
                    "name": window_name,
                    "controls": [
                        {
                            "name": control.name,
                            "max_value": control.max_value,
                            "value": control.value,
                            "default_value": control.default_value,
                        }
                        for control in controls
                    ],
                }
                for window_name, controls in self._windows.items()
            ],
        }
        self._write_json_atomic(self.spec_path, payload)

    def _write_state(self) -> None:
        payload = {
            "version": 1,
            "updated_ms": int(time.time() * 1000),
            "windows": {
                window_name: {control.name: control.value for control in controls}
                for window_name, controls in self._windows.items()
            },
            "last_action": None,
        }
        self._write_json_atomic(self.state_path, payload)

    def _read_stderr_tail(self, max_chars: int = 1200) -> str:
        try:
            text = self.stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        text = text.strip()
        if not text:
            return ""
        return text[-max_chars:]

    def _find_control(self, window_name: str, control_name: str) -> _Control:
        for control in self._windows[window_name]:
            if control.name == control_name:
                return control
        raise KeyError(control_name)

    @staticmethod
    def _clamp(value: int, max_value: int) -> int:
        return max(0, min(int(max_value), int(value)))

    @staticmethod
    def _write_json_atomic(path: Path, payload: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)

    @staticmethod
    def _resolve_command() -> Optional[List[str]]:
        env_bin = os.environ.get("HM_UI_BIN")
        if env_bin:
            return [env_bin]
        exe = shutil.which("hm-ui")
        if exe:
            return [exe]
        repo_root = Path(__file__).resolve().parents[2]
        runfiles_dir = os.environ.get("RUNFILES_DIR")
        if runfiles_dir:
            for candidate in (
                Path(runfiles_dir) / "hockeymom" / "hm-ui" / "hm-ui-bin",
                Path(runfiles_dir) / "hm-ui" / "hm-ui-bin",
            ):
                if candidate.exists() and os.access(candidate, os.X_OK):
                    return [str(candidate)]
        for candidate in (
            repo_root / "bazel-bin" / "hm-ui" / "hm-ui-bin",
            repo_root / "hm-ui" / "target" / "release" / "hm-ui",
            repo_root / "hm-ui" / "target" / "debug" / "hm-ui",
        ):
            if candidate.exists() and os.access(candidate, os.X_OK):
                return [str(candidate)]
        if os.environ.get("HM_UI_ALLOW_CARGO_RUN") == "1" and shutil.which("cargo"):
            return [
                "cargo",
                "run",
                "--locked",
                "--manifest-path",
                str(repo_root / "hm-ui" / "Cargo.toml"),
                "--",
            ]
        maybe_bazel_runfile = Path(sys.argv[0]).resolve().parent / "hm-ui" / "hm-ui-bin"
        if maybe_bazel_runfile.exists() and os.access(maybe_bazel_runfile, os.X_OK):
            return [str(maybe_bazel_runfile)]
        return None


class HmUiDialog:
    """CameraControlDialog-compatible handle backed by the hm-ui sidecar."""

    def __init__(
        self,
        manager: HmUiProcess,
        window_name: str,
        *,
        on_change: Optional[Callable[[int], None]] = None,
        initial_size: Tuple[int, int] = (900, 640),
        position: Optional[Tuple[int, int]] = None,
    ) -> None:
        del initial_size, position
        self.window_name = window_name
        self._manager = manager
        self._on_change = on_change

    def open(self) -> None:
        self._manager.add_window(self.window_name)

    def add_slider(self, name: str, max_value: int, initial_value: int) -> None:
        self._manager.add_slider(self.window_name, name, max_value, initial_value)

    def get_value(self, name: str) -> int:
        return self._manager.get_value(self.window_name, name)

    def set_value(self, name: str, value: int, *, notify: bool = True) -> None:
        changed = self._manager.set_value(self.window_name, name, value, notify=notify)
        if changed and self._on_change is not None:
            self._on_change(value)

    def show(self) -> None:
        if self._manager.poll() and self._on_change is not None:
            self._on_change(0)
        if self._manager.closed:
            raise RuntimeError("hm-ui was closed")

    def consume_actions(self) -> List[str]:
        return self._manager.consume_actions()
