import io
import sys
import tempfile
from contextlib import redirect_stdout
from importlib import util
from pathlib import Path


def should_video_clipper_blink_text_at_event_moment_from_timestamps():
    spec = util.spec_from_file_location("video_clipper_mod", "hmlib/cli/video_clipper.py")
    video_clipper = util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(video_clipper)  # type: ignore

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        ts_path = base / "times.txt"
        # Format: start end event_moment
        ts_path.write_text("00:00:10 00:00:15 00:00:12\n", encoding="utf-8")

        out = io.StringIO()
        argv_prev = sys.argv[:]
        try:
            sys.argv = [
                "video_clipper.py",
                "--input",
                str(base / "dummy.mp4"),
                "--timestamps",
                str(ts_path),
                "--temp-dir",
                str(base / "temp"),
                "--dry-run",
                "--notk",
                "--blink-event-text",
                "--blink-event-label",
                "SOG",
                "Test Label",
            ]
            with redirect_stdout(out):
                video_clipper.main()
        finally:
            sys.argv = argv_prev

        txt = out.getvalue()
        assert "fontcolor=yellow" in txt
        assert "fontsize=135" in txt
        assert "x=10:y=10" in txt
        assert "between(t,1.000,3.000)" in txt


def should_video_clipper_delay_goal_blink_by_one_second():
    spec = util.spec_from_file_location("video_clipper_mod", "hmlib/cli/video_clipper.py")
    video_clipper = util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(video_clipper)  # type: ignore

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        ts_path = base / "times.txt"
        # Format: start end event_moment
        ts_path.write_text("00:00:10 00:00:15 00:00:12\n", encoding="utf-8")

        out = io.StringIO()
        argv_prev = sys.argv[:]
        try:
            sys.argv = [
                "video_clipper.py",
                "--input",
                str(base / "dummy.mp4"),
                "--timestamps",
                str(ts_path),
                "--temp-dir",
                str(base / "temp"),
                "--dry-run",
                "--notk",
                "--blink-event-text",
                "--blink-event-label",
                "GOAL",
                "Test Label",
            ]
            with redirect_stdout(out):
                video_clipper.main()
        finally:
            sys.argv = argv_prev

        txt = out.getvalue()
        # Event is at t=2s relative to clip start; GOAL should begin blinking at t=3s.
        assert "between(t,3.000,5.000)" in txt
