import sys
import threading
import time
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


def _ensure_repo_on_path():
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def should_sideband_queue_block_on_put_until_capacity_available():
    _ensure_repo_on_path()
    from hmlib.utils.containers import SidebandQueue

    warn_after = 0.05
    durations = []
    stderr_buffer = StringIO()

    with redirect_stderr(stderr_buffer):
        queue = SidebandQueue(
            name="test-sideband-queue",
            warn_after=warn_after,
            repeat_warn=False,
            max_size=1,
        )

        queue.put("first")

        def producer():
            start = time.time()
            queue.put("second")
            durations.append(time.time() - start)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        time.sleep(warn_after * 2)
        assert thread.is_alive(), "Producer should still be blocked waiting for space"

        assert queue.get() == "first"

        thread.join(timeout=1.0)
        assert not thread.is_alive(), "Producer should finish once space is available"

        assert queue.get() == "second"

    assert durations, "Producer thread should have recorded a wait duration"
    assert durations[0] >= warn_after - 0.01

    stderr_output = stderr_buffer.getvalue()
    expected_wait_fragment = f"waiting >= {warn_after:.2f}s for free slot"
    assert expected_wait_fragment in stderr_output
    assert "resumed after waiting" in stderr_output
    assert "to put item" in stderr_output
