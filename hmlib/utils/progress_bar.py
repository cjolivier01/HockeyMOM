"""Rich terminal progress bar and table utilities for HockeyMOM CLIs.

Provides a rich-based, stationary progress UI used by many command-line
tools in :mod:`hmlib`.

@see @ref ProgressBar "ProgressBar" for the main iteration helper.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import sys
import time
from collections import OrderedDict
from typing import Any, Callable, Iterator, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, ProgressColumn, TextColumn
from rich.progress import Progress as RichProgress
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

progress_out = sys.stderr
logging_out = sys.stdout

# Shared rich console used for both progress bars and log output.
RICH_CONSOLE = Console(file=progress_out, stderr=True)


class FramesColumn(ProgressColumn):
    """Right-aligned ``completed/total`` frames column with fixed width.

    @brief Shows iteration progress in terms of frames, reserving space for
    up to eight digits for both completed and total counts.
    """

    def render(self, task) -> Text:  # type: ignore[override]
        completed = int(task.completed)
        total = task.total
        if total is None or (isinstance(total, (int, float)) and total <= 0):
            total_str = " " * 8
        else:
            try:
                total_str = f"{int(total):>8}"
            except Exception:
                total_str = " " * 8
        text = f"{completed:>8}/{total_str}"
        return Text(text, style="white")


def _get_terminal_width():
    """Return the current terminal width in columns."""
    return shutil.get_terminal_size().columns




class CallbackStreamHandler(logging.StreamHandler):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        # Use the handler's own formatter to format the record
        message = self.format(record)
        # Call the callback with the formatted message
        self.callback(message)


class ScrollOutput:
    """Maintain a simple scrolling text buffer for log output.

    @brief Collects recent log lines and forwards them into a sink callback.
    This is used together with :class:`ProgressBar` to feed a scrolling log
    area inside the rich-based progress UI.

    @param lines Maximum number of lines to retain in the buffer.
    """

    def __init__(self, lines=10):
        self.capture = []
        self.lines = lines
        self._sink: Optional[Callable[[str], None]] = None

    def write(self, msg):
        if msg == "\n":
            return
        text = msg.rstrip("\n")
        if not text:
            return
        self.capture.append(text)
        if len(self.capture) > self.lines:
            self.capture.pop(0)
        if self._sink is not None:
            try:
                self._sink(text)
            except Exception:
                # Best-effort; avoid breaking logging on sink errors.
                RICH_CONSOLE.print(text)

    def flush(self):
        pass

    def reset(self):
        """Legacy no-op kept for API compatibility."""
        return

    def refresh(self):
        """Legacy no-op kept for API compatibility."""
        return

    def register_logger(self, logger) -> ScrollOutput:
        """
        Register a callback stream handler which
        redirects logging to this class' "write" function.
        """
        # Remove all existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create and add our custom handler
        callback_handler = CallbackStreamHandler(self.write)
        callback_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(callback_handler)
        return self

    # Internal: set by ProgressBar when using curses
    def _attach_curses_ui(self, ui: object | None):
        """Legacy no-op kept for API compatibility."""
        return

    # Optional: used by fallback ANSI renderer to align right border
    def _set_column_width(self, column_width: int | None):
        """Legacy no-op kept for API compatibility."""
        return

    def set_sink(self, sink: Callable[[str], None]) -> ScrollOutput:
        """Register a callback that receives each log line."""
        self._sink = sink
        return self


class ProgressBar:
    """Rich-based progress bar with a status table and scrolling log area.

    @brief Wraps an iterator (or manual counter) and renders a stationary
    progress UI using :mod:`rich`. The UI contains:

      - A two-column status table built from ``table_map``.
      - A progress bar with percentage and `completed/total` frames.
      - A scrolling log area fed by :class:`ScrollOutput`.

    @param table_map Initial key-value mapping displayed in the status table.
    @param total Total number of iterations expected.
    @param iterator Optional iterator to wrap; if provided, the bar is iterable.
    @param scroll_output Optional :class:`ScrollOutput` used for log lines.
    @param bar_length Explicit bar width (currently unused; reserved for future).
    @param update_rate Refresh interval in iterations.
    @param table_callback Optional callback to mutate ``table_map`` on refresh.
    @param use_curses Deprecated flag; kept for API compatibility only.
    """

    def __init__(
        self,
        table_map: OrderedDict[Any, Any] = OrderedDict(),
        total: int = 0,
        iterator: Optional[Iterator[Any]] = None,
        scroll_output: Optional[ScrollOutput] = None,
        bar_length: Optional[int] = None,
        update_rate: int = 1,
        table_callback: Optional[Callable] = None,
        use_curses: bool = True,
    ):
        self._total = total
        self._counter: int = 0
        self.table_map = table_map
        self.original_stdout = logging_out
        self.scroll_output = scroll_output
        self.iterator = iterator
        self.update_rate = update_rate
        self.table_callbacks: List[Callable] = []
        self.bar_length = bar_length
        self._use_curses_requested = use_curses  # Deprecated, no effect
        if not self.bar_length:
            self.terminal_width = _get_terminal_width()
            # Re-evaluate the terminal width periodically to handle resizes.
            self.terminal_width_interval = 250
        else:
            self.terminal_width = None
        # Delay initializing/rendering the rich UI until after a warm-up
        # period so that early startup text does not interfere with it.
        self._start_threshold: int = 50
        if table_callback is not None:
            self.add_table_callback(table_callback)

        # Rich progress UI setup
        self._console = RICH_CONSOLE
        self._progress = RichProgress(
            TextColumn("Progress:", justify="left", style="white"),
            BarColumn(bar_width=None, complete_style="bright_green", finished_style="bright_green"),
            FramesColumn(),
            console=self._console,
            transient=False,
        )
        description = ""  # description column is a fixed label
        total_value = self._total if self._total > 0 else None
        self._rich_task = self._progress.add_task(description=description, total=total_value)
        self._rich_started: bool = False
        self._live: Optional[Live] = None
        self._line_count: int = 0
        self._log_lines: List[str] = []
        self._log_max_lines: int = scroll_output.lines if scroll_output is not None else 4

        if self.scroll_output is not None:
            # Route ScrollOutput lines into this ProgressBar's log buffer.
            self.scroll_output.set_sink(self._append_log_line)

    def add_table_callback(self, callback: Callable):
        self.table_callbacks.append(callback)

    def set_iterator(self, iterator: Iterator[Any]) -> Iterator[Any]:
        self.iterator = iterator
        return iter(self)

    def _run_callbacks(self, table_map: OrderedDict[Any, Any]):
        for cb in self.table_callbacks:
            cb(table_map)

    def _format_description(self) -> str:
        """Format the current table_map into a single-line description."""
        if not self.table_map:
            return ""
        parts = [f"{key}: {value}" for key, value in self.table_map.items()]
        return " | ".join(parts)

    def _append_log_line(self, line: str) -> None:
        """Append a log line to the in-memory buffer used for the log area."""
        self._log_lines.append(line)
        if len(self._log_lines) > self._log_max_lines:
            self._log_lines = self._log_lines[-self._log_max_lines :]

    def _build_table(self) -> Table:
        """Build a two-column-of-entries rich Table from the current table_map.

        Layout:
          [label_1] [value_1]   [label_2] [value_2]
        """
        table = Table.grid(expand=True, padding=(0, 1))
        # Black background for labels and values; text white for readability.
        table.style = "white on black"
        table.add_column(justify="left", style="bold white", ratio=1)
        table.add_column(justify="right", style="white", ratio=1)
        table.add_column(justify="left", style="bold white", ratio=1)
        table.add_column(justify="right", style="white", ratio=1)

        items = list(self.table_map.items())
        if not items:
            return table

        half = (len(items) + 1) // 2
        left_items = items[:half]
        right_items = items[half:]

        # Pad right side to match left length
        while len(right_items) < len(left_items):
            right_items.append(("", ""))

        for (k1, v1), (k2, v2) in zip(left_items, right_items):
            table.add_row(str(k1), str(v1), str(k2), str(v2))
        return table

    def _build_log_table(self) -> Table:
        """Build a table containing the scrolling log area."""
        log_table = Table.grid(expand=True, padding=(0, 1))
        # Do not set a background style here so that per-line rich markup
        # colors and the enclosing panel's background can show through.
        log_table.style = ""
        log_table.add_column(justify="left")
        # Take the last N lines for display
        for line in self._log_lines[-self._log_max_lines :]:
            log_table.add_row(line)
        return log_table

    def _build_layout(self) -> Group:
        """Compose the status table, progress bar, and log area inside a single bordered panel."""
        status_table = self._build_table()
        log_table = self._build_log_table()

        status_panel = Panel(
            status_table,
            border_style="black",
            style="white on dark_blue",
            padding=(0, 1),
        )
        progress_panel = Panel(
            self._progress,
            border_style="black",
            style="white on grey35",
            padding=(0, 1),
        )
        log_panel = Panel(
            log_table,
            border_style="black",
            # Gray background for the scrolling log area; individual lines
            # can still use rich markup colors on top of this.
            style="on grey23",
            padding=(0, 1),
        )

        # Horizontal ASCII separators between sections
        sep = Rule(style="black")

        body = Group(status_panel, sep, progress_panel, sep, log_panel)
        outer = Panel(
            body,
            border_style="black",
            padding=(0, 1),
        )
        return outer

    @property
    def total(self) -> int:
        return self._total

    @property
    def current(self) -> int:
        return self._counter

    def _ensure_rich_started(self) -> None:
        if not self._rich_started:
            # Start the rich.Progress and Live render loop the first time we render.
            self._progress.start()
            self._live = Live(
                self._build_layout(),
                console=self._console,
                screen=True,
                auto_refresh=False,
            )
            self._live.start()
            self._rich_started = True

    def _render_rich(self, final: bool = False) -> None:
        """Render or update the rich progress bar and surrounding layout."""
        self._ensure_rich_started()
        # Allow callers to update table_map before we format the description/table.
        self._run_callbacks(self.table_map)
        # Progress bar label is a fixed prefix; tabular values are shown only
        # in the status table above, not in the bar description.
        description = ""
        if self._total > 0:
            completed = min(self._counter, self._total)
            total_val = self._total
        else:
            completed = self._counter
            total_val = None
        self._progress.update(self._rich_task, completed=completed, total=total_val, description=description)
        if final and total_val is not None:
            # Ensure we show a fully-complete bar if requested.
            self._progress.update(self._rich_task, completed=total_val, description=description)

        # Render the composed layout (status table + progress bar + log panel)
        # in-place using Live so the UI appears stationary in the terminal.
        renderable = self._build_layout()
        if self._live is not None:
            self._live.update(renderable, refresh=True)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = None
        # Drive the wrapped iterator when provided; otherwise behave as a manual counter.
        if self.iterator is not None:
            try:
                next_item = next(self.iterator)
            except StopIteration:
                # Final update on completion
                self.refresh(final=True)
                self.close()
                raise
        else:
            if self._counter >= self._total:
                self.refresh(final=True)
                self.close()
                raise StopIteration()

        if not self.bar_length and self._counter % self.terminal_width_interval == 0:
            self.terminal_width = _get_terminal_width()

        if self._counter % self.update_rate == 0:
            self.refresh()
            if next_item is None:
                time.sleep(0.001)  # Simulating work by sleeping

        self._counter += 1
        return next_item

    def close(self):
        # Tear down the rich progress / Live context if it was started.
        if getattr(self, "_rich_started", False):
            try:
                if self._live is not None:
                    self._live.stop()
            except Exception:
                pass
            try:
                self._progress.stop()
            except Exception:
                pass
            self._rich_started = False

    def refresh(self, final: bool = False):
        # Avoid starting the rich UI until after the warm-up threshold so
        # that early startup output does not interfere with the layout.
        if not getattr(self, "_rich_started", False) and self._counter < self._start_threshold:
            # Still allow table callbacks to keep table_map up to date.
            self._run_callbacks(self.table_map)
            return
        # Single rich-backed rendering path once started
        self._render_rich(final=final)

def convert_seconds_to_hms(total_seconds: Any) -> str:
    hours = int(total_seconds // 3600)  # Calculate the number of hours
    minutes = int((total_seconds % 3600) // 60)  # Calculate the remaining minutes
    seconds = int(total_seconds % 60)  # Calculate the remaining seconds

    # Format the time in "HH:MM:SS" format
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def convert_hms_to_seconds(timestr: str) -> float:
    """
    Convert a time string in HH:MM:ss.ssss format to total seconds as a float.
    'HH:' may be omitted and 'MM' can be more than 59, as can 'ss' be more than 59.

    Parameters:
    - timestr (str): Time string in "HH:MM:ss.ssss" or "MM:ss.ssss" format.

    Returns:
    - float: Total seconds represented by the input string.
    """
    # Split the time string by colon
    parts = timestr.split(":")

    # Initialize seconds, minutes, and hours
    seconds = 0.0
    minutes = 0
    hours = 0

    # Depending on the number of components, assign values
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    elif len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
    elif len(parts) == 1:
        seconds = float(parts[0])
    else:
        raise ValueError("Invalid time format")

    # Calculate total seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


# Example usage
if __name__ == "__main__":
    table_map = OrderedDict({"Key1": "Value1", "Key2": "Value2", "Key3": "Value3"})
    # Redirect stdout to scroll output handler
    scroll_output = ScrollOutput()
    progress_bar = ProgressBar(total=20, table_map=table_map, scroll_output=scroll_output)

    # with contextlib.redirect_stdout(scroll_output):
    with contextlib.redirect_stderr(scroll_output):
        counter = 0
        for _ in progress_bar:
            print(f"Simulated stdout message for iteration {counter}", file=sys.stderr)
            counter += 1
