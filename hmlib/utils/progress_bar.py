"""Rich terminal progress bar and table utilities for HockeyMOM CLIs.

Provides ANSI-based and optional curses-based status views used by many
command-line tools in :mod:`hmlib`.

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
from rich.panel import Panel
from rich.table import Table
from rich.progress import BarColumn, Progress as RichProgress, TaskProgressColumn, TextColumn, ProgressColumn
from rich.live import Live
from rich.rule import Rule
from rich.text import Text

# Optional curses support
try:
    # import curses  # type: ignore
    # import queue
    # import threading

    raise Exception()
except Exception:  # pragma: no cover - curses may not be available
    curses = None  # type: ignore
    queue = None  # type: ignore
    threading = None  # type: ignore

progress_out = sys.stderr
logging_out = sys.stdout

# Shared rich console used for both progress bars and log output.
RICH_CONSOLE = Console(file=progress_out, stderr=True)


class FramesColumn(ProgressColumn):
    """Right-aligned 'completed/total' frames column with fixed width."""

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
    width = shutil.get_terminal_size().columns
    return width


def write_dict_in_columns(data_dict, out_file, table_width: int) -> int:
    lines_out = 0
    # Treat table_width as the full box width including three vertical borders.
    # Each column then gets roughly half of the interior width.
    column_width = max(1, (table_width - 3) // 2)
    # Create list of formatted key-value strings
    kv_pairs = [f"{key}: {value}"[:column_width] for key, value in data_dict.items()]

    # Ensure the list has an even number of elements
    if len(kv_pairs) % 2 != 0:
        kv_pairs.append("")

    # Calculate the number of rows needed for two columns
    num_rows = len(kv_pairs) // 2

    # Prepare horizontal border and empty row format
    border_top = "\x1b[2K┌" + "─" * column_width + "┬" + "─" * column_width + "┐\n"
    row_format = "\x1b[2K│{:<" + str(column_width) + "}│{:<" + str(column_width) + "}│\n"
    border_bottom = "\x1b[2K└" + "─" * column_width + "┴" + "─" * column_width + "┘\n"

    # Print the top border
    out_file.write(border_top)
    lines_out += 1

    # Print each row
    for i in range(num_rows):
        left = kv_pairs[i]
        right = kv_pairs[i + num_rows]
        out_file.write(row_format.format(left, right))
        lines_out += 1

    # Print the bottom border
    out_file.write(border_bottom)
    lines_out += 1
    return lines_out


def _isatty() -> bool:
    try:
        return progress_out.isatty()
    except Exception:
        return False


class _CursesUI:
    """
    Thin wrapper that manages a curses screen with a fixed
    header for progress + table and a scrolling log window below.

    All curses drawing occurs in the thread that calls `render()`/`drain_logs()`;
    producers must enqueue log lines via `enqueue_log()` to avoid cross-thread curses calls.
    """

    def __init__(self):
        if curses is None:
            raise RuntimeError("curses is not available")
        # Curses init
        self._stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        try:
            self._stdscr.keypad(True)
        except Exception:
            pass
        try:
            curses.curs_set(0)
        except Exception:
            pass
        # Colors
        self._has_colors = False
        self._hdr_attr = 0
        try:
            if curses.has_colors():
                curses.start_color()
                try:
                    curses.use_default_colors()
                except Exception:
                    pass
                # 1: white on magenta (purple background)
                try:
                    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_MAGENTA)
                    self._hdr_attr = curses.color_pair(1)
                    self._has_colors = True
                except Exception:
                    self._hdr_attr = 0
        except Exception:
            self._hdr_attr = 0
        self._height, self._width = self._stdscr.getmaxyx()
        # Windows; created on first render when we know header height
        self._header_win = None
        self._log_win = None
        # Thread-safe queue for log lines
        self._log_queue = queue.Queue() if queue is not None else None
        self._lock = threading.Lock() if threading is not None else None
        self._last_header_height = 0

    def close(self):
        # Tear down curses; safe to call multiple times
        try:
            if curses is None:
                return
            try:
                if self._stdscr is not None:
                    self._stdscr.keypad(False)
            except Exception:
                pass
            curses.echo()
            curses.nocbreak()
            curses.endwin()
        except Exception:
            # Best-effort cleanup
            pass

    # -------- Layout management --------
    def _ensure_layout(self, header_height: int):
        # Resize detection
        h, w = self._stdscr.getmaxyx()
        if (h, w) != (self._height, self._width):
            self._height, self._width = h, w
            try:
                curses.resizeterm(h, w)
            except Exception:
                pass
        # Recreate windows if needed
        if (
            self._header_win is None
            or self._log_win is None
            or header_height != self._last_header_height
        ):
            self._last_header_height = header_height
            if self._header_win is not None:
                del self._header_win
            if self._log_win is not None:
                del self._log_win
            header_h = min(max(1, header_height), max(1, self._height - 1))
            log_h = max(1, self._height - header_h)
            self._header_win = curses.newwin(header_h, self._width, 0, 0)
            # Apply purple background to header area if colors available
            try:
                if self._hdr_attr:
                    self._header_win.bkgd(" ", self._hdr_attr)
            except Exception:
                pass
            self._log_win = curses.newwin(log_h, self._width, header_h, 0)
            try:
                self._log_win.scrollok(True)
                self._log_win.idlok(True)
            except Exception:
                pass

    # -------- Rendering --------
    def _format_table_lines(self, data_dict: OrderedDict[Any, Any], table_width: int) -> List[str]:
        # table_width is the full box width including three vertical borders.
        column_width = max(1, (table_width - 3) // 2)
        kv_pairs = [f"{key}: {value}"[:column_width] for key, value in data_dict.items()]
        if len(kv_pairs) % 2 != 0:
            kv_pairs.append("")
        num_rows = len(kv_pairs) // 2
        lines: List[str] = []
        lines.append("┌" + "─" * column_width + "┬" + "─" * column_width + "┐")
        for i in range(num_rows):
            left = kv_pairs[i]
            right = kv_pairs[i + num_rows]
            lines.append(f"│{left:<{column_width}}│{right:<{column_width}}│")
        lines.append("└" + "─" * column_width + "┴" + "─" * column_width + "┘")
        return lines

    def render(
        self,
        table_map: OrderedDict[Any, Any],
        current: int,
        total: int,
        percent: float,
    ):
        # Use an effective table width that is 80% of the current terminal
        # width, capped at 120 columns. Keep it odd so that two columns plus
        # three borders line up cleanly.
        effective_width = min(max(int(self._width * 0.8), 20), 120)
        if effective_width % 2 == 0:
            effective_width -= 1
        # Compute header height: table box + 1 progress line
        table_lines = self._format_table_lines(table_map, effective_width)
        header_height = len(table_lines) + 1
        self._ensure_layout(header_height)

        # Draw header
        # Fill header area with background color, then draw
        try:
            if self._hdr_attr:
                self._header_win.bkgd(" ", self._hdr_attr)
        except Exception:
            pass
        self._header_win.erase()
        maxw = effective_width
        y = 0
        # Top border
        if table_lines:
            try:
                self._header_win.addnstr(y, 0, table_lines[0], maxw)
            except Exception:
                pass
            y += 1
        # Rows (without bottom border)
        for line in table_lines[1:-1]:
            try:
                self._header_win.addnstr(y, 0, line, maxw)
            except Exception:
                pass
            y += 1

        # Progress bar line inside the same 2-column box
        colw = max(1, (effective_width - 3) // 2)
        total_content = colw * 2
        prefix = "Progress: "
        suffix = f" {percent:.1f}% Complete {current}/{total}"
        reserved = len(prefix) + len(suffix) + 2  # two for bar delimiters
        bar_len = max(1, total_content - reserved)
        filled_length = 0 if total <= 0 else int(bar_len * current // max(1, total))
        bar = "█" * filled_length + "-" * (bar_len - filled_length)
        content = f"{prefix}|{bar}|{suffix}"
        content = content[:total_content]
        content = f"{content:<{total_content}}"
        left = content[:colw]
        right = content[colw : colw * 2]
        progress_line = f"│{left}│{right}│"
        try:
            self._header_win.addnstr(y, 0, progress_line, maxw)
        except Exception:
            pass
        y += 1

        # Bottom border to close the box
        if table_lines:
            try:
                self._header_win.addnstr(y, 0, table_lines[-1], maxw)
            except Exception:
                pass
        try:
            self._header_win.noutrefresh()
        except Exception:
            pass

    # -------- Logs --------
    def enqueue_log(self, msg: str):
        if self._log_queue is None:
            return
        # Normalize to lines; avoid pushing empty newlines alone
        for part in msg.splitlines():
            if part:
                self._log_queue.put(part)

    def drain_logs(self):
        if self._log_win is None or self._log_queue is None:
            return
        maxw = max(1, self._width - 1)
        drained = False
        while True:
            try:
                line = self._log_queue.get_nowait()
            except Exception:
                break
            drained = True
            try:
                self._log_win.addnstr(line + "\n", maxw)
            except Exception:
                # As a fallback, try to ensure we progress even on encoding issues
                try:
                    safe = line.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
                    self._log_win.addnstr(safe + "\n", maxw)
                except Exception:
                    pass
        if drained:
            try:
                self._log_win.noutrefresh()
            except Exception:
                pass
        try:
            curses.doupdate()
        except Exception:
            pass


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

    @param lines: Maximum number of lines to retain in the buffer.
    """

    def __init__(self, lines=10):
        self.capture = []
        self.lines = lines
        self._curses_ui: Optional[_CursesUI] = None
        self._column_width: Optional[int] = None
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
        # In curses mode, reset is a no-op
        if self._curses_ui is not None:
            return
        progress_out.write(f"\x1b[0G\x1b[{self.lines}A")
        progress_out.flush()

    def refresh(self):
        if self._curses_ui is not None:
            # Rendering handled by ProgressBar via CursesUI.drain_logs()
            return
        for i in reversed(range(self.lines)):
            cap_count = len(self.capture)
            if i < cap_count:
                line = self.capture[cap_count - i - 1]
            else:
                line = ""
            # Use a single full-width log line; avoid a vertical divider in the middle
            inner_width = self._column_width
            if inner_width is None or inner_width <= 0:
                try:
                    # Fallback to current terminal width minus borders
                    inner_width = max(1, _get_terminal_width() - 2)
                except Exception:
                    inner_width = 78
            # Normalize and truncate the line to fit
            text = line.replace("\n", " ")
            if len(text) > inner_width:
                text = text[:inner_width]
            progress_out.write(f"\x1b[2K│{text:<{inner_width}}│\n")
        progress_out.flush()

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
    def _attach_curses_ui(self, ui: _CursesUI | None):
        self._curses_ui = ui
        # When using curses, we don't manage ANSI box widths
        if ui is not None:
            self._column_width = None

    # Optional: used by fallback ANSI renderer to align right border
    def _set_column_width(self, column_width: int | None):
        # Treat column_width as the total interior width (without outer borders)
        self._column_width = column_width if column_width and column_width > 0 else None

    def set_sink(self, sink: Callable[[str], None]) -> ScrollOutput:
        """Register a callback that receives each log line."""
        self._sink = sink
        return self


class ProgressBar:
    """
    This class wraps an iterator (or manual counter) and renders progress
    alongside arbitrary key-value statistics in a terminal-friendly format.

    @param table_map: Initial key-value mapping displayed in the header table.
    @param total: Total number of iterations expected.
    @param iterator: Optional iterator to wrap; if provided, the bar is iterable.
    @param scroll_output: Optional :class:`ScrollOutput` used for log lines.
    @param bar_length: Explicit bar width; inferred from terminal when ``None``.
    @param update_rate: Refresh interval in iterations.
    @param table_callback: Optional callback to mutate ``table_map`` on refresh.
    @param use_curses: Enable curses-based UI when available.
    @see @ref hmlib.utils.profiler.HmProfiler "HmProfiler" for complementary profiling.
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
        self._use_curses_requested = use_curses
        # Legacy curses support is disabled; ProgressBar now uses rich.Progress
        # for all rendering, but we retain these attributes for API stability.
        self._curses_ui: Optional[_CursesUI] = None
        if not self.bar_length:
            self.terminal_width = _get_terminal_width()
            # Re-evaluate the terminal width periodically to handle resizes.
            self.terminal_width_interval = 250
        else:
            self.terminal_width = None
        # Delay initializing/rendering the rich UI until after a warm-up
        # period so that early startup text does not interfere with it.
        self._start_threshold: int = 250
        if table_callback is not None:
            self.add_table_callback(table_callback)

        # Rich progress UI setup
        self._console = RICH_CONSOLE
        self._progress = RichProgress(
            TextColumn("Progress:", justify="left", style="white"),
            BarColumn(bar_width=None, complete_style="bright_green", finished_style="bright_green"),
            TaskProgressColumn(),
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

    def _maybe_init_curses(self):
        # Legacy no-op: curses UI has been replaced by rich.Progress.
        return

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

    def _refresh_fallback(self):
        # Compute a column width consistent with the table box
        table_width = self._get_bar_width()
        if self.scroll_output is not None:
            # Inform ScrollOutput so it can close the box on the right
            try:
                inner_width = max(1, table_width - 2)
                self.scroll_output._set_column_width(inner_width)
            except Exception:
                pass

        if self._counter > 0:
            if self.scroll_output is not None:
                self.scroll_output.reset()
            progress_out.write(f"\x1b[0G\x1b[{self._line_count}A")
        self._line_count = 0
        self.print_table()
        self.print_progress_bar()
        if self.scroll_output is not None:
            self.scroll_output.refresh()

    def _get_bar_width(self):
        # When bar_length is not provided, scale the bar to 80% of the current
        # terminal width with a hard cap of 120 columns so it doesn't dominate
        # very wide terminals.
        if self.bar_length:
            return self.bar_length
        # terminal_width is kept up-to-date in __next__; guard defensively here.
        if self.terminal_width is None:
            try:
                self.terminal_width = _get_terminal_width()
            except Exception:
                self.terminal_width = 80
        width = int(self.terminal_width * 0.8)
        width = min(max(width, 20), 120)
        # Ensure an odd width so that two equal columns plus three borders
        # add up exactly to this width.
        if width % 2 == 0:
            width -= 1
        if width < 5:
            width = 5
        return width

    def print_progress_bar(self):
        # Build a progress line sized to the table's two-column width and close with right border
        if self._total - self._counter < self.update_rate:
            percent = 100
        else:
            percent = (self._counter / self._total) * 100

        table_width = self._get_bar_width()
        colw = max(1, (table_width - 3) // 2)
        total_content = colw * 2

        # Estimate bar length to fit within content width after fixed text
        suffix = f" {percent:.1f}% Complete {self._counter}/{self._total}"
        prefix = "Progress: "
        # Reserve 2 characters for bar delimiters '|' '|'
        reserved = len(prefix) + len(suffix) + 2
        bar_len = max(1, total_content - reserved)

        # Compute filled length and build bar
        try:
            filled_length = int(bar_len * self._counter // max(1, self._total))
        except Exception:
            filled_length = 0
        bar = "█" * filled_length + "-" * (bar_len - filled_length)
        content = f"{prefix}|{bar}|{suffix}"
        # Truncate if still too long and pad to fit
        content = content[:total_content]
        content = f"{content:<{total_content}}"

        # Split across two columns and add borders to match table width
        left = content[:colw]
        right = content[colw : colw * 2]
        # Progress bar line
        progress_out.write(f"\r\x1b[2K│{left}│{right}│\n")
        # Horizontal separator below the progress bar to visually separate
        # it from the scrolling log area.
        bottom = "└" + "─" * colw + "┴" + "─" * colw + "┘"
        progress_out.write(f"\x1b[2K{bottom}\n")
        progress_out.flush()
        self._line_count += 2

    def print_table(self):
        self._line_count += write_dict_in_columns(
            self.table_map, progress_out, self._get_bar_width()
        )
        progress_out.flush()

    @contextlib.contextmanager
    def stdout_redirect(self):
        # Redirect stdout to the scroll output handler (curses-aware)
        with contextlib.redirect_stdout(self.scroll_output):
            yield


class ProgressBarWith:
    def __init__(self, dataloader):
        self._len = len(dataloader)
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return None


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
