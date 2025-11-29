from __future__ import annotations

"""Rich terminal progress bar and table utilities for HockeyMOM CLIs.

Provides ANSI-based and optional curses-based status views used by many
command-line tools in :mod:`hmlib`.

@see @ref ProgressBar "ProgressBar" for the main iteration helper.
"""

import contextlib
import logging
import os
import shutil
import sys
import threading as _threading
import time
from collections import OrderedDict, deque
from typing import Any, Callable, Iterator, List, Optional

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


def _get_terminal_width():
    width = shutil.get_terminal_size().columns
    return width


def write_dict_in_columns(data_dict, out_file, table_width: int) -> int:
    lines_out = 0
    column_width = (table_width - 2) // 2
    # Create list of formatted key-value strings
    kv_pairs = [f"{key}: {value}"[:column_width] for key, value in data_dict.items()]

    # Ensure the list has an even number of elements
    if len(kv_pairs) % 2 != 0:
        kv_pairs.append("")

    # Calculate the number of rows needed for two columns
    num_rows = len(kv_pairs) // 2

    # Prepare horizontal border and empty row format
    border_top = "\x1b[2K┌" + "─" * column_width + "┬" + "─" * column_width + "┐\n"
    row_format = (
        "\x1b[2K│{:<" + str(column_width) + "}│{:<" + str(column_width) + "}│\n"
    )
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
                    self._header_win.bkgd(' ', self._hdr_attr)
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
        column_width = max(1, (table_width - 2) // 2)
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
        # Compute header height: table box + 1 progress line
        table_lines = self._format_table_lines(table_map, self._width)
        header_height = len(table_lines) + 1
        self._ensure_layout(header_height)

        # Draw header
        # Fill header area with background color, then draw
        try:
            if self._hdr_attr:
                self._header_win.bkgd(' ', self._hdr_attr)
        except Exception:
            pass
        self._header_win.erase()
        maxw = self._width - 1
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
        colw = max(1, (self._width - 2) // 2)
        total_content = colw * 2
        prefix = "Progress: "
        suffix = f" {percent:.1f}% Complete {current}/{total}"
        reserved = len(prefix) + len(suffix) + 2  # two for bar delimiters
        bar_len = max(1, total_content - reserved)
        filled_length = 0 if total <= 0 else int(bar_len * current // max(1, total))
        bar = "█" * filled_length + "-" * (bar_len - filled_length)
        content = f"{prefix}|{bar}|{suffix}"
        content = content[: total_content]
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

    def __init__(self, lines=4):
        self.capture = []
        self.lines = lines
        self._curses_ui: Optional[_CursesUI] = None
        self._column_width: Optional[int] = None

    def write(self, msg):
        # If curses is active, enqueue logs and return immediately
        if self._curses_ui is not None:
            # Avoid calling curses from writer threads; just enqueue
            self._curses_ui.enqueue_log(msg)
            return
        if msg == "\n":
            return
        self.capture.append(msg)
        if len(self.capture) > self.lines:
            self.capture.pop(0)

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
                line.replace("\n", "\r")
            else:
                line = ""
            if self._column_width is not None and self._column_width > 0:
                colw = self._column_width
                # Fit content across two columns; include middle divider to match header box width
                text = line
                left = text[:colw]
                right = text[colw : colw * 2]
                progress_out.write(
                    f"\x1b[2K│{left:<{colw}}│{right:<{colw}}│\n"
                )
            else:
                progress_out.write(f"\x1b[2K│" + line + "│\n")
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
        self._column_width = column_width if column_width and column_width > 0 else None


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
        self._curses_ui: Optional[_CursesUI] = None
        if not self.bar_length:
            self.terminal_width = _get_terminal_width()
            self.terminal_width_interval = 100
        else:
            self.terminal_width = None
        if table_callback is not None:
            self.add_table_callback(table_callback)

    def add_table_callback(self, callback: Callable):
        self.table_callbacks.append(callback)

    def set_iterator(self, iterator: Iterator[Any]) -> Iterator[Any]:
        self.iterator = iterator
        return iter(self)

    def _run_callbacks(self, table_map: OrderedDict[Any, Any]):
        for cb in self.table_callbacks:
            cb(table_map)

    @property
    def total(self) -> int:
        return self._total

    @property
    def current(self) -> int:
        return self._counter

    def __iter__(self):
        return self

    def __next__(self):
        next_item = None
        if self.iterator is not None:
            try:
                next_item = next(self.iterator)
            except StopIteration:
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
            self._run_callbacks(table_map=self.table_map)
            self.refresh()
            if next_item is None:
                time.sleep(0.001)  # Simulating work by sleeping

        self._counter += 1
        return next_item

    def _maybe_init_curses(self):
        if self._curses_ui is not None:
            return
        if not self._use_curses_requested:
            return
        if curses is None or not _isatty():
            return
        try:
            self._curses_ui = _CursesUI()
            if self.scroll_output is not None:
                self.scroll_output._attach_curses_ui(self._curses_ui)
        except Exception:
            # Fallback on failure
            self._curses_ui = None

    def close(self):
        if self._curses_ui is not None:
            try:
                # Final drain of logs and render one last time
                self._curses_ui.render(
                    self.table_map, self._counter, max(1, self._total), 100.0
                )
                self._curses_ui.drain_logs()
            except Exception:
                pass
            self._curses_ui.close()
            self._curses_ui = None
            if self.scroll_output is not None:
                self.scroll_output._attach_curses_ui(None)

    def refresh(self, final: bool = False):
        # Attempt to use curses if available
        if self._curses_ui is None:
            self._maybe_init_curses()

        if self._curses_ui is not None:
            # Curses path: render header and progress; drain logs
            total = max(1, self._total)
            percent = 100.0 if final else (self._counter / total) * 100.0
            try:
                self._curses_ui.render(self.table_map, self._counter, total, percent)
                self._curses_ui.drain_logs()
            except Exception:
                # If curses rendering fails mid-run, silently fallback to non-curses printing
                self.close()
                self._refresh_fallback()
            return

        # Fallback path using ANSI printing
        self._refresh_fallback()

    def _refresh_fallback(self):
        # Compute a column width consistent with the table box
        table_width = self._get_bar_width()
        column_width = max(1, (table_width - 2) // 2)
        if self.scroll_output is not None:
            # Inform ScrollOutput so it can close the box on the right
            try:
                self.scroll_output._set_column_width(column_width)
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
        return self.bar_length if self.bar_length else min(self.terminal_width - 10, 80)

    def print_progress_bar(self):
        # Build a progress line sized to the table's two-column width and close with right border
        counter = self._counter + 1
        if self._total - self._counter < self.update_rate:
            percent = 100
        else:
            percent = (self._counter / self._total) * 100

        table_width = self._get_bar_width()
        colw = max(1, (table_width - 2) // 2)
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
        content = content[: total_content]
        content = f"{content:<{total_content}}"

        # Split across two columns and add borders to match table width
        left = content[:colw]
        right = content[colw : colw * 2]
        progress_out.write(f"\r\x1b[2K│{left}│{right}│\n")
        progress_out.flush()
        self._line_count += 1

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
    progress_bar = ProgressBar(
        total=20, table_map=table_map, scroll_output=scroll_output
    )

    # with contextlib.redirect_stdout(scroll_output):
    with contextlib.redirect_stderr(scroll_output):
        counter = 0
        for _ in progress_bar:
            print(f"Simulated stdout message for iteration {counter}", file=sys.stderr)
            counter += 1
