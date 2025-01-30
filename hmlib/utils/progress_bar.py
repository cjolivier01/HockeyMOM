from __future__ import annotations

import contextlib
import logging
import shutil
import sys
import time
from collections import OrderedDict
from typing import Any, Callable, Iterator, List, Optional

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
    """
    Class for maintaining a dynamioc text scrolling area
    """

    def __init__(self, lines=4):
        self.capture = []
        self.lines = lines

    def write(self, msg):
        if msg == "\n":
            return
        self.capture.append(msg)
        if len(self.capture) > self.lines:
            self.capture.pop(0)

    def flush(self):
        pass

    def reset(self):
        progress_out.write(f"\x1b[0G\x1b[{self.lines}A")
        progress_out.flush()

    def refresh(self):
        for i in reversed(range(self.lines)):
            cap_count = len(self.capture)
            if i < cap_count:
                line = self.capture[cap_count - i - 1]
                line.replace("\n", "\r")
            else:
                line = ""
            progress_out.write(f"\x1b[2K│ " + line + "\n")
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


class ProgressBar:
    def __init__(
        self,
        table_map: OrderedDict[Any, Any] = OrderedDict(),
        total: int = 0,
        iterator: Optional[Iterator[Any]] = None,
        scroll_output: Optional[ScrollOutput] = None,
        bar_length: Optional[int] = None,
        update_rate: int = 1,
        table_callback: Optional[Callable] = None,
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
                self.refresh()
                raise
        else:
            if self._counter >= self._total:
                self.refresh()
                raise StopIteration()

        if not self.bar_length and self._counter % self.terminal_width_interval == 0:
            self.terminal_width = _get_terminal_width()

        if self._counter % self.update_rate == 0:
            self._run_callbacks(table_map=self.table_map)
            self.refresh()
            if next_item is None:
                time.sleep(0.1)  # Simulating work by sleeping
        self._counter += 1
        return next_item

    def refresh(self):
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
        counter = self._counter + 1
        if self._total - self._counter < self.update_rate:
            # There won't be another update, so fill the bar up all of the way
            percent = 100
        else:
            percent = (self._counter / self._total) * 100
        bar_length = self._get_bar_width()
        filled_length = int(bar_length * self._counter // self._total)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        progress_out.write(
            f"\r\x1b[2KProgress: |{bar}| {percent:.1f}% Complete {self._counter}/{self._total}\n"
        )
        progress_out.flush()
        self._line_count += 1

    def print_table(self):
        self._line_count += write_dict_in_columns(
            self.table_map, progress_out, self._get_bar_width()
        )
        progress_out.flush()

    @contextlib.contextmanager
    def stdout_redirect(self):
        # with contextlib.redirect_stdout(self.original_stdout):
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
