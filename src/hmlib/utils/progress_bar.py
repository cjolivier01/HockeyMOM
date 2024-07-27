import sys
import time
import shutil
from typing import Any, Callable, Dict, Iterator, List, Optional
import contextlib
import logging
from itertools import cycle
from collections import OrderedDict


progress_out = sys.stderr
logging_out = sys.stdout


def _get_terminal_width():
    width = shutil.get_terminal_size().columns
    return width


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
    def __init__(self, lines=4):
        self.capture = []
        self.lines = lines

    def write(self, msg):
        if msg == "\n":
            return
        self.capture.append(msg)
        if len(self.capture) > self.lines:
            self.capture.pop(0)
        # self.refresh()

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
            progress_out.write(f"\x1b[2K{i}: " + line + "\n")
        # progress_out.write(f"\x1b[0G\x1b[{self.lines}A")
        progress_out.flush()

    def my_callback(self, message):
        # Here you handle the message. For now, we'll just print it.
        # print(f"Callback received log: {message}", file=sys.stderr)
        self.write(message)

    def register_logger(self, logger):

        # Remove all existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create and add our custom handler
        callback_handler = CallbackStreamHandler(self.my_callback)
        callback_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(callback_handler)


class ProgressBar:
    def __init__(
        self,
        table_map: OrderedDict[Any, Any] = OrderedDict(),
        total: Optional[int] = 0,
        iterator: Optional[Iterator[Any]] = None,
        scroll_output: Optional[ScrollOutput] = None,
        bar_length: Optional[int] = None,
        update_rate: int = 1,
        table_callback: Optional[Callable] = None,
    ):
        self.total = total
        self.counter = 0
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

    def _run_callbacks(self, table_map: OrderedDict[Any, Any]):
        for cb in self.table_callbacks:
            cb(table_map)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = None
        if self.iterator is not None:
            next_item = next(self.iterator)
        else:
            if self.counter >= self.total:
                raise StopIteration

        if not self.bar_length and self.counter % self.terminal_width_interval == 0:
            self.terminal_width = _get_terminal_width()

        if self.counter % self.update_rate == 0:
            self._run_callbacks(table_map=self.table_map)

            if self.counter > 0:
                if self.scroll_output is not None:
                    self.scroll_output.reset()
                progress_out.write(f"\x1b[0G\x1b[{self._line_count}A")

            self._line_count = 0
            self.print_table()
            self.print_progress_bar()
            if self.scroll_output is not None:
                self.scroll_output.refresh()
            if next_item is None:
                time.sleep(0.1)  # Simulating work by sleeping
        self.counter += 1
        return next_item

    def _get_bar_width(self):
        return self.bar_length if self.bar_length else min(self.terminal_width - 10, 80)

    def print_progress_bar(self):
        if self.total - self.counter < self.update_rate:
            # There won't be another update, so fill the bar up all of the way
            percent = 100
        else:
            percent = (self.counter / self.total) * 100
        bar_length = self._get_bar_width()
        filled_length = int(bar_length * self.counter // self.total)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        progress_out.write(
            f"\r\x1b[2KProgress: |{bar}| {percent:.1f}% Complete {self.counter}/{self.total}\n"
        )
        progress_out.flush()
        self._line_count += 1

    def print_table(self):
        rows = min(3, len(self.table_map))
        progress_out.write("\x1b[2K____________________________\n")
        self._line_count += 1
        if rows:
            keys = list(self.table_map.keys())
            for i in range(rows):
                key = keys[i]
                value = self.table_map[key]
                progress_out.write(f"\x1b[2K{key}: {value}\n")
                self._line_count += 1
            progress_out.write("\x1b[2K____________________________\n")
        progress_out.flush()
        self._line_count += 1

    @contextlib.contextmanager
    def stdout_redirect(self):
        # with contextlib.redirect_stdout(self.original_stdout):
        with contextlib.redirect_stdout(self.scroll_output):
            yield


# Example usage
if __name__ == "__main__":
    table_map = {"Key1": "Value1", "Key2": "Value2", "Key3": "Value3"}
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
