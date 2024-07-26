import sys
import time
from typing import Optional
import contextlib
from itertools import cycle


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
        sys.stderr.write(f"\x1b[0G\x1b[{self.lines}A")
        sys.stderr.flush()

    def refresh(self):
mv         for i in reversed(range(self.lines)):
            cap_count = len(self.capture)
            if i < cap_count:
                line = self.capture[cap_count - i - 1]
                line.replace("\n", "\r")
            else:
                line = ""
            sys.stderr.write(f"\x1b[2K{i}: " + line + "\n")
        # sys.stderr.write(f"\x1b[0G\x1b[{self.lines}A")
        # sys.stderr.flush()


class ProgressBar:
    def __init__(self, total, table_map, scroll_output: Optional[ScrollOutput] = None):
        self.total = total
        self.counter = 0
        self.table_map = table_map
        self.original_stdout = sys.stdout
        self.scroll_output = scroll_output

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter >= self.total:
            raise StopIteration
        self.counter += 1
        if self.counter > 1:
            sys.stderr.write(f"\x1b[0G\x1b[{self._line_count}A")
        self._line_count = 0
        if self.scroll_output is not None:
            self.scroll_output.reset()
        self.print_table()
        self.print_progress_bar()
        if self.scroll_output is not None:
            self.scroll_output.refresh()
        time.sleep(0.1)  # Simulating work by sleeping

    def print_progress_bar(self):
        percent = (self.counter / self.total) * 100
        bar_length = 30
        filled_length = int(bar_length * self.counter // self.total)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        sys.stderr.write(f"\r\x1b[2KProgress: |{bar}| {percent:.1f}% Complete\n")
        sys.stderr.flush()
        self._line_count += 1

    def print_table(self):
        rows = min(3, len(self.table_map))
        sys.stderr.write("\x1b[2K----------------------------\n")
        self._line_count += 1
        keys = list(self.table_map.keys())
        for i in range(rows):
            key = keys[i]
            value = self.table_map[key]
            sys.stderr.write(f"\x1b[2K{key}: {value}\n")
            self._line_count += 1
        sys.stderr.write("\x1b[2K----------------------------\n")
        sys.stderr.flush()
        self._line_count += 1

    @contextlib.contextmanager
    def stdout_redirect(self):
        with contextlib.redirect_stdout(self.original_stdout):
            yield


# Example usage
if __name__ == "__main__":
    table_map = {"Key1": "Value1", "Key2": "Value2", "Key3": "Value3"}
    # Redirect stdout to scroll output handler
    scroll_output = ScrollOutput()
    progress_bar = ProgressBar(
        total=20, table_map=table_map, scroll_output=scroll_output
    )

    with contextlib.redirect_stdout(scroll_output):
        counter = 0
        for _ in progress_bar:
            print(f"Simulated stdout message for iteration {counter}")
            counter += 1
