# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import time
import contextlib

from .log import logger


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

        self.duration = 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0
        self.duration = 0.0


class TimeTracker:
    def __init__(
        self,
        name: str,
        timer: Timer,
        print_interval: int = 20,
        batch_size: int = 1,
        reset_each_print: bool = True,
    ):
        self._timer = timer
        self.batch_size = batch_size
        self.name = name
        self._reset_each_print = reset_each_print
        if not hasattr(self._timer, "entry_count"):
            self._timer.entry_count = 0
            self._timer.print_interval = print_interval

    def __enter__(self):
        self._timer.entry_count += 1
        self._timer.tic()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._timer.toc()
        if self._timer.entry_count % self._timer.print_interval == 0:
            logger.info(
                "{} ({:.2f} fps)".format(
                    self.name,
                    self.batch_size * 1.0 / max(1e-5, self._timer.average_time),
                )
            )
            if self._reset_each_print:
                self._timer.clear()
