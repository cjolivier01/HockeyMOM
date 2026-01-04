from __future__ import annotations

import multiprocessing
import queue
import sys
import time
from threading import Condition, Lock
from typing import Any, Dict, Optional


class LLNode:
    def __init__(self, value: Any):
        self.value: Any = value
        self.next: LLNode | None = None


class LinkedList:
    def __init__(self) -> None:
        self.head: Optional[LLNode] = None
        self.tail: Optional[LLNode] = None
        self._size: int = 0
        self._mu = Lock()

    def push_back(self, value: int) -> None:
        """Add an element to the end of the list."""
        new_node = LLNode(value)
        with self._mu:
            if not self.head:
                self.head = new_node
                self.tail = new_node
            else:
                self.tail.next = new_node
                self.tail = new_node
            self._size += 1

    def push_front(self, value: Any) -> None:
        """Add an element to the beginning of the list."""
        new_node = LLNode(value)
        with self._mu:
            if not self.head:
                self.head = new_node
                self.tail = new_node
            else:
                new_node.next = self.head
                self.head = new_node
            self._size += 1

    def pop_back(self) -> Any:
        """Remove the last element from the list and return its value."""
        with self._mu:
            if not self.head:
                raise IndexError("pop from empty list")
            removed_value = self.tail.value
            if self.head == self.tail:
                self.head = None
                self.tail = None
            else:
                current = self.head
                while current.next != self.tail:
                    current = current.next
                current.next = None
                self.tail = current
            self._size -= 1
        return removed_value

    def pop_front(self) -> Any:
        """Remove the first element from the list and return its value."""
        with self._mu:
            if not self.head:
                raise IndexError("pop from empty list")
            removed_value = self.head.value
            self.head = self.head.next
            if not self.head:
                self.tail = None
            self._size -= 1
        return removed_value

    def __len__(self) -> int:
        return self._size

    def __iter__(self):
        with self._mu:
            current = self.head
            while current:
                yield current.value
                with self._mu:
                    current = current.next

    def __repr__(self) -> str:
        return "LinkedList([" + ", ".join(map(str, self)) + "])"


class SidebandQueue:
    def __init__(
        self,
        name: Optional[str] = None,
        warn_after: Optional[float] = None,
        repeat_warn: bool = False,
        max_size: int = -1,
    ):
        """Create a SidebandQueue.

        Args:
            name: Optional name for the queue used in printed warnings.
            warn_after: Optional default seconds to wait before printing a warning when blocking in `get()`.
            repeat_warn: If True, repeat the warning every `warn_after` seconds while still waiting.
            max_size: Maximum number of in-flight items allowed before `put()` blocks (-1 for infinite).
        """
        if max_size == 0 or max_size < -1:
            raise ValueError("max_size must be -1 (unbounded) or > 0")

        self._q = queue.Queue()
        self._counter = 1
        self._map: Dict[int, Any] = {}
        self._lock = Lock()
        self._not_full = Condition(self._lock)
        self._name = name or f"SidebandQueue@{id(self)}"
        self._warn_after_default = warn_after
        self._repeat_warn_default = repeat_warn
        self._max_size = max_size
        self._closed = False

    def put(
        self,
        obj: Any,
        block: bool = True,
        timeout: Optional[float] = None,
        warn_after: Optional[float] = None,
        repeat_warn: Optional[bool] = None,
    ):
        warn_after = self._warn_after_default if warn_after is None else warn_after
        repeat_warn = self._repeat_warn_default if repeat_warn is None else repeat_warn

        if self._max_size == -1:
            with self._lock:
                if self._closed:
                    raise ValueError("SidebandQueue is closed")
                ctr = self._counter
                self._counter += 1
                assert ctr not in self._map
                self._map[ctr] = obj
            self._q.put(ctr)
            return

        if not block:
            with self._not_full:
                if self._closed:
                    raise ValueError("SidebandQueue is closed")
                if len(self._map) >= self._max_size:
                    raise queue.Full
                ctr = self._counter
                self._counter += 1
                assert ctr not in self._map
                self._map[ctr] = obj
            self._q.put(ctr)
            return

        start: Optional[float] = time.time() if timeout is not None else None
        warned = False
        ctr: Optional[int] = None

        while ctr is None:
            with self._not_full:
                if self._closed:
                    raise ValueError("SidebandQueue is closed")
                if len(self._map) < self._max_size:
                    ctr = self._counter
                    self._counter += 1
                    assert ctr not in self._map
                    self._map[ctr] = obj
                    break

                if timeout is not None:
                    elapsed = time.time() - start if start is not None else 0.0
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        raise queue.Full
                    wait_for = remaining if warn_after is None else min(remaining, warn_after)
                    notified = self._not_full.wait(timeout=wait_for)
                else:
                    if warn_after is None:
                        self._not_full.wait()
                        notified = True
                    else:
                        if start is None:
                            start = time.time()
                        notified = self._not_full.wait(timeout=warn_after)

                if warn_after is not None and (not notified) and ((not warned) or repeat_warn):
                    print(
                        f"Warning: SidebandQueue '{self._name}' waiting >= {warn_after:.2f}s for free slot",
                        file=sys.stderr,
                    )
                    warned = True

        self._q.put(ctr)
        if warned:
            total_wait = time.time() - start if start is not None else 0.0
            print(
                f"SidebandQueue '{self._name}' resumed after waiting {total_wait:.2f}s to put item",
                file=sys.stderr,
            )

    def get(
        self,
        block: bool = True,
        timeout: Optional[float] = None,
        warn_after: Optional[float] = None,
        repeat_warn: Optional[bool] = None,
    ) -> Any:
        """Get next item from queue.

        Args:
            block: Whether to block waiting for an item.
            timeout: Optional total timeout in seconds for the get operation.
            warn_after: Optional per-queue or per-call seconds to wait before printing a warning.
            repeat_warn: If True, repeat the warning every `warn_after` seconds while still waiting.

        Behavior:
            - If `warn_after` is provided (or set on the queue), the method will poll in
              intervals of up to `warn_after` seconds. If a poll times out, a warning
              containing the queue name is printed. If `repeat_warn` is False the warning
              is printed only once; otherwise it is printed every time a `warn_after`
              segment elapses while still waiting.
            - If the get eventually succeeds after waiting, and at least one warning was
              issued, a resumed message is printed with the total waited time.
        """
        # Resolve defaults
        if warn_after is None:
            warn_after = self._warn_after_default
        if repeat_warn is None:
            repeat_warn = self._repeat_warn_default

        # Fast-path: no warning logic requested, just call underlying queue.
        if warn_after is None:
            ctr = self._q.get(block=block, timeout=timeout) if block else self._q.get_nowait()
            with self._lock:
                val = self._map.pop(ctr)
                if self._max_size > -1:
                    self._not_full.notify()
                return val

        # If not blocking, warning logic doesn't make sense; just try non-blocking get.
        if not block:
            ctr = self._q.get_nowait()
            with self._lock:
                return self._map.pop(ctr)

        start = time.time()
        warned = False
        try:
            while True:
                # compute remaining total timeout (if any)
                if timeout is not None:
                    elapsed = time.time() - start
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        raise queue.Empty
                    seg_timeout = min(warn_after, remaining)
                else:
                    seg_timeout = warn_after

                try:
                    ctr = self._q.get(block=True, timeout=seg_timeout)
                    with self._lock:
                        val = self._map.pop(ctr)
                        if self._max_size > -1:
                            self._not_full.notify()
                    if warned:
                        total_wait = time.time() - start
                        print(
                            f"SidebandQueue '{self._name}' resumed after waiting {total_wait:.2f}s",
                            file=sys.stderr,
                        )
                        warned = False
                    return val
                except queue.Empty:
                    # segment timed out without getting an item
                    if ((not warned) or repeat_warn) and seg_timeout >= warn_after:
                        print(
                            f"Warning: SidebandQueue '{self._name}' waiting >= {seg_timeout:.2f}s for item",
                            file=sys.stderr,
                        )
                        warned = True
                    # loop again (will respect overall timeout if provided)
        except queue.Empty:
            # propagate empty to caller
            raise

    def qsize(self):
        return len(self._map)

    def close(self) -> None:
        with self._not_full:
            self._closed = True
            self._not_full.notify_all()


class IterableQueue:
    def __init__(self, queue):
        self.queue = queue

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration()
        return item


def create_queue(
    mp: bool,
    name: Optional[str] = None,
    warn_after: Optional[float] = None,
    repeat_warn: bool = False,
    max_size: int = -1,
):
    if mp:
        assert False
        return multiprocessing.Queue()
    else:
        return SidebandQueue(
            name=name,
            warn_after=warn_after if warn_after is not None else 300.0,
            repeat_warn=repeat_warn,
            max_size=max_size,
        )
