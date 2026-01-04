import time
from threading import Lock, Thread
from typing import Optional

from .containers import create_queue


class SimpleCachedIterator:
    """Single-threaded lookahead iterator with a small in-memory cache.

    This implementation is kept for compatibility but does not overlap
    producer and consumer work; for performance-sensitive paths the
    threaded variant below is preferred.
    """

    def __init__(
        self,
        iterator,
        cache_size: int = 2,
        pre_callback_fn: callable = None,
        name: Optional[str] = None,
    ):
        self._iterator = iterator
        self._pre_callback_fn = pre_callback_fn
        self._eof_reached = False
        self._stopped = False
        self._cache_size = cache_size
        self._cache_primed: bool = False
        self._prime_cache_lock = Lock()
        self._sentinel = object()

        if cache_size:
            self._q = create_queue(mp=False, name="SimpleCachedIterator" if name is None else name)
        else:
            self._q = None
            return

    def _prime_cache(self):
        if not self._cache_size or self._cache_primed:
            return
        with self._prime_cache_lock:
            if self._cache_primed:
                return
            self._cache_primed = True
            for _ in range(self._cache_size):
                try:
                    item = next(self._iterator)
                    if self._pre_callback_fn is not None:
                        item = self._pre_callback_fn(item)
                    self._q.put(item)
                except StopIteration:
                    self._eof_reached = True
                    self._q.put(self._sentinel)
                    break

    def __iter__(self):
        return self

    def __next__(self):
        self._prime_cache()
        if self._q is None:
            result_item = next(self._iterator)
            if self._pre_callback_fn is not None:
                result_item = self._pre_callback_fn(result_item)
            return result_item

        if self._stopped:
            raise StopIteration

        result_item = self._q.get()
        if result_item is self._sentinel:
            self._stopped = True
            raise StopIteration
        if isinstance(result_item, Exception):
            self._stopped = True
            raise result_item

        try:
            if not self._eof_reached:
                cached_item = next(self._iterator)
                if self._pre_callback_fn is not None:
                    cached_item = self._pre_callback_fn(cached_item)
                self._q.put(cached_item)
        except StopIteration:
            self._eof_reached = True
            self._q.put(self._sentinel)
        except Exception as ex:
            self._eof_reached = True
            self._q.put(ex)
            self._q.put(self._sentinel)

        return result_item


class ThreadedCachedIterator:
    """Background-prefetching iterator with a bounded cache.

    Items are pulled from the wrapped iterator on a worker thread and
    delivered to the consumer via an internal queue. This allows I/O-
    bound workloads (video decode, preprocessing) to overlap with GPU
    compute in the hmtrack pipelines.
    """

    def __init__(
        self,
        iterator,
        cache_size: int = 2,
        pre_callback_fn: callable = None,
        name: Optional[str] = None,
    ):
        self._iterator = iterator
        self._pre_callback_fn = pre_callback_fn
        self._cache_size = max(0, int(cache_size or 0))
        self._stopped = False

        self._q = None
        self._pull_queue_to_worker = None
        self._pull_thread: Optional[Thread] = None
        self._name = name or "ThreadedCachedIterator"

        if self._cache_size <= 0:
            return

        self._q = create_queue(mp=False, name=self._name)
        self._pull_queue_to_worker = create_queue(mp=False, name=f"{self._name}.ctrl")
        self._pull_thread = Thread(target=self._pull_worker, name=self._name)
        self._pull_thread.daemon = True

        self._cache_primed: bool = False
        self._prime_cache_lock = Lock()
        self._pull_thread.start()

    def _prime_cache(self):
        with self._prime_cache_lock:
            if self._cache_primed:
                return
            self._cache_primed = True
            for _ in range(self._cache_size):
                self._pull_queue_to_worker.put("ok")

    def _pull_worker(self):
        try:
            self._prime_cache()
            while True:
                msg = self._pull_queue_to_worker.get()
                if msg is None:
                    # Shutdown signal from main thread.
                    if self._q is not None:
                        self._q.put(StopIteration())
                    break

                try:
                    item = next(self._iterator)
                except StopIteration as ex:
                    if self._q is not None:
                        self._q.put(ex)
                    break
                except Exception as ex:
                    if self._q is not None:
                        self._q.put(ex)
                    break

                if self._pre_callback_fn is not None:
                    try:
                        item = self._pre_callback_fn(item)
                    except Exception as ex:
                        if self._q is not None:
                            self._q.put(ex)
                        break
                if self._q is not None:
                    self._q.put(item)
        finally:
            return

    def _stop(self):
        if self._pull_thread is None:
            return
        try:
            if self._pull_queue_to_worker is not None:
                self._pull_queue_to_worker.put(None)
            self._pull_thread.join(timeout=1.0)
        except Exception:
            pass
        finally:
            self._pull_thread = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._cache_size <= 0 or self._q is None:
            result_item = next(self._iterator)
            if isinstance(result_item, Exception):
                raise result_item
            if self._pre_callback_fn is not None:
                result_item = self._pre_callback_fn(result_item)
            return result_item

        if self._stopped:
            raise StopIteration

        start_wait = time.time()
        result_item = self._q.get()
        wait_time = time.time() - start_wait
        if wait_time > 0.01:
            # Optional debug hook; keep silent in production.
            # print(f"ThreadedCachedIterator waited {wait_time*1000:.2f} ms")
            pass

        if isinstance(result_item, StopIteration):
            self._stopped = True
            self._stop()
            raise StopIteration

        if isinstance(result_item, Exception):
            self._stopped = True
            self._stop()
            raise result_item

        # Request the next item to keep the cache warm.
        if self._pull_queue_to_worker is not None:
            self._pull_queue_to_worker.put("ok")

        return result_item

    def __del__(self):
        self._stop()


CachedIterator = SimpleCachedIterator
