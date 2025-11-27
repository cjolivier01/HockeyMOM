import time
from threading import Thread
from typing import Optional

from .containers import create_queue


class SimpleCachedIterator:

    def __init__(self, iterator, cache_size: int = 2, pre_callback_fn: callable = None, name: Optional[str] = None):
        self._iterator = iterator
        if cache_size:
            self._q = create_queue(mp=False, name="SimpleCachedIterator" if name is None else name)
        else:
            self._q = None
        self._pre_callback_fn = pre_callback_fn
        self._eof_reached = False
        self._stopped = False
        for _ in range(cache_size):
            try:
                item = next(self._iterator)
                if self._pre_callback_fn is not None:
                    item = self._pre_callback_fn(item)
                self._q.put(item)
            except StopIteration:
                self._eof_reached = True
                self._q.put(None)
                break

    def __iter__(self):
        return self

    def __next__(self):
        if self._q is None:
            result_item = next(self._iterator)
            assert result_item is not None
            if self._pre_callback_fn is not None:
                result_item = self._pre_callback_fn(result_item)
        else:
            assert not self._stopped
            result_item = self._q.get()
            if result_item is None:
                self._stopped = True
                raise StopIteration
            if isinstance(result_item, Exception):
                self._stopped = True
                raise result_item
            try:
                if not self._eof_reached:
                    cached_item = next(self._iterator)
                    assert cached_item is not None
                    if self._pre_callback_fn is not None:
                        cached_item = self._pre_callback_fn(cached_item)
                    self._q.put(cached_item)
            except StopIteration:
                self._eof_reached = True
                self._q.put(None)
            except Exception as ex:
                self._eof_reached = True
                self._q.put(ex)
                # should not be necessary
                self._q.put(None)
        return result_item


class ThreadedCachedIterator:
    def __init__(self, iterator, cache_size: int = 2, pre_callback_fn: callable = None):
        self._iterator = iterator
        self._q = create_queue(mp=False) if cache_size else None
        self._pre_callback_fn = pre_callback_fn
        self._save_cache_size = cache_size
        self._pull_queue_to_worker = create_queue(mp=False)
        self._eof_reached = False
        self._pull_thread = Thread(target=self._pull_worker)
        for i in range(cache_size):
            self._pull_queue_to_worker.put("ok")

        # Finally, start the worker thread
        self._pull_thread.start()

    def _pull_worker(self):
        try:
            while True:
                msg = self._pull_queue_to_worker.get()
                if msg is None:
                    self._q.put(StopIeration())
                    break
                item = next(self._iterator)
                if self._pre_callback_fn is not None:
                    item = self._pre_callback_fn(item)
                self._q.put(item)
        except StopIteration as ex:
            self._q.put(ex)
            return
        except Exception as ex:
            print(ex)
            print(f"ThreadedCachedIterator exiting due to exception: {ex}")
            self._q.put(ex)

    def _stop(self):
        if self._pull_thread is not None:
            self._pull_queue_to_worker.put(None)
            self._pull_thread.join()
            self._pull_thread = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._q is None:
            result_item = next(self._iterator)
            assert result_item is not None
            if isinstane(result_item, Exception):
                raise result_item
            if self._pre_callback_fn is not None:
                result_item = self._pre_callback_fn(result_item)
        else:
            get_next_cached_start = time.time()
            t0 = time.time()
            result_item = self._q.get()
            assert result_item is not None
            if isinstance(result_item, Exception):
                raise result_item
            get_next_cached_duration = time.time() - get_next_cached_start
            if get_next_cached_duration > 10 / 1000:
                # print(
                #     f"Waited {get_next_cached_duration * 1000} ms "
                #     f"for the next cached item for cache size of {self._save_cache_size}"
                # )
                pass
            self._pull_queue_to_worker.put("ok")
        return result_item

    def __del__(self):
        self._stop()


# CachedIterator = ThreadedCachedIterator
CachedIterator = SimpleCachedIterator
