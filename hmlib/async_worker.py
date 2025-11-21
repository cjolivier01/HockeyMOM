"""Base class for simple background worker threads with queues.

An :class:`AsyncWorker` owns a thread plus optional incoming / outgoing
queues and a callback hook for processed items.

@see @ref hmlib.utils.progress_bar.ProgressBar "ProgressBar" for integrating workers
     into rich CLI progress UIs.
"""

import queue
import threading
import traceback


class AsyncWorker:
    """Simple queued worker that executes work on a dedicated thread.

    Subclasses are expected to override :meth:`run`, :meth:`send` and
    :meth:`receive` to implement custom behavior.

    @param name: Descriptive worker name (used as the thread name).
    @param callback: Optional callable invoked for each delivered item.
    """

    def __init__(self, name: str, callback: callable = None, **kwargs):
        self._kwargs = kwargs
        self._name = name
        self._thread = None
        self._incoming_queue = None
        self._outgoing_queue = None
        self._callback = callback

    def start(self):
        self._incoming_queue = queue.Queue()
        if self._callback is None:
            self._outgoing_queue = queue.Queue()
        self._thread = threading.Thread(
            name=self._name,
            target=self.run,
        )
        self._thread.start()

    def on_before_stop_join(self):
        """Hook invoked before joining the worker thread.

        Override this in subclasses to abort blocking operations (e.g. I/O)
        prior to :meth:`stop` waiting on ``join()``.
        """
        pass

    def stop(self):
        if self._thread is not None:
            self._incoming_queue.put(None)
            self.on_before_stop_join()
            self._thread.join()
            self._thread = None
            self._incoming_queue = None
            self._outgoing_queue = None

    def run(self, **kwargs):
        """Worker entry point executed on the background thread.

        Subclasses must implement this method, typically looping on
        :meth:`receive` and calling :meth:`deliver_item` for results.
        """
        assert False and "Not implemented!"

    def send(**kwargs):
        """Enqueue work for the worker.

        Subclasses should implement this method and forward inputs into
        the worker's incoming queue.
        """
        assert False and "Not implemented!"

    def receive(**kwargs):
        """Receive an item from the worker.

        Subclasses should implement this method to return processed
        results or propagated exceptions from the worker thread.
        """
        assert False and "Not implemented!"

    def deliver_item(self, item):
        """Deliver a processed item via callback and/or outgoing queue.

        @param item: Result or exception produced by the worker.
        """
        if self._callback is not None:
            self._callback(item)
        if self._outgoing_queue is not None:
            self._outgoing_queue.put(item)

    def _run(self):
        """Internal wrapper that runs :meth:`run` and forwards exceptions."""
        try:
            self.run(**self._kwargs)
        except Exception as e:
            if not isinstance(e, StopIteration):
                print(e)
                traceback.print_exc()
            self._deliver_item(e)
        try:
            self._deliver_item(StopIteration())
        except:
            pass
