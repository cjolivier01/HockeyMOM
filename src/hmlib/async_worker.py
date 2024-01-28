import queue
import threading
import traceback


class AsyncWorker:
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
        """
        Called before join when stopping, allows the worker to abort any blocking operation
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
        # Implement in derived class
        assert False and "Not implemented!"

    def send(**kwargs):
        # Implement in derived class
        assert False and "Not implemented!"

    def receive(**kwargs):
        # Implement in derived class
        assert False and "Not implemented!"

    def deliver_item(self, item):
        if self._callback is not None:
            self._callback(item)
        if self._outgoing_queue is not None:
            self._outgoing_queue.put(item)

    def _run(self):
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
