"""Helpers for raising exceptions across threads using CPython APIs.

These utilities are primarily used to interrupt long-running loops from a
worker thread by injecting a :class:`KeyboardInterrupt` into the main thread.

@see @ref hmlib.utils.progress_bar.ProgressBar "ProgressBar" for cooperative
     cancellation patterns in long-running CLIs.
"""

import ctypes
import threading
import time

from hmlib.log import get_logger


def get_main_thread_id():
    """Return the identifier of the Python main thread."""
    return threading.main_thread().ident


def raise_exception_in_thread(thread_id=get_main_thread_id(), exception=KeyboardInterrupt):
    """Raise an exception asynchronously in the given thread.

    @param thread_id: Target thread identifier (defaults to the main thread).
    @param exception: Exception type to inject (defaults to ``KeyboardInterrupt``).
    """
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_id), ctypes.py_object(exception)
    )
    if res == 0:
        raise ValueError("Invalid thread ID")
    elif res > 1:
        # Reset the exception in case of failure
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


if __name__ == "__main__":
    #
    # Example usage
    #

    # Worker thread to trigger the exception
    def worker():
        time.sleep(2)  # Allow main thread to run for a while
        main_thread_id = threading.main_thread().ident
        raise_exception_in_thread(main_thread_id, KeyboardInterrupt)

    logger = get_logger(__name__)

    # Main thread function
    def main():
        logger.info("Main thread started")
        try:
            while True:
                time.sleep(1)  # Simulate main thread workload
                logger.info("Main thread is running")
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt caught in main thread!")

    # Start the worker thread
    worker_thread = threading.Thread(target=worker)
    worker_thread.start()

    # Run the main function
    main()
