import tkinter as tk
from threading import Thread
from typing import Optional


class ControlledApp:
    def __init__(self, root):
        self.root = root if root is not None else tk.Tk()
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.stop)  # Handle window close button

    def stop(self):
        self.running = False
        self.root.destroy()  # This ensures the window closes properly

    def run(self):
        while self.running:
            self.root.update_idletasks()
            self.root.update()


_ROOT: Optional[ControlledApp] = None


def get_tk_root():
    global _ROOT
    if _ROOT is None:
        _ROOT = ControlledApp(root=None)
    thread = Thread(target=_ROOT.run)
    return _ROOT.root


def stop_tk():
    global _ROOT
    if _ROOT is not None:
        _ROOT.stop()
