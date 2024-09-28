"""
Persistent tracking state database
"""
from typing import List

import pandas as pd


class Player:
    def __init__(self, tracking_id: int) -> None:
        self.current_tracking_id: int = tracking_id
        self.other_tracking_ids: List[int] = []


class TrackingDatabase:

    def __init__(self) -> None:
        pass
