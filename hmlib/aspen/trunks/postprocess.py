from typing import Any, Dict


from .base import Trunk


class CamPostProcessTrunk(Trunk):
    """
    Feeds results into CamTrackHead to render/save video and produce outputs.

    Expects in context:
      - postprocessor: CamTrackHead instance
      - data: dict from MMTrackingTrunk/PoseTrunk
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    def forward(self, context: Dict[str, Any]):  # type: ignore[override]
        if not self.enabled:
            return {}
        postprocessor = context.get("postprocessor")
        if postprocessor is None:
            return {}
        data: Dict[str, Any] = context.get("data", {})
        postprocessor.process_tracking(results=data, context=context)
        return {}

    def input_keys(self):
        return {"postprocessor", "data", "rink_profile"}

    def output_keys(self):
        return set()
