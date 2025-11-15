import unittest

import torch

from hmlib.aspen.trunks.tracker import TrackerTrunk


class TrackerTrunkConfigTest(unittest.TestCase):
    def test_default_tracker_class_instantiates_cpu_tracker(self):
        trunk = TrackerTrunk(enabled=True)
        trunk._ensure_tracker(image_size=torch.Size([1, 3, 720, 1280]))
        self.assertIsNotNone(trunk._hm_tracker)
        self.assertEqual(trunk.tracker_class, "hockeymom.core.HmTracker")
        self.assertEqual(trunk._hm_tracker.__class__.__name__, "HmTracker")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_tracker_class_instantiates_cuda_tracker(self):
        trunk = TrackerTrunk(
            enabled=True,
            tracker_class="hockeymom.core.HmByteTrackerCuda",
            tracker_kwargs={"device": "cuda:0"},
        )
        trunk._ensure_tracker(image_size=torch.Size([1, 3, 720, 1280]))
        self.assertIsNotNone(trunk._hm_tracker)
        self.assertEqual(trunk.tracker_class, "hockeymom.core.HmByteTrackerCuda")
        self.assertEqual(trunk._hm_tracker.__class__.__name__, "HmByteTrackerCuda")


if __name__ == "__main__":
    unittest.main()
