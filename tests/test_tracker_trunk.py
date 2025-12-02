import unittest

import torch

from hmlib.aspen.plugins.tracker_plugin import TrackerPlugin


class TrackerPluginConfigTest(unittest.TestCase):
    def test_default_tracker_class_instantiates_cpu_tracker(self):
        trunk = TrackerPlugin(enabled=True)
        trunk._ensure_tracker(image_size=torch.Size([1, 3, 720, 1280]))
        self.assertIsNotNone(trunk._hm_tracker)
        self.assertEqual(trunk.tracker_class, "hockeymom.core.HmTracker")
        self.assertEqual(trunk._hm_tracker.__class__.__name__, "HmTracker")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_tracker_class_instantiates_cuda_tracker(self):
        trunk = TrackerPlugin(
            enabled=True,
            tracker_class="hockeymom.core.HmByteTrackerCuda",
            tracker_kwargs={"device": "cuda:0"},
        )
        trunk._ensure_tracker(image_size=torch.Size([1, 3, 720, 1280]))
        self.assertIsNotNone(trunk._hm_tracker)
        self.assertEqual(trunk.tracker_class, "hockeymom.core.HmByteTrackerCuda")
        self.assertEqual(trunk._hm_tracker.__class__.__name__, "HmByteTrackerCuda")

    def test_prepare_tracker_inputs_with_static_limits(self):
        trunk = TrackerPlugin(enabled=True)
        trunk._static_tracker_max_detections = 4
        trunk._static_tracker_max_tracks = 8
        bboxes = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        labels = torch.arange(5, dtype=torch.long)
        scores = torch.tensor([0.1, 0.9, 0.3, 0.5, 0.8], dtype=torch.float32)
        payload = trunk._prepare_tracker_inputs(3, bboxes, labels, scores)
        self.assertIn("num_detections", payload)
        self.assertEqual(int(payload["num_detections"].item()), 4)
        self.assertEqual(payload["bboxes"].shape, (4, 4))
        self.assertEqual(payload["labels"].shape[0], 4)
        self.assertTrue(torch.all(payload["bboxes"][0] == bboxes[1]))
        self.assertTrue(torch.all(payload["bboxes"][3] == bboxes[4]))


if __name__ == "__main__":
    unittest.main()
