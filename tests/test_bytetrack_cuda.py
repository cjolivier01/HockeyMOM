import unittest

import torch

from hockeymom.core import HmByteTrackConfig, HmTracker, HmByteTrackerCuda


def _make_data(device: torch.device, frame_id: int, boxes, labels, scores):
    return {
        "frame_id": torch.tensor([frame_id], dtype=torch.long, device=device),
        "bboxes": torch.tensor(boxes, dtype=torch.float32, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "scores": torch.tensor(scores, dtype=torch.float32, device=device),
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class ByteTrackCudaTest(unittest.TestCase):
    def test_cuda_tracker_matches_cpu(self):
        config = HmByteTrackConfig()
        cpu_tracker = HmTracker(config)
        gpu_tracker = HmByteTrackerCuda(config, device="cuda:0")

        frames = [
            (
                [[10.0, 10.0, 30.0, 40.0], [100.0, 100.0, 140.0, 160.0]],
                [1, 1],
                [0.9, 0.85],
            ),
            (
                [[12.0, 12.0, 32.0, 42.0], [103.0, 103.0, 143.0, 163.0]],
                [1, 1],
                [0.92, 0.8],
            ),
        ]

        for frame_id, (boxes, labels, scores) in enumerate(frames):
            cpu_data = _make_data(torch.device("cpu"), frame_id, boxes, labels, scores)
            gpu_data = _make_data(torch.device("cuda"), frame_id, boxes, labels, scores)

            cpu_res = cpu_tracker.track(cpu_data.copy())
            gpu_res = gpu_tracker.track(gpu_data)

            self.assertTrue(torch.equal(cpu_res["ids"], gpu_res["ids"].cpu()))
            self.assertTrue(torch.allclose(cpu_res["bboxes"], gpu_res["bboxes"].cpu(), atol=1e-3))
            self.assertTrue(torch.allclose(cpu_res["scores"], gpu_res["scores"].cpu(), atol=1e-3))


if __name__ == "__main__":
    unittest.main()
