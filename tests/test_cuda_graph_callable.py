import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def should_replay_cuda_graph_outputs():
    from hmlib.utils.cuda_graph import CudaGraphCallable

    def fn(x: torch.Tensor) -> torch.Tensor:
        return (x * 2) + 1

    x0 = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    cg = CudaGraphCallable(fn, (x0,), warmup=1, name="test_fn")

    y0 = cg(x0).detach().cpu()
    assert torch.allclose(y0, fn(x0).detach().cpu())

    x1 = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4) + 10
    y1 = cg(x1).detach().cpu()
    assert torch.allclose(y1, fn(x1).detach().cpu())
    assert cg.stats.captures == 1
    assert cg.stats.replays >= 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def should_recapture_on_signature_change():
    from hmlib.utils.cuda_graph import CudaGraphCallable

    def fn(x: torch.Tensor) -> torch.Tensor:
        return x + 3

    x0 = torch.zeros((2, 3), device="cuda", dtype=torch.float16)
    cg = CudaGraphCallable(fn, (x0,), warmup=0, name="sig_change")
    assert cg.stats.captures == 1

    _ = cg(x0)
    x1 = torch.zeros((2, 4), device="cuda", dtype=torch.float16)
    _ = cg(x1)
    assert cg.stats.captures == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def should_support_multiple_tensor_outputs():
    from hmlib.utils.cuda_graph import CudaGraphCallable

    def fn(x: torch.Tensor):
        return x + 1, x - 1

    x = torch.randn(8, device="cuda", dtype=torch.float32)
    cg = CudaGraphCallable(fn, (x,), warmup=0, name="multi_out")
    a, b = cg(x)
    assert torch.allclose(a, x + 1)
    assert torch.allclose(b, x - 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def should_prune_detections_static_with_fixed_outputs():
    from hmlib.tracking_utils.segm_boundaries import SegmBoundaries

    H, W = 64, 64
    mask = torch.ones((H, W), device="cuda", dtype=torch.bool)
    centroid = torch.tensor([W / 2, H / 2], device="cuda", dtype=torch.float32)
    segm = SegmBoundaries(segment_mask=mask, centroid=centroid, max_detections_in_mask=5)

    det_bboxes = torch.tensor(
        [
            [10, 10, 20, 20],
            [11, 11, 21, 21],
            [12, 12, 22, 22],
            [13, 13, 23, 23],
            [14, 14, 24, 24],
            [15, 15, 25, 25],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    labels = torch.arange(det_bboxes.shape[0], device="cuda", dtype=torch.int64)
    scores = torch.linspace(0.1, 0.6, det_bboxes.shape[0], device="cuda", dtype=torch.float32)

    out_b, out_l, out_s, num_valid = segm.prune_detections_static(det_bboxes, labels, scores)
    assert out_b.shape == (5, 4)
    assert out_l.shape == (5,)
    assert out_s.shape == (5,)
    assert isinstance(num_valid, torch.Tensor)
    assert int(num_valid.detach().cpu()) == 5
