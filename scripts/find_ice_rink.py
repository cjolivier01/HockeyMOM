import os
from pathlib import Path

import torch

from hmlib.hm_opts import hm_opts
from hmlib.segm.ice_rink import main
from hmlib.utils.gpu import GpuAllocator

if __name__ == "__main__":
    opts = hm_opts()
    opts.parser.add_argument("--device", default="cuda:0", type=str, help="Device used for inference")
    args = opts.parse()

    this_path = Path(os.path.dirname(__file__))
    root_dir = os.path.realpath(this_path / ".." / ".." / "..")

    if "cuda" in args.device and ":" not in args.device:
        gpu_allocator = GpuAllocator(gpus=args.gpus)
        device: torch.device = (
            torch.device("cuda", gpu_allocator.allocate_fast())
            if not gpu_allocator.is_single_lowmem_gpu(low_threshold_mb=1024 * 10)
            else torch.device("cpu")
        )
    else:
        device = torch.device(args.device)
    main(args, device=device)
