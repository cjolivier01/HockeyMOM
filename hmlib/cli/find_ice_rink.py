import os
from pathlib import Path

import torch

from hmlib.hm_opts import hm_opts
from hmlib.segm.ice_rink import main
from hmlib.utils.gpu import GpuAllocator

if __name__ == "__main__":
    main()
