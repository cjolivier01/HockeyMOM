#!/bin/bash
PYTHONPATH=$(pwd) python scripts/hm_cuda_container.py --tag hm $@
