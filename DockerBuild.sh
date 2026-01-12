#!/bin/bash
PYTHONPATH=$(pwd) python scripts/hm_cuda_container.py build --tag hm $@
