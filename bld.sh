#!/bin/bash
if [ "$CONDA_DEFAULT_ENV" != "ubuntu" ]; then
  mamba activate ubuntu
fi
set -e
cd DCNv2
./make.sh
cd -
cd external/pt_autograph
python setup.py develop
cd -
cd external/fast_pytorch_kmeans
python setup.py develop
cd -
cd models/mixsort
python setup.py develop
cd -
