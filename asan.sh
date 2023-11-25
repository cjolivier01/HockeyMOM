#!/bin/bash
ASAN_OPTIONS=symbolize=1 \
  LD_PRELOAD=$(clang -print-file-name=libclang_rt.asan-x86_64.so) \
  ASAN_SYMBOLIZER_PATH="$(which llvm-symbolizer)" \
  OMP_NUM_THREADS=16 \
  PYTHONPATH=$(pwd):$(pwd)/models/mixsort \
  $@