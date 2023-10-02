#!/bin/bash
LD_PRELOAD=$(clang -print-file-name=libclang_rt.asan-x86_64.so) $@
