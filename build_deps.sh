#!/bin/bash
cd external/hugin
bazelisk build //:install_tree -- --prefix=$CONDA_PREFIX

