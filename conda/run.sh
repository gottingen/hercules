#!/bin/bash
set -e

export CONDA_BLD_PATH=build

if [ -d "build" ]; then
    rm -rf build
fi

mkdir build
conda build .